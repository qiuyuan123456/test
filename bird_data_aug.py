#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Value-grounding data augmentation for Text-to-SQL (BIRD/SQLite), LLM-assisted rewriting.

Augmentation strategies:
  A1 - Value substitution: replace SQL filter literals with real DB values,
       then rewrite the NL question (and evidence if it mentions old values).
  A2 - Question paraphrase: keep SQL unchanged, rewrite question with synonyms/rephrasing.
  B1 - Evidence paraphrase: paraphrase evidence while preserving all domain anchors.

Key design points:
  - SQL rewriting uses sqlglot AST node replacement (precise, no accidental global replace).
  - Table aliases are resolved to real table names before value sampling.
  - BETWEEN sampling supports semantic modes: free / superset / subset / near.
  - Checkpoint file enables resume after crash without reprocessing.
  - API calls use exponential-backoff retry to survive rate-limits and transient errors.
  - Output is appended incrementally so partial results are never lost.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp
from sqlalchemy import create_engine

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("Missing dependency: openai. Install with `pip install openai`") from e

try:
    _possible_paths = [
        os.path.join(os.path.dirname(__file__), "../QBridge_new/src"),
        os.path.join(os.path.dirname(__file__), "../../QBridge_new/src"),
        "/data1/lzs/QBridge_new/src",
    ]
    HAS_SCHEMA_ENGINE = False
    for _p in _possible_paths:
        if os.path.exists(_p):
            sys.path.insert(0, _p)
            try:
                from schema_engine_sqlite import SchemaEngine  # type: ignore
                HAS_SCHEMA_ENGINE = True
                break
            except ImportError:
                continue
    if not HAS_SCHEMA_ENGINE:
        print("[WARNING] SchemaEngine not found, falling back to simple schema format.")
except Exception:
    HAS_SCHEMA_ENGINE = False


# ---------------------------------------------------------
# I/O
# ---------------------------------------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    """Append-mode write: safe to call incrementally after partial runs."""
    if not items:
        return
    with open(path, "a", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------------------------------------
# Checkpoint: resume support
# ---------------------------------------------------------

def _ckpt_path(output_path: str) -> str:
    base, _ = os.path.splitext(output_path)
    return base + ".ckpt"


def load_checkpoint(output_path: str) -> Set[str]:
    """Return set of already-processed keys formatted as '{question_id}:{aug_type}'."""
    path = _ckpt_path(output_path)
    if not os.path.exists(path):
        return set()
    seen: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seen.add(line)
    return seen


def save_checkpoint_entry(output_path: str, question_id: Any, aug_type: str) -> None:
    with open(_ckpt_path(output_path), "a", encoding="utf-8") as f:
        f.write(f"{question_id}:{aug_type}\n")


# ---------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------

def connect_sqlite(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def safe_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def resolve_db_path(db_root: str, db_id: str) -> str:
    p1 = os.path.join(db_root, db_id, f"{db_id}.sqlite")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(db_root, f"{db_id}.sqlite")
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"Cannot find sqlite for db_id={db_id} under db_root={db_root}")


def get_schema_text(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    tables = [r["name"] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()]
    lines: List[str] = ["Tables:"]
    for t in tables:
        cols = cur.execute(f"PRAGMA table_info({safe_ident(t)})").fetchall()
        lines.append(f"- {t}({', '.join(c['name'] for c in cols)})")
    fk_lines: List[str] = []
    for t in tables:
        fks = cur.execute(f"PRAGMA foreign_key_list({safe_ident(t)})").fetchall()
        for fk in fks:
            fk_lines.append(f"- {t}.{fk['from']} -> {fk['table']}.{fk['to']}")
    if fk_lines:
        lines.append("Foreign Keys:")
        lines.extend(fk_lines)
    return "\n".join(lines)


def get_mschema_str(db_path: str, db_id: str) -> str:
    if HAS_SCHEMA_ENGINE:
        try:
            db_engine = create_engine(f"sqlite:///{db_path}")
            se = SchemaEngine(engine=db_engine, schema="main", db_name=db_id)  # type: ignore
            return se.mschema.to_mschema(selected_tables=None, selected_columns=None)
        except Exception as e:
            print(f"[WARNING] mschema failed for {db_id}: {e}, falling back to simple schema")
    conn = connect_sqlite(db_path)
    result = get_schema_text(conn)
    conn.close()
    return result


def exec_sql(conn: sqlite3.Connection, sql: str) -> Tuple[bool, Optional[int], Optional[str]]:
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return True, len(rows), None
    except Exception as e:
        return False, None, str(e)


def find_candidate_tables_for_column(conn: sqlite3.Connection, column: str) -> List[str]:
    cur = conn.cursor()
    tables = [r["name"] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()]
    target = str(column).strip('"`[]').lower()
    out: List[str] = []
    for t in tables:
        cols = [str(r["name"]).strip('"`[]').lower()
                for r in cur.execute(f"PRAGMA table_info({safe_ident(t)})").fetchall()]
        if target in cols:
            out.append(t)
    return out


def sample_values_by_freq(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    limit_pool: int = 200,
    k: int = 16,
) -> List[Tuple[Any, int]]:
    """
    Frequency-weighted sampling: common values appear more often, but rare values
    can still be selected. This produces more "natural" substitutions than uniform
    random sampling while avoiding overfit to the most-frequent value.
    """
    cur = conn.cursor()
    rows = cur.execute(f"""
        SELECT {safe_ident(column)} AS v, COUNT(*) AS cnt
        FROM {safe_ident(table)}
        WHERE {safe_ident(column)} IS NOT NULL
        GROUP BY {safe_ident(column)}
        ORDER BY cnt DESC
        LIMIT {int(limit_pool)}
    """).fetchall()

    pool = [(r["v"], int(r["cnt"])) for r in rows if r["v"] is not None]
    if not pool:
        return []

    values  = [p[0] for p in pool]
    weights = [max(1, p[1]) for p in pool]
    freq_map: Dict[Any, int] = {v: c for v, c in pool}  # O(1) lookup avoids O(n^2) scan

    out: List[Tuple[Any, int]] = []
    seen: Set[Any] = set()
    for _ in range(min(k * 4, len(values) * 3)):
        v = random.choices(values, weights=weights, k=1)[0]
        if v in seen:
            continue
        seen.add(v)
        out.append((v, freq_map[v]))
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------
# SQL AST: alias resolution / binding extraction / precise rewrite
# ---------------------------------------------------------

def normalize_sql(sql: str) -> str:
    try:
        return sqlglot.parse_one(sql, read="sqlite").sql(dialect="sqlite")
    except Exception:
        return re.sub(r"\s+", " ", sql.strip())


def build_alias_map(tree: exp.Expression) -> Dict[str, str]:
    """Map alias -> real table name so we sample from the correct table."""
    alias_map: Dict[str, str] = {}

    def record(t: exp.Expression) -> None:
        if isinstance(t, exp.Table) and t.name:
            alias_map[t.alias_or_name] = t.name

    from_expr = tree.args.get("from")
    if from_expr:
        for e in from_expr.find_all(exp.Table):
            record(e)
    for j in tree.find_all(exp.Join):
        if isinstance(j.this, exp.Table):
            record(j.this)
    return alias_map


@dataclass
class Binding:
    resolved_table: Optional[str]
    column: str
    op: str                          # '=', '!=', '>', '>=', '<', '<=', 'IN', 'BETWEEN'
    old_values: List[Any]
    is_string: bool
    literal_nodes: List[exp.Literal] # direct AST node refs for surgical replacement


def extract_bindings(sql: str) -> Tuple[Optional[exp.Expression], List[Binding]]:
    """
    Parse SQL and collect filter bindings with direct AST literal node references.
    We store node ids so rewrite_sql_by_binding can do a surgical swap without
    touching the rest of the tree.
    """
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None, []

    alias_map = build_alias_map(tree)
    bindings: List[Binding] = []

    def resolve_table(col: exp.Column) -> Optional[str]:
        if not col.table:
            return None

        t = col.table.name if hasattr(col.table, "name") else str(col.table)
        t = t.strip('"`[]')

        # 兼容大小写别名
        return (
            alias_map.get(t)
            or alias_map.get(t.lower())
            or alias_map.get(t.upper())
            or None   # 注意：不要回退成 t，否则会把别名当真实表
        )

    # Single-literal binary predicates
    # (A1 originally focused on '=', now expanded to include common comparison operators)
    binary_ops: List[Tuple[Any, str]] = [
        (exp.EQ, "="),
        (exp.NEQ, "!="),
        (exp.GT, ">"),
        (exp.GTE, ">="),
        (exp.LT, "<"),
        (exp.LTE, "<="),
    ]
    for op_cls, op_name in binary_ops:
        for node in tree.find_all(op_cls):
            l, r = node.left, node.right
            if isinstance(l, exp.Column) and isinstance(r, exp.Literal):
                bindings.append(Binding(resolve_table(l), l.name, op_name, [r.this], r.is_string, [r]))
            elif isinstance(r, exp.Column) and isinstance(l, exp.Literal):
                bindings.append(Binding(resolve_table(r), r.name, op_name, [l.this], l.is_string, [l]))

    for node in tree.find_all(exp.In):
        if isinstance(node.this, exp.Column):
            lits: List[exp.Literal] = []
            vals: List[Any] = []
            is_str = False
            ok = True
            for e in node.expressions:
                if not isinstance(e, exp.Literal):
                    ok = False
                    break
                lits.append(e)
                vals.append(e.this)
                is_str = is_str or e.is_string
            if ok and lits:
                bindings.append(Binding(
                    resolve_table(node.this), node.this.name, "IN", vals, is_str, lits))

    for node in tree.find_all(exp.Between):
        if isinstance(node.this, exp.Column):
            low  = node.args.get("low")
            high = node.args.get("high")
            if isinstance(low, exp.Literal) and isinstance(high, exp.Literal):
                bindings.append(Binding(
                    resolve_table(node.this), node.this.name, "BETWEEN",
                    [low.this, high.this], low.is_string or high.is_string, [low, high],
                ))

    return tree, bindings


def make_literal(value: Any, is_string: bool) -> exp.Expression:
    if value is None:
        return exp.Null()
    return exp.Literal.string(str(value)) if is_string else exp.Literal.number(str(value))


def rewrite_sql_by_binding(tree: exp.Expression, binding: Binding, new_values: List[Any]) -> str:
    """
    Replace exactly the literal nodes referenced by this binding using node-id matching.
    tree.transform() returns a deep copy, so the original AST is unchanged and we can
    safely reuse it for other bindings within the same item.
    """
    if len(binding.literal_nodes) != len(new_values):
        return tree.sql(dialect="sqlite")

    idx_map = {id(n): make_literal(v, binding.is_string)
               for n, v in zip(binding.literal_nodes, new_values)}
    new_tree = tree.transform(lambda node: idx_map.get(id(node), node))
    return new_tree.sql(dialect="sqlite")


# ---------------------------------------------------------
# Range sampling for BETWEEN
# ---------------------------------------------------------

def is_floatable(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def pick_between_values(
    pool: List[Tuple[Any, int]],
    old_low: Any,
    old_high: Any,
    mode: str,
) -> Optional[List[Any]]:
    """
    Semantic range modes for BETWEEN substitution:
      free     - any two values from pool (no constraint)
      superset - new range contains the original range (tests wider coverage)
      subset   - new range is within the original range (tests narrower filtering)
      near     - new range stays within +/-50% of original range width (locality-preserving)
    Only numeric ranges are supported; string BETWEEN falls back to the caller.
    """
    vals = [v for v, _ in pool]
    if len(vals) < 2 or not (is_floatable(old_low) and is_floatable(old_high)):
        return None

    old_l, old_h = float(old_low), float(old_high)
    if old_l > old_h:
        old_l, old_h = old_h, old_l

    nums = sorted([(v, float(v)) for v in vals if is_floatable(v)], key=lambda t: t[1])
    if len(nums) < 2:
        return None

    def ordered(a: Any, b: Any) -> List[Any]:
        try:
            return [a, b] if float(a) <= float(b) else [b, a]
        except Exception:
            return [a, b]

    if mode == "free":
        a, b = random.sample(nums, 2)
        return ordered(a[0], b[0])

    if mode == "superset":
        lows  = [v for v, fv in nums if fv <= old_l]
        highs = [v for v, fv in nums if fv >= old_h]
        if not lows or not highs:
            return None
        return [random.choice(lows), random.choice(highs)]

    if mode == "subset":
        lows  = [v for v, fv in nums if fv >= old_l]
        highs = [v for v, fv in nums if fv <= old_h]
        if not lows or not highs:
            return None
        return ordered(random.choice(lows), random.choice(highs))

    if mode == "near":
        w = max(1e-9, (old_h - old_l) * 0.5)
        candidates = [v for v, fv in nums if (old_l - w) <= fv <= (old_h + w)]
        if len(candidates) < 2:
            return None
        a, b = random.sample(candidates, 2)
        return ordered(a, b)

    return None


# ---------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------

A1_SYSTEM = (
    "You are a data augmentation assistant for Text-to-SQL (SQLite). "
    "Your job is to align the natural language question (and optionally the evidence) with an updated SQL "
    "where only the filter literal values have changed. "
    "Keep SQL structure, tables, joins, aggregation, and selected columns unchanged. "
    "Output JSON only."
)

A1_USER_TMPL = """Schema:
{schema_text}

Original question:
{question}

Original evidence (may be empty):
{evidence}

Original SQL (SQLite):
{sql}

Updated SQL (only literal values changed, structure unchanged):
{new_sql}

Replaced literals:
{replaced_json}

Task:
1. Rewrite the question to match the updated SQL. Keep it natural and fluent.
2. If the evidence is non-empty AND contains any of the old literal values, update those occurrences to the new values. Otherwise keep the evidence exactly unchanged.
Do NOT mention schema names explicitly in the question.
Return ONE-LINE JSON only:
{{"new_question":"...", "new_evidence":"..."}}"""

A2_SYSTEM = (
    "You are a data augmentation assistant for Text-to-SQL (SQLite). "
    "Rewrite the natural language question using paraphrases and synonymous expressions "
    "while keeping the intent exactly the same as the SQL. "
    "Do NOT change the SQL. Output JSON only."
)

A2_USER_TMPL = """Schema:
{schema_text}

Original question:
{question}

Gold SQL (SQLite, do not change):
{sql}

Task:
Rewrite the question with different wording (paraphrase / synonyms / abbreviation expansion), keeping meaning fully consistent with the SQL filters and constraints.
Return ONE-LINE JSON only:
{{"new_question":"..."}}"""

VER_SYSTEM = (
    "You are a strict verifier for Text-to-SQL data augmentation. "
    "Decide whether the question matches the SQL intent and constraints. "
    "Output JSON only."
)

VER_USER_TMPL = """Schema:
{schema_text}

Question:
{question}

SQL (SQLite):
{sql}

Criteria:
- PASS only if the question implies the same constraints and literal values as the SQL (common alias/expansion allowed).
- FAIL if ambiguous or mismatched.
Return ONE-LINE JSON only:
{{"verdict":"PASS" or "FAIL","reason":"one short sentence"}}"""

B1_SYSTEM = (
    "You are a data augmentation assistant for Text-to-SQL (SQLite). "
    "Paraphrase evidence text while preserving ALL anchors. "
    "Anchors include: table names, column names, quoted strings, numbers, dates, codes, "
    "backtick-quoted terms, and formulas. "
    "Output JSON only."
)

B1_USER_TMPL = """Schema:
{schema_text}

Evidence:
{evidence}

Task:
Produce ONE paraphrased version that is semantically equivalent and more concise, keeping ALL anchors unchanged.
Return ONE-LINE JSON only:
{{"new_evidence":"..."}}"""

# Dedicated verifier for B1: checks semantic equivalence of paraphrased evidence.
B1_VER_SYSTEM = (
    "You are a strict verifier for evidence paraphrase in Text-to-SQL. "
    "Decide whether the paraphrased evidence is fully semantically equivalent to the original. "
    "Output JSON only."
)

B1_VER_USER_TMPL = """Original evidence:
{original_evidence}

Paraphrased evidence:
{new_evidence}

Criteria:
- PASS if all domain anchors are preserved and semantics are fully equivalent.
- FAIL if any value, name, formula, or domain rule has been changed or lost.
Return ONE-LINE JSON only:
{{"verdict":"PASS" or "FAIL","reason":"one short sentence"}}"""


# ---------------------------------------------------------
# LLM call + exponential-backoff retry + JSON extraction
# ---------------------------------------------------------

def extract_first_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    # Incremental scan: find first { ... } pair that parses cleanly
    for s in [m.start() for m in re.finditer(r"\{", text)]:
        for e in range(len(text), s + 1, -1):
            if text[e - 1] != "}":
                continue
            try:
                return json.loads(text[s:e])
            except Exception:
                continue
    raise ValueError("No valid JSON object found in LLM output")


def llm_json(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 5,
) -> Dict[str, Any]:
    """
    Call LLM with exponential-backoff retry.
    Jitter is added to avoid thundering-herd when multiple workers hit the same rate limit.
    Raises on persistent failure so callers can decide whether to skip or abort.
    """
    delay = 2.0
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            return extract_first_json_object(text)
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                jitter = random.uniform(0, delay * 0.3)
                time.sleep(delay + jitter)
                delay = min(delay * 2, 60.0)
    raise last_exc


def _replace_literal_mentions(text: str, old_values: List[Any], new_values: List[Any]) -> str:
    """
    Best-effort textual replacement used as a fallback when A1 LLM rewriting fails.
    Replaces quoted and unquoted literal mentions; if no match exists, text is unchanged.
    """
    out = text
    for old_v, new_v in zip(old_values, new_values):
        old_s = str(old_v)
        new_s = str(new_v)
        # Replace quoted forms first
        out = out.replace(f"'{old_s}'", f"'{new_s}'")
        out = out.replace(f"\"{old_s}\"", f"\"{new_s}\"")
        # Then replace standalone occurrences (number/code tokens)
        out = re.sub(rf"(?<!\w){re.escape(old_s)}(?!\w)", new_s, out)
    return out


def verify_alignment(
    client: OpenAI,
    model: str,
    schema_text: str,
    question: str,
    sql: str,
) -> Tuple[bool, str]:
    user = VER_USER_TMPL.format(schema_text=schema_text, question=question, sql=sql)
    try:
        out = llm_json(client, model, VER_SYSTEM, user, temperature=0.0, max_tokens=128)
        return str(out.get("verdict", "")).strip().upper() == "PASS", str(out.get("reason", ""))
    except Exception as e:
        return False, f"verifier_error: {e}"


def verify_b1(
    client: OpenAI,
    model: str,
    original_ev: str,
    new_ev: str,
) -> Tuple[bool, str]:
    user = B1_VER_USER_TMPL.format(original_evidence=original_ev, new_evidence=new_ev)
    try:
        out = llm_json(client, model, B1_VER_SYSTEM, user, temperature=0.0, max_tokens=128)
        return str(out.get("verdict", "")).strip().upper() == "PASS", str(out.get("reason", ""))
    except Exception as e:
        return False, f"verifier_error: {e}"


# ---------------------------------------------------------
# Augmentation A1: Value substitution
# ---------------------------------------------------------

def do_A1_for_item(
    client: OpenAI,
    model: str,
    conn: sqlite3.Connection,
    schema_text: str,
    item: Dict[str, Any],
    max_aug_per_item: int,
    require_non_empty: bool,
    verify: bool,
    sleep_s: float,
    range_mode: str,
    allow_string_between: bool,
    global_seen_sql: Set[str],
) -> List[Dict[str, Any]]:
    sql = item["query"]
    q   = item["question"]
    ev  = (item.get("evidence") or "").strip()

    tree, bindings = extract_bindings(sql)
    if tree is None or not bindings:
        return []

    # Randomize binding order so different items explore different substitutions
    random.shuffle(bindings)
    out: List[Dict[str, Any]] = []

    for b in bindings:
        if len(out) >= max_aug_per_item:
            break

        fallback_tables = find_candidate_tables_for_column(conn, b.column)
        if b.resolved_table:
            # Even when alias resolution gives a table, keep fallback candidates.
            # Complex SQL/subqueries can produce imperfect table resolution.
            table_candidates = [b.resolved_table] + [t for t in fallback_tables if t != b.resolved_table]
        else:
            table_candidates = fallback_tables
        if not table_candidates:
            continue
        random.shuffle(table_candidates)

        table = None
        pool: List[Tuple[Any, int]] = []
        for t in table_candidates:
            sampled = sample_values_by_freq(conn, t, b.column, k=24)
            if sampled:
                table = t
                pool = sampled
                break
        if table is None or not pool:
            continue

        for _ in range(16):
            if len(out) >= max_aug_per_item:
                break

            if b.op in {"=", "!=", ">", ">=", "<", "<="}:
                new_v = random.choice([v for v, _ in pool])
                if str(new_v) == str(b.old_values[0]):
                    continue
                new_values = [new_v]

            elif b.op == "IN":
                if len(pool) < len(b.old_values):
                    continue
                new_values = [v for v, _ in random.sample(pool, k=len(b.old_values))]

            elif b.op == "BETWEEN":
                if b.is_string and not allow_string_between:
                    continue
                new_values = pick_between_values(pool, b.old_values[0], b.old_values[1], range_mode)
                if new_values is None:
                    continue
            else:
                continue

            new_sql = rewrite_sql_by_binding(tree, b, new_values)
            norm = normalize_sql(new_sql).upper()
            if norm == normalize_sql(sql).upper():
                continue
            # Cross-item dedup: skip if this exact SQL was already generated for another item
            if norm in global_seen_sql:
                continue

            ok, rowcount, _ = exec_sql(conn, new_sql)
            if not ok or (require_non_empty and rowcount == 0):
                continue

            replaced = {
                "column":     f"{table}.{b.column}",
                "op":         b.op,
                "old_values": [str(x) for x in b.old_values],
                "new_values": [str(x) for x in new_values],
            }

            user_prompt = A1_USER_TMPL.format(
                schema_text=schema_text,
                question=q,
                evidence=ev,
                sql=sql,
                new_sql=new_sql,
                replaced_json=json.dumps(replaced, ensure_ascii=False),
            )

            try:
                gen    = llm_json(client, model, A1_SYSTEM, user_prompt, temperature=0.2, max_tokens=300)
                new_q  = str(gen.get("new_question", "")).strip()
                # Keep LLM-updated evidence if provided; fall back to original otherwise
                new_ev = str(gen.get("new_evidence", ev)).strip() or ev

                if not new_q:
                    # Fallback: if model returns malformed/empty new_question, keep question text
                    # and only do deterministic literal mention replacement.
                    new_q = _replace_literal_mentions(q, b.old_values, new_values)
                    new_ev = _replace_literal_mentions(new_ev, b.old_values, new_values) if ev else new_ev

                if verify:
                    v_ok, v_reason = verify_alignment(client, model, schema_text, new_q, new_sql)
                    if not v_ok:
                        continue
                else:
                    v_reason = ""

                global_seen_sql.add(norm)

                aug = dict(item)
                aug["question"]           = new_q
                aug["query"]              = new_sql
                if ev:
                    aug["evidence"]       = new_ev
                aug["aug_type"]           = "A1_value_substitution"
                aug["source_question_id"] = item.get("question_id")
                aug["replaced"]           = replaced
                aug["exec_rowcount"]      = rowcount
                if verify:
                    aug["verifier_reason"] = v_reason

                out.append(aug)
                if sleep_s > 0:
                    time.sleep(sleep_s)

            except Exception:
                # Fallback path: keep augmentation even if A1 rewriting call fails.
                # This avoids all-A1-zero runs due to transient API issues.
                new_q = _replace_literal_mentions(q, b.old_values, new_values)
                new_ev = _replace_literal_mentions(ev, b.old_values, new_values) if ev else ev
                if verify:
                    v_ok, v_reason = verify_alignment(client, model, schema_text, new_q, new_sql)
                    if not v_ok:
                        continue
                else:
                    v_reason = ""

                global_seen_sql.add(norm)
                aug = dict(item)
                aug["question"]           = new_q
                aug["query"]              = new_sql
                if ev:
                    aug["evidence"]       = new_ev
                aug["aug_type"]           = "A1_value_substitution"
                aug["source_question_id"] = item.get("question_id")
                aug["replaced"]           = replaced
                aug["exec_rowcount"]      = rowcount
                if verify:
                    aug["verifier_reason"] = v_reason
                out.append(aug)
                if sleep_s > 0:
                    time.sleep(sleep_s)

    return out


# ---------------------------------------------------------
# Augmentation A2: Question paraphrase
# ---------------------------------------------------------

def do_A2_for_item(
    client: OpenAI,
    model: str,
    schema_text: str,
    item: Dict[str, Any],
    verify: bool,
    sleep_s: float,
    n_per_item: int = 1,
    max_attempts: int = 5,
) -> List[Dict[str, Any]]:
    sql = item["query"]
    q   = item["question"]
    out: List[Dict[str, Any]] = []
    seen_q: Set[str] = {q.lower().strip()}

    for _ in range(n_per_item * max_attempts):
        if len(out) >= n_per_item:
            break

        user_prompt = A2_USER_TMPL.format(schema_text=schema_text, question=q, sql=sql)
        try:
            gen   = llm_json(client, model, A2_SYSTEM, user_prompt, temperature=0.7, max_tokens=256)
            new_q = str(gen.get("new_question", "")).strip()

            if not new_q or new_q.lower().strip() in seen_q:
                continue
            if len(new_q) < max(8, int(len(q) * 0.5)):
                continue

            if verify:
                v_ok, v_reason = verify_alignment(client, model, schema_text, new_q, sql)
                if not v_ok:
                    continue
            else:
                v_reason = ""

            seen_q.add(new_q.lower().strip())
            aug = dict(item)
            aug["question"]           = new_q
            aug["aug_type"]           = "A2_question_paraphrase"
            aug["source_question_id"] = item.get("question_id")
            if verify:
                aug["verifier_reason"] = v_reason
            out.append(aug)
            if sleep_s > 0:
                time.sleep(sleep_s)

        except Exception:
            continue

    return out


# ---------------------------------------------------------
# Augmentation B1: Evidence paraphrase
# ---------------------------------------------------------

def do_B1_for_item(
    client: OpenAI,
    model: str,
    schema_text: str,
    item: Dict[str, Any],
    verify: bool,
    sleep_s: float,
) -> List[Dict[str, Any]]:
    evidence = (item.get("evidence") or "").strip()
    if not evidence:
        return []

    user_prompt = B1_USER_TMPL.format(schema_text=schema_text, evidence=evidence)
    try:
        gen    = llm_json(client, model, B1_SYSTEM, user_prompt, temperature=0.5, max_tokens=256)
        new_ev = str(gen.get("new_evidence", "")).strip()

        if not new_ev or new_ev.lower() == evidence.lower():
            return []

        if verify:
            # Dedicated evidence equivalence check rather than the question-SQL verifier
            v_ok, v_reason = verify_b1(client, model, evidence, new_ev)
            if not v_ok:
                return []
        else:
            v_reason = ""

        aug = dict(item)
        aug["evidence"]           = new_ev
        aug["aug_type"]           = "B1_evidence_paraphrase"
        aug["source_question_id"] = item.get("question_id")
        if verify:
            aug["verifier_reason"] = v_reason
        if sleep_s > 0:
            time.sleep(sleep_s)
        return [aug]

    except Exception:
        return []


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="BIRD Text-to-SQL data augmentation")
    ap.add_argument("--input_jsonl",  required=True,  help="Input JSONL path")
    ap.add_argument("--output_jsonl", required=True,  help="Output JSONL path (append mode)")
    ap.add_argument("--db_root",      required=True,  help="Root directory containing SQLite databases")

    ap.add_argument("--model",    default="deepseek-v3.2")
    ap.add_argument("--base_url", default="",  help="OpenAI-compatible API base URL")
    ap.add_argument("--api_key",  default="",  help="API key (falls back to OPENAI_API_KEY env var)")
    ap.add_argument("--seed",     type=int, default=13)

    ap.add_argument("--do_A1", action="store_true", help="Value substitution augmentation")
    ap.add_argument("--do_A2", action="store_true", help="Question paraphrase augmentation")
    ap.add_argument("--do_B1", action="store_true", help="Evidence paraphrase augmentation")

    ap.add_argument("--max_aug_per_item",  type=int, default=4, help="Max A1 augmentations per item")
    ap.add_argument("--a2_per_item",       type=int, default=1, help="Max A2 paraphrases per item")
    ap.add_argument("--require_non_empty", action="store_true", help="Skip A1 results that return 0 rows")
    ap.add_argument("--verify",            action="store_true", help="Use LLM verifier to filter outputs")
    ap.add_argument("--include_original",  action="store_true", help="Write original items to output first")

    ap.add_argument("--range_mode", default="free", choices=["free", "superset", "subset", "near"],
                    help="Semantic mode for BETWEEN value sampling")
    ap.add_argument("--allow_string_between", action="store_true",
                    help="Allow BETWEEN substitution on string literals (not recommended)")

    ap.add_argument("--sleep_s",   type=float, default=0.0, help="Sleep between LLM calls (seconds)")
    ap.add_argument("--max_items", type=int,   default=0,   help="Process at most N items (0=all)")
    ap.add_argument("--resume",    action="store_true",
                    help="Resume from checkpoint if output already exists")

    args = ap.parse_args()
    random.seed(args.seed)

    client_kwargs: Dict[str, Any] = {}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    client = OpenAI(**client_kwargs)

    items = read_jsonl(args.input_jsonl)

    # Normalize BIRD's uppercase "SQL" field to the canonical "query" key
    for it in items:
        if "db_id" not in it or "question" not in it:
            raise ValueError(f"Missing db_id or question in record: {it}")
        if "query" not in it and "SQL" not in it:
            raise ValueError(f"Missing 'query' or 'SQL' field in record: {it}")
        if "SQL" in it and "query" not in it:
            it["query"] = it["SQL"]

    if args.max_items > 0:
        items = items[:args.max_items]

    # Checkpoint: determine which (question_id, aug_type) pairs were already completed
    done_keys: Set[str] = set()
    if args.resume:
        done_keys = load_checkpoint(args.output_jsonl)
        if done_keys:
            print(f"[Resume] {len(done_keys)} (item, aug_type) pairs already done, skipping.")
    elif os.path.exists(args.output_jsonl):
        # Fresh run: clear stale output and checkpoint from a previous attempt
        os.remove(args.output_jsonl)
        ck = _ckpt_path(args.output_jsonl)
        if os.path.exists(ck):
            os.remove(ck)

    if args.include_original:
        write_jsonl(args.output_jsonl, items)

    # Global SQL dedup across all items prevents duplicate augmentations
    global_seen_sql: Set[str] = set()

    # Group by db_id to open each SQLite file only once
    by_db: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        by_db.setdefault(it["db_id"], []).append(it)

    total_aug = 0
    aug_type_counts: Dict[str, int] = {}

    for db_id, group in tqdm(list(by_db.items()), desc="Databases"):
        db_path     = resolve_db_path(args.db_root, db_id)
        conn        = connect_sqlite(db_path)
        schema_text = get_mschema_str(db_path, db_id)

        for it in tqdm(group, desc=f"  {db_id}", leave=False):
            qid = it.get("question_id", id(it))

            if args.do_A1:
                key = f"{qid}:A1"
                if key not in done_keys:
                    results = do_A1_for_item(
                        client=client, model=args.model, conn=conn,
                        schema_text=schema_text, item=it,
                        max_aug_per_item=args.max_aug_per_item,
                        require_non_empty=args.require_non_empty,
                        verify=args.verify, sleep_s=args.sleep_s,
                        range_mode=args.range_mode,
                        allow_string_between=args.allow_string_between,
                        global_seen_sql=global_seen_sql,
                    )
                    append_jsonl(args.output_jsonl, results)
                    save_checkpoint_entry(args.output_jsonl, qid, "A1")
                    total_aug += len(results)
                    aug_type_counts["A1_value_substitution"] = (
                        aug_type_counts.get("A1_value_substitution", 0) + len(results))

            if args.do_A2:
                key = f"{qid}:A2"
                if key not in done_keys:
                    results = do_A2_for_item(
                        client=client, model=args.model,
                        schema_text=schema_text, item=it,
                        verify=args.verify, sleep_s=args.sleep_s,
                        n_per_item=args.a2_per_item,
                    )
                    append_jsonl(args.output_jsonl, results)
                    save_checkpoint_entry(args.output_jsonl, qid, "A2")
                    total_aug += len(results)
                    aug_type_counts["A2_question_paraphrase"] = (
                        aug_type_counts.get("A2_question_paraphrase", 0) + len(results))

            if args.do_B1:
                key = f"{qid}:B1"
                if key not in done_keys:
                    results = do_B1_for_item(
                        client=client, model=args.model,
                        schema_text=schema_text, item=it,
                        verify=args.verify, sleep_s=args.sleep_s,
                    )
                    append_jsonl(args.output_jsonl, results)
                    save_checkpoint_entry(args.output_jsonl, qid, "B1")
                    total_aug += len(results)
                    aug_type_counts["B1_evidence_paraphrase"] = (
                        aug_type_counts.get("B1_evidence_paraphrase", 0) + len(results))

        conn.close()

    n_items = len(items)
    print(f"\n[统计]")
    print(f"输入样本数  : {n_items}")
    print(f"新增增强数  : {total_aug}")
    if n_items:
        print(f"平均增强倍数: {total_aug / n_items:.2f}")
    print(f"\n增强类型分布:")
    for t, c in aug_type_counts.items():
        print(f"  {t}: {c}")
    if args.verify:
        print(f"\n(已通过 LLM verifier 过滤)")
    print(f"\n[完成] 结果已追加写入: {args.output_jsonl}")
    print(f"[断点文件] {_ckpt_path(args.output_jsonl)}")


if __name__ == "__main__":
    main()

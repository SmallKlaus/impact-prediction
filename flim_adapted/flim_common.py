"""
flim_common.py — Shared utilities for the FLIM baseline
====
Java method extraction, token normalization, metric helpers, JSONL IO.
Metric helpers are copied verbatim from diagnose_ce.py so numbers are
computed identically across baselines.
"""
from __future__ import annotations
import json, math, re
from collections import defaultdict
from pathlib import Path

# ── Token normalization (camelCase / snake_case aware) ────
_CAMEL_1 = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_2 = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_NONWORD = re.compile(r"[^A-Za-z0-9]+")

_STOP = {
    "the","a","an","and","or","of","to","in","for","is","are","be","on","with",
    "this","that","it","as","we","by","if","then","else","at","from","not",
    "public","private","protected","static","final","void","return","new",
    "class","int","long","string","boolean","null","true","false","import",
    "package","throws","extends","implements",
}

def normalize_tokens(text: str) -> list[str]:
    text = _CAMEL_1.sub(" ", _CAMEL_2.sub(" ", text or ""))
    toks = [t.lower() for t in _NONWORD.split(text)]
    return [t for t in toks if len(t) > 1 and t not in _STOP and not t.isdigit()]

def tokens_as_text(text: str) -> str:
    return " ".join(normalize_tokens(text))

# ── Java method extraction ────
_KEYWORDS = {"if","for","while","switch","catch","synchronized","return",
             "new","do","try","else","case","throw","assert","super","this"}

METHOD_HEADER_RE = re.compile(
    r"(?:(?:public|protected|private|static|final|synchronized|abstract|"
    r"default|native|strictfp)\s+)*"
    r"(?:<[^<>]{0,80}>\s*)?"
    r"[\w$.\[\]<>,?\s]{1,120}?\s"
    r"([\w$]+)\s*\("
)

def _find_block_end(src: str, open_idx: int) -> int:
    """Index of the '}' matching the '{' at open_idx (comment/string aware)."""
    depth, i, n = 0, open_idx, len(src)
    in_s = in_c = line_c = block_c = False
    while i < n:
        ch  = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if line_c:
            if ch == "\n": line_c = False
        elif block_c:
            if ch == "*" and nxt == "/": block_c = False; i += 1
        elif in_s:
            if ch == "\\": i += 1
            elif ch == '"': in_s = False
        elif in_c:
            if ch == "\\": i += 1
            elif ch == "'": in_c = False
        else:
            if   ch == "/" and nxt == "/": line_c  = True; i += 1
            elif ch == "/" and nxt == "*": block_c = True; i += 1
            elif ch == '"': in_s = True
            elif ch == "'": in_c = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: return i
        i += 1
    return -1

def extract_java_methods(source: str, max_functions: int = 50,
                         min_chars: int = 30) -> list[str]:
    """Extract method bodies (header included). Best-effort regex+brace parser."""
    methods, pos = [], 0
    while len(methods) < max_functions:
        m = METHOD_HEADER_RE.search(source, pos)
        if not m: break
        name = m.group(1)
        if name in _KEYWORDS:
            pos = m.end(); continue
        # find matching ')' of the parameter list
        i, depth = m.end() - 1, 0
        while i < len(source):
            if source[i] == "(": depth += 1
            elif source[i] == ")":
                depth -= 1
                if depth == 0: break
            i += 1
        if i >= len(source): break
        # skip 'throws ...' then expect '{' (skip abstract/interface ';')
        j = i + 1
        while j < len(source) and source[j] not in "{;":
            j += 1
        if j >= len(source) or source[j] == ";":
            pos = m.end(); continue
        end = _find_block_end(source, j)
        if end == -1:
            pos = m.end(); continue
        body = source[m.start():end + 1].strip()
        if len(body) >= min_chars:
            methods.append(body)
        pos = end + 1
    return methods

def window_chunks(source: str, window_tokens: int = 200,
                  max_functions: int = 50) -> list[str]:
    words = (source or "").split()
    out = [" ".join(words[s:s + window_tokens])
           for s in range(0, len(words), window_tokens)]
    return [w for w in out if w][:max_functions]

def extract_functions(class_text: str, cfg: dict) -> list[str]:
    """Segment a file into 'functions' per FLIM. Falls back to fixed windows."""
    mode   = cfg.get("mode", "java")
    max_f  = cfg.get("max_functions", 50)
    funcs: list[str] = []
    if mode == "java":
        funcs = extract_java_methods(class_text, max_f,
                                     cfg.get("min_function_chars", 30))
    if cfg.get("include_file_header", True):
        header = " ".join((class_text or "").split()[: cfg.get("window_tokens", 200)])
        if header: funcs = [header] + funcs
    if len(funcs) <= (1 if cfg.get("include_file_header", True) else 0):
        funcs = window_chunks(class_text, cfg.get("window_tokens", 200), max_f)
    return funcs[:max_f] if funcs else [""]

# ── JSONL ────
def load_jsonl_by_issue(path: Path) -> dict[str, list[dict]]:
    by_issue: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: rec = json.loads(line)
            except json.JSONDecodeError: continue
            jid = rec.get("jira_id", "")
            if jid: by_issue[jid].append(rec)
    return dict(by_issue)

# ── Metrics (identical to diagnose_ce.py) ────
def recall_at_k(scores, labels, k):
    if not any(labels): return 0.0
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    return sum(lbl for _, lbl in ranked[:k]) / sum(labels)

def ndcg_at_k(scores, labels, k):
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    dcg  = sum(lbl / math.log2(i + 2) for i, (_, lbl) in enumerate(ranked[:k]))
    idcg = sum(lbl / math.log2(i + 2)
               for i, lbl in enumerate(sorted(labels, reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0

def average_precision(scores, labels):
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos, cum, ap = 0, 0, 0.0
    for i, (_, lbl) in enumerate(ranked):
        if lbl:
            cum += 1; n_pos += 1; ap += cum / (i + 1)
    return ap / n_pos if n_pos else 0.0

def mean_reciprocal_rank(scores, labels):
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    for i, (_, lbl) in enumerate(ranked):
        if lbl: return 1.0 / (i + 1)
    return 0.0

def aggregate(values):
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "n": 0}
    n = len(values); mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return {"mean": round(mean, 4), "min": round(min(values), 4),
            "max": round(max(values), 4), "std": round(std, 4), "n": n}

def project_of(jira_id: str) -> str:
    prefix  = jira_id.split("-")[0].upper()
    mapping = {"FLINK": "flink", "KAFKA": "kafka", "HADOOP": "hadoop_common",
               "HDFS": "hdfs", "MAPREDUCE": "mapreduce", "YARN": "yarn"}
    return mapping.get(prefix, prefix.lower())

FEATURE_COLUMNS = [
    "f_sem_max", "f_sem_mean", "f_sem_top3", "f_sem_header",
    "f_lex_tfidf", "f_lex_id", "f_cls_name", "f_path_overlap",
]
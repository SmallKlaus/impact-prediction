"""
flim_common.py — Shared utilities for the FLIM baseline
====
Java method extraction, token normalization, metric helpers, JSONL IO.
Metric helpers are copied verbatim from diagnose_ce.py so numbers are
computed identically across baselines.
"""
from __future__ import annotations
import json, math, re, sqlite3
from collections import defaultdict
from pathlib import Path

# ── javalang availability (pure-Python AST parser for Java) ────
# Install: pip install javalang
# If unavailable, extract_java_methods_ast falls back to the regex extractor.
_JAVALANG_AVAILABLE = False
try:
    import javalang as _javalang  # type: ignore
    _JAVALANG_AVAILABLE = True
except ImportError:
    pass

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

def _build_line_offsets(source: str) -> list[int]:
    """Return a list where offsets[i] = char index of the start of line i (0-indexed)."""
    offsets = []
    pos = 0
    for line in source.splitlines(keepends=True):
        offsets.append(pos)
        pos += len(line)
    return offsets


def extract_java_methods_ast(source: str, max_functions: int = 50,
                              min_chars: int = 30) -> list[str]:
    """
    Extract method bodies using javalang's AST parser.

    Algorithm
    ---------
    1. Parse source to AST with javalang.
    2. Build a line→char-offset map (needed because javalang positions are
       line/column, not absolute character indices).
    3. For each MethodDeclaration / ConstructorDeclaration with a body,
       convert its start position to a char offset, then use _find_block_end
       to locate the closing brace.
    4. Slice [start : end+1] from the raw source — this preserves every token
       (modifiers, generics, annotations, Javadoc *if it immediately precedes
       the declaration position*) exactly as the original FLIM AST extractor did.

    Falls back to extract_java_methods (regex) on any parse error so the
    pipeline never crashes on unusual Java syntax (records, sealed classes,
    text blocks in Java ≥ 14).
    """
    if not _JAVALANG_AVAILABLE:
        return extract_java_methods(source, max_functions, min_chars)

    try:
        tree = _javalang.parse.parse(source)
    except Exception:
        # Unsupported syntax, encoding issues, etc.
        return extract_java_methods(source, max_functions, min_chars)

    line_offsets = _build_line_offsets(source)
    src_len = len(source)
    methods: list[str] = []

    for _, node in tree:
        if not isinstance(
            node,
            (_javalang.tree.MethodDeclaration,
             _javalang.tree.ConstructorDeclaration),
        ):
            continue
        if node.position is None:
            continue
        # Abstract methods and interface default stubs have no body
        if getattr(node, "body", None) is None:
            continue

        ln, col = node.position.line, node.position.column
        if ln - 1 >= len(line_offsets):
            continue
        start = line_offsets[ln - 1] + (col - 1)
        if start >= src_len:
            continue

        # Find the opening brace of the method body from the declaration start
        brace = source.find("{", start)
        if brace == -1:
            continue
        end = _find_block_end(source, brace)
        if end == -1:
            continue

        body = source[start : end + 1].strip()
        if len(body) >= min_chars:
            methods.append(body)
        if len(methods) >= max_functions:
            break

    if not methods:
        # javalang parsed OK but no methods found (empty class, pure interface, …)
        return extract_java_methods(source, max_functions, min_chars)

    return methods


# ── Source cache helpers (SQLite) ────
# The cache is built locally by build_source_cache.py and transferred to the
# server.  All server-side scripts look up raw Java source via these helpers;
# if the cache is absent or a key is missing, they fall back to class_text.

def open_source_cache(cache_path: str | Path | None) -> sqlite3.Connection | None:
    """
    Open the SQLite source cache and return a connection.
    Returns None if cache_path is None or empty — downstream code treats
    None as "no cache available, use class_text summaries".

    check_same_thread=False: the connection is opened in the main process
    and used only there (DataLoader workers do not call extract_functions).
    """
    if not cache_path:
        return None
    try:
        conn = sqlite3.connect(str(cache_path), check_same_thread=False)
        # Smoke-test: make sure the expected table exists
        conn.execute("SELECT 1 FROM source_cache LIMIT 1")
        return conn
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Could not open source cache at %s: %s — falling back to class_text",
            cache_path, exc,
        )
        return None


def lookup_source(cache_conn: sqlite3.Connection | None,
                  sha: str, class_path: str) -> str | None:
    """
    Return raw Java source for (sha, class_path) from the SQLite cache.
    Returns None on any miss (cache absent, key not found, status != 'ok').
    """
    if cache_conn is None or not sha or not class_path:
        return None
    try:
        row = cache_conn.execute(
            "SELECT source, status FROM source_cache"
            " WHERE sha=? AND class_path=?",
            (sha, class_path),
        ).fetchone()
        if row is not None and row[1] == "ok" and row[0]:
            return row[0]
    except Exception:
        pass
    return None


def window_chunks(source: str, window_tokens: int = 200,
                  max_functions: int = 50) -> list[str]:
    words = (source or "").split()
    out = [" ".join(words[s:s + window_tokens])
           for s in range(0, len(words), window_tokens)]
    return [w for w in out if w][:max_functions]

def extract_functions(class_text: str, cfg: dict,
                      raw_source: str | None = None) -> list[str]:
    """
    Segment a file into 'functions' per FLIM.  Falls back to fixed windows.

    Parameters
    ----------
    class_text  : structured summary from the JSONL dataset (always available)
    cfg         : functions block from config_flim.json
    raw_source  : full Java source recovered from git (may be None if the cache
                  is absent or the file was not found at sha_before)

    Behaviour
    ---------
    • mode="ast"  — preferred when raw_source is available; uses the javalang
                    AST extractor on the real source.  Automatically falls back
                    to the regex extractor if javalang is not installed or the
                    source cannot be parsed (e.g. Java 17+ syntax).
    • mode="java" — always uses the regex+brace extractor on (raw_source or
                    class_text), no AST dependency.
    • mode="window" — skips method extraction entirely; only window chunks.

    The file-header pseudo-function (first `header_tokens` words) is always
    prepended when include_file_header=True.  If no real methods are found
    after extraction, the whole source is split into overlapping windows so
    the pipeline always produces a non-empty segment list.
    """
    mode      = cfg.get("mode", "java")
    max_f     = cfg.get("max_functions", 50)
    min_chars = cfg.get("min_function_chars", cfg.get("min_func_length", 30))
    # source to extract from: prefer real Java over summary
    source = raw_source if raw_source is not None else class_text

    funcs: list[str] = []

    if mode == "ast":
        if raw_source is not None:
            # Full source available → use AST parser (with regex fallback inside)
            funcs = extract_java_methods_ast(source, max_f, min_chars)
        else:
            # No raw source → regex on the summary (degraded, logged by caller)
            funcs = extract_java_methods(class_text, max_f, min_chars)
    elif mode == "java":
        funcs = extract_java_methods(source, max_f, min_chars)
    # mode == "window": skip method extraction, go straight to chunking below

    if cfg.get("include_file_header", True):
        n_header = cfg.get("header_tokens", cfg.get("window_tokens", 200))
        header   = " ".join((source or "").split()[:n_header])
        if header:
            funcs = [header] + funcs

    # Fallback: if only the header was found (or nothing at all), use windows
    n_real = len(funcs) - (1 if cfg.get("include_file_header", True) and funcs else 0)
    if n_real <= 0:
        wt    = cfg.get("fallback_window", cfg.get("window_tokens", 200))
        funcs = window_chunks(source, wt, max_f)

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
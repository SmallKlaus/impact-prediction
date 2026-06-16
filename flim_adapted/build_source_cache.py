"""
build_source_cache.py — LOCAL script: recover raw Java source via git show
====
Run this on your development machine (or any machine with access to the
project git repositories).  It produces a single SQLite database
(source_cache.sqlite) that maps every (sha_before, class_path) pair found
in your dataset splits to the raw Java source at that commit.

The cache is then transferred to the GPU server and passed to the FLIM
pipeline via:
    flim_finetune.py  --source-cache  /path/to/source_cache.sqlite
    flim_features.py  --source-cache  /path/to/source_cache.sqlite

Without the cache, both scripts fall back to class_text summaries —
the cache dramatically improves function-extraction fidelity.

────────────────────────────────────────────────────────────────────
Schema (SQLite)
    CREATE TABLE source_cache (
        sha        TEXT NOT NULL,
        class_path TEXT NOT NULL,
        source     TEXT,
        status     TEXT NOT NULL DEFAULT 'ok',
        PRIMARY KEY (sha, class_path)
    )
    status = 'ok'      → source column holds raw Java text
    status = 'missing' → git show returned non-zero (file not at that sha)
    status = 'error'   → unexpected exception (git not found, etc.)
────────────────────────────────────────────────────────────────────

Usage example
    python build_source_cache.py \\
        --jsonl  train.jsonl val_500.jsonl test_full.jsonl \\
        --repos-dir  /path/to/repos \\
        --repo-map   '{"FLINK":"flink","KAFKA":"kafka","HADOOP":"hadoop","HBASE":"hbase","HDDS":"hdds","CAMEL":"camel"}' \\
        --output  source_cache.sqlite \\
        --workers 8

Repo layout assumption
    <repos_dir>/<repo_name>/.git    (bare or normal git repos both work)
    e.g. /path/to/repos/flink/.git

Repo map
    Maps the uppercase JIRA project prefix to the subdirectory name under
    repos_dir.  Pass as inline JSON string OR as a path to a JSON file.
    Example:  '{"FLINK":"flink","KAFKA":"kafka","HADOOP":"hadoop_common"}'
    Any prefix not in the map falls back to the lowercase prefix itself.

Incremental runs
    Already-cached (sha, class_path) pairs — regardless of status — are
    skipped on re-runs, so you can safely re-run after partial failures or
    after adding new JSONL splits.

Performance
    With --workers 8 and local git repos, expect ~500-2000 lookups/second
    depending on repo size and disk speed.  Full dataset typically finishes
    in < 30 minutes on an SSD.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import logging
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── SQLite helpers ────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS source_cache (
    sha        TEXT NOT NULL,
    class_path TEXT NOT NULL,
    source     TEXT,
    status     TEXT NOT NULL DEFAULT 'ok',
    PRIMARY KEY (sha, class_path)
);
CREATE INDEX IF NOT EXISTS idx_sha ON source_cache (sha);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.executescript(DDL)
    conn.commit()
    return conn


def load_existing_keys(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    rows = conn.execute("SELECT sha, class_path FROM source_cache").fetchall()
    return {(r[0], r[1]) for r in rows}


def batch_insert(conn: sqlite3.Connection,
                 rows: list[tuple[str, str, str | None, str]]) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO source_cache (sha, class_path, source, status) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()


# ── JSONL scanning ────────────────────────────────────────────────────────────

def scan_jsonl_files(
    jsonl_paths: list[Path],
) -> dict[tuple[str, str], str]:
    """
    Returns a dict  {(sha, class_path): jira_id}  for every unique pair found
    across all provided JSONL files.  jira_id is stored so we can derive the
    project (and therefore the right git repo) from it.
    """
    pairs: dict[tuple[str, str], str] = {}
    for path in jsonl_paths:
        log.info("Scanning %s ...", path)
        n_lines = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sha = rec.get("sha_before", "")
                cp  = rec.get("class_path", "")
                jid = rec.get("jira_id", "")
                if sha and cp and jid:
                    pairs.setdefault((sha, cp), jid)
                n_lines += 1
        log.info("  → %d lines scanned, running unique pairs: %d",
                 n_lines, len(pairs))
    return pairs


# ── Project → repo directory mapping ─────────────────────────────────────────

def load_repo_map(repo_map_arg: str) -> dict[str, str]:
    """
    Accepts either an inline JSON string or a path to a JSON file.
    Keys are UPPER-CASE JIRA prefixes; values are repo subdirectory names.
    """
    repo_map_arg = repo_map_arg.strip()
    if repo_map_arg.startswith("{"):
        raw = json.loads(repo_map_arg)
    else:
        raw = json.load(open(repo_map_arg, encoding="utf-8"))
    return {k.upper(): v for k, v in raw.items()}


def jira_prefix(jira_id: str) -> str:
    return jira_id.split("-")[0].upper() if jira_id else ""


def resolve_repo(jira_id: str, repos_dir: Path,
                 repo_map: dict[str, str]) -> Path | None:
    prefix  = jira_prefix(jira_id)
    subdir  = repo_map.get(prefix, prefix.lower())
    repo    = repos_dir / subdir
    if not repo.is_dir():
        return None
    return repo


# ── git show ─────────────────────────────────────────────────────────────────

def git_show(repo_dir: Path, sha: str, class_path: str) -> tuple[str | None, str]:
    """
    Runs `git -C <repo_dir> show <sha>:<class_path>`.
    Returns (source_text, status) where status ∈ {'ok', 'missing', 'error'}.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "show", f"{sha}:{class_path}"],
            capture_output=True,
            timeout=15,
        )
        if result.returncode == 0:
            # Decode with errors='replace' so lone surrogates never crash us
            return result.stdout.decode("utf-8", errors="replace"), "ok"
        else:
            return None, "missing"
    except subprocess.TimeoutExpired:
        log.debug("git show timeout: %s:%s in %s", sha[:8], class_path, repo_dir)
        return None, "error"
    except Exception as exc:
        log.debug("git show exception: %s", exc)
        return None, "error"


# ── Worker (for ThreadPoolExecutor) ──────────────────────────────────────────

def fetch_one(
    task: tuple[str, str, str, Path | None],
) -> tuple[str, str, str | None, str]:
    """
    task = (sha, class_path, jira_id, repo_dir_or_None)
    Returns (sha, class_path, source_or_None, status).
    """
    sha, cp, jira_id, repo_dir = task
    if repo_dir is None:
        return sha, cp, None, "error"
    source, status = git_show(repo_dir, sha, cp)
    return sha, cp, source, status


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a SQLite source cache from git repositories."
    )
    ap.add_argument(
        "--jsonl", required=True, nargs="+",
        help="One or more JSONL dataset splits to scan for (sha, class_path) pairs.",
    )
    ap.add_argument(
        "--repos-dir", required=True,
        help="Parent directory containing all project git repositories as subdirectories.",
    )
    ap.add_argument(
        "--repo-map", required=True,
        help=(
            'JSON mapping JIRA prefix → repo subdirectory name. '
            'Inline: \'{"FLINK":"flink","KAFKA":"kafka",...}\' '
            'or a path to a JSON file.'
        ),
    )
    ap.add_argument(
        "--output", default="source_cache.sqlite",
        help="Output SQLite path (default: source_cache.sqlite).",
    )
    ap.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel git show workers (default: 8).",
    )
    ap.add_argument(
        "--batch-size", type=int, default=500,
        help="DB insert batch size (default: 500).",
    )
    args = ap.parse_args()

    repos_dir = Path(args.repos_dir)
    if not repos_dir.is_dir():
        log.error("--repos-dir does not exist: %s", repos_dir)
        sys.exit(1)

    repo_map = load_repo_map(args.repo_map)
    log.info("Repo map: %s", repo_map)

    # ── Scan JSONL files ──
    jsonl_paths = [Path(p) for p in args.jsonl]
    for p in jsonl_paths:
        if not p.is_file():
            log.error("JSONL file not found: %s", p)
            sys.exit(1)

    all_pairs = scan_jsonl_files(jsonl_paths)
    log.info("Total unique (sha, class_path) pairs: %d", len(all_pairs))

    # ── Open / initialise DB ──
    db_path = Path(args.output)
    conn    = init_db(db_path)
    log.info("Database: %s", db_path.resolve())

    existing = load_existing_keys(conn)
    log.info("Already cached (will skip): %d", len(existing))

    todo = [(sha, cp, jid) for (sha, cp), jid in all_pairs.items()
            if (sha, cp) not in existing]
    log.info("Pairs to fetch             : %d", len(todo))

    if not todo:
        log.info("Nothing to do — cache is already complete.")
        conn.close()
        return

    # ── Resolve repo dirs ──
    tasks: list[tuple[str, str, str, Path | None]] = []
    unresolved: set[str] = set()
    for sha, cp, jid in todo:
        rdir = resolve_repo(jid, repos_dir, repo_map)
        if rdir is None:
            prefix = jira_prefix(jid)
            if prefix not in unresolved:
                log.warning(
                    "No repo directory found for JIRA prefix '%s' "
                    "(jira_id=%s). These pairs will be stored as 'error'. "
                    "Check --repo-map and --repos-dir.", prefix, jid,
                )
                unresolved.add(prefix)
        tasks.append((sha, cp, jid, rdir))

    # ── Parallel fetch ──
    log.info("Starting parallel fetch with %d workers ...", args.workers)
    results_buffer: list[tuple[str, str, str | None, str]] = []
    n_ok = n_miss = n_err = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, t): t for t in tasks}
        pbar = tqdm(concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="git show",
                    unit="file",
                    dynamic_ncols=True)
        for fut in pbar:
            sha, cp, source, status = fut.result()
            results_buffer.append((sha, cp, source, status))
            if status == "ok":
                n_ok += 1
            elif status == "missing":
                n_miss += 1
            else:
                n_err += 1
            pbar.set_postfix(ok=n_ok, miss=n_miss, err=n_err)

            # Flush to DB in batches to bound memory
            if len(results_buffer) >= args.batch_size:
                batch_insert(conn, results_buffer)
                results_buffer.clear()

    # Flush remainder
    if results_buffer:
        batch_insert(conn, results_buffer)
        results_buffer.clear()

    conn.close()

    # ── Summary ──
    total = n_ok + n_miss + n_err
    log.info("═══ build_source_cache complete ═══")
    log.info("Fetched    : %d", total)
    log.info("  ok       : %d  (%.1f%%) — raw Java stored", n_ok,  100 * n_ok  / max(total, 1))
    log.info("  missing  : %d  (%.1f%%) — file not at sha", n_miss, 100 * n_miss / max(total, 1))
    log.info("  error    : %d  (%.1f%%) — git/IO failure",  n_err,  100 * n_err  / max(total, 1))
    log.info("Database   : %s  (%.1f MB)",
             db_path.resolve(),
             db_path.stat().st_size / 1e6)
    log.info(
        "Transfer this file to your GPU server and pass it via "
        "--source-cache to flim_finetune.py and flim_features.py."
    )


if __name__ == "__main__":
    main()
"""
diagnose_bm25.py — BM25 Baseline with Hyperparameter Tuning
==================================================================
Runs BM25 (TF-IDF) retrieval with an automated Grid Search over k1 and b 
using the validation set, followed by evaluation on the test set.

Usage:
    pip install rank-bm25
    python diagnose_bm25.py --val-jsonl /data1/amine/IMPACT_TRAINING_SAMPLES/SAMPLES_V4/val_full.jsonl --test-jsonl /data1/amine/IMPACT_TRAINING_SAMPLES/SAMPLES_V4/test_full.jsonl --output /data1/amine/baselines/diagnosis_bm25.json
"""
from __future__ import annotations
import argparse
import json
import logging
import math
import re
import itertools
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
from rank_bm25 import BM25Okapi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── SE-Specific Tokenization ───────────────────────────────────────────────

def tokenize_for_code(text: str) -> list[str]:
    """Splits CamelCase, snake_case, and non-alphanumerics, then lowercases."""
    if not text:
        return []
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    return text.lower().split()

# ── Metric Helpers ─────────────────────────────────────────────────────────

def recall_at_k(scores, labels, k):
    if not any(labels): return 0.0
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    return sum(lbl for _, lbl in ranked[:k]) / sum(labels)

def ndcg_at_k(scores, labels, k):
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    dcg  = sum(lbl / math.log2(i+2) for i, (_, lbl) in enumerate(ranked[:k]))
    idcg = sum(lbl / math.log2(i+2) for i, lbl in enumerate(sorted(labels, reverse=True)[:k]))
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

def aggregate(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "n": 0}
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean)**2 for v in values) / n)
    return {"mean": round(mean, 4), "min": round(min(values), 4),
            "max": round(max(values), 4), "std": round(std, 4), "n": n}

def project_of(jira_id: str) -> str:
    prefix  = jira_id.split("-")[0].upper()
    mapping = {
        "FLINK": "flink", "KAFKA": "kafka", "HADOOP": "hadoop",
        "HDFS":  "hadoop",  "MAPREDUCE": "hadoop", "YARN": "hadoop",
    }
    return mapping.get(prefix, prefix.lower())

# ── Data Loading ───────────────────────────────────────────────────────────

def load_issues(jsonl_path: Path, desc: str):
    by_issue = defaultdict(list)
    log.info(f"Loading {desc} from {jsonl_path}...")
    
    # Pre-calculate file length for accurate progress bar
    with open(jsonl_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
        
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc=f"Reading {desc}"):
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            by_issue[rec["jira_id"]].append(rec)
            
    log.info(f"Loaded {len(by_issue)} unique issues for {desc}.")
    return by_issue

# ── Core Evaluation Loop ───────────────────────────────────────────────────

def score_issues(issues, k1: float, b: float, ks: list, desc: str):
    """Runs BM25 over a set of issues and returns a dict of per_issue metrics."""
    per_issue = {}
    
    for jid, records in tqdm(issues.items(), desc=desc, leave=False):
        labels = [int(r.get("label", 0)) for r in records]
        if not any(labels):
            continue 
        
        # 1. Prepare Corpus
        corpus_texts = [r.get("class_text", "") for r in records]
        tokenized_corpus = [tokenize_for_code(doc) for doc in corpus_texts]
        
        # 2. Fit BM25 with specific parameters
        bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        
        # 3. Prepare Query
        chunks = records[0].get("feature_chunks", [])
        if chunks:
            raw_query = " ".join([c.get("text", "") for c in chunks])
        else:
            raw_query = records[0].get("feature_text", "")
        tokenized_query = tokenize_for_code(raw_query)
        
        # 4. Score & Evaluate
        scores = bm25.get_scores(tokenized_query).tolist()
        n_pos_total = sum(labels)
        
        entry = {
            "project": project_of(jid),
            "n_candidates": len(records),
            "n_pos_total": n_pos_total,
        }
        
        for k in ks:
            entry[f"bm25_recall@{k}"] = round(recall_at_k(scores, labels, k), 4)
            if k <= 20:
                entry[f"bm25_ndcg@{k}"] = round(ndcg_at_k(scores, labels, k), 4)
        
        entry["bm25_map"] = round(average_precision(scores, labels), 4)
        entry["bm25_mrr"] = round(mean_reciprocal_rank(scores, labels), 4)
        per_issue[jid] = entry
        
    return per_issue

# ── Main Script ────────────────────────────────────────────────────────────

def run_pipeline(val_jsonl: Path, test_jsonl: Path, output_path: Path, worst_n: int):
    # 1. Load Data
    val_issues = load_issues(val_jsonl, "Validation Set")
    test_issues = load_issues(test_jsonl, "Test Set")

    # 2. Hyperparameter Grid Search (Tuning)
    k1_grid = [0.5, 1.0, 1.2, 1.5, 2.0]
    b_grid = [0.3, 0.5, 0.75, 0.9]
    grid = list(itertools.product(k1_grid, b_grid))
    
    log.info("=" * 60)
    log.info(f"Phase 1: Hyperparameter Tuning on Validation Set ({len(grid)} combinations)")
    log.info("=" * 60)
    
    best_mrr = -1.0
    best_params = {"k1": 1.5, "b": 0.75} # Default fallbacks
    
    # Progress bar for the grid search
    for k1, b in tqdm(grid, desc="Tuning BM25 Parameters"):
        val_metrics = score_issues(val_issues, k1, b, ks=[10], desc=f"Eval k1={k1}, b={b}")
        
        # Calculate Mean MRR for this grid point
        if not val_metrics:
            continue
        current_mean_mrr = sum(e["bm25_mrr"] for e in val_metrics.values()) / len(val_metrics)
        
        if current_mean_mrr > best_mrr:
            best_mrr = current_mean_mrr
            best_params = {"k1": k1, "b": b}
            
    log.info(f"Tuning Complete. Best Validation MRR: {best_mrr:.4f}")
    log.info(f"Selected Hyperparameters: k1 = {best_params['k1']}, b = {best_params['b']}")

    # 3. Final Evaluation on Test Set
    log.info("=" * 60)
    log.info("Phase 2: Final Evaluation on Test Set")
    log.info("=" * 60)
    
    KS = [1, 3, 5, 10, 20, 50, 80, 100]
    test_results = score_issues(
        test_issues, 
        k1=best_params["k1"], 
        b=best_params["b"], 
        ks=KS, 
        desc="Scoring Test Set"
    )

    # 4. Global & Project Aggregation
    all_ents = list(test_results.values())
    N = len(all_ents)
    def gmean(key): return round(sum(e[key] for e in all_ents) / N, 4) if N else 0.0

    global_metrics = {
        "n_issues": N,
        **{f"bm25_recall@{k}": gmean(f"bm25_recall@{k}") for k in KS},
        "bm25_ndcg@10": gmean("bm25_ndcg@10"),
        "bm25_map": gmean("bm25_map"),
        "bm25_mrr": gmean("bm25_mrr"),
    }

    # Per Project
    proj_issues = defaultdict(list)
    for jid, e in test_results.items():
        proj_issues[e["project"]].append(jid)

    per_project = {}
    for proj, jids in sorted(proj_issues.items()):
        ents = [test_results[j] for j in jids]
        per_project[proj] = {
            "n_issues": len(jids),
            **{f"bm25_recall@{k}": aggregate([e[f"bm25_recall@{k}"] for e in ents]) for k in KS},
            "bm25_ndcg@10": aggregate([e["bm25_ndcg@10"] for e in ents]),
            "bm25_map": aggregate([e["bm25_map"] for e in ents]),
            "bm25_mrr": aggregate([e["bm25_mrr"] for e in ents]),
        }

    # Worst N
    worst = sorted(test_results.items(), key=lambda x: x[1]["bm25_mrr"])[:worst_n]
    worst_issues = [
        {
            "jira_id": jid,
            "project": e["project"],
            "bm25_mrr": e["bm25_mrr"],
            "bm25_recall@10": e["bm25_recall@10"],
            "n_candidates": e["n_candidates"],
            "n_pos_total": e["n_pos_total"],
        }
        for jid, e in worst
    ]

    # 5. Save to JSON
    report = {
        "val_jsonl": str(val_jsonl),
        "test_jsonl": str(test_jsonl),
        "hyperparameters_tuned": best_params,
        "best_val_mrr": best_mrr,
        "global": global_metrics,
        "per_project": per_project,
        "worst_issues": worst_issues,
        "per_issue": test_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 6. Print Console Summary
    W = 70
    print("\n" + "=" * W)
    print(f"GLOBAL METRICS (BM25 TUNED: k1={best_params['k1']}, b={best_params['b']})")
    print("=" * W)
    print(f"  Issues evaluated : {N}")
    for k in [1, 5, 10, 20, 50, 100]:
        print(f"  Recall@{k:<3}       : {global_metrics[f'bm25_recall@{k}']:.4f}")
    print(f"  NDCG@10          : {global_metrics['bm25_ndcg@10']:.4f}")
    print(f"  MAP              : {global_metrics['bm25_map']:.4f}")
    print(f"  MRR              : {global_metrics['bm25_mrr']:.4f}")

    print("\n" + "=" * W)
    print("PER-PROJECT SUMMARY")
    print("=" * W)
    print(f"  {'Project':<16} {'Issues':>6}  {'MRR':>6}  {'R@10':>6}  {'R@50':>6}")
    print("  " + "-" * 55)
    for proj, stats in sorted(per_project.items()):
        print(f"  {proj:<16} {stats['n_issues']:>6}  "
              f"{stats['bm25_mrr']['mean']:>6.4f}  "
              f"{stats['bm25_recall@10']['mean']:>6.4f}  "
              f"{stats['bm25_recall@50']['mean']:>6.4f}")

    print(f"\nReport saved to: {output_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tuned BM25 baseline")
    parser.add_argument("--val-jsonl", required=True, help="Path to val_full.jsonl for tuning")
    parser.add_argument("--test-jsonl", required=True, help="Path to test_full.jsonl for evaluation")
    parser.add_argument("--output", required=True, help="Path to save diagnosis_bm25.json")
    parser.add_argument("--worst-n", type=int, default=20)
    args = parser.parse_args()
    
    run_pipeline(Path(args.val_jsonl), Path(args.test_jsonl), Path(args.output), args.worst_n)
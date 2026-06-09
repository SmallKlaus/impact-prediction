"""
diagnose_ce.py — Per-Issue Cross-Encoder Diagnostic
====================================================
Loads a trained cross-encoder checkpoint, runs inference on a CE JSONL
(produced by build_ce_samples.py), and produces a comprehensive JSON report
with readable printed tables.

The key feature of this diagnostic compared to the bi-encoder's diagnose.py
is the side-by-side CE vs. bi-encoder comparison on exactly the same
candidate window, plus end-to-end pipeline recall estimates.

Report sections
---------------
1.  GLOBAL METRICS       — CE and bi-encoder baseline, macro-averaged
2.  IMPROVEMENT TABLE    — Delta CE minus bi-encoder for every metric
3.  PIPELINE RECALL      — End-to-end recall accounting for bi-encoder ceiling
4.  MRR DISTRIBUTION     — How many issues have MRR=1, >=0.5, <0.5, =0
5.  PER-PROJECT SUMMARY  — Macro-averaged metrics and deltas per project
6.  WORST-N ISSUES       — Lowest CE MRR issues for error analysis
7.  per_issue (JSON)     — Full per-issue detail (last key, largest)

Usage:
    python diagnose_ce.py \\
        --checkpoint  /data1/amine/checkpoints/ce_run_001/best_model_ce.pt \\
        --config      /data1/amine/checkpoints/ce_run_001/config_ce.json \\
        --test-jsonl  /data1/amine/IMPACT_TRAINING_SAMPLES/CE_SAMPLES/ce_test.jsonl \\
        --output      /data1/amine/checkpoints/ce_run_001/diagnosis_ce.json \\
        [--batch-size 16] \\
        [--worst-n    20]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import ssl
import urllib3
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from cross_encoder_model   import build_cross_encoder
from cross_encoder_dataset import CrossEncoderDataset, ce_collate_fn

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Metric helpers ─────────────────────────────────────────────────────────

def recall_at_k(scores, labels, k):
    if not any(labels): return 0.0
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    return sum(lbl for _, lbl in ranked[:k]) / sum(labels)

def ndcg_at_k(scores, labels, k):
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    dcg  = sum(lbl / math.log2(i+2) for i, (_, lbl) in enumerate(ranked[:k]))
    idcg = sum(lbl / math.log2(i+2)
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

def aggregate(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "n": 0}
    n    = len(values)
    mean = sum(values) / n
    std  = math.sqrt(sum((v - mean)**2 for v in values) / n)
    return {"mean": round(mean, 4), "min": round(min(values), 4),
            "max": round(max(values), 4), "std": round(std, 4), "n": n}

def project_of(jira_id: str) -> str:
    prefix  = jira_id.split("-")[0].upper()
    mapping = {
        "FLINK": "flink", "KAFKA": "kafka", "HADOOP": "hadoop_common",
        "HDFS":  "hdfs",  "MAPREDUCE": "mapreduce", "YARN": "yarn",
    }
    return mapping.get(prefix, prefix.lower())


# ── Inference ──────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    """
    Returns parallel lists:
        ce_scores, be_scores, labels, jira_ids, class_paths,
        n_pos_totals, n_pos_in_topks
    """
    model.eval()
    ce_scores, be_scores, labels = [], [], []
    jira_ids, class_paths        = [], []
    n_pos_totals, n_pos_in_topks = [], []

    for batch in tqdm(loader, desc="Inference", leave=True):
        batch  = batch.to(device)
        logits = model(batch.input_ids, batch.attention_mask, batch.chunk_mask)
        probs  = torch.sigmoid(logits).cpu().tolist()

        ce_scores.extend(probs)
        be_scores.extend(batch.biencoder_scores)
        labels.extend(batch.labels.int().cpu().tolist())
        jira_ids.extend(batch.jira_ids)
        class_paths.extend(batch.class_paths)

        # n_pos_total and n_pos_in_topk come from the dataset samples directly
        for idx in range(len(probs)):
            # Retrieve from dataset via jira_id + class_path lookup
            n_pos_totals.append(0)   # filled below from dataset
            n_pos_in_topks.append(0)

    return (ce_scores, be_scores, labels, jira_ids, class_paths,
            n_pos_totals, n_pos_in_topks)


# ── Diagnose ───────────────────────────────────────────────────────────────

def diagnose(
    checkpoint_path: Path,
    config:          dict,
    test_jsonl:      Path,
    output_path:     Path,
    batch_size:      int,
    worst_n:         int,
):
    tcfg   = config["training"]
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    log.info("Device: %s", device)

    # ── Load model ─────────────────────────────────────────────────────
    model_name = config["model"]["model_name"]
    tokenizer  = AutoTokenizer.from_pretrained(model_name)

    model = build_cross_encoder(config["model"]).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    log.info("Checkpoint loaded: %s", checkpoint_path)

    # ── Build dataloader ────────────────────────────────────────────────
    log.info("Loading test set: %s", test_jsonl)
    test_ds = CrossEncoderDataset(
        test_jsonl, tokenizer,
        max_chunk_tokens = tcfg.get("max_chunk_tokens", 512),
        max_class_tokens = tcfg.get("max_class_tokens", 256),
        max_chunks       = tcfg.get("max_chunks", 4),
    )
    loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = ce_collate_fn,
        num_workers = tcfg.get("num_workers", 4),
        pin_memory  = True,
    )
    log.info("Test samples: %d", len(test_ds))

    # ── Build n_pos lookup from raw samples ────────────────────────────
    # Each JSONL record has n_pos_total and n_pos_in_topk
    pos_total_by_jid:  dict[str, int] = {}
    pos_topk_by_jid:   dict[str, int] = {}
    for s in test_ds.samples:
        jid = s.get("jira_id", "")
        pos_total_by_jid[jid] = s.get("n_pos_total",   0)
        pos_topk_by_jid[jid]  = s.get("n_pos_in_topk", 0)

    # ── Run inference ───────────────────────────────────────────────────
    model.eval()
    ce_scores_all, be_scores_all = [], []
    labels_all, jira_ids_all, class_paths_all = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=True):
            batch  = batch.to(device)
            logits = model(batch.input_ids, batch.attention_mask,
                           batch.chunk_mask)
            probs  = torch.sigmoid(logits).cpu().tolist()
            ce_scores_all.extend(probs)
            be_scores_all.extend(batch.biencoder_scores)
            labels_all.extend(batch.labels.int().cpu().tolist())
            jira_ids_all.extend(batch.jira_ids)
            class_paths_all.extend(batch.class_paths)

    # ── Group by issue ──────────────────────────────────────────────────
    ce_by_jid: dict[str, list] = defaultdict(list)
    be_by_jid: dict[str, list] = defaultdict(list)
    lb_by_jid: dict[str, list] = defaultdict(list)

    for ce, be, lb, jid in zip(ce_scores_all, be_scores_all,
                                labels_all, jira_ids_all):
        ce_by_jid[jid].append(ce)
        be_by_jid[jid].append(be)
        lb_by_jid[jid].append(lb)

    KS = [1, 3, 5, 10, 20]

    # ── Per-issue metrics ───────────────────────────────────────────────
    per_issue: dict[str, dict] = {}

    for jid in ce_by_jid:
        ce, be, lb = ce_by_jid[jid], be_by_jid[jid], lb_by_jid[jid]
        if not any(lb):
            continue

        n_pos_total  = pos_total_by_jid.get(jid, sum(lb))
        n_pos_in_topk = pos_topk_by_jid.get(jid, sum(lb))
        ceiling       = n_pos_in_topk / n_pos_total if n_pos_total > 0 else 0.0

        entry: dict = {
            "project":          project_of(jid),
            "n_candidates":     len(ce),
            "n_pos_in_window":  sum(lb),
            "n_pos_total":      n_pos_total,
            "n_pos_in_topk":    n_pos_in_topk,
            "bi_ceiling":       round(ceiling, 4),
        }

        # CE metrics (denominator = positives in CE window)
        for k in KS:
            entry[f"ce_recall@{k}"]  = round(recall_at_k(ce, lb, k), 4)
        for k in KS:
            if k <= 10:
                entry[f"ce_ndcg@{k}"] = round(ndcg_at_k(ce, lb, k), 4)
        entry["ce_map"] = round(average_precision(ce, lb), 4)
        entry["ce_mrr"] = round(mean_reciprocal_rank(ce, lb), 4)

        # Bi-encoder baseline (same window, same labels)
        for k in KS:
            entry[f"be_recall@{k}"]  = round(recall_at_k(be, lb, k), 4)
        entry["be_map"] = round(average_precision(be, lb), 4)
        entry["be_mrr"] = round(mean_reciprocal_rank(be, lb), 4)

        # Improvement delta (CE minus bi-encoder within window)
        for k in KS:
            entry[f"delta_recall@{k}"] = round(
                entry[f"ce_recall@{k}"] - entry[f"be_recall@{k}"], 4
            )
        entry["delta_mrr"] = round(entry["ce_mrr"] - entry["be_mrr"], 4)
        entry["delta_map"] = round(entry["ce_map"] - entry["be_map"], 4)

        # Pipeline recall: CE recall within window × bi-encoder ceiling
        # Answers "of ALL positives in the codebase, how many does the
        # full two-stage pipeline surface in the CE's top-k?"
        for k in KS:
            entry[f"pipeline_recall@{k}"] = round(
                entry[f"ce_recall@{k}"] * ceiling, 4
            )

        per_issue[jid] = entry

    log.info("Per-issue metrics computed for %d issues", len(per_issue))

    # ── Per-project aggregates ──────────────────────────────────────────
    proj_issues: dict[str, list[str]] = defaultdict(list)
    for jid, e in per_issue.items():
        proj_issues[e["project"]].append(jid)

    per_project: dict[str, dict] = {}
    for proj, jids in sorted(proj_issues.items()):
        ents = [per_issue[j] for j in jids]
        per_project[proj] = {
            "n_issues":    len(jids),
            "bi_ceiling":  aggregate([e["bi_ceiling"] for e in ents]),
            **{f"ce_recall@{k}":       aggregate([e[f"ce_recall@{k}"]  for e in ents]) for k in KS},
            **{f"be_recall@{k}":       aggregate([e[f"be_recall@{k}"]  for e in ents]) for k in KS},
            **{f"pipeline_recall@{k}": aggregate([e[f"pipeline_recall@{k}"] for e in ents]) for k in KS},
            "ce_ndcg@10":  aggregate([e["ce_ndcg@10"] for e in ents]),
            "ce_map":      aggregate([e["ce_map"]     for e in ents]),
            "ce_mrr":      aggregate([e["ce_mrr"]     for e in ents]),
            "be_mrr":      aggregate([e["be_mrr"]     for e in ents]),
            "delta_mrr":   aggregate([e["delta_mrr"]  for e in ents]),
        }

    # ── Global aggregate ────────────────────────────────────────────────
    all_ents = list(per_issue.values())
    N        = len(all_ents)

    def gmean(key): return round(sum(e[key] for e in all_ents) / N, 4) if N else 0.0

    global_metrics = {
        "n_issues":      N,
        "mean_bi_ceiling": gmean("bi_ceiling"),
        **{f"ce_recall@{k}":       gmean(f"ce_recall@{k}") for k in KS},
        **{f"be_recall@{k}":       gmean(f"be_recall@{k}") for k in KS},
        **{f"pipeline_recall@{k}": gmean(f"pipeline_recall@{k}") for k in KS},
        "ce_ndcg@10":  gmean("ce_ndcg@10"),
        "ce_map":      gmean("ce_map"),
        "ce_mrr":      gmean("ce_mrr"),
        "be_mrr":      gmean("be_mrr"),
        "delta_mrr":   gmean("delta_mrr"),
        "delta_map":   gmean("delta_map"),
        **{f"delta_recall@{k}": gmean(f"delta_recall@{k}") for k in KS},
    }

    # ── MRR distribution ────────────────────────────────────────────────
    mrr_vals = [e["ce_mrr"] for e in all_ents]
    mrr_dist = {
        "mrr_eq_1":   sum(1 for v in mrr_vals if v == 1.0),
        "mrr_gte_0.5": sum(1 for v in mrr_vals if 0.5 <= v < 1.0),
        "mrr_lt_0.5":  sum(1 for v in mrr_vals if 0.0 < v < 0.5),
        "mrr_eq_0":    sum(1 for v in mrr_vals if v == 0.0),
        "mrr_mean":    round(sum(mrr_vals) / N, 4) if N else 0.0,
        "mrr_std":     round(math.sqrt(
                           sum((v - sum(mrr_vals)/N)**2 for v in mrr_vals) / N
                       ), 4) if N else 0.0,
    }

    # ── Worst-N ─────────────────────────────────────────────────────────
    worst = sorted(per_issue.items(), key=lambda x: x[1]["ce_mrr"])[:worst_n]
    worst_issues = [
        {
            "jira_id":       jid,
            "project":       e["project"],
            "ce_mrr":        e["ce_mrr"],
            "be_mrr":        e["be_mrr"],
            "delta_mrr":     e["delta_mrr"],
            "ce_recall@5":   e["ce_recall@5"],
            "be_recall@5":   e["be_recall@5"],
            "bi_ceiling":    e["bi_ceiling"],
            "n_candidates":  e["n_candidates"],
            "n_pos_in_window": e["n_pos_in_window"],
        }
        for jid, e in worst
    ]

    # ── Assemble report ─────────────────────────────────────────────────
    report = {
        "checkpoint":    str(checkpoint_path),
        "test_jsonl":    str(test_jsonl),
        "global":        global_metrics,
        "mrr_distribution": mrr_dist,
        "per_project":   per_project,
        "worst_issues":  worst_issues,
        "per_issue":     per_issue,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Print summary ────────────────────────────────────────────────────
    W = 72
    print("\n" + "=" * W)
    print("GLOBAL METRICS  (within bi-encoder top-K window)")
    print("=" * W)
    print(f"  Issues evaluated      : {N}")
    print(f"  Mean bi-encoder ceiling: {global_metrics['mean_bi_ceiling']:.4f}  "
          f"(fraction of positives inside top-K)")
    print()
    print(f"  {'Metric':<18}  {'CrossEnc':>9}  {'BiEncoder':>9}  {'Delta':>9}")
    print("  " + "-" * 50)
    for k in KS:
        ce_v  = global_metrics[f"ce_recall@{k}"]
        be_v  = global_metrics[f"be_recall@{k}"]
        d     = global_metrics[f"delta_recall@{k}"]
        arrow = "▲" if d > 0 else ("▼" if d < 0 else " ")
        print(f"  {'Recall@'+str(k):<18}  {ce_v:>9.4f}  {be_v:>9.4f}  "
              f"{arrow}{abs(d):>8.4f}")
    print(f"  {'NDCG@10':<18}  {global_metrics['ce_ndcg@10']:>9.4f}  {'—':>9}  {'':>9}")
    print(f"  {'MAP':<18}  {global_metrics['ce_map']:>9.4f}  {'—':>9}  "
          f"  {global_metrics['delta_map']:>+9.4f}")
    for label, ce_k, be_k, d_k in [
        ("MRR", "ce_mrr", "be_mrr", "delta_mrr"),
    ]:
        ce_v = global_metrics[ce_k]
        be_v = global_metrics[be_k]
        d    = global_metrics[d_k]
        arrow = "▲" if d > 0 else ("▼" if d < 0 else " ")
        print(f"  {label:<18}  {ce_v:>9.4f}  {be_v:>9.4f}  "
              f"{arrow}{abs(d):>8.4f}")

    print("\n" + "=" * W)
    print("PIPELINE RECALL  (end-to-end: bi-encoder ceiling × CE recall)")
    print("=" * W)
    for k in KS:
        print(f"  Pipeline Recall@{k:<3}    : "
              f"{global_metrics[f'pipeline_recall@{k}']:.4f}")

    print("\n" + "=" * W)
    print("MRR DISTRIBUTION  (cross-encoder)")
    print("=" * W)
    print(f"  MRR = 1.0  (1st hit at rank 1) : "
          f"{mrr_dist['mrr_eq_1']:4d}  "
          f"({100*mrr_dist['mrr_eq_1']/max(N,1):.1f}%)")
    print(f"  MRR in [0.5, 1.0)              : "
          f"{mrr_dist['mrr_gte_0.5']:4d}  "
          f"({100*mrr_dist['mrr_gte_0.5']/max(N,1):.1f}%)")
    print(f"  MRR in (0, 0.5)                : "
          f"{mrr_dist['mrr_lt_0.5']:4d}  "
          f"({100*mrr_dist['mrr_lt_0.5']/max(N,1):.1f}%)")
    print(f"  MRR = 0.0  (no hit)            : "
          f"{mrr_dist['mrr_eq_0']:4d}  "
          f"({100*mrr_dist['mrr_eq_0']/max(N,1):.1f}%)")
    print(f"  Mean / Std                     : "
          f"{mrr_dist['mrr_mean']:.4f} / {mrr_dist['mrr_std']:.4f}")

    print("\n" + "=" * W)
    print("PER-PROJECT SUMMARY")
    print("=" * W)
    hdr = f"  {'Project':<16} {'N':>4}  {'Ceiling':>7}  "
    hdr += f"{'CE-MRR':>7}  {'BE-MRR':>7}  {'ΔMRR':>7}  "
    hdr += f"{'CE-R@5':>7}  {'Pipe-R@5':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for proj, stats in sorted(per_project.items()):
        print(
            f"  {proj:<16} {stats['n_issues']:>4}  "
            f"{stats['bi_ceiling']['mean']:>7.4f}  "
            f"{stats['ce_mrr']['mean']:>7.4f}  "
            f"{stats['be_mrr']['mean']:>7.4f}  "
            f"{stats['delta_mrr']['mean']:>+7.4f}  "
            f"{stats['ce_recall@5']['mean']:>7.4f}  "
            f"{stats['pipeline_recall@5']['mean']:>8.4f}"
        )

    print("\n" + "=" * W)
    print(f"WORST {worst_n} ISSUES BY CE MRR")
    print("=" * W)
    print(f"  {'Jira ID':<22} {'Proj':<10} "
          f"{'CE-MRR':>7}  {'BE-MRR':>7}  {'ΔMRR':>7}  "
          f"{'Ceiling':>7}  {'Cands':>5}  {'Pos':>3}")
    print("  " + "-" * 70)
    for w in worst_issues:
        print(
            f"  {w['jira_id']:<22} {w['project']:<10} "
            f"{w['ce_mrr']:>7.4f}  {w['be_mrr']:>7.4f}  "
            f"{w['delta_mrr']:>+7.4f}  "
            f"{w['bi_ceiling']:>7.4f}  "
            f"{w['n_candidates']:>5}  {w['n_pos_in_window']:>3}"
        )

    print(f"\nFull report saved to: {output_path}\n")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-issue cross-encoder reranker diagnostic"
    )
    parser.add_argument("--checkpoint",  required=True,
                        help="CE checkpoint (best_model_ce.pt)")
    parser.add_argument("--config",      required=True,
                        help="config_ce.json saved in output_dir")
    parser.add_argument("--test-jsonl",  required=True,
                        help="CE test JSONL produced by build_ce_samples.py")
    parser.add_argument("--output",      required=True,
                        help="Path to write diagnosis_ce.json")
    parser.add_argument("--batch-size",  type=int, default=16)
    parser.add_argument("--worst-n",     type=int, default=20)
    args = parser.parse_args()

    for p, name in [
        (args.checkpoint, "--checkpoint"),
        (args.config,     "--config"),
        (args.test_jsonl, "--test-jsonl"),
    ]:
        if not Path(p).exists():
            import sys; sys.exit(f"{name} not found: {p}")

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    diagnose(
        checkpoint_path = Path(args.checkpoint),
        config          = config,
        test_jsonl      = Path(args.test_jsonl),
        output_path     = Path(args.output),
        batch_size      = args.batch_size,
        worst_n         = args.worst_n,
    )


if __name__ == "__main__":
    main()

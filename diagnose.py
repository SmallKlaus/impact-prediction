"""
diagnose.py — Per-Issue and Per-Project Retrieval Diagnostic
=============================================================
Loads a trained checkpoint and runs inference on the full-codebase val set,
then produces:

  1. Per-issue metrics (R@5/10/20/50, NDCG@10, MAP, n_candidates, n_positives)
  2. Per-project aggregates (macro-averaged across issues in that project)
  3. Global aggregate (same as training logs, for sanity check)
  4. Worst-N issues ranked by R@10 (the cases dragging the average down)
  5. Distribution summary: how many issues score R@10 = 0, <0.5, >=0.5, = 1.0

All results are written to a JSON report and printed as a readable table.

Usage:
    python diagnose.py \
        --checkpoint  /data/amine/checkpoints/run_002/best_model.pt \
        --config      /data/amine/checkpoints/run_002/config.json \
        --val-jsonl   /data/amine/IMPACT_TRAINING_SAMPLES/val_full.jsonl \
        --output      /data/amine/checkpoints/run_002/diagnosis.json \
        [--batch-size 16] \
        [--worst-n    20]

The --config flag expects the config.json saved alongside the checkpoint
(it is written automatically by train.py into output_dir/config.json).
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

from model   import build_model
from dataset import ImpactSampleDataset, collate_fn
from loss    import build_loss

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Metric helpers (mirrors train.py exactly) ─────────────────────────────────

def recall_at_k(scores: list, labels: list, k: int) -> float:
    if not any(labels):
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    return sum(lbl for _, lbl in ranked[:k]) / sum(labels)


def ndcg_at_k(scores: list, labels: list, k: int) -> float:
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    dcg  = sum(lbl / math.log2(i + 2) for i, (_, lbl) in enumerate(ranked[:k]))
    idcg = sum(lbl / math.log2(i + 2)
               for i, lbl in enumerate(sorted(labels, reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(scores: list, labels: list) -> float:
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos, cum_pos, ap = 0, 0, 0.0
    for i, (_, lbl) in enumerate(ranked):
        if lbl:
            cum_pos += 1
            n_pos   += 1
            ap      += cum_pos / (i + 1)
    return ap / n_pos if n_pos else 0.0


def mean_reciprocal_rank(scores: list, labels: list) -> float:
    """MRR: reciprocal rank of the first positive in the ranked list."""
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    for i, (_, lbl) in enumerate(ranked):
        if lbl:
            return 1.0 / (i + 1)
    return 0.0


def project_of(jira_id: str) -> str:
    """Infer project name from Jira ID prefix."""
    prefix = jira_id.split("-")[0].upper()
    mapping = {
        "FLINK":     "flink",
        "KAFKA":     "kafka",
        "HADOOP":    "hadoop_common",
        "HDFS":      "hdfs",
        "MAPREDUCE": "mapreduce",
        "YARN":      "yarn",
    }
    return mapping.get(prefix, prefix.lower())


def aggregate(values: list[float]) -> dict:
    """Mean, min, max, std of a list."""
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "n": 0}
    n    = len(values)
    mean = sum(values) / n
    std  = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return {
        "mean": round(mean, 4),
        "min":  round(min(values), 4),
        "max":  round(max(values), 4),
        "std":  round(std, 4),
        "n":    n,
    }


# ── Inference pass ────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model:   torch.nn.Module,
    loader:  DataLoader,
    device:  torch.device,
) -> tuple[list[float], list[int], list[str], list[str]]:
    """
    Returns parallel lists of (scores, labels, jira_ids, class_paths).
    """
    model.eval()
    all_scores, all_labels, all_jira_ids, all_class_paths = [], [], [], []

    pbar = tqdm(loader, desc="Inference", leave=True)
    for batch in pbar:
        batch = batch.to(device)
        logits, _, _ = model(
            batch.feature_input_ids,
            batch.feature_attention_mask,
            batch.feature_chunk_mask,
            batch.class_input_ids,
            batch.class_attention_mask,
        )
        probs = torch.sigmoid(logits).cpu().tolist()
        lbls  = batch.labels.int().cpu().tolist()

        all_scores.extend(probs)
        all_labels.extend(lbls)
        all_jira_ids.extend(batch.jira_ids)
        all_class_paths.extend(batch.class_paths)

    return all_scores, all_labels, all_jira_ids, all_class_paths


# ── Main diagnostic ───────────────────────────────────────────────────────────

def diagnose(
    checkpoint_path: Path,
    config:          dict,
    val_jsonl:       Path,
    output_path:     Path,
    batch_size:      int,
    worst_n:         int,
):
    tcfg   = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load tokenizer and model ──────────────────────────────────────────────
    model_name = config["model"]["model_name"]
    log.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    log.info("Building model ...")
    model = build_model(config["model"]).to(device)

    log.info("Loading checkpoint: %s", checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device)
    # Checkpoint may be a full training state dict or just model weights
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
        log.info("  Loaded from full checkpoint (epoch %d)", state.get("epoch", "?"))
    else:
        model.load_state_dict(state)
        log.info("  Loaded model weights directly")

    # ── Build val dataloader ──────────────────────────────────────────────────
    log.info("Loading val set: %s", val_jsonl)
    val_ds = ImpactSampleDataset(
        jsonl_path       = val_jsonl,
        tokenizer        = tokenizer,
        max_chunk_tokens = tcfg.get("max_chunk_tokens", 512),
        max_class_tokens = tcfg.get("max_class_tokens", 512),
        max_chunks       = tcfg.get("max_chunks", 8),
        label_smoothing  = 0.0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = tcfg.get("num_workers", 4),
        pin_memory  = True,
    )
    log.info("Val samples: %d", len(val_ds))

    # ── Run inference ─────────────────────────────────────────────────────────
    all_scores, all_labels, all_jira_ids, all_class_paths = run_inference(
        model, val_loader, device
    )

    # ── Group by issue ────────────────────────────────────────────────────────
    issue_scores:  dict[str, list] = defaultdict(list)
    issue_labels:  dict[str, list] = defaultdict(list)

    for score, label, jid in zip(all_scores, all_labels, all_jira_ids):
        issue_scores[jid].append(score)
        issue_labels[jid].append(label)

    KS = [5, 10, 20, 50, 80, 100]

    # ── Per-issue metrics ─────────────────────────────────────────────────────
    per_issue: dict[str, dict] = {}
    for jid in issue_scores:
        s, l = issue_scores[jid], issue_labels[jid]
        if not any(l):
            continue
        entry = {
            "project":      project_of(jid),
            "n_candidates": len(s),
            "n_positives":  sum(l),
            "map":          round(average_precision(s, l), 4),
            "mrr":          round(mean_reciprocal_rank(s, l), 4),
        }
        for k in KS:
            entry[f"recall@{k}"]  = round(recall_at_k(s, l, k), 4)
        for k in KS:
            if k <= 20:
                entry[f"ndcg@{k}"] = round(ndcg_at_k(s, l, k), 4)
        per_issue[jid] = entry

    log.info("Computed metrics for %d issues", len(per_issue))

    # ── Per-project aggregates ────────────────────────────────────────────────
    project_issues: dict[str, list[str]] = defaultdict(list)
    for jid, entry in per_issue.items():
        project_issues[entry["project"]].append(jid)

    per_project: dict[str, dict] = {}
    for proj, jids in sorted(project_issues.items()):
        proj_entries = [per_issue[j] for j in jids]
        per_project[proj] = {
            "n_issues": len(jids),
            **{f"recall@{k}": aggregate([e[f"recall@{k}"] for e in proj_entries])
               for k in KS},
            "ndcg@10":  aggregate([e["ndcg@10"]  for e in proj_entries]),
            "map":      aggregate([e["map"]       for e in proj_entries]),
            "mrr":      aggregate([e["mrr"]       for e in proj_entries]),
        }

    # ── Global aggregate ──────────────────────────────────────────────────────
    all_entries = list(per_issue.values())
    global_metrics = {
        "n_issues": len(all_entries),
        **{f"recall@{k}": round(sum(e[f"recall@{k}"] for e in all_entries)
                                / len(all_entries), 4)
           for k in KS},
        **{f"ndcg@{k}": round(sum(e[f"ndcg@{k}"] for e in all_entries)
                               / len(all_entries), 4)
           for k in [k for k in KS if k <= 20]},
        "map": round(sum(e["map"] for e in all_entries) / len(all_entries), 4),
        "mrr": round(sum(e["mrr"] for e in all_entries) / len(all_entries), 4),
    }

    # ── R@10 distribution ─────────────────────────────────────────────────────
    r10_values = [e["recall@10"] for e in all_entries]
    distribution = {
        "r10_eq_0":        sum(1 for v in r10_values if v == 0.0),
        "r10_lt_0.5":      sum(1 for v in r10_values if 0.0 < v < 0.5),
        "r10_gte_0.5":     sum(1 for v in r10_values if 0.5 <= v < 1.0),
        "r10_eq_1":        sum(1 for v in r10_values if v == 1.0),
        "r10_mean":        round(sum(r10_values) / len(r10_values), 4),
        "r10_std":         round(math.sqrt(
                               sum((v - sum(r10_values)/len(r10_values))**2
                                   for v in r10_values) / len(r10_values)), 4),
    }

    # ── Worst-N issues ────────────────────────────────────────────────────────
    worst = sorted(per_issue.items(), key=lambda x: x[1]["recall@10"])[:worst_n]
    worst_issues = [
        {
            "jira_id":      jid,
            "project":      entry["project"],
            "recall@10":    entry["recall@10"],
            "recall@50":    entry["recall@50"],
            "mrr":          entry["mrr"],
            "map":          entry["map"],
            "n_candidates": entry["n_candidates"],
            "n_positives":  entry["n_positives"],
        }
        for jid, entry in worst
    ]

    # ── Assemble report ───────────────────────────────────────────────────────
    report = {
        "checkpoint":    str(checkpoint_path),
        "val_jsonl":     str(val_jsonl),
        "global":        global_metrics,
        "distribution":  distribution,
        "per_project":   per_project,
        "worst_issues":  worst_issues,
        "per_issue":     per_issue,   # full detail, last since it's large
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GLOBAL METRICS")
    print("=" * 70)
    print(f"  Issues evaluated : {global_metrics['n_issues']}")
    for k in KS:
        print(f"  Recall@{k:<3}       : {global_metrics[f'recall@{k}']:.4f}")
    print(f"  NDCG@10          : {global_metrics['ndcg@10']:.4f}")
    print(f"  MAP              : {global_metrics['map']:.4f}")
    print(f"  MRR              : {global_metrics['mrr']:.4f}")

    print("\n" + "=" * 70)
    print("R@10 DISTRIBUTION")
    print("=" * 70)
    n = len(all_entries)
    print(f"  R@10 = 0.0       : {distribution['r10_eq_0']:4d}  ({100*distribution['r10_eq_0']/n:.1f}%)")
    print(f"  R@10 in (0, 0.5) : {distribution['r10_lt_0.5']:4d}  ({100*distribution['r10_lt_0.5']/n:.1f}%)")
    print(f"  R@10 in [0.5, 1) : {distribution['r10_gte_0.5']:4d}  ({100*distribution['r10_gte_0.5']/n:.1f}%)")
    print(f"  R@10 = 1.0       : {distribution['r10_eq_1']:4d}  ({100*distribution['r10_eq_1']/n:.1f}%)")
    print(f"  Mean / Std       : {distribution['r10_mean']:.4f} / {distribution['r10_std']:.4f}")

    print("\n" + "=" * 85)
    print("PER-PROJECT SUMMARY  (macro-avg R@50 across issues)")
    print("=" * 85)
    print(f"  {'Project':<18} {'Issues':>6}  {'R@10':>6}  {'R@50':>6}  {'R@80':>6}  {'R@100':>6}  {'MAP':>6}  {'MRR':>6}")
    print("  " + "-" * 75)
    for proj, stats in sorted(per_project.items()):
        print(
            f"  {proj:<18} {stats['n_issues']:>6}  "
            f"{stats['recall@10']['mean']:>6.4f}  "
            f"{stats['recall@50']['mean']:>6.4f}  "
            f"{stats['recall@80']['mean']:>6.4f}  "
            f"{stats['recall@100']['mean']:>6.4f}  "
            f"{stats['map']['mean']:>6.4f}  "
            f"{stats['mrr']['mean']:>6.4f}"
        )

    print("\n" + "=" * 80)
    print(f"WORST {worst_n} ISSUES BY R@10")
    print("=" * 80)
    print(f"  {'Jira ID':<20} {'Project':<16} {'R@10':>6}  {'R@50':>6}  "
          f"{'MRR':>6}  {'MAP':>6}  {'Cands':>6}  {'Pos':>4}")
    print("  " + "-" * 78)
    for w in worst_issues:
        print(
            f"  {w['jira_id']:<20} {w['project']:<16} "
            f"{w['recall@10']:>6.4f}  {w['recall@50']:>6.4f}  "
            f"{w['mrr']:>6.4f}  {w['map']:>6.4f}  "
            f"{w['n_candidates']:>6}  {w['n_positives']:>4}"
        )

    print(f"\nFull report saved to: {output_path}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Per-issue retrieval diagnostic")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to checkpoint (.pt file)")
    parser.add_argument("--config",      required=True,
                        help="Path to config.json (saved in output_dir by train.py)")
    parser.add_argument("--val-jsonl",   required=True,
                        help="Path to val_full.jsonl (full-codebase val set)")
    parser.add_argument("--output",      required=True,
                        help="Path to write diagnosis.json")
    parser.add_argument("--batch-size",  type=int, default=16,
                        help="Inference batch size (default 16)")
    parser.add_argument("--worst-n",     type=int, default=20,
                        help="Number of worst issues to highlight (default 20)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    config_path     = Path(args.config)
    val_jsonl       = Path(args.val_jsonl)
    output_path     = Path(args.output)

    for p, name in [
        (checkpoint_path, "--checkpoint"),
        (config_path,     "--config"),
        (val_jsonl,       "--val-jsonl"),
    ]:
        if not p.exists():
            import sys; sys.exit(f"{name} not found: {p}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    diagnose(
        checkpoint_path = checkpoint_path,
        config          = config,
        val_jsonl       = val_jsonl,
        output_path     = output_path,
        batch_size      = args.batch_size,
        worst_n         = args.worst_n,
    )


if __name__ == "__main__":
    main()
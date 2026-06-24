"""
diagnose_multigpu.py — Per-Issue and Per-Project Retrieval Diagnostic (Multi-GPU)
==================================================================================
Loads a trained checkpoint and runs inference on the full-codebase val set,
then produces:

  1. Per-issue metrics (R@5/10/20/50, NDCG@10, MAP, n_candidates, n_positives)
  2. Per-project aggregates (macro-averaged across issues in that project)
  3. Global aggregate (same as training logs, for sanity check)
  4. Worst-N issues ranked by R@10 (the cases dragging the average down)
  5. Distribution summary: how many issues score R@10 = 0, <0.5, >=0.5, = 1.0

All results are written to a JSON report and printed as a readable table.

Usage:
    python diagnose_multigpu.py \\
        --checkpoint  /data1/amine/checkpoints/run_016/best_model.pt \\
        --config      /data1/amine/checkpoints/run_016/config.json \\
        --val-jsonl   /data1/amine/IMPACT_TRAINING_SAMPLES/SAMPLES_V4/test_full.jsonl \\
        --output      /data1/amine/checkpoints/run_016/diagnosis.json \\
        [--batch-size 64]     # per-GPU batch size; effective total = 64 × 4 = 256
        [--worst-n    20] \\
        [--no-amp] \\          # disable FP16 if you see NaN/Inf
        [--gpu-ids 0,1,2,3]   # restrict to specific GPUs (default: all available)

The --config flag expects the config.json saved alongside the checkpoint
(it is written automatically by train.py into output_dir/config.json).

Performance notes
-----------------
  1. AMP (FP16): halves activation memory and runs ~1.5–2× faster on
     Ampere/Turing GPUs.  Disabled on CPU/MPS automatically. Pass
     --no-amp to force FP32 (useful for debugging numerics).

  2. Deferred GPU→CPU transfer: probabilities accumulate as GPU tensors
     inside the loop, then a single torch.cat(...).cpu() at the end does
     one host-device sync instead of one per batch.

  3. DataLoader: persistent_workers keeps worker processes alive between
     batches (no fork overhead), prefetch_factor=4 means 4 batches are
     pre-tokenised and sitting in shared memory.  num_workers is set to
     min(cpu_count, 4 × n_gpus) so workers scale with GPU count.

  4. DataParallel (multi-GPU): When ≥2 CUDA GPUs are available, the model
     is wrapped with nn.DataParallel.  Each forward pass, DP:
       (a) scatters the batch evenly across all GPU devices (dim 0 split),
       (b) runs the model in parallel on each GPU,
       (c) gathers outputs back onto the primary GPU (cuda:0).
     No process spawning or torchrun needed — this is inference only.

     With 4× RTX 2080 Ti (11 GB each):
       - --batch-size 64  →  64 samples per GPU  →  256 samples per step
       - Each GPU uses ~8–9 GB VRAM with AMP enabled
       - Expected throughput gain: ~3.5–4× vs single GPU
       - Total wall-clock estimate: ~30–45 min vs 2–3 h single-GPU

  5. Checkpoint compatibility: if the checkpoint was saved from a
     DataParallel-wrapped model (keys prefixed with "module."), that prefix
     is stripped automatically before loading into the bare model.  So
     checkpoints from both plain and DP training are handled transparently.

  6. GPU selection: use --gpu-ids 0,1 to restrict to GPUs 0 and 1, or set
     CUDA_VISIBLE_DEVICES=0,1 in the environment before launching.
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
import torch.nn as nn
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


# ── Evaluation metrics ────────────────────────────────────────────────────────

def recall_at_k(scores: list[float], labels: list[int], k: int) -> float:
    if not any(labels):
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    return sum(lbl for _, lbl in ranked[:k]) / sum(labels)


def ndcg_at_k(scores: list[float], labels: list[int], k: int) -> float:
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    dcg  = sum(lbl / math.log2(i + 2) for i, (_, lbl) in enumerate(ranked[:k]))
    idcg = sum(lbl / math.log2(i + 2)
               for i, lbl in enumerate(sorted(labels, reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(scores: list[float], labels: list[int]) -> float:
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos, cum_pos, ap = 0, 0, 0.0
    for i, (_, lbl) in enumerate(ranked):
        if lbl:
            cum_pos += 1
            n_pos   += 1
            ap      += cum_pos / (i + 1)
    return ap / n_pos if n_pos else 0.0


def mean_reciprocal_rank(scores: list[float], labels: list[int]) -> float:
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    for i, (_, lbl) in enumerate(ranked):
        if lbl:
            return 1.0 / (i + 1)
    return 0.0


def project_of(jira_id: str) -> str:
    prefix  = jira_id.split("-")[0].upper()
    mapping = {
        "FLINK":     "flink",
        "KAFKA":     "kafka",
        "HADOOP":    "hadoop",
        "HDFS":      "hadoop",
        "MAPREDUCE": "hadoop",
        "YARN":      "hadoop",
    }
    return mapping.get(prefix, prefix.lower())


def aggregate(values: list[float]) -> dict:
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


def recall_distribution(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"eq_0": 0, "lt_0.5": 0, "gte_0.5": 0, "eq_1": 0,
                "mean": 0.0, "std": 0.0, "n": 0}
    mean = sum(values) / n
    std  = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return {
        "eq_0":    sum(1 for v in values if v == 0.0),
        "lt_0.5":  sum(1 for v in values if 0.0 < v < 0.5),
        "gte_0.5": sum(1 for v in values if 0.5 <= v < 1.0),
        "eq_1":    sum(1 for v in values if v == 1.0),
        "mean":    round(mean, 4),
        "std":     round(std, 4),
        "n":       n,
    }


def compute_ranking_metrics(
    all_scores:   list[float],
    all_labels:   list[int],
    all_jira_ids: list[str],
    ks:           list[int] = [5, 10, 20, 50, 80, 100],
) -> dict:
    issue_scores: dict[str, list] = defaultdict(list)
    issue_labels: dict[str, list] = defaultdict(list)

    for score, label, jid in zip(all_scores, all_labels, all_jira_ids):
        issue_scores[jid].append(score)
        issue_labels[jid].append(label)

    metrics: dict[str, list] = {f"recall@{k}": [] for k in ks}
    metrics.update({f"ndcg@{k}": [] for k in ks if k <= 20})
    metrics["map"] = []
    metrics["mrr"] = []

    for jid in issue_scores:
        s = issue_scores[jid]
        l = issue_labels[jid]
        if not any(l):
            continue
        for k in ks:
            metrics[f"recall@{k}"].append(recall_at_k(s, l, k))
        for k in ks:
            if k <= 20:
                metrics[f"ndcg@{k}"].append(ndcg_at_k(s, l, k))
        metrics["map"].append(average_precision(s, l))
        metrics["mrr"].append(mean_reciprocal_rank(s, l))

    return {k: (sum(v) / len(v) if v else 0.0) for k, v in metrics.items()}


def compute_binary_metrics(all_scores: list[float], all_labels: list[int]) -> dict:
    try:
        from sklearn.metrics import roc_auc_score, f1_score
        auc   = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.0
        preds = [1 if s >= 0.5 else 0 for s in all_scores]
        f1    = f1_score(all_labels, preds, zero_division=0)
        return {"auc_roc": auc, "f1_at_0.5": f1}
    except ImportError:
        return {}


# ── VRAM helper ───────────────────────────────────────────────────────────────

def log_vram(label: str, gpu_ids: list[int]) -> None:
    """Log allocated / total VRAM for each GPU in gpu_ids."""
    log.info("VRAM — %s:", label)
    for gid in gpu_ids:
        props = torch.cuda.get_device_properties(gid)
        alloc = torch.cuda.memory_allocated(gid) / 1e9
        total = props.total_memory / 1e9
        log.info("  GPU %d (%s): %.1f / %.1f GB", gid, props.name, alloc, total)


# ── Inference pass ────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model:    torch.nn.Module,
    loader:   DataLoader,
    device:   torch.device,
    use_amp:  bool = True,
) -> tuple[list[float], list[int], list[str], list[str]]:
    """
    Returns parallel lists of (scores, labels, jira_ids, class_paths).

    Works transparently with both a bare model and an nn.DataParallel-wrapped
    model — DP scatter/gather is handled inside the model() call.

    Key perf notes
    ──────────────
    • AMP (use_amp=True): FP16 forward pass — ~1.5–2× faster, half VRAM.
      Sigmoid and list operations stay in FP32 automatically.

    • Deferred GPU→CPU transfer: logits stay on-device as a list of tensors;
      a single torch.cat(...).cpu() at the end is one CUDA sync for the whole
      dataset instead of one per batch.

    • Non-blocking .to(): overlaps H→D copy with the previous iteration's
      GPU compute when pin_memory=True on the DataLoader.

    • With DataParallel: DP scatters the batch across all GPUs and gathers
      logits back onto the primary GPU (cuda:0) before returning.  The code
      below is identical to single-GPU — DP is transparent to the caller.
    """
    model.eval()

    prob_chunks:     list[torch.Tensor] = []
    label_chunks:    list[torch.Tensor] = []
    all_jira_ids:    list[str]          = []
    all_class_paths: list[str]          = []

    # autocast is a no-op on CPU/MPS; only meaningful on CUDA.
    amp_ctx = torch.cuda.amp.autocast() if use_amp and device.type == "cuda" \
              else torch.inference_mode()

    pbar = tqdm(loader, desc="Inference", leave=True)
    for batch in pbar:
        # Move all batch tensors to the primary device.
        # With DataParallel, DP will then scatter them to the other GPUs.
        batch = batch.to(device, non_blocking=True)

        with amp_ctx:
            logits, _, _ = model(
                batch.feature_input_ids,
                batch.feature_attention_mask,
                batch.feature_chunk_mask,
                batch.class_input_ids,
                batch.class_attention_mask,
            )

        # logits are gathered on the primary GPU by DP; safe to accumulate here.
        prob_chunks.append(torch.sigmoid(logits).detach())
        label_chunks.append(batch.labels.int().detach())
        all_jira_ids.extend(batch.jira_ids)
        all_class_paths.extend(batch.class_paths)

    # Single host-device sync for the entire dataset.
    all_scores = torch.cat(prob_chunks).cpu().tolist()
    all_labels = torch.cat(label_chunks).cpu().tolist()

    return all_scores, all_labels, all_jira_ids, all_class_paths


# ── Main diagnostic ───────────────────────────────────────────────────────────

def diagnose(
    checkpoint_path: Path,
    config:          dict,
    val_jsonl:       Path,
    output_path:     Path,
    batch_size:      int,
    worst_n:         int,
    use_amp:         bool = True,
    gpu_ids:         list[int] | None = None,
):
    tcfg = config["training"]

    # ── GPU / device setup ────────────────────────────────────────────────────
    if torch.cuda.is_available():
        available = list(range(torch.cuda.device_count()))
        if gpu_ids is None:
            gpu_ids = available
        else:
            invalid = [g for g in gpu_ids if g not in available]
            if invalid:
                raise ValueError(
                    f"Requested GPU IDs {invalid} not available. "
                    f"Available: {available}"
                )
        n_gpus     = len(gpu_ids)
        primary    = gpu_ids[0]
        device     = torch.device(f"cuda:{primary}")
        log.info(
            "Device: CUDA — %d GPU(s) [%s], primary: cuda:%d",
            n_gpus, ", ".join(f"cuda:{g}" for g in gpu_ids), primary,
        )
    elif torch.backends.mps.is_available():
        device  = torch.device("mps")
        gpu_ids = []
        n_gpus  = 0
        log.info("Device: MPS (Apple Silicon)")
    else:
        device  = torch.device("cpu")
        gpu_ids = []
        n_gpus  = 0
        log.info("Device: CPU")

    # AMP is only effective on CUDA
    if device.type == "cuda" and use_amp:
        log.info("AMP: enabled (FP16 inference)")
    elif device.type != "cuda" and use_amp:
        log.info("AMP: disabled (only effective on CUDA)")
        use_amp = False
    else:
        log.info("AMP: disabled (--no-amp flag)")

    # ── Load tokenizer and model ──────────────────────────────────────────────
    model_name = config["model"]["model_name"]
    log.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    log.info("Building model ...")
    model = build_model(config["model"]).to(device)

    log.info("Loading checkpoint: %s", checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device)

    # Extract the raw state dict (handle both bare and "model_state"-keyed saves)
    raw_state = state.get("model_state", state)

    # Strip "module." prefix produced by DataParallel when saving.
    # This makes the checkpoint compatible regardless of whether it was saved
    # from a plain model or a DataParallel-wrapped one.
    if any(k.startswith("module.") for k in raw_state.keys()):
        log.info("  Stripping 'module.' prefix from checkpoint keys "
                 "(checkpoint was saved from a DataParallel model)")
        raw_state = {k[len("module."):]: v for k, v in raw_state.items()}

    model.load_state_dict(raw_state)
    if "model_state" in state:
        log.info("  Loaded from full checkpoint (epoch %d)", state.get("epoch", "?"))
    else:
        log.info("  Loaded model weights directly")

    # Log VRAM before DataParallel expansion
    if device.type == "cuda":
        log_vram("after model load (before DataParallel)", gpu_ids)

    # ── Wrap with DataParallel ────────────────────────────────────────────────
    if n_gpus > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        log.info(
            "DataParallel: model replicated across %d GPUs [%s]",
            n_gpus, ", ".join(f"cuda:{g}" for g in gpu_ids),
        )
    else:
        log.info("DataParallel: disabled (single device)")

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

    # Effective batch size: each GPU processes `batch_size` samples per step.
    # Total samples per step = batch_size × n_gpus.
    effective_batch = batch_size * max(n_gpus, 1)

    # DataLoader workers: 4 per GPU, capped at cpu_count.
    # More workers keep the GPUs fed during tokenisation.
    n_workers = min(os.cpu_count() or 4, max(16, 4 * max(n_gpus, 1)))
    log.info(
        "DataLoader: %d workers  |  per-GPU batch %d  |  effective batch %d",
        n_workers, batch_size, effective_batch,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size         = effective_batch,
        shuffle            = False,
        collate_fn         = collate_fn,
        num_workers        = n_workers,
        pin_memory         = device.type == "cuda",  # required for non_blocking=True
        persistent_workers = n_workers > 0,
        prefetch_factor    = 4 if n_workers > 0 else None,
    )
    log.info("Val samples: %d", len(val_ds))

    # ── Run inference ─────────────────────────────────────────────────────────
    all_scores, all_labels, all_jira_ids, all_class_paths = run_inference(
        model, val_loader, device, use_amp=use_amp
    )

    if device.type == "cuda":
        log_vram("after inference", gpu_ids)

    # ── Group by issue ────────────────────────────────────────────────────────
    log.info("Grouping %d predictions by issue ...", len(all_scores))
    issue_scores: dict[str, list] = defaultdict(list)
    issue_labels: dict[str, list] = defaultdict(list)

    for score, label, jid in tqdm(
        zip(all_scores, all_labels, all_jira_ids),
        total=len(all_scores),
        desc="Group by issue",
        leave=True,
    ):
        issue_scores[jid].append(score)
        issue_labels[jid].append(label)
    log.info("  → %d distinct issues seen", len(issue_scores))

    KS      = [5, 10, 20, 50, 80, 100]
    KS_DIST = [10, 20, 50, 80, 100]
    KS_NDCG = [k for k in KS if k <= 20]

    # ── Per-issue metrics ─────────────────────────────────────────────────────
    log.info("Computing per-issue metrics ...")
    per_issue: dict[str, dict] = {}
    n_skipped = 0
    for jid in tqdm(list(issue_scores.keys()), desc="Per-issue metrics", leave=True):
        s, l = issue_scores[jid], issue_labels[jid]
        if not any(l):
            n_skipped += 1
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
        for k in KS_NDCG:
            entry[f"ndcg@{k}"] = round(ndcg_at_k(s, l, k), 4)
        per_issue[jid] = entry

    log.info("Computed metrics for %d issues (skipped %d with no positives)",
             len(per_issue), n_skipped)

    # ── Per-project aggregates ────────────────────────────────────────────────
    project_issues: dict[str, list[str]] = defaultdict(list)
    for jid, entry in per_issue.items():
        project_issues[entry["project"]].append(jid)
    log.info("Aggregating across %d projects: %s",
             len(project_issues), sorted(project_issues.keys()))

    per_project: dict[str, dict] = {}
    for proj, jids in tqdm(sorted(project_issues.items()),
                           desc="Per-project aggregates", leave=True):
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
           for k in KS_NDCG},
        "map": round(sum(e["map"] for e in all_entries) / len(all_entries), 4),
        "mrr": round(sum(e["mrr"] for e in all_entries) / len(all_entries), 4),
    }

    # ── Recall distributions ──────────────────────────────────────────────────
    distribution: dict[str, dict] = {}
    for k in KS_DIST:
        values = [e[f"recall@{k}"] for e in all_entries]
        distribution[f"recall@{k}"] = recall_distribution(values)

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
        "per_issue":     per_issue,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing report to %s ...", output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("  → wrote %d issues, %d projects", len(per_issue), len(per_project))

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

    print("\n" + "=" * 78)
    print("RECALL DISTRIBUTIONS")
    print("=" * 78)
    n = len(all_entries)
    print(f"  {'k':>5} | {'=0':>10}  {'(0,0.5)':>10}  {'[0.5,1)':>10}  {'=1':>10}  "
          f"{'mean':>6}  {'std':>6}")
    print("  " + "-" * 74)
    for k in KS_DIST:
        d = distribution[f"recall@{k}"]
        print(
            f"  R@{k:<3} | "
            f"{d['eq_0']:>4d} ({100*d['eq_0']/n:>4.1f}%)  "
            f"{d['lt_0.5']:>4d} ({100*d['lt_0.5']/n:>4.1f}%)  "
            f"{d['gte_0.5']:>4d} ({100*d['gte_0.5']/n:>4.1f}%)  "
            f"{d['eq_1']:>4d} ({100*d['eq_1']/n:>4.1f}%)  "
            f"{d['mean']:>6.4f}  {d['std']:>6.4f}"
        )

    print("\n" + "=" * 85)
    print("PER-PROJECT SUMMARY  (macro-avg R@50 across issues)")
    print("=" * 85)
    print(f"  {'Project':<18} {'Issues':>6}  {'R@10':>6}  {'R@50':>6}  {'R@80':>6}  "
          f"{'R@100':>6}  {'MAP':>6}  {'MRR':>6}")
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
    parser = argparse.ArgumentParser(description="Per-issue retrieval diagnostic (multi-GPU)")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to checkpoint (.pt file)")
    parser.add_argument("--config",      required=True,
                        help="Path to config.json saved by train.py")
    parser.add_argument("--val-jsonl",   required=True,
                        help="Path to val_full.jsonl (full-codebase val set)")
    parser.add_argument("--output",      required=True,
                        help="Path to write diagnosis.json")
    parser.add_argument("--batch-size",  type=int, default=64,
                        help="Per-GPU batch size (default 64). "
                             "With AMP, each RTX 2080 Ti uses ~8-9 GB VRAM at this size. "
                             "Effective total = batch_size × n_gpus (256 across 4 GPUs). "
                             "Use 32 if you OOM on a single GPU, 128 for more throughput.")
    parser.add_argument("--worst-n",     type=int, default=20,
                        help="Number of worst issues to highlight (default 20)")
    parser.add_argument("--no-amp",      action="store_true",
                        help="Disable AMP/FP16 inference (use if you see NaN/Inf)")
    parser.add_argument("--gpu-ids",     type=str, default=None,
                        help="Comma-separated CUDA GPU IDs to use "
                             "(e.g. '0,1,2,3' for all four 2080 Tis, or '0,1' for two). "
                             "Defaults to all available CUDA GPUs. "
                             "Alternatively, set CUDA_VISIBLE_DEVICES before launching.")
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

    # Parse optional GPU-ID list
    gpu_ids: list[int] | None = None
    if args.gpu_ids is not None:
        try:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
        except ValueError:
            import sys; sys.exit(
                f"--gpu-ids must be a comma-separated list of integers "
                f"(e.g. '0,1,2,3'), got: {args.gpu_ids!r}"
            )

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    diagnose(
        checkpoint_path = checkpoint_path,
        config          = config,
        val_jsonl       = val_jsonl,
        output_path     = output_path,
        batch_size      = args.batch_size,
        worst_n         = args.worst_n,
        use_amp         = not args.no_amp,
        gpu_ids         = gpu_ids,
    )


if __name__ == "__main__":
    main()
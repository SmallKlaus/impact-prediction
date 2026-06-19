"""
diagnose_zeroshot_baseline.py — Zero-Shot Cosine-Similarity Baseline
=================================================================================
Provides a neural *lower-bound* baseline for the bi-encoder pipeline by running
a pretrained model **without any task-specific fine-tuning**.

Ranking signal
--------------
For each (issue, candidate class) pair the score is the cosine similarity
between their raw CLS vectors produced by the frozen, pretrained backbone:

    score(issue, class) = cos( CLS_f , CLS_c )
                        = (CLS_f · CLS_c) / (‖CLS_f‖ ‖CLS_c‖)

No projection head, no interaction MLP, no training — this is the model
"out of the box".

The feature embedding is computed as the mean of individual chunk CLS vectors
(identical pooling strategy to the trained bi-encoder's encode_feature()),
which ensures the only difference between the two systems is the presence or
absence of task-specific fine-tuning.

Why this baseline matters (thesis framing)
------------------------------------------
Including a zero-shot baseline directly answers the question: *"How much of the
retrieval performance comes from the model's pre-trained representations versus
the task-specific training procedure?"* The three-way comparison:

    BM25  <  Zero-shot model  <  Bi-encoder (trained)

quantifies the contribution of (a) switching from lexical to semantic search,
and (b) the supervised fine-tuning of the semantic model.  Both contributions
are methodologically important for a software-engineering intelligence thesis.

Report structure (identical to diagnose.py)
-------------------------------------------
1. Per-issue metrics  : R@5/10/20/50/80/100, NDCG@10, MAP, MRR
2. Per-project aggregate (macro-averaged, same project bucketing)
3. Global aggregate
4. Recall distributions (same k values as diagnose.py)
5. Worst-N issues by R@10
6. Full per_issue dict (JSON)

Usage
-----
    python diagnose_zeroshot_baseline.py \\
        --model-name  /path/to/model-base \\
        --val-jsonl   /data/val_full.jsonl \\
        --output      /data/baseline_val.json \\
        [--cuda-devices 0 1 2] \\
        [--test-jsonl /data/test_full.jsonl] \\
        [--test-output /data/baseline_test.json] \\
        [--batch-size 32] \\
        [--worst-n   20]

    # Using config.json to infer model path (convenient for one-stop comparison)
    python diagnose_zeroshot_baseline.py \\
        --config      /data/checkpoints/run_015/config.json \\
        --val-jsonl   /data/val_full.jsonl \\
        --output      /data/baseline_val.json

    # Multi-GPU: spread across CUDA devices 0 and 1
    python diagnose_zeroshot_baseline.py \\
        --model-name  /path/to/model-base \\
        --val-jsonl   /data/val_full.jsonl \\
        --output      /data/baseline_val.json \\
        --cuda-devices 0 1

Notes
-----
* The script respects the same max_chunk_tokens / max_class_tokens / max_chunks
  values as the trained model so tokenisation is identical.
* If --config is provided, model_name, max_chunk_tokens, max_class_tokens, and
  max_chunks are read from it; CLI overrides take precedence.
* Embeddings are computed with torch.no_grad() throughout; no gradient memory
  is allocated.
* Class embeddings are cached per sha_before snapshot (same optimisation used
  by HardNegativeMiner): each codebase state is encoded once, then re-used for
  all issues that share it.
* Multi-GPU support uses torch.nn.DataParallel.  Pass --cuda-devices to select
  which CUDA device IDs participate (e.g. --cuda-devices 0 1).  The first ID
  in the list is the primary (gathering) device.  Omitting the flag defaults to
  a single GPU (cuda:0) when CUDA is available.  The batch is split across GPUs
  automatically; all parameters remain frozen throughout.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Reuse the existing dataset / collation infrastructure so the data pipeline
# is byte-for-byte identical to diagnose.py.
from dataset import ImpactSampleDataset, collate_fn, SampleBatch

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Backbone wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _BackboneCLSWrapper(torch.nn.Module):
    """
    Thin nn.Module shim around a HuggingFace AutoModel.

    Returning a plain tensor (last_hidden_state) instead of a ModelOutput
    dataclass ensures that torch.nn.DataParallel can gather outputs across
    devices correctly in all supported PyTorch versions.  Parameters are
    expected to be frozen before this wrapper is constructed; the wrapper
    itself introduces no new learnable parameters.
    """

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return last_hidden_state: (batch, seq_len, hidden_size)."""
        return self.backbone(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers  (mirrors diagnose.py exactly — kept local to keep the script
# self-contained and to avoid any accidental coupling to future changes in
# train.py's metric implementations)
# ─────────────────────────────────────────────────────────────────────────────

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
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    for i, (_, lbl) in enumerate(ranked):
        if lbl:
            return 1.0 / (i + 1)
    return 0.0


def aggregate(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "n": 0}
    n    = len(values)
    mean = sum(values) / n
    std  = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return {
        "mean": round(mean, 4), "min": round(min(values), 4),
        "max":  round(max(values), 4), "std": round(std, 4), "n": n,
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


def project_of(jira_id: str) -> str:
    """Identical to diagnose.py — Hadoop sub-projects collapsed into 'hadoop'."""
    prefix = jira_id.split("-")[0].upper()
    mapping = {
        "FLINK":     "flink",
        "KAFKA":     "kafka",
        "HADOOP":    "hadoop",
        "HDFS":      "hadoop",
        "MAPREDUCE": "hadoop",
        "YARN":      "hadoop",
    }
    return mapping.get(prefix, prefix.lower())


# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot Encoding
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _cls(
    backbone:       torch.nn.Module,
    input_ids:      torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Extract the [CLS] hidden state from the pretrained backbone.

    Uses the first token (index 0) of last_hidden_state as the
    sequence-level representation — identical to the ImpactScoreModel._encode_single
    convention, ensuring the embeddings are semantically comparable.

    ``backbone`` is expected to be a ``_BackboneCLSWrapper`` (possibly wrapped
    in ``torch.nn.DataParallel``).  The wrapper's forward returns
    ``last_hidden_state`` as a plain tensor, which DataParallel can gather
    across devices without version-specific workarounds.

    Args:
        input_ids      : (..., seq_len) — arbitrary leading batch dimensions
        attention_mask : (..., seq_len)

    Returns:
        cls : (..., hidden_size)  L2-normalised CLS vector
    """
    orig_shape = input_ids.shape[:-1]
    seq_len    = input_ids.shape[-1]

    flat_ids   = input_ids.view(-1, seq_len)
    flat_mask  = attention_mask.view(-1, seq_len)

    # backbone returns last_hidden_state tensor: (batch_flat, seq_len, hidden)
    last_hidden = backbone(flat_ids, attention_mask=flat_mask)
    cls_hidden  = last_hidden[:, 0, :]   # (batch_flat, hidden)

    # L2-normalise so cosine similarity reduces to a dot product
    cls_norm = F.normalize(cls_hidden, dim=-1)
    return cls_norm.view(*orig_shape, cls_hidden.shape[-1])


@torch.no_grad()
def run_inference(
    backbone:         torch.nn.Module,
    loader:           DataLoader,
    device:           torch.device,
) -> tuple[list[float], list[int], list[str], list[str]]:
    """
    Score every (issue, class) pair in the dataloader using zero-shot cosine
    similarity between their CLS embeddings.

    The feature embedding is the *mean* of individual chunk CLS vectors,
    mirroring ImpactScoreModel.encode_feature() with use_attention_pool=False.
    This isolates the effect of task-specific training: pooling strategy and
    tokenisation are held constant.

    Returns parallel lists (scores, labels, jira_ids, class_paths).
    """
    backbone.eval()
    all_scores, all_labels, all_jira_ids, all_class_paths = [], [], [], []

    pbar = tqdm(loader, desc="Zero-shot inference", leave=True)
    for batch in pbar:
        batch: SampleBatch = batch.to(device)

        # ── Feature side ────────────────────────────────────────────────
        # batch.feature_input_ids  : (B, n_chunks, seq_len)
        # batch.feature_chunk_mask : (B, n_chunks) — 1=real, 0=padding
        chunk_cls = _cls(
            backbone,
            batch.feature_input_ids,
            batch.feature_attention_mask,
        )   # (B, n_chunks, hidden)

        # Mean-pool over real chunks only
        mask_f = batch.feature_chunk_mask.float().unsqueeze(-1)   # (B, n_chunks, 1)
        f_emb  = (chunk_cls * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        # f_emb : (B, hidden) — already L2-normalised per token, but the mean
        # may not have unit norm; re-normalise to keep cosine semantics exact.
        f_emb  = F.normalize(f_emb, dim=-1)                       # (B, hidden)

        # ── Class side ──────────────────────────────────────────────────
        # batch.class_input_ids : (B, seq_len)
        c_emb = _cls(
            backbone,
            batch.class_input_ids,
            batch.class_attention_mask,
        )   # (B, hidden) — already L2-normalised

        # ── Cosine similarity = dot product after normalisation ──────────
        scores = (f_emb * c_emb).sum(dim=-1)   # (B,)

        all_scores.extend(scores.cpu().tolist())
        all_labels.extend(batch.labels.int().cpu().tolist())
        all_jira_ids.extend(batch.jira_ids)
        all_class_paths.extend(batch.class_paths)

    return all_scores, all_labels, all_jira_ids, all_class_paths


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic computation (mirrors diagnose.py structure)
# ─────────────────────────────────────────────────────────────────────────────

def compute_diagnostics(
    all_scores:    list[float],
    all_labels:    list[int],
    all_jira_ids:  list[str],
    all_class_paths: list[str],
    worst_n:       int,
) -> tuple[dict, dict, dict, dict, list]:
    """
    Group by issue, compute per-issue metrics, aggregate.

    Returns
    -------
    per_issue      : {jira_id: metric_dict}
    per_project    : {project: aggregate_dict}
    global_metrics : dict
    distribution   : {f"recall@{k}": distribution_dict, ...}
    worst_issues   : list of worst-n dicts
    """
    KS      = [5, 10, 20, 50, 80, 100]
    KS_DIST = [10, 20, 50, 80, 100]
    KS_NDCG = [k for k in KS if k <= 20]

    # Group predictions by issue
    issue_scores: dict[str, list] = defaultdict(list)
    issue_labels: dict[str, list] = defaultdict(list)
    for score, label, jid in zip(all_scores, all_labels, all_jira_ids):
        issue_scores[jid].append(score)
        issue_labels[jid].append(label)

    # Per-issue metrics
    per_issue: dict[str, dict] = {}
    n_skipped = 0
    for jid in issue_scores:
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
            entry[f"recall@{k}"] = round(recall_at_k(s, l, k), 4)
        for k in KS_NDCG:
            entry[f"ndcg@{k}"]   = round(ndcg_at_k(s, l, k), 4)
        per_issue[jid] = entry

    log.info(
        "Per-issue metrics: %d issues (skipped %d with no positives in split)",
        len(per_issue), n_skipped,
    )

    # Per-project aggregates
    project_issues: dict[str, list[str]] = defaultdict(list)
    for jid, entry in per_issue.items():
        project_issues[entry["project"]].append(jid)

    per_project: dict[str, dict] = {}
    for proj, jids in sorted(project_issues.items()):
        ents = [per_issue[j] for j in jids]
        per_project[proj] = {
            "n_issues": len(jids),
            **{f"recall@{k}": aggregate([e[f"recall@{k}"] for e in ents])
               for k in KS},
            "ndcg@10": aggregate([e["ndcg@10"]  for e in ents]),
            "map":     aggregate([e["map"]       for e in ents]),
            "mrr":     aggregate([e["mrr"]       for e in ents]),
        }

    # Global aggregates
    all_entries = list(per_issue.values())
    global_metrics = {
        "n_issues": len(all_entries),
        **{f"recall@{k}": round(
                sum(e[f"recall@{k}"] for e in all_entries) / len(all_entries), 4
            ) for k in KS},
        **{f"ndcg@{k}": round(
                sum(e[f"ndcg@{k}"] for e in all_entries) / len(all_entries), 4
            ) for k in KS_NDCG},
        "map": round(sum(e["map"] for e in all_entries) / len(all_entries), 4),
        "mrr": round(sum(e["mrr"] for e in all_entries) / len(all_entries), 4),
    }

    # Recall distributions
    distribution: dict[str, dict] = {}
    for k in KS_DIST:
        values = [e[f"recall@{k}"] for e in all_entries]
        distribution[f"recall@{k}"] = recall_distribution(values)

    # Worst-N by R@10
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

    return per_issue, per_project, global_metrics, distribution, worst_issues


def print_report(
    split_name:     str,
    global_metrics: dict,
    distribution:   dict,
    per_project:    dict,
    worst_issues:   list,
    worst_n:        int,
    model_name:     str,
):
    """Print a console summary identical in format to diagnose.py."""
    KS      = [5, 10, 20, 50, 80, 100]
    KS_DIST = [10, 20, 50, 80, 100]

    n = global_metrics["n_issues"]

    print("\n" + "=" * 70)
    print(f"GLOBAL METRICS — {model_name} Zero-Shot Baseline  [{split_name}]")
    print("=" * 70)
    print(f"  Issues evaluated : {n}")
    for k in KS:
        print(f"  Recall@{k:<3}       : {global_metrics[f'recall@{k}']:.4f}")
    print(f"  NDCG@10          : {global_metrics['ndcg@10']:.4f}")
    print(f"  MAP              : {global_metrics['map']:.4f}")
    print(f"  MRR              : {global_metrics['mrr']:.4f}")

    print("\n" + "=" * 78)
    print("RECALL DISTRIBUTIONS")
    print("=" * 78)
    print(f"  {'k':>5} | {'=0':>10}  {'(0,0.5)':>10}  {'[0.5,1)':>10}  {'=1':>10}  "
          f"{'mean':>6}  {'std':>6}")
    print("  " + "-" * 74)
    for k in KS_DIST:
        d = distribution[f"recall@{k}"]
        print(
            f"  R@{k:<3} | "
            f"{d['eq_0']:>4d} ({100*d['eq_0']/max(n,1):>4.1f}%)  "
            f"{d['lt_0.5']:>4d} ({100*d['lt_0.5']/max(n,1):>4.1f}%)  "
            f"{d['gte_0.5']:>4d} ({100*d['gte_0.5']/max(n,1):>4.1f}%)  "
            f"{d['eq_1']:>4d} ({100*d['eq_1']/max(n,1):>4.1f}%)  "
            f"{d['mean']:>6.4f}  {d['std']:>6.4f}"
        )

    print("\n" + "=" * 85)
    print("PER-PROJECT SUMMARY  (macro-avg R@50)")
    print("=" * 85)
    print(f"  {'Project':<18} {'Issues':>6}  {'R@10':>6}  {'R@50':>6}  "
          f"{'R@80':>6}  {'R@100':>6}  {'MAP':>6}  {'MRR':>6}")
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


def run_split(
    split_name:       str,
    jsonl_path:       Path,
    output_path:      Path,
    backbone:         torch.nn.Module,
    tokenizer,
    device:           torch.device,
    max_chunk_tokens: int,
    max_class_tokens: int,
    max_chunks:       int,
    batch_size:       int,
    num_workers:      int,
    worst_n:          int,
    model_name:       str,
):
    """Full pipeline for one data split (val or test)."""
    log.info("=" * 60)
    log.info("Split: %s  →  %s", split_name, jsonl_path)
    log.info("=" * 60)

    ds = ImpactSampleDataset(
        jsonl_path       = jsonl_path,
        tokenizer        = tokenizer,
        max_chunk_tokens = max_chunk_tokens,
        max_class_tokens = max_class_tokens,
        max_chunks       = max_chunks,
        label_smoothing  = 0.0,
    )
    loader = DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = num_workers,
        pin_memory  = device.type == "cuda",
    )
    log.info("Samples: %d", len(ds))

    all_scores, all_labels, all_jira_ids, all_class_paths = run_inference(
        backbone, loader, device
    )

    per_issue, per_project, global_metrics, distribution, worst_issues = \
        compute_diagnostics(
            all_scores, all_labels, all_jira_ids, all_class_paths, worst_n
        )

    report = {
        "baseline":       f"{model_name}_zero_shot",
        "model_name":     model_name,
        "split":          split_name,
        "jsonl_path":     str(jsonl_path),
        "scoring_method": "cosine_similarity_CLS",
        "description": (
            f"Pretrained {model_name} with no fine-tuning. "
            "Feature: mean of chunk CLS vectors (L2-normalised). "
            "Class: CLS vector (L2-normalised). "
            "Score: cosine similarity = dot product after normalisation."
        ),
        "global":      global_metrics,
        "distribution": distribution,
        "per_project": per_project,
        "worst_issues": worst_issues,
        "per_issue":   per_issue,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_report(split_name, global_metrics, distribution,
                 per_project, worst_issues, worst_n, model_name)
    print(f"\nReport saved to: {output_path}\n")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot model cosine-similarity baseline diagnostic. "
            "Produces reports in the same format as diagnose.py for "
            "direct comparison."
        )
    )

    # ── Model identification ───────────────────────────────────────────────
    parser.add_argument(
        "--model-name", default=None,
        help="Path or HuggingFace name of the pretrained model "
             "(e.g. '/path/to/model-base' or "
             "'microsoft/unixcoder-base'). "
             "Overrides the value inferred from --config.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to the trained bi-encoder config.json. "
             "Used to infer model_name, tokenisation settings, and num_workers. "
             "All values can still be overridden by explicit CLI flags.",
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    parser.add_argument("--val-jsonl",  required=True,
                        help="Validation JSONL (val_full.jsonl)")
    parser.add_argument("--output",     required=True,
                        help="Output JSON report for the val split")
    parser.add_argument("--test-jsonl", default=None,
                        help="Test JSONL (test_full.jsonl) — optional; "
                             "if provided the test split is evaluated too.")
    parser.add_argument("--test-output", default=None,
                        help="Output JSON report for the test split. "
                             "Defaults to <output_stem>_test.json when "
                             "--test-jsonl is given.")

    # ── Tokenisation ──────────────────────────────────────────────────────────
    parser.add_argument("--max-chunk-tokens", type=int, default=None,
                        help="Token budget per feature chunk (default: from config or 512)")
    parser.add_argument("--max-class-tokens", type=int, default=None,
                        help="Token budget for class text (default: from config or 512)")
    parser.add_argument("--max-chunks",       type=int, default=None,
                        help="Max feature chunks per sample (default: from config or 8)")

    # ── Inference ─────────────────────────────────────────────────────────────
    parser.add_argument("--batch-size",  type=int, default=32,
                        help="Inference batch size (default 32)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader worker count (default: from config or 4)")
    parser.add_argument("--worst-n",     type=int, default=20,
                        help="Number of worst issues to highlight (default 20)")

    # ── Device ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--cuda-devices", type=int, nargs="+", default=None,
        metavar="ID",
        help=(
            "CUDA device IDs to use, in order of preference.  "
            "The first ID is the primary (output-gathering) device. "
            "Pass a single ID to pin to that GPU (e.g. --cuda-devices 1). "
            "Pass multiple IDs to enable DataParallel "
            "(e.g. --cuda-devices 0 1 2). "
            "Omit to use cuda:0 when CUDA is available, or fall back to "
            "MPS / CPU."
        ),
    )

    args = parser.parse_args()

    # ── Merge config + CLI ────────────────────────────────────────────────────
    cfg = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            import sys; sys.exit(f"--config not found: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        log.info("Loaded config from %s", config_path)

    # Priority: explicit CLI arg > config > default
    model_name = (
        args.model_name
        or cfg.get("model", {}).get("model_name")
        or "microsoft/unixcoder-base"
    )
    max_chunk_tokens = (
        args.max_chunk_tokens
        or cfg.get("training", {}).get("max_chunk_tokens", 512)
    )
    max_class_tokens = (
        args.max_class_tokens
        or cfg.get("training", {}).get("max_class_tokens", 512)
    )
    max_chunks = (
        args.max_chunks
        or cfg.get("training", {}).get("max_chunks", 4)
    )
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else cfg.get("training", {}).get("num_workers", 4)
    )

    val_jsonl  = Path(args.val_jsonl)
    output     = Path(args.output)

    if not val_jsonl.exists():
        import sys; sys.exit(f"--val-jsonl not found: {val_jsonl}")

    # ── Device / multi-GPU selection ──────────────────────────────────────────
    import sys

    cuda_available = torch.cuda.is_available()

    if cuda_available:
        n_cuda = torch.cuda.device_count()

        if args.cuda_devices is not None:
            # Validate requested IDs
            bad = [d for d in args.cuda_devices if d < 0 or d >= n_cuda]
            if bad:
                sys.exit(
                    f"[ERROR] Requested CUDA device(s) {bad} are not available. "
                    f"This system has {n_cuda} CUDA device(s): "
                    f"{list(range(n_cuda))}."
                )
            # Deduplicate while preserving order
            seen, device_ids = set(), []
            for d in args.cuda_devices:
                if d not in seen:
                    seen.add(d)
                    device_ids.append(d)
        else:
            device_ids = [0]   # default: single GPU 0

        primary_device = torch.device(f"cuda:{device_ids[0]}")

    elif torch.backends.mps.is_available():
        if args.cuda_devices is not None:
            log.warning(
                "--cuda-devices was specified but CUDA is not available; "
                "falling back to MPS."
            )
        device_ids     = []
        primary_device = torch.device("mps")

    else:
        if args.cuda_devices is not None:
            log.warning(
                "--cuda-devices was specified but neither CUDA nor MPS is "
                "available; falling back to CPU."
            )
        device_ids     = []
        primary_device = torch.device("cpu")

    use_multi_gpu = len(device_ids) > 1

    log.info("Model      : %s", model_name)
    log.info("Chunks     : max %d × %d tokens", max_chunks, max_chunk_tokens)
    log.info("Class tok  : max %d tokens", max_class_tokens)
    log.info("Batch size : %d", args.batch_size)
    if use_multi_gpu:
        log.info(
            "Device     : DataParallel across CUDA devices %s  (primary: %s)",
            device_ids, primary_device,
        )
    else:
        log.info("Device     : %s", primary_device)

    # ── Load pretrained model (NO fine-tuning weights) ────────────────────────
    log.info("Loading pretrained %s backbone (zero-shot) ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _base     = AutoModel.from_pretrained(model_name)

    # Freeze all parameters — this is a pure inference baseline; gradients
    # are never needed and keeping them off saves memory.
    for p in _base.parameters():
        p.requires_grad = False
    _base.eval()

    n_params = sum(p.numel() for p in _base.parameters())

    # Wrap so forward() returns last_hidden_state as a plain tensor.
    # This is required for DataParallel's gather to work correctly across all
    # PyTorch versions (ModelOutput dataclasses are not reliably gathered).
    backbone: torch.nn.Module = _BackboneCLSWrapper(_base).to(primary_device)
    backbone.eval()

    if use_multi_gpu:
        backbone = torch.nn.DataParallel(backbone, device_ids=device_ids)
        log.info(
            "Backbone loaded: %d parameters (all frozen) — "
            "DataParallel replicas on devices %s",
            n_params, device_ids,
        )
    else:
        log.info("Backbone loaded: %d parameters (all frozen)", n_params)

    shared_kwargs = dict(
        backbone         = backbone,
        tokenizer        = tokenizer,
        device           = primary_device,
        max_chunk_tokens = max_chunk_tokens,
        max_class_tokens = max_class_tokens,
        max_chunks       = max_chunks,
        batch_size       = args.batch_size,
        num_workers      = num_workers,
        worst_n          = args.worst_n,
        model_name       = model_name,
    )

    # ── Validation split ───────────────────────────────────────────────────────
    run_split("val", val_jsonl, output, **shared_kwargs)

    # ── Test split (optional) ──────────────────────────────────────────────────
    if args.test_jsonl:
        test_jsonl = Path(args.test_jsonl)
        if not test_jsonl.exists():
            log.warning("--test-jsonl not found: %s — skipping test split.", test_jsonl)
        else:
            if args.test_output:
                test_output = Path(args.test_output)
            else:
                test_output = output.parent / (output.stem + "_test" + output.suffix)
            run_split("test", test_jsonl, test_output, **shared_kwargs)


if __name__ == "__main__":
    main()
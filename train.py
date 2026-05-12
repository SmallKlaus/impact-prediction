"""
train.py — Training Loop with Frozen Warmup and Gradual Unfreeze
=================================================================
Training schedule:
  Phase 1 (warmup_epochs):     Backbone fully frozen.
                                Only projection heads and interaction MLP train.
                                Higher LR for new layers (lr_head).

  Phase 2 (unfreeze_epoch onwards): Top n_unfreeze_layers backbone layers thawed.
                                Backbone gets a much lower LR (lr_backbone)
                                to avoid catastrophic forgetting.
                                New layers keep lr_head.

Evaluation metrics (computed each epoch on validation set):
  - Recall@k for k in [5, 10, 20, 50]
  - NDCG@k   for k in [5, 10, 20]
  - MAP
  - AUC-ROC, F1@0.5 (for completeness)

Recall@k and NDCG@k are the primary metrics since the downstream task is
ranking classes by impact probability — classification metrics are secondary.

Usage:
    python train.py --config config.json

config.json example:
    {
      "train_jsonl":        "data/training_samples/train.jsonl",
      "val_jsonl":          "data/training_samples/val.jsonl",
      "output_dir":         "checkpoints/run_001",
      "model": {
        "model_name":         "microsoft/unixcoder-base",
        "proj_dim":           256,
        "interaction_hidden": 256,
        "dropout":            0.1,
        "freeze_backbone":    true,
        "n_unfreeze_layers":  0
      },
      "loss": {
        "focal_alpha":        0.75,
        "focal_gamma":        2.0,
        "lambda_contrastive": 0.1,
        "contrastive_temp":   0.07
      },
      "training": {
        "batch_size":         32,
        "warmup_epochs":      2,
        "total_epochs":       10,
        "unfreeze_epoch":     3,
        "n_unfreeze_layers":  2,
        "lr_head":            1e-4,
        "lr_backbone":        1e-5,
        "weight_decay":       0.01,
        "max_grad_norm":      1.0,
        "max_chunk_tokens":   512,
        "max_class_tokens":   512,
        "max_chunks":         8,
        "label_smoothing":    0.0,
        "num_workers":        4,
        "eval_every_n_steps": 500,
        "patience":           3
      }
    }
"""

from __future__ import annotations
from tqdm import tqdm
import os
import ssl
import requests
import urllib3

# Disable urllib3 SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Disable standard library SSL checks
ssl._create_default_https_context = ssl._create_unverified_context

# Force requests to ignore SSL verification
os.environ['CURL_CA_BUNDLE'] = ''


import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model   import ImpactScoreModel, build_model
from dataset import build_dataloaders, SampleBatch
from loss    import CombinedLoss, build_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Evaluation metrics ────────────────────────────────────────────────────────

def recall_at_k(scores: list[float], labels: list[int], k: int) -> float:
    """Recall@k: fraction of positives appearing in top-k ranked items."""
    if not any(labels):
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    top_k_labels = [lbl for _, lbl in ranked[:k]]
    n_pos = sum(labels)
    return sum(top_k_labels) / n_pos


def ndcg_at_k(scores: list[float], labels: list[int], k: int) -> float:
    """NDCG@k (binary relevance)."""
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    dcg = sum(
        lbl / math.log2(i + 2)
        for i, (_, lbl) in enumerate(ranked[:k])
    )
    ideal = sorted(labels, reverse=True)
    idcg = sum(
        lbl / math.log2(i + 2)
        for i, lbl in enumerate(ideal[:k])
    )
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(scores: list[float], labels: list[int]) -> float:
    """Average Precision for a single query."""
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos, cum_pos, ap = 0, 0, 0.0
    for i, (_, lbl) in enumerate(ranked):
        if lbl:
            cum_pos += 1
            n_pos   += 1
            ap      += cum_pos / (i + 1)
    return ap / n_pos if n_pos else 0.0


def compute_ranking_metrics(
    all_scores: list[float],
    all_labels: list[int],
    all_jira_ids: list[str],
    ks: list[int] = [5, 10, 20, 50],
) -> dict:
    """
    Compute per-issue ranking metrics, then macro-average across issues.

    Groups predictions by jira_id so each issue is one query and its
    candidate classes are the ranked items.
    """
    from collections import defaultdict
    issue_scores: dict[str, list] = defaultdict(list)
    issue_labels: dict[str, list] = defaultdict(list)

    for score, label, jid in zip(all_scores, all_labels, all_jira_ids):
        issue_scores[jid].append(score)
        issue_labels[jid].append(label)

    metrics: dict[str, list] = {f"recall@{k}": [] for k in ks}
    metrics.update({f"ndcg@{k}": [] for k in ks if k <= 20})
    metrics["map"] = []

    for jid in issue_scores:
        s = issue_scores[jid]
        l = issue_labels[jid]
        if not any(l):
            continue   # skip issues with no positives in this split
        for k in ks:
            metrics[f"recall@{k}"].append(recall_at_k(s, l, k))
        for k in ks:
            if k <= 20:
                metrics[f"ndcg@{k}"].append(ndcg_at_k(s, l, k))
        metrics["map"].append(average_precision(s, l))

    return {k: (sum(v) / len(v) if v else 0.0) for k, v in metrics.items()}


def compute_binary_metrics(
    all_scores: list[float],
    all_labels: list[int],
) -> dict:
    """AUC-ROC and F1@0.5 threshold."""
    try:
        from sklearn.metrics import roc_auc_score, f1_score
        auc  = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.0
        preds = [1 if s >= 0.5 else 0 for s in all_scores]
        f1    = f1_score(all_labels, preds, zero_division=0)
        return {"auc_roc": auc, "f1_at_0.5": f1}
    except ImportError:
        return {}


# ── Evaluation pass ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:      ImpactScoreModel,
    loader:     DataLoader,
    loss_fn:    CombinedLoss,
    device:     torch.device,
) -> dict:
    model.eval()

    all_scores, all_labels, all_jira_ids = [], [], []
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        batch = batch.to(device)

        logits, f_proj, c_proj = model(
            batch.feature_input_ids,
            batch.feature_attention_mask,
            batch.feature_chunk_mask,
            batch.class_input_ids,
            batch.class_attention_mask,
        )
        _, breakdown = loss_fn(logits, f_proj, c_proj, batch.labels)
        total_loss  += breakdown["loss_total"]
        n_batches   += 1

        probs = torch.sigmoid(logits).cpu().tolist()
        lbls  = batch.labels.int().cpu().tolist()
        all_scores.extend(probs)
        all_labels.extend(lbls)
        all_jira_ids.extend(batch.jira_ids)

    avg_loss     = total_loss / max(n_batches, 1)
    rank_metrics = compute_ranking_metrics(all_scores, all_labels, all_jira_ids)
    bin_metrics  = compute_binary_metrics(all_scores, all_labels)

    return {"val_loss": avg_loss, **rank_metrics, **bin_metrics}


# ── Optimizer construction ────────────────────────────────────────────────────

def build_optimizer(
    model:        ImpactScoreModel,
    lr_head:      float,
    lr_backbone:  float,
    weight_decay: float,
) -> AdamW:
    """
    Two parameter groups:
      - Backbone (encoder): low LR to avoid catastrophic forgetting
      - New layers (projection, interaction): high LR for fast adaptation

    Backbone params with requires_grad=False are excluded automatically
    by AdamW (no gradient, no update).
    """
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    new_params      = (
        list(model.projection.parameters()) +
        list(model.interaction.parameters()) +
        (list(model.attn_pool.parameters()) if model.attn_pool else [])
    )

    param_groups = [
        {"params": new_params,      "lr": lr_head,     "name": "new_layers"},
        {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
    ]
    # Filter out empty groups
    param_groups = [g for g in param_groups if g["params"]]

    return AdamW(param_groups, weight_decay=weight_decay)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(config: dict):
    tcfg = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    log.info("Device: %s", device)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Tokenizer and data ────────────────────────────────────────────────────
    model_name = config["model"]["model_name"]
    log.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    log.info("Building dataloaders ...")
    train_loader, val_loader = build_dataloaders(
        train_path       = config["train_jsonl"],
        val_path         = config["val_jsonl"],
        tokenizer        = tokenizer,
        batch_size       = tcfg["batch_size"],
        max_chunk_tokens = tcfg.get("max_chunk_tokens", 512),
        max_class_tokens = tcfg.get("max_class_tokens", 512),
        max_chunks       = tcfg.get("max_chunks", 8),
        label_smoothing  = tcfg.get("label_smoothing", 0.0),
        num_workers      = tcfg.get("num_workers", 4),
    )
    log.info("Train samples: %d  |  Val samples: %d",
             len(train_loader.dataset), len(val_loader.dataset))

    # ── Model, loss, optimizer ────────────────────────────────────────────────
    log.info("Building model ...")
    model   = build_model(config["model"]).to(device)
    loss_fn = build_loss(config["loss"])

    frozen, total = model.frozen_param_count()
    log.info("Backbone params: %d frozen / %d total", frozen, total)

    optimizer = build_optimizer(
        model,
        lr_head      = tcfg["lr_head"],
        lr_backbone  = tcfg["lr_backbone"],
        weight_decay = tcfg["weight_decay"],
    )

    total_steps = len(train_loader) * tcfg["total_epochs"]
    scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # ── Training state ────────────────────────────────────────────────────────
    best_recall_at_10 = 0.0
    patience_counter  = 0
    global_step       = 0
    unfrozen          = False

    log.info("Starting training for %d epochs (warmup=%d, unfreeze_epoch=%d) ...",
             tcfg["total_epochs"], tcfg["warmup_epochs"], tcfg["unfreeze_epoch"])

    for epoch in range(1, tcfg["total_epochs"] + 1):

        # ── Unfreeze backbone after warmup ────────────────────────────────────
        if epoch == tcfg["unfreeze_epoch"] and not unfrozen:
            n = tcfg.get("n_unfreeze_layers", 2)
            log.info("Epoch %d: unfreezing top %d backbone layers.", epoch, n)
            model.unfreeze_top_layers(n)
            unfrozen = True

            # Rebuild optimizer to include newly unfrozen params
            optimizer = build_optimizer(
                model,
                lr_head      = tcfg["lr_head"],
                lr_backbone  = tcfg["lr_backbone"],
                weight_decay = tcfg["weight_decay"],
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max  = len(train_loader) * (tcfg["total_epochs"] - epoch + 1),
                eta_min = 1e-6,
            )
            frozen, total = model.frozen_param_count()
            log.info("After unfreeze: %d frozen / %d backbone params", frozen, total)

        # ── Train epoch ───────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{tcfg['total_epochs']:02d}", leave=False)

        for batch in pbar:
            batch = batch.to(device)

            logits, f_proj, c_proj = model(
                batch.feature_input_ids,
                batch.feature_attention_mask,
                batch.feature_chunk_mask,
                batch.class_input_ids,
                batch.class_attention_mask,
            )

            loss, breakdown = loss_fn(logits, f_proj, c_proj, batch.labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()

            epoch_loss  += breakdown["loss_total"]
            n_batches   += 1
            global_step += 1

            pbar.set_postfix({
                "loss": f"{breakdown['loss_total']:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.1e}"
            })

            if global_step % tcfg.get("eval_every_n_steps", 500) == 0:
                log.info(
                    "  step=%d  focal=%.4f  nce=%.4f  total=%.4f",
                    global_step,
                    breakdown["loss_focal"],
                    breakdown["loss_contrastive"],
                    breakdown["loss_total"],
                )

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ────────────────────────────────────────────────────────
        metrics = evaluate(model, val_loader, loss_fn, device)

        log.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  "
            "R@10=%.4f  R@20=%.4f  NDCG@10=%.4f  MAP=%.4f",
            epoch, tcfg["total_epochs"],
            avg_train_loss, metrics["val_loss"],
            metrics.get("recall@10", 0),
            metrics.get("recall@20", 0),
            metrics.get("ndcg@10", 0),
            metrics.get("map", 0),
        )

        # Save checkpoint
        ckpt_path = output_dir / f"checkpoint_epoch{epoch:02d}.pt"
        torch.save({
            "epoch":          epoch,
            "global_step":    global_step,
            "model_state":    model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics":        metrics,
        }, ckpt_path)

        # Best model tracking (primary metric: Recall@10)
        r10 = metrics.get("recall@10", 0.0)
        if r10 > best_recall_at_10:
            best_recall_at_10 = r10
            patience_counter  = 0
            best_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            log.info("  -> New best Recall@10=%.4f  saved to %s", r10, best_path)
        else:
            patience_counter += 1
            log.info(
                "  -> No improvement (patience %d/%d)",
                patience_counter, tcfg["patience"],
            )
            if patience_counter >= tcfg["patience"]:
                log.info("Early stopping triggered.")
                break

        # Save metrics log
        metrics_log_path = output_dir / "metrics_log.jsonl"
        with open(metrics_log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, "train_loss": avg_train_loss,
                                **metrics}) + "\n")

    log.info("Training complete. Best Recall@10: %.4f", best_recall_at_10)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train impact score BiEncoder")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    train(config)


if __name__ == "__main__":
    main()

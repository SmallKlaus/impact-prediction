"""
train_ce.py — Cross-Encoder Reranker Training
=============================================
Trains on top-K candidates produced by build_ce_samples.py. The bi-encoder's
top-K already contains the hard negatives (false positives the bi-encoder
ranked highly), so no further mining is needed.

Evaluation metrics (within the top-K reranking window):
    Recall@k  for k in [1, 3, 5, 10, 20]
    NDCG@k    for k in [5, 10]
    MRR, MAP
    Primary early-stopping metric: MRR

At the start of each validation pass the bi-encoder baseline (same candidates,
ranked by biencoder_score field) is evaluated too, so every epoch log shows
both numbers side-by-side and the improvement is always visible.

Usage:
    python train_ce.py --config config_ce.json
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
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from cross_encoder_model   import CrossEncoder, build_cross_encoder
from cross_encoder_dataset import (CrossEncoderDataset, CEBatch,
                                   ce_collate_fn, build_ce_dataloaders)
from loss import FocalLoss

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Ranking metrics ────────────────────────────────────────────────────────

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


def compute_ranking_metrics(
    all_scores:   list[float],
    all_labels:   list[int],
    all_jira_ids: list[str],
    ks:           list[int] = [1, 3, 5, 10, 20],
) -> dict:
    issue_scores: dict[str, list] = defaultdict(list)
    issue_labels: dict[str, list] = defaultdict(list)
    for s, l, jid in zip(all_scores, all_labels, all_jira_ids):
        issue_scores[jid].append(s)
        issue_labels[jid].append(l)

    bucket: dict[str, list] = {f"recall@{k}": [] for k in ks}
    bucket.update({f"ndcg@{k}": [] for k in ks if k <= 10})
    bucket["map"] = []
    bucket["mrr"] = []

    for jid in issue_scores:
        s, l = issue_scores[jid], issue_labels[jid]
        if not any(l): continue
        for k in ks:
            bucket[f"recall@{k}"].append(recall_at_k(s, l, k))
        for k in ks:
            if k <= 10:
                bucket[f"ndcg@{k}"].append(ndcg_at_k(s, l, k))
        bucket["map"].append(average_precision(s, l))
        bucket["mrr"].append(mean_reciprocal_rank(s, l))

    return {k: (sum(v) / len(v) if v else 0.0) for k, v in bucket.items()}


# ── Evaluation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:   CrossEncoder,
    loader:  DataLoader,
    loss_fn: FocalLoss,
    device:  torch.device,
    desc:    str = "Validating",
) -> tuple[dict, dict]:
    """
    Returns (ce_metrics, biencoder_baseline_metrics).
    The bi-encoder baseline is computed from the biencoder_score metadata
    already stored in each sample — zero additional inference cost.
    """
    model.eval()

    ce_scores, be_scores = [], []
    all_labels, all_jira_ids = [], []
    total_loss = 0.0
    n_batches  = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        batch  = batch.to(device)
        logits = model(batch.input_ids, batch.attention_mask, batch.chunk_mask)
        loss   = loss_fn(logits, batch.labels)

        total_loss += loss.item()
        n_batches  += 1

        ce_scores.extend(torch.sigmoid(logits).cpu().tolist())
        be_scores.extend(batch.biencoder_scores)
        all_labels.extend(batch.labels.int().cpu().tolist())
        all_jira_ids.extend(batch.jira_ids)

    avg_loss    = total_loss / max(n_batches, 1)
    ce_metrics  = compute_ranking_metrics(ce_scores,  all_labels, all_jira_ids)
    be_metrics  = compute_ranking_metrics(be_scores,  all_labels, all_jira_ids)

    ce_metrics["val_loss"] = avg_loss
    return ce_metrics, be_metrics


# ── Logging helpers ────────────────────────────────────────────────────────

def _log_class_balance(dataset: CrossEncoderDataset, split: str):
    n_pos = sum(1 for s in dataset.samples if int(s.get("label", 0)) == 1)
    n_neg = len(dataset.samples) - n_pos
    pos_rate = 100 * n_pos / max(len(dataset.samples), 1)
    log.info("  %-6s  %d samples  |  %d pos  %d neg  (%.2f%% positive)",
             split, len(dataset.samples), n_pos, n_neg, pos_rate)


def _log_comparison(epoch: int, total: int,
                    train_loss: float,
                    ce: dict, be: dict):
    log.info(
        "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
        epoch, total, train_loss, ce["val_loss"],
    )
    log.info(
        "           %8s  %8s  %8s  %8s  %8s  %8s  %8s",
        "", "R@1", "R@3", "R@5", "R@10", "NDCG@10", "MRR",
    )
    log.info(
        "  CE:      %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f",
        ce.get("recall@1",0), ce.get("recall@3",0), ce.get("recall@5",0),
        ce.get("recall@10",0), ce.get("ndcg@10",0), ce.get("mrr",0),
    )
    log.info(
        "  BiEnc:   %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f",
        be.get("recall@1",0), be.get("recall@3",0), be.get("recall@5",0),
        be.get("recall@10",0), be.get("ndcg@10",0), be.get("mrr",0),
    )
    mrr_delta = ce.get("mrr", 0) - be.get("mrr", 0)
    r5_delta  = ce.get("recall@5", 0) - be.get("recall@5", 0)
    log.info(
        "  Delta:   %+.4f MRR   %+.4f R@5",
        mrr_delta, r5_delta,
    )


# ── Training loop ──────────────────────────────────────────────────────────

def train(config: dict):
    tcfg   = config["training"]
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    log.info("Device: %s", device)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config_ce.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Tokenizer and data ─────────────────────────────────────────────
    model_name = config["model"]["model_name"]
    log.info("Loading tokenizer: %s", model_name)
    tokenizer  = AutoTokenizer.from_pretrained(model_name)

    log.info("Building dataloaders ...")
    train_loader, val_loader = build_ce_dataloaders(
        train_path       = config["ce_train_jsonl"],
        val_path         = config["ce_val_jsonl"],
        tokenizer        = tokenizer,
        batch_size       = tcfg["batch_size"],
        max_chunk_tokens = tcfg.get("max_chunk_tokens", 512),
        max_class_tokens = tcfg.get("max_class_tokens", 256),
        max_chunks       = tcfg.get("max_chunks", 4),
        num_workers      = tcfg.get("num_workers", 4),
    )
    _log_class_balance(train_loader.dataset, "train")
    _log_class_balance(val_loader.dataset,   "val")

    # ── Model and loss ─────────────────────────────────────────────────
    log.info("Building cross-encoder ...")
    model = build_cross_encoder(
        config               = config["model"],
        biencoder_checkpoint = config.get("biencoder_checkpoint"),
    ).to(device)

    frozen, total_params = model.frozen_param_count()
    log.info("Backbone params: %d frozen / %d total", frozen, total_params)

    lcfg    = config.get("loss", {})
    loss_fn = FocalLoss(
        alpha = lcfg.get("focal_alpha", 0.85),
        gamma = lcfg.get("focal_gamma", 2.0),
    )

    # ── Optimizer and scheduler ────────────────────────────────────────
    optimizer   = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = tcfg["lr"],
        weight_decay = tcfg.get("weight_decay", 0.01),
    )
    total_steps = len(train_loader) * tcfg["total_epochs"]
    scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # ── Training state ─────────────────────────────────────────────────
    best_mrr         = 0.0
    patience_counter = 0
    global_step      = 0
    unfrozen         = False

    log.info("=" * 60)
    log.info("Starting training for %d epochs", tcfg["total_epochs"])
    log.info("=" * 60)

    for epoch in range(1, tcfg["total_epochs"] + 1):

        # ── Optional backbone unfreeze ─────────────────────────────────
        unfreeze_epoch = tcfg.get("unfreeze_epoch")
        if unfreeze_epoch and epoch == unfreeze_epoch and not unfrozen:
            n = tcfg.get("n_unfreeze_layers", 2)
            log.info("Epoch %d: unfreezing top %d backbone layers.", epoch, n)
            model.unfreeze_top_layers(n)
            unfrozen = True
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr           = tcfg["lr"],
                weight_decay = tcfg.get("weight_decay", 0.01),
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max   = len(train_loader) * (tcfg["total_epochs"] - epoch + 1),
                eta_min = 1e-6,
            )
            frozen, _ = model.frozen_param_count()
            log.info("After unfreeze: %d frozen backbone params", frozen)

        # ── Train epoch ────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        pbar = tqdm(train_loader,
                    desc  = f"Epoch {epoch:02d}/{tcfg['total_epochs']:02d}",
                    leave = False)

        for batch in pbar:
            batch  = batch.to(device)
            logits = model(batch.input_ids, batch.attention_mask, batch.chunk_mask)
            loss   = loss_fn(logits, batch.labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),
                                     tcfg.get("max_grad_norm", 1.0))
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            n_batches   += 1
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr":   f"{scheduler.get_last_lr()[0]:.1e}",
            })

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ─────────────────────────────────────────────────
        ce_metrics, be_metrics = evaluate(model, val_loader, loss_fn, device)
        _log_comparison(epoch, tcfg["total_epochs"], avg_train_loss,
                        ce_metrics, be_metrics)

        # Save checkpoint
        torch.save({
            "epoch":           epoch,
            "global_step":     global_step,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "ce_metrics":      ce_metrics,
            "be_metrics":      be_metrics,
        }, output_dir / f"checkpoint_epoch{epoch:02d}.pt")

        # Best model: primary metric is MRR
        current_mrr = ce_metrics.get("mrr", 0.0)
        if current_mrr > best_mrr:
            best_mrr         = current_mrr
            patience_counter = 0
            best_path        = output_dir / "best_model_ce.pt"
            torch.save(model.state_dict(), best_path)
            log.info("  -> New best MRR=%.4f  saved to %s",
                     current_mrr, best_path)
        else:
            patience_counter += 1
            log.info("  -> No improvement in MRR (patience %d/%d)",
                     patience_counter, tcfg["patience"])
            if patience_counter >= tcfg["patience"]:
                log.info("Early stopping triggered.")
                break

        with open(output_dir / "metrics_log_ce.jsonl", "a") as f:
            f.write(json.dumps({
                "epoch":      epoch,
                "train_loss": avg_train_loss,
                **{f"ce_{k}":  v for k, v in ce_metrics.items()},
                **{f"be_{k}":  v for k, v in be_metrics.items()},
            }) + "\n")

    log.info("=" * 60)
    log.info("Training complete.  Best MRR: %.4f", best_mrr)
    log.info("=" * 60)

    # ── Final test evaluation ──────────────────────────────────────────
    test_path = config.get("ce_test_jsonl")
    if test_path and Path(test_path).exists():
        best_path = output_dir / "best_model_ce.pt"
        if not best_path.exists():
            log.warning("ce_test_jsonl set but best_model_ce.pt not found — skipping.")
        else:
            log.info("=" * 60)
            log.info("FINAL TEST EVALUATION")
            log.info("Checkpoint : %s", best_path)
            log.info("Test set   : %s", test_path)
            log.info("=" * 60)

            model.load_state_dict(torch.load(best_path, map_location=device))

            test_ds = CrossEncoderDataset(
                test_path, tokenizer,
                max_chunk_tokens = tcfg.get("max_chunk_tokens", 512),
                max_class_tokens = tcfg.get("max_class_tokens", 256),
                max_chunks       = tcfg.get("max_chunks", 4),
            )
            test_loader = DataLoader(
                test_ds,
                batch_size  = tcfg["batch_size"] * 2,
                shuffle     = False,
                collate_fn  = ce_collate_fn,
                num_workers = tcfg.get("num_workers", 4),
                pin_memory  = True,
            )
            _log_class_balance(test_ds, "test")

            ce_test, be_test = evaluate(
                model, test_loader, loss_fn, device, desc="Testing"
            )

            log.info("TEST RESULTS")
            _log_comparison(0, 0, 0.0, ce_test, be_test)

            results_path = output_dir / "test_metrics_ce.json"
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump({
                    "checkpoint":   str(best_path),
                    "test_jsonl":   test_path,
                    "best_val_mrr": best_mrr,
                    **{f"ce_test_{k}": v for k, v in ce_test.items()},
                    **{f"be_test_{k}": v for k, v in be_test.items()},
                }, f, indent=2)
            log.info("Test metrics saved to %s", results_path)


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train cross-encoder reranker")
    parser.add_argument("--config", required=True, help="Path to config_ce.json")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    train(config)


if __name__ == "__main__":
    main()

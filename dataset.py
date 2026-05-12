"""
dataset.py — JSONL Dataset and Collation
=========================================
Reads the training_samples.jsonl produced by build_training_samples.py,
tokenizes feature and class texts, and packs them into padded batches.

Feature side:
    Each sample has feature_chunks: [{text, token_count}, ...]
    We tokenize each chunk independently and stack them into a
    (n_chunks, seq_len) tensor. Batches are padded to the maximum
    number of chunks seen in the batch.

Class side:
    class_text tokenized as a single sequence, truncated to max_class_len.

The collate_fn handles variable chunk counts across samples in the same batch
by padding the chunk dimension with zero tensors and producing a chunk_mask.
"""
from __future__ import annotations
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase


@dataclass
class SampleBatch:
    """Typed container returned by the collate function."""
    feature_input_ids:      torch.Tensor   # (batch, n_chunks, seq_len)
    feature_attention_mask: torch.Tensor   # (batch, n_chunks, seq_len)
    feature_chunk_mask:     torch.Tensor   # (batch, n_chunks) — real chunks
    class_input_ids:        torch.Tensor   # (batch, class_seq_len)
    class_attention_mask:   torch.Tensor   # (batch, class_seq_len)
    labels:                 torch.Tensor   # (batch,) float32
    # Metadata — not used in the forward pass but useful for analysis
    jira_ids:               list[str]
    class_paths:            list[str]
    is_hard_negative:       list[bool]
    dependency_scores:      list[float]

    def to(self, device: torch.device) -> "SampleBatch":
        return SampleBatch(
            feature_input_ids      = self.feature_input_ids.to(device),
            feature_attention_mask = self.feature_attention_mask.to(device),
            feature_chunk_mask     = self.feature_chunk_mask.to(device),
            class_input_ids        = self.class_input_ids.to(device),
            class_attention_mask   = self.class_attention_mask.to(device),
            labels                 = self.labels.to(device),
            jira_ids               = self.jira_ids,
            class_paths            = self.class_paths,
            is_hard_negative       = self.is_hard_negative,
            dependency_scores      = self.dependency_scores,
        )


class ImpactSampleDataset(Dataset):
    """
    Reads training_samples.jsonl and tokenizes on the fly.

    Args:
        jsonl_path       : path to training_samples.jsonl
        tokenizer        : HuggingFace tokenizer
        max_chunk_tokens : token budget per feature chunk (default 512)
        max_class_tokens : token budget for class description (default 512)
        max_chunks       : cap on number of chunks per sample (default 8)
                           chunks beyond this are dropped (rare in practice)
        label_smoothing  : if > 0, soft labels: pos=1-eps, neg=eps
        issues_filter    : optional set of jira_ids to load (for debugging)
    """

    def __init__(
        self,
        jsonl_path:       str | Path,
        tokenizer:        PreTrainedTokenizerBase,
        max_chunk_tokens: int   = 512,
        max_class_tokens: int   = 512,
        max_chunks:       int   = 8,
        label_smoothing:  float = 0.0,
        issues_filter:    Optional[set] = None,
    ):
        self.tokenizer        = tokenizer
        self.max_chunk_tokens = max_chunk_tokens
        self.max_class_tokens = max_class_tokens
        self.max_chunks       = max_chunks
        self.label_smoothing  = label_smoothing

        self.samples: list[dict] = []
        self._load(Path(jsonl_path), issues_filter)

    def _load(self, path: Path, issues_filter: Optional[set]):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if issues_filter and rec.get("jira_id") not in issues_filter:
                    continue
                self.samples.append(rec)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict with tokenized tensors for one sample.
        Tokenization happens here (in the worker process) to parallelise it.
        """
        rec = self.samples[idx]

        # ── Feature side ─────────────────────────────────────────────────────
        chunks = rec.get("feature_chunks", [])
        if not chunks:
            # Fallback: treat the full text as a single chunk
            chunks = [{"text": rec.get("feature_text", ""), "chunk_idx": 0}]
        chunks = chunks[:self.max_chunks]

        chunk_encodings = []
        for chunk in chunks:
            enc = self.tokenizer(
                chunk["text"],
                max_length     = self.max_chunk_tokens,
                truncation     = True,
                padding        = "max_length",
                return_tensors = "pt",
            )
            chunk_encodings.append({
                "input_ids":      enc["input_ids"].squeeze(0),      # (seq_len,)
                "attention_mask": enc["attention_mask"].squeeze(0),  # (seq_len,)
            })

        # Stack into (n_chunks, seq_len)
        feature_input_ids      = torch.stack([c["input_ids"]      for c in chunk_encodings])
        feature_attention_mask = torch.stack([c["attention_mask"] for c in chunk_encodings])

        # ── Class side ───────────────────────────────────────────────────────
        class_enc = self.tokenizer(
            rec.get("class_text", ""),
            max_length     = self.max_class_tokens,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt",
        )

        # ── Label ────────────────────────────────────────────────────────────
        raw_label = float(rec.get("label", 0))
        if self.label_smoothing > 0:
            if raw_label == 1.0:
                label = 1.0 - self.label_smoothing
            else:
                label = self.label_smoothing
        else:
            label = raw_label

        return {
            "feature_input_ids":      feature_input_ids,
            "feature_attention_mask": feature_attention_mask,
            "n_chunks":               len(chunk_encodings),
            "class_input_ids":        class_enc["input_ids"].squeeze(0),
            "class_attention_mask":   class_enc["attention_mask"].squeeze(0),
            "label":                  torch.tensor(label, dtype=torch.float32),
            # Metadata
            "jira_id":          rec.get("jira_id", ""),
            "class_path":       rec.get("class_path", ""),
            "is_hard_negative": rec.get("is_hard_negative", False),
            "dependency_score": rec.get("dependency_score", 0.0),
        }


def collate_fn(batch: list[dict]) -> SampleBatch:
    """
    Pads the chunk dimension across samples in a batch.

    Different samples may have different numbers of chunks (1 for short
    feature descriptions, up to max_chunks for long ones). We pad to the
    maximum chunk count seen in this batch and build a chunk_mask.
    """
    max_chunks   = max(item["n_chunks"] for item in batch)
    seq_len_feat = batch[0]["feature_input_ids"].shape[-1]
    seq_len_cls  = batch[0]["class_input_ids"].shape[-1]
    batch_size   = len(batch)

    # Pre-allocate padded tensors
    feat_ids    = torch.zeros(batch_size, max_chunks, seq_len_feat, dtype=torch.long)
    feat_mask   = torch.zeros(batch_size, max_chunks, seq_len_feat, dtype=torch.long)
    chunk_mask  = torch.zeros(batch_size, max_chunks,               dtype=torch.long)
    cls_ids     = torch.zeros(batch_size, seq_len_cls,              dtype=torch.long)
    cls_mask    = torch.zeros(batch_size, seq_len_cls,              dtype=torch.long)
    labels      = torch.zeros(batch_size,                           dtype=torch.float32)

    for i, item in enumerate(batch):
        n = item["n_chunks"]
        feat_ids[i, :n]   = item["feature_input_ids"]
        feat_mask[i, :n]  = item["feature_attention_mask"]
        chunk_mask[i, :n] = 1
        cls_ids[i]        = item["class_input_ids"]
        cls_mask[i]       = item["class_attention_mask"]
        labels[i]         = item["label"]

    return SampleBatch(
        feature_input_ids      = feat_ids,
        feature_attention_mask = feat_mask,
        feature_chunk_mask     = chunk_mask,
        class_input_ids        = cls_ids,
        class_attention_mask   = cls_mask,
        labels                 = labels,
        jira_ids               = [item["jira_id"]          for item in batch],
        class_paths            = [item["class_path"]       for item in batch],
        is_hard_negative       = [item["is_hard_negative"] for item in batch],
        dependency_scores      = [item["dependency_score"] for item in batch],
    )


def build_dataloaders(
    train_path:       str | Path,
    val_path:         str | Path,
    tokenizer:        PreTrainedTokenizerBase,
    batch_size:       int   = 32,
    max_chunk_tokens: int   = 512,
    max_class_tokens: int   = 512,
    max_chunks:       int   = 8,
    label_smoothing:  float = 0.0,
    num_workers:      int   = 4,
    val_issues_filter: Optional[set] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    Args:
        train_path / val_path : paths to JSONL files (produced by the
                                train/val split script or manually split)
        val_issues_filter     : optional set of jira_ids for validation
                                (useful to hold out whole issues, not samples)
    """
    train_ds = ImpactSampleDataset(
        jsonl_path       = train_path,
        tokenizer        = tokenizer,
        max_chunk_tokens = max_chunk_tokens,
        max_class_tokens = max_class_tokens,
        max_chunks       = max_chunks,
        label_smoothing  = label_smoothing,
    )
    val_ds = ImpactSampleDataset(
        jsonl_path       = val_path,
        tokenizer        = tokenizer,
        max_chunk_tokens = max_chunk_tokens,
        max_class_tokens = max_class_tokens,
        max_chunks       = max_chunks,
        label_smoothing  = 0.0,          # no smoothing for validation
        issues_filter    = val_issues_filter,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = num_workers,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size * 2,   # no grad → can double batch size
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = num_workers,
        pin_memory  = True,
    )

    return train_loader, val_loader

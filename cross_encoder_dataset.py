"""
cross_encoder_dataset.py — Dataset and Collation for the Cross-Encoder
=======================================================================
Each sample in the CE JSONL contains a feature description (stored as
individual chunks) and a candidate class text. For each chunk we produce
one joint encoding:

    [CLS] chunk_i_text [SEP] class_text [SEP]

This gives the cross-encoder full cross-attention between every chunk of
the feature and the complete class. The class text is never truncated;
only the chunk side is truncated if it exceeds max_chunk_tokens (rare in
practice since chunks are already token-bounded upstream).

The collate function pads the chunk dimension to the maximum n_chunks seen
in the batch and returns a chunk_mask so the model ignores padding chunks.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase


@dataclass
class CEBatch:
    """Typed batch container for the cross-encoder."""
    input_ids:        torch.Tensor   # (batch, n_chunks, seq_len)
    attention_mask:   torch.Tensor   # (batch, n_chunks, seq_len)
    chunk_mask:       torch.Tensor   # (batch, n_chunks) — 1=real, 0=pad
    labels:           torch.Tensor   # (batch,) float32
    # Metadata (not used in forward pass)
    jira_ids:         list[str]
    class_paths:      list[str]
    biencoder_scores: list[float]
    biencoder_ranks:  list[int]

    def to(self, device: torch.device) -> "CEBatch":
        return CEBatch(
            input_ids        = self.input_ids.to(device),
            attention_mask   = self.attention_mask.to(device),
            chunk_mask       = self.chunk_mask.to(device),
            labels           = self.labels.to(device),
            jira_ids         = self.jira_ids,
            class_paths      = self.class_paths,
            biencoder_scores = self.biencoder_scores,
            biencoder_ranks  = self.biencoder_ranks,
        )


class CrossEncoderDataset(Dataset):
    """
    Reads a CE JSONL (output of build_ce_samples.py) and tokenises each
    (feature_chunks, class_text) pair into stacked (n_chunks, seq_len) tensors.

    Args:
        jsonl_path       : path to ce_train/val/test.jsonl
        tokenizer        : HuggingFace tokenizer
        max_chunk_tokens : token budget per chunk–class pair sequence (default 512)
        max_class_tokens : hard cap on class text tokens before chunk is added;
                           the tokenizer handles joint truncation, but this avoids
                           class text alone exceeding the budget (default 256)
        max_chunks       : cap on number of feature chunks per sample (default 4)
        issues_filter    : optional set of jira_ids to restrict loading
    """

    def __init__(
        self,
        jsonl_path:       str | Path,
        tokenizer:        PreTrainedTokenizerBase,
        max_chunk_tokens: int  = 512,
        max_class_tokens: int  = 256,
        max_chunks:       int  = 4,
        issues_filter:    Optional[set] = None,
    ):
        self.tokenizer        = tokenizer
        self.max_chunk_tokens = max_chunk_tokens
        self.max_class_tokens = max_class_tokens
        self.max_chunks       = max_chunks
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
        rec = self.samples[idx]

        # ── Feature chunks ────────────────────────────────────────────────
        chunks = rec.get("feature_chunks", [])
        if not chunks:
            # Fallback: treat the flat feature_text as a single chunk
            chunks = [{"text": rec.get("feature_text", "")}]
        chunks = chunks[: self.max_chunks]

        class_text = rec.get("class_text", "")

        # Truncate class text to max_class_tokens before forming pairs,
        # so it always fits alongside even a short chunk.
        class_enc_check = self.tokenizer(
            class_text,
            max_length     = self.max_class_tokens,
            truncation     = True,
            add_special_tokens = False,
        )
        class_text_truncated = self.tokenizer.decode(
            class_enc_check["input_ids"], skip_special_tokens=True
        )

        # ── Encode each (chunk, class) pair ───────────────────────────────
        pair_encodings = []
        for chunk in chunks:
            enc = self.tokenizer(
                chunk.get("text", ""),
                class_text_truncated,
                max_length     = self.max_chunk_tokens,
                truncation     = "only_first",   # only truncate the chunk side
                padding        = "max_length",
                return_tensors = "pt",
            )
            pair_encodings.append({
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })

        input_ids      = torch.stack([e["input_ids"]      for e in pair_encodings])
        attention_mask = torch.stack([e["attention_mask"] for e in pair_encodings])

        return {
            "input_ids":        input_ids,       # (n_chunks, seq_len)
            "attention_mask":   attention_mask,  # (n_chunks, seq_len)
            "n_chunks":         len(pair_encodings),
            "label":            torch.tensor(float(rec.get("label", 0)),
                                             dtype=torch.float32),
            "jira_id":          rec.get("jira_id",          ""),
            "class_path":       rec.get("class_path",       ""),
            "biencoder_score":  float(rec.get("biencoder_score", 0.0)),
            "biencoder_rank":   int(rec.get("biencoder_rank",    0)),
        }


def ce_collate_fn(batch: list[dict]) -> CEBatch:
    """
    Pads the chunk dimension to the maximum n_chunks in the batch.
    Produces chunk_mask to mark real vs. padding chunks.
    """
    max_chunks = max(item["n_chunks"] for item in batch)
    seq_len    = batch[0]["input_ids"].shape[-1]
    B          = len(batch)

    ids    = torch.zeros(B, max_chunks, seq_len, dtype=torch.long)
    mask   = torch.zeros(B, max_chunks, seq_len, dtype=torch.long)
    cmask  = torch.zeros(B, max_chunks,           dtype=torch.long)
    labels = torch.zeros(B,                        dtype=torch.float32)

    for i, item in enumerate(batch):
        n = item["n_chunks"]
        ids[i,   :n] = item["input_ids"]
        mask[i,  :n] = item["attention_mask"]
        cmask[i, :n] = 1
        labels[i]    = item["label"]

    return CEBatch(
        input_ids        = ids,
        attention_mask   = mask,
        chunk_mask       = cmask,
        labels           = labels,
        jira_ids         = [b["jira_id"]         for b in batch],
        class_paths      = [b["class_path"]      for b in batch],
        biencoder_scores = [b["biencoder_score"]  for b in batch],
        biencoder_ranks  = [b["biencoder_rank"]   for b in batch],
    )


def build_ce_dataloaders(
    train_path:       str | Path,
    val_path:         str | Path,
    tokenizer:        PreTrainedTokenizerBase,
    batch_size:       int = 32,
    max_chunk_tokens: int = 512,
    max_class_tokens: int = 256,
    max_chunks:       int = 4,
    num_workers:      int = 4,
) -> tuple[DataLoader, DataLoader]:
    train_ds = CrossEncoderDataset(
        train_path, tokenizer, max_chunk_tokens, max_class_tokens, max_chunks
    )
    val_ds = CrossEncoderDataset(
        val_path,   tokenizer, max_chunk_tokens, max_class_tokens, max_chunks
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=ce_collate_fn, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        collate_fn=ce_collate_fn, num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader

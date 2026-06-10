"""
cross_encoder_model.py — Cross-Encoder Reranker with Chunk Pooling
===================================================================
Architecture:
    For each feature chunk i:
        [CLS] chunk_i [SEP] class_text [SEP]
                  ↓  UniXcoder encoder
            CLS hidden state  (768-dim)

    Chunk CLS vectors  →  ChunkAttentionPool  →  pooled (768-dim)
                                                       ↓
                                              Dropout → Linear(768, 1)
                                                       ↓
                                                  logit → sigmoid → P(impact)

Why chunk pooling instead of simple truncation:
    A Jira description can exceed 800 tokens; a Java class 600+. Truncating
    the combined sequence with "longest_first" destroys the middle and tail
    of both sides and defeats the purpose of paying for cross-attention.

    The matching signal is local: a single sentence mentioning a class name
    or method is sufficient evidence of impact. Encoding each (chunk_i,
    full_class) pair independently ensures the class text is always fully
    preserved in cross-attention with every chunk of the feature. Learned
    attention pooling then weights which chunk carries the strongest signal.

Initialisation:
    backbone can be loaded from either raw UniXcoder or the bi-encoder's
    fine-tuned weights via build_cross_encoder(biencoder_checkpoint=...).
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

log = logging.getLogger(__name__)


class ChunkAttentionPool(nn.Module):
    """
    Learnable attention pooling over per-chunk CLS representations.

    For each chunk i, a scalar attention score is computed; scores are
    softmax-normalised (masked to ignore padding chunks) and used as
    weights for a weighted sum of the CLS vectors.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(
        self,
        chunk_cls:  torch.Tensor,   # (batch, n_chunks, hidden)
        chunk_mask: torch.Tensor,   # (batch, n_chunks) — 1=real, 0=pad
    ) -> torch.Tensor:              # (batch, hidden)
        scores  = self.attn(chunk_cls).squeeze(-1)             # (batch, n_chunks)
        scores  = scores.masked_fill(chunk_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)                    # (batch, n_chunks)
        pooled  = (weights.unsqueeze(-1) * chunk_cls).sum(1)   # (batch, hidden)
        return pooled


class CrossEncoder(nn.Module):
    """
    Cross-encoder impact score predictor with chunk pooling.

    Forward:
        logit = model(input_ids, attention_mask, chunk_mask)
        input_ids      : (batch, n_chunks, seq_len)
        attention_mask : (batch, n_chunks, seq_len)
        chunk_mask     : (batch, n_chunks)  — 1 for real chunks, 0 for padding

    Returns raw logit (batch,); apply sigmoid for probability.
    """

    def __init__(
        self,
        model_name:        str   = "microsoft/unixcoder-base",
        dropout:           float = 0.1,
        freeze_backbone:   bool  = False,
        n_unfreeze_layers: int   = 0,
    ):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden_size     = self.encoder.config.hidden_size
        self.attn_pool  = ChunkAttentionPool(hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self._apply_freeze(freeze_backbone, n_unfreeze_layers)

    # ── Backbone freezing / unfreezing ─────────────────────────────────────

    def _apply_freeze(self, freeze_backbone: bool, n_unfreeze_layers: int):
        if not freeze_backbone:
            return
        for p in self.encoder.parameters():
            p.requires_grad = False
        if n_unfreeze_layers > 0:
            self._unfreeze_top_n(n_unfreeze_layers)
        if hasattr(self.encoder, "pooler") and self.encoder.pooler is not None:
            for p in self.encoder.pooler.parameters():
                p.requires_grad = True

    def unfreeze_top_layers(self, n: int):
        self._unfreeze_top_n(n)

    def _unfreeze_top_n(self, n: int):
        layers = None
        for attr in ["encoder.layer", "transformer.layer", "layers"]:
            obj = self.encoder
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                layers = obj
                break
            except AttributeError:
                continue
        if layers is None:
            for p in self.encoder.parameters():
                p.requires_grad = True
            return
        total = len(layers)
        for i, layer in enumerate(layers):
            if i >= total - n:
                for p in layer.parameters():
                    p.requires_grad = True

    def frozen_param_count(self) -> tuple[int, int]:
        total  = sum(p.numel() for p in self.encoder.parameters())
        frozen = sum(p.numel() for p in self.encoder.parameters()
                     if not p.requires_grad)
        return frozen, total

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:      torch.Tensor,   # (batch, n_chunks, seq_len)
        attention_mask: torch.Tensor,   # (batch, n_chunks, seq_len)
        chunk_mask:     torch.Tensor,   # (batch, n_chunks)
    ) -> torch.Tensor:                  # (batch,) raw logit
        batch, n_chunks, seq_len = input_ids.shape

        # Flatten: encode all (chunk, class) pairs in one transformer pass
        flat_ids  = input_ids.view(-1, seq_len)        # (batch*n_chunks, seq_len)
        flat_mask = attention_mask.view(-1, seq_len)

        outputs    = self.encoder(flat_ids, attention_mask=flat_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]            # (batch*n_chunks, hidden)
        cls_hidden = cls_hidden.view(batch, n_chunks, -1)          # (batch, n_chunks, hidden)

        # Pool chunk representations with learned attention weights
        pooled = self.attn_pool(cls_hidden, chunk_mask)            # (batch, hidden)

        # Classify
        return self.classifier(pooled).squeeze(-1)                 # (batch,)


# ── Factory ────────────────────────────────────────────────────────────────

def build_cross_encoder(
    config:               dict,
    biencoder_checkpoint: Optional[str | Path] = None,
) -> CrossEncoder:
    """
    Build a CrossEncoder from a config dict.

    If biencoder_checkpoint is provided the backbone is initialised from
    the bi-encoder's fine-tuned weights (only encoder.* keys are loaded;
    the attention pool and classifier head are always randomly initialised).

    Typical config:
        {
          "model_name":         "/path/to/unixcoder-base",
          "dropout":            0.1,
          "freeze_backbone":    false,
          "n_unfreeze_layers":  0
        }
    """
    model = CrossEncoder(
        model_name        = config.get("model_name",        "microsoft/unixcoder-base"),
        dropout           = config.get("dropout",           0.1),
        freeze_backbone   = config.get("freeze_backbone",   False),
        n_unfreeze_layers = config.get("n_unfreeze_layers", 0),
    )

    if biencoder_checkpoint:
        ckpt_path = Path(biencoder_checkpoint)
        log.info("Initialising backbone from bi-encoder: %s", ckpt_path)
        state = torch.load(ckpt_path, map_location="cpu")
        if "model_state" in state:
            state = state["model_state"]
        encoder_state = {
            k[len("encoder."):] if k.startswith("encoder.") else k: v
            for k, v in state.items()
            if k.startswith("encoder.")
        }
        missing, _ = model.encoder.load_state_dict(encoder_state, strict=False)
        if missing:
            log.warning("  Missing encoder keys (expected for new pooler): %s",
                        missing[:3])
        log.info("  Backbone initialised from bi-encoder (%d keys).",
                 len(encoder_state))

    return model

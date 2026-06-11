"""
model.py — BiEncoder Impact Score Model
========================================
Architecture:
  Shared UniXcoder backbone (or any HuggingFace encoder)
    ↓
  Feature side : N chunks → encode each → mean-pool CLS vectors → projection
  Class side   : single sequence → encode → CLS → projection
    ↓
  Interaction  : concat(f, c, f-c, f*c) → MLP → sigmoid → P(impact)

The shared-weight design is intentional: UniXcoder was pretrained on a joint
NL+code space, so the same encoder already handles both feature descriptions
(NL-heavy) and class descriptions (code-heavy) without needing separate towers.

Key design choices:
  - Shared backbone, NOT separate towers
  - Projection head: Linear → LayerNorm (no activation before norm — avoids
    dead neurons in the projection before the interaction MLP)
  - Interaction features: [f, c, f-c, f*c] (InferSent-style, consistently
    outperforms dot-product-only on semantic matching tasks)
  - Chunk pooling: mean over CLS vectors (attention-weighted pool available
    as alternative via use_attention_pool=True)
  - Inference mode: encode_class() can be called once per sha_before and
    results cached; encode_feature() called per issue at prediction time
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Optional


class ProjectionHead(nn.Module):
    """
    Linear projection from encoder hidden size to proj_dim,
    followed by LayerNorm.

    No non-linearity before the norm — keeps the projected space
    smooth for the interaction layer's element-wise operations.
    """
    def __init__(self, hidden_size: int, proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear  = nn.Linear(hidden_size, proj_dim)
        self.norm    = nn.LayerNorm(proj_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.dropout(self.linear(x)))


class AttentionPool(nn.Module):
    """
    Learnable attention pooling over a sequence of chunk CLS vectors.
    Computes a weighted mean where weights are learned from the chunk embeddings.
    Used as an alternative to simple mean pooling for multi-chunk features.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, chunk_embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            chunk_embeds : (batch, n_chunks, dim)
            mask         : (batch, n_chunks) — 1 for real chunks, 0 for padding
        Returns:
            pooled       : (batch, dim)
        """
        scores = self.attn(chunk_embeds).squeeze(-1)          # (batch, n_chunks)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)                   # (batch, n_chunks)
        pooled  = (weights.unsqueeze(-1) * chunk_embeds).sum(dim=1)
        return pooled


class InteractionMLP(nn.Module):
    """
    MLP over the InferSent-style interaction vector [f, c, f-c, f*c].
    Input dimension is always 4 * proj_dim.
 
    Controlled by four config parameters:
 
        interaction_hidden      Width of the first hidden layer.
                                The second hidden layer (n_layers=2) is always
                                interaction_hidden // 2.
 
        n_layers                1  →  original single-hidden-layer MLP.
                                2  →  deeper MLP (H → H//2 → 1).
 
        activation              "relu"  →  original behaviour.
                                "gelu"  →  smoother, matches UniXcoder internals.
 
        use_layernorm           False  →  original behaviour.
                                True   →  LayerNorm after each activation.
 
    Dropout is placed immediately before the final linear regardless of depth.
    """
 
    def __init__(
        self,
        proj_dim:      int,
        hidden_dim:    int,
        dropout:       float = 0.1,
        n_layers:      int   = 1,
        activation:    str   = "relu",
        use_layernorm: bool  = False,
    ):
        super().__init__()
 
        def act() -> nn.Module:
            return nn.GELU() if activation == "gelu" else nn.ReLU()
 
        hidden_dims = [hidden_dim] if n_layers == 1 else [hidden_dim, hidden_dim // 2]
 
        layers: list[nn.Module] = []
        in_dim = 4 * proj_dim
 
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act())
            if use_layernorm:
                layers.append(nn.LayerNorm(h_dim))
            in_dim = h_dim
 
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_dim, 1))
 
        self.mlp = nn.Sequential(*layers)

    def forward(self, f: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f : (batch, proj_dim) feature representation
            c : (batch, proj_dim) class representation
        Returns:
            logit : (batch,) raw logit (before sigmoid)
        """
        interaction = torch.cat([f, c, f - c, f * c], dim=-1)
        return self.mlp(interaction).squeeze(-1)


class ImpactScoreModel(nn.Module):
    """
    Bi-encoder impact score predictor.

    Forward pass for training:
        logit, f_proj, c_proj = model(
            feature_input_ids,      # (batch, n_chunks, seq_len)
            feature_attention_mask, # (batch, n_chunks, seq_len)
            feature_chunk_mask,     # (batch, n_chunks) — which chunks are real
            class_input_ids,        # (batch, seq_len)
            class_attention_mask,   # (batch, seq_len)
        )

    f_proj and c_proj are returned for the optional contrastive loss.
    logit is the raw scalar before sigmoid; pass through sigmoid for probability.

    Inference helpers:
        class_embed = model.encode_class(input_ids, attention_mask)
        feat_embed  = model.encode_feature(input_ids, attention_mask, chunk_mask)
        prob        = model.score(feat_embed, class_embed)
    """

    def __init__(
        self,
        model_name:          str   = "microsoft/unixcoder-base",
        proj_dim:            int   = 256,
        interaction_hidden:  int   = 512,
        interaction_layers:  int   = 1,
        interaction_activation: str   = "relu",
        interaction_layernorm: bool  = False,
        dropout:             float = 0.1,
        use_attention_pool:  bool  = False,
        freeze_backbone:     bool  = True,
        n_unfreeze_layers:   int   = 0,
    ):
        super().__init__()
        self.proj_dim           = proj_dim
        self.use_attention_pool = use_attention_pool

        # Shared encoder backbone
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size  = self.encoder.config.hidden_size

        # Shared projection head (same weights for feature and class sides)
        self.projection = ProjectionHead(hidden_size, proj_dim, dropout)

        # Chunk pooling (only used on the feature side)
        self.attn_pool = AttentionPool(proj_dim) if use_attention_pool else None

        # Interaction MLP
        self.interaction = InteractionMLP(
            proj_dim=proj_dim,
            hidden_dim=interaction_hidden,
            dropout=dropout,
            n_layers=interaction_layers,
            activation=interaction_activation,
            use_layernorm=interaction_layernorm
        )

        # Apply initial freezing
        self._apply_freeze(freeze_backbone, n_unfreeze_layers)

    # ── Backbone freezing / unfreezing ────────────────────────────────────────

    def _apply_freeze(self, freeze_backbone: bool, n_unfreeze_layers: int):
        """
        Freeze the backbone except for the top n_unfreeze_layers transformer
        blocks (counted from the output side).

        Call this once at init, then call unfreeze_top_layers() progressively
        during training.
        """
        if not freeze_backbone:
            return

        # Freeze all backbone params first
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Unfreeze the top N transformer layers if requested
        if n_unfreeze_layers > 0:
            self._unfreeze_top_n(n_unfreeze_layers)

        # Always keep the pooler unfrozen (it contributes to CLS quality)
        if hasattr(self.encoder, 'pooler') and self.encoder.pooler is not None:
            for p in self.encoder.pooler.parameters():
                p.requires_grad = True

    def unfreeze_top_layers(self, n: int):
        """
        Progressively unfreeze the top N transformer layers.
        Call this after the projection heads have converged (typically epoch 2-3).
        """
        self._unfreeze_top_n(n)

    def _unfreeze_top_n(self, n: int):
        """Unfreeze the last n encoder layers (works for BERT-style models)."""
        # Try common attribute names for the transformer layer stack
        layers = None
        for attr in ['encoder.layer', 'transformer.layer', 'layers']:
            parts = attr.split('.')
            obj   = self.encoder
            try:
                for part in parts:
                    obj = getattr(obj, part)
                layers = obj
                break
            except AttributeError:
                continue

        if layers is None:
            # Fallback: unfreeze all encoder params
            for p in self.encoder.parameters():
                p.requires_grad = True
            return

        total = len(layers)
        for i, layer in enumerate(layers):
            if i >= total - n:
                for p in layer.parameters():
                    p.requires_grad = True

    def frozen_param_count(self) -> tuple[int, int]:
        """Returns (frozen_count, total_count) for the backbone."""
        total  = sum(p.numel() for p in self.encoder.parameters())
        frozen = sum(p.numel() for p in self.encoder.parameters()
                     if not p.requires_grad)
        return frozen, total

    # ── Encoding helpers ──────────────────────────────────────────────────────

    def _encode_single(
        self,
        input_ids:      torch.Tensor,   # (..., seq_len)
        attention_mask: torch.Tensor,   # (..., seq_len)
    ) -> torch.Tensor:
        """
        Run the encoder on a single sequence and return the projected CLS vector.
        Handles arbitrary leading batch dimensions by flattening.
        """
        orig_shape = input_ids.shape[:-1]       # e.g. (batch,) or (batch, n_chunks)
        seq_len    = input_ids.shape[-1]

        flat_ids   = input_ids.view(-1, seq_len)
        flat_mask  = attention_mask.view(-1, seq_len)

        outputs    = self.encoder(flat_ids, attention_mask=flat_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # (batch_flat, hidden)

        projected  = self.projection(cls_hidden)         # (batch_flat, proj_dim)
        return projected.view(*orig_shape, self.proj_dim)

    def encode_feature(
        self,
        input_ids:      torch.Tensor,   # (batch, n_chunks, seq_len)
        attention_mask: torch.Tensor,   # (batch, n_chunks, seq_len)
        chunk_mask:     torch.Tensor,   # (batch, n_chunks) int/bool
    ) -> torch.Tensor:
        """
        Encode a multi-chunk feature description.
        Each chunk is encoded independently, then pooled across chunks.

        Returns:
            f : (batch, proj_dim)
        """
        # chunk_embeds: (batch, n_chunks, proj_dim)
        chunk_embeds = self._encode_single(input_ids, attention_mask)

        if self.use_attention_pool:
            return self.attn_pool(chunk_embeds, chunk_mask.float())
        else:
            # Mean pool over real (non-padding) chunks
            mask_f = chunk_mask.float().unsqueeze(-1)          # (batch, n_chunks, 1)
            summed = (chunk_embeds * mask_f).sum(dim=1)        # (batch, proj_dim)
            count  = mask_f.sum(dim=1).clamp(min=1)            # (batch, 1)
            return summed / count

    def encode_class(
        self,
        input_ids:      torch.Tensor,   # (batch, seq_len)
        attention_mask: torch.Tensor,   # (batch, seq_len)
    ) -> torch.Tensor:
        """
        Encode a class description.
        Returns:
            c : (batch, proj_dim)
        """
        return self._encode_single(input_ids, attention_mask)

    def score(
        self,
        f: torch.Tensor,   # (batch, proj_dim)
        c: torch.Tensor,   # (batch, proj_dim)
    ) -> torch.Tensor:
        """
        Compute impact probability from pre-computed embeddings.
        Returns:
            prob : (batch,) in [0, 1]
        """
        logit = self.interaction(f, c)
        return torch.sigmoid(logit)

    # ── Full forward pass (training) ──────────────────────────────────────────

    def forward(
        self,
        feature_input_ids:      torch.Tensor,   # (batch, n_chunks, seq_len)
        feature_attention_mask: torch.Tensor,   # (batch, n_chunks, seq_len)
        feature_chunk_mask:     torch.Tensor,   # (batch, n_chunks)
        class_input_ids:        torch.Tensor,   # (batch, seq_len)
        class_attention_mask:   torch.Tensor,   # (batch, seq_len)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logit  : (batch,)       raw logit for BCEWithLogitsLoss / FocalLoss
            f_proj : (batch, proj_dim)  for contrastive loss
            c_proj : (batch, proj_dim)  for contrastive loss
        """
        f_proj = self.encode_feature(
            feature_input_ids, feature_attention_mask, feature_chunk_mask
        )
        c_proj = self.encode_class(class_input_ids, class_attention_mask)
        logit  = self.interaction(f_proj, c_proj)
        return logit, f_proj, c_proj


def build_model(config: dict) -> ImpactScoreModel:
    """
    Convenience constructor from a config dict.
    Typical config:
        {
          "model_name":         "microsoft/unixcoder-base",
          "proj_dim":           256,
          "interaction_hidden": 256,
          "interaction_layers":     1     ← original
          "interaction_activation": "relu"← original
          "interaction_layernorm":  false ← original
          "dropout":            0.1,
          "use_attention_pool": false,
          "freeze_backbone":    true,
          "n_unfreeze_layers":  0
        }
    """
    return ImpactScoreModel(
        model_name          = config.get("model_name", "microsoft/unixcoder-base"),
        proj_dim            = config.get("proj_dim", 256),
        interaction_hidden  = config.get("interaction_hidden", 256),
        interaction_layers     = config.get("interaction_layers",     1),
        interaction_activation = config.get("interaction_activation", "relu"),
        interaction_layernorm  = config.get("interaction_layernorm",  False),
        dropout             = config.get("dropout", 0.1),
        use_attention_pool  = config.get("use_attention_pool", False),
        freeze_backbone     = config.get("freeze_backbone", True),
        n_unfreeze_layers   = config.get("n_unfreeze_layers", 0),
    )

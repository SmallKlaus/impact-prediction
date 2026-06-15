"""
flim_model.py — CodeBERT bi-encoder used by FLIM
====
Mirrors the Model class in FLIM's CodeBERT-finetune/run.py:
a single shared RoBERTa encoder for NL (issue text) and PL (function code),
pooled representation, cosine-similarity scoring, in-batch-negative
cross-entropy fine-tuning objective.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class FlimEncoder(nn.Module):
    def __init__(self, model_name: str, scale: float = 20.0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.scale   = scale

    def embed(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out[0]                                   # (B, L, H)
        # CLS pooling (as in CodeBERT/GraphCodeBERT code-search), L2-normalized
        vec = hidden[:, 0, :]
        return F.normalize(vec, p=2, dim=-1)

    def forward(self, nl_ids, nl_mask, code_ids, code_mask):
        """In-batch-negative InfoNCE (NL → code), as in FLIM's run.py."""
        nl_vec   = self.embed(nl_ids, nl_mask)            # (B, H)
        code_vec = self.embed(code_ids, code_mask)        # (B, H)
        logits   = nl_vec @ code_vec.t() * self.scale     # (B, B)
        labels   = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)


def build_flim_encoder(model_cfg: dict) -> FlimEncoder:
    return FlimEncoder(model_cfg["model_name"], model_cfg.get("scale", 20.0))
"""
hard_negative_miner.py — Model-driven hard-negative mining for the impact bi-encoder
=====================================================================================
After each epoch (from `start_epoch` onwards) we replace the *hard negative* slice
of the training set with samples that the current model itself ranks too high.
This closes the train/inference gap and mirrors the schedule used in DPR / ANCE /
RocketQA.

Workflow per mining round
-------------------------
1. Read `train_full.jsonl` (produced by step_11 style enumeration) once at the
   start of training. Each line is one (jira_id, class) pair carrying the same
   fields as the regular train samples (feature_text, feature_chunks, class_text,
   sha_before, ...).
2. Group candidates by `sha_before`. Classes are shared across issues that point
   at the same commit, so we encode each codebase snapshot exactly once per
   mining round.
3. For each training issue:
     - encode its feature description once (chunks → pooled embedding)
     - score it against every candidate class in its `sha_before` snapshot via
       the model's interaction MLP
     - drop positives (the issue's own labelled-1 class_paths)
     - take the top-K highest-scoring remaining classes as the new hard negs
4. Materialise those classes back into full sample dicts (with feature_chunks,
   class_text, label=0, is_hard_negative=True) so they slot straight into the
   training dataset.

Performance notes
-----------------
* Bi-encoder embeddings are pre-computed per (sha_before, class), so the inner
  loop over issues is a cheap matmul + interaction MLP call, not a transformer
  pass. The expensive transformer pass scales with `unique sha_before × classes
  per snapshot`, not `issues × classes`.
* All tensor work stays on the training device; only the final
  (top_k_per_issue) indices are pulled to host.
* For very large codebases the candidate batch is sliced — `embed_batch_size`
  controls peak VRAM.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from model import ImpactScoreModel

log = logging.getLogger(__name__)


# ── Candidate pool index ──────────────────────────────────────────────────────

@dataclass
class CandidatePool:
    """Records loaded from train_full.jsonl, indexed for fast retrieval.

    Two views over the same underlying record list:
      * `by_sha[sha_before]`           → list of full record dicts
      * `by_jira_positives[jira_id]`   → set of positive class_paths for that issue
      * `by_jira_sha[jira_id]`         → sha_before of that issue (for grouping)
      * `by_jira_feature[jira_id]`     → (feature_text, feature_chunks)  — used to
                                          stamp the mined sample dicts
    """
    by_sha:           dict[str, list[dict]]
    by_jira_positives: dict[str, set[str]]
    by_jira_sha:      dict[str, str]
    by_jira_feature:  dict[str, tuple[str, list[dict]]]

    @classmethod
    def from_jsonl(cls, path: str | Path, restrict_to_jira_ids: Optional[set[str]] = None) -> "CandidatePool":
        path = Path(path)
        log.info("Loading candidate pool from %s ...", path)

        # 1. Quick pre-count to enable exact (n / total) tracking in the progress bar
        log.info("Calculating total samples...")
        with open(path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        by_sha:           dict[str, list[dict]] = defaultdict(list)
        by_jira_positives: dict[str, set[str]]  = defaultdict(set)
        by_jira_sha:      dict[str, str]        = {}
        by_jira_feature:  dict[str, tuple[str, list[dict]]] = {}

        # Deduplicate classes within a sha_before — train_full.jsonl repeats
        # every class once per issue at that sha; we only want each class once.
        seen_in_sha: dict[str, set[str]] = defaultdict(set)

        n_lines = 0
        # 2. Wrap the file reader in tqdm with the total line count
        with open(path, encoding="utf-8") as f:
            for line in tqdm(f, total=total_lines, desc="Loading candidate pool", unit=" samples"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_lines += 1

                jid = rec.get("jira_id", "")
                if restrict_to_jira_ids is not None and jid not in restrict_to_jira_ids:
                    continue

                sha  = rec.get("sha_before", "")
                cpath = rec.get("class_path", "")

                # Track per-issue positives even before deduping
                if int(rec.get("label", 0)) == 1:
                    by_jira_positives[jid].add(cpath)

                by_jira_sha[jid] = sha

                if jid not in by_jira_feature:
                    chunks = rec.get("feature_chunks", [])
                    text = "" if chunks else rec.get("feature_text", "")
                    by_jira_feature[jid] = (text, chunks)

                # Only the first occurrence of (sha, cpath) is kept in the candidate list
                if cpath in seen_in_sha[sha]:
                    continue
                seen_in_sha[sha].add(cpath)
                slim_rec = {
                    "sha_before":       sha,
                    "class_path":       cpath,
                    "class_fqn":        rec.get("class_fqn", ""),
                    "class_text":       rec.get("class_text", ""),
                    "dependency_score": rec.get("dependency_score", 0.0),
                    "dependency_fwd":   rec.get("dependency_fwd", 0.0),
                    "dependency_rev":   rec.get("dependency_rev", 0.0),
                    "depth_fwd":        rec.get("depth_fwd"),
                    "depth_rev":        rec.get("depth_rev"),
                    "is_seed":          rec.get("is_seed", False),
                }
                by_sha[sha].append(slim_rec)

        log.info(
            "  Candidate pool: %d lines  |  %d unique sha_before  |  %d jira_ids  |  %d unique (sha,class) pairs",
            n_lines, len(by_sha), len(by_jira_sha),
            sum(len(v) for v in by_sha.values()),
        )

        return cls(
            by_sha=dict(by_sha),
            by_jira_positives=dict(by_jira_positives),
            by_jira_sha=by_jira_sha,
            by_jira_feature=by_jira_feature,
        )


# ── Encoding helpers ──────────────────────────────────────────────────────────

@torch.no_grad()
def _encode_classes_for_sha(
    model:            ImpactScoreModel,
    tokenizer:        PreTrainedTokenizerBase,
    records:          list[dict],
    device:           torch.device,
    max_class_tokens: int,
    embed_batch_size: int,
) -> torch.Tensor:
    """Encode every candidate class for one sha_before into a (N, proj_dim) tensor."""
    model.eval()
    embeds: list[torch.Tensor] = []

    for start in range(0, len(records), embed_batch_size):
        chunk = records[start : start + embed_batch_size]
        texts = [r.get("class_text", "") for r in chunk]
        enc = tokenizer(
            texts,
            max_length     = max_class_tokens,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt",
        )
        ids  = enc["input_ids"].to(device, non_blocking=True)
        mask = enc["attention_mask"].to(device, non_blocking=True)
        c    = model.encode_class(ids, mask)              # (B, proj_dim)
        embeds.append(c)

    return torch.cat(embeds, dim=0)                       # (N, proj_dim)


@torch.no_grad()
def _encode_feature_for_jira(
    model:            ImpactScoreModel,
    tokenizer:        PreTrainedTokenizerBase,
    feature_chunks:   list[dict],
    feature_text:     str,
    device:           torch.device,
    max_chunk_tokens: int,
    max_chunks:       int,
) -> torch.Tensor:
    """Encode one issue's feature description into a (proj_dim,) tensor."""
    model.eval()

    chunks = feature_chunks if feature_chunks else [{"text": feature_text}]
    chunks = chunks[:max_chunks]
    texts  = [c.get("text", "") for c in chunks]

    enc = tokenizer(
        texts,
        max_length     = max_chunk_tokens,
        truncation     = True,
        padding        = "max_length",
        return_tensors = "pt",
    )
    ids  = enc["input_ids"].unsqueeze(0).to(device, non_blocking=True)   # (1, n_chunks, seq_len)
    mask = enc["attention_mask"].unsqueeze(0).to(device, non_blocking=True)
    chunk_mask = torch.ones(1, len(chunks), dtype=torch.long, device=device)

    f = model.encode_feature(ids, mask, chunk_mask)        # (1, proj_dim)
    return f.squeeze(0)


@torch.no_grad()
def _score_feature_against_classes(
    model:        ImpactScoreModel,
    f_embed:      torch.Tensor,              # (proj_dim,)
    class_embeds: torch.Tensor,              # (N, proj_dim)
    score_batch:  int = 4096,
) -> torch.Tensor:
    """Run the interaction MLP for one feature against N class embeddings.

    Returns logits as a (N,) CPU tensor.
    """
    model.eval()
    N = class_embeds.size(0)
    scores: list[torch.Tensor] = []
    f_proj = f_embed.unsqueeze(0)                          # (1, proj_dim)

    for start in range(0, N, score_batch):
        end   = min(start + score_batch, N)
        c_blk = class_embeds[start:end]                    # (b, proj_dim)
        f_blk = f_proj.expand(c_blk.size(0), -1)           # (b, proj_dim)
        logit = model.interaction(f_blk, c_blk)            # (b,)
        scores.append(logit.detach().cpu())

    return torch.cat(scores, dim=0)


# ── Sample materialisation ────────────────────────────────────────────────────

def _make_hard_neg_record(
    candidate_record: dict,
    jira_id:          str,
    feature_text:     str,
    feature_chunks:   list[dict],
) -> dict:
    """Stamp a candidate class record into a training sample dict.

    `candidate_record` is the (per-sha, deduped) class record from
    train_full.jsonl; we attach the *current* issue's feature side and force
    label=0, is_hard_negative=True so it loads identically to a static HN.
    """
    return {
        "jira_id":          jira_id,
        "sha_before":       candidate_record.get("sha_before", ""),
        "class_path":       candidate_record.get("class_path", ""),
        "class_fqn":        candidate_record.get("class_fqn", ""),
        "feature_text":     feature_text,
        "feature_chunks":   feature_chunks,
        "class_text":       candidate_record.get("class_text", ""),
        "label":            0,
        "is_hard_negative": True,
        "dependency_score": candidate_record.get("dependency_score", 0.0),
        "dependency_fwd":   candidate_record.get("dependency_fwd", 0.0),
        "dependency_rev":   candidate_record.get("dependency_rev", 0.0),
        "depth_fwd":        candidate_record.get("depth_fwd"),
        "depth_rev":        candidate_record.get("depth_rev"),
        "is_seed":          candidate_record.get("is_seed", False),
        "mined":            True,
    }


# ── Public entry point ────────────────────────────────────────────────────────

class HardNegativeMiner:
    """Holds the candidate pool + encoder-cache lifetime across the run."""

    def __init__(
        self,
        candidate_pool:   CandidatePool,
        tokenizer:        PreTrainedTokenizerBase,
        max_chunk_tokens: int  = 512,
        max_class_tokens: int  = 512,
        max_chunks:       int  = 8,
        embed_batch_size: int  = 64,
    ):
        self.pool             = candidate_pool
        self.tokenizer        = tokenizer
        self.max_chunk_tokens = max_chunk_tokens
        self.max_class_tokens = max_class_tokens
        self.max_chunks       = max_chunks
        self.embed_batch_size = embed_batch_size

    def mine(
        self,
        model:                ImpactScoreModel,
        device:               torch.device,
        per_issue_top_k:      dict[str, int],
        jira_ids:             list[str],
    ) -> dict[str, list[dict]]:
        """Run one mining pass.

        Args:
            model            : current model (will be flipped to eval, then back to its caller-state)
            device           : training device
            per_issue_top_k  : {jira_id: how_many_hard_negs_to_return}
            jira_ids         : explicit list of jira_ids to mine (training issues only)

        Returns:
            {jira_id: [hard_neg_sample_dict, ...]}
        """
        was_training = model.training
        model.eval()

        # Group requested issues by sha_before so each codebase is encoded once
        issues_by_sha: dict[str, list[str]] = defaultdict(list)
        missing: list[str] = []
        for jid in jira_ids:
            sha = self.pool.by_jira_sha.get(jid)
            if sha is None:
                missing.append(jid)
                continue
            if sha not in self.pool.by_sha:
                missing.append(jid)
                continue
            issues_by_sha[sha].append(jid)

        if missing:
            log.warning(
                "Hard-neg mining: %d training jira_ids have no candidate-pool entry — they will keep their previous hard negatives.",
                len(missing),
            )

        mined: dict[str, list[dict]] = {}

        sha_order = list(issues_by_sha.keys())
        pbar = tqdm(sha_order, desc="HN mining (by sha)", leave=False)

        for sha in pbar:
            records = self.pool.by_sha[sha]
            class_embeds = _encode_classes_for_sha(
                model            = model,
                tokenizer        = self.tokenizer,
                records          = records,
                device           = device,
                max_class_tokens = self.max_class_tokens,
                embed_batch_size = self.embed_batch_size,
            )                                                          # (N, proj_dim)
            class_paths = [r.get("class_path", "") for r in records]

            for jid in issues_by_sha[sha]:
                top_k = per_issue_top_k.get(jid, 0)
                if top_k <= 0:
                    mined[jid] = []
                    continue

                feature_text, feature_chunks = self.pool.by_jira_feature[jid]
                f_embed = _encode_feature_for_jira(
                    model            = model,
                    tokenizer        = self.tokenizer,
                    feature_chunks   = feature_chunks,
                    feature_text     = feature_text,
                    device           = device,
                    max_chunk_tokens = self.max_chunk_tokens,
                    max_chunks       = self.max_chunks,
                )

                scores = _score_feature_against_classes(
                    model        = model,
                    f_embed      = f_embed,
                    class_embeds = class_embeds,
                )                                                       # (N,) CPU

                positives = self.pool.by_jira_positives.get(jid, set())

                # Mask out positives so they never become hard negs.
                # Use a large negative value, then top-k by score.
                if positives:
                    mask = torch.tensor(
                        [(cp in positives) for cp in class_paths],
                        dtype=torch.bool,
                    )
                    scores = scores.masked_fill(mask, float("-inf"))

                # We may have fewer non-positive candidates than top_k
                k = min(top_k, scores.numel() - len(positives))
                if k <= 0:
                    mined[jid] = []
                    continue

                top_idx = torch.topk(scores, k=k).indices.tolist()

                mined[jid] = [
                    _make_hard_neg_record(
                        candidate_record = records[i],
                        jira_id          = jid,
                        feature_text     = feature_text,
                        feature_chunks   = feature_chunks,
                    )
                    for i in top_idx
                ]

            # Free the (N, proj_dim) buffer before moving to the next sha
            del class_embeds
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if was_training:
            model.train()

        # Fill jira_ids that had no candidates with empty lists so callers don't
        # have to handle KeyError
        for jid in jira_ids:
            mined.setdefault(jid, [])

        return mined


# ── Config helper ─────────────────────────────────────────────────────────────

def build_miner_from_config(
    config:    dict,
    tokenizer: PreTrainedTokenizerBase,
    restrict_to_jira_ids: Optional[set[str]] = None,
) -> Optional[HardNegativeMiner]:
    """Build a miner from the `hard_negative_mining` config block, or return None
    if mining is disabled / not configured."""
    hnm_cfg = config.get("hard_negative_mining", {}) or {}
    if not hnm_cfg.get("enabled", False):
        return None

    train_full = hnm_cfg.get("train_full_jsonl")
    if not train_full:
        raise ValueError(
            "hard_negative_mining.enabled=true but train_full_jsonl is missing from config."
        )

    pool = CandidatePool.from_jsonl(train_full, restrict_to_jira_ids=restrict_to_jira_ids)

    tcfg = config.get("training", {}) or {}
    return HardNegativeMiner(
        candidate_pool   = pool,
        tokenizer        = tokenizer,
        max_chunk_tokens = tcfg.get("max_chunk_tokens", 512),
        max_class_tokens = tcfg.get("max_class_tokens", 512),
        max_chunks       = tcfg.get("max_chunks", 8),
        embed_batch_size = hnm_cfg.get("embed_batch_size", 64),
    )

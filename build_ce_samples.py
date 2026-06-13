"""
build_ce_samples.py — Cross-Encoder Sample Generation with Reciprocal Rank Fusion
==================================================================================
Combines a bi-encoder and a BM25 index via Reciprocal Rank Fusion (RRF) to
produce richer candidate lists for cross-encoder training and evaluation.

Motivation
----------
The bi-encoder alone misses the positives for ~30 test issues entirely (ceiling=0)
because these issues' semantics do not surface the right classes in the top-80.
BM25 recovers some of those positives through lexical matching (class names,
identifiers, method names mentioned in the issue description). RRF fuses both
rankings without requiring calibrated scores from either system.

RRF score for candidate i:
    rrf(i) = 1 / (k + rank_biencoder(i))  +  1 / (k + rank_bm25(i))

where k=60 (Cormack et al. 2009) and ranks are 1-based over the full candidate
pool for that issue's sha_before snapshot.

Pipeline per issue
------------------
1. Encode all candidate classes with bi-encoder   → bi_scores  (N,)
2. Score all candidate classes with BM25          → bm25_scores (N,)
3. Convert both to 1-based ranks over N
4. Compute RRF score for every candidate
5. Take top-K by RRF score (default K=100)
6. Write one JSONL record per candidate, preserving both component scores

Output JSONL fields
-------------------
    jira_id           Jira issue key
    class_path        Relative path of the candidate class
    feature_text      Flattened feature description (chunks joined)
    feature_chunks    Individual feature chunks [{text, ...}, ...]
    class_text        Class source text
    label             1 = truly impacted, 0 = false positive
    biencoder_score   Sigmoid probability from bi-encoder (CE baseline)
    biencoder_rank    1-based rank by bi-encoder over the full pool
    bm25_rank         1-based rank by BM25 over the full pool
    rrf_score         Combined RRF score used for final selection
    rrf_rank          1-based rank by RRF (position in output, 1..top_k)
    from_bm25_rescue  True if NOT in bi-encoder top-K but rescued by BM25
    n_pos_total       Total ground-truth positives for this issue
    n_pos_in_topk     Positives that landed in the RRF top-K window

Usage (requires: pip install rank-bm25)
---------------------------------------
    python build_ce_samples.py \
        --input      .../SAMPLES_V4/train_2000.jsonl \
        --checkpoint .../run_015/best_model.pt \
        --config     .../run_015/config.json \
        --output     .../CE_SAMPLES_V2/ce_train.jsonl \
        --top-k      100

    python build_ce_samples.py --input .../val_full.jsonl  ... --output ce_val.jsonl
    python build_ce_samples.py --input .../test_full.jsonl ... --output ce_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import ssl
import urllib3
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoTokenizer

from model import build_model, ImpactScoreModel

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── BM25 helpers ───────────────────────────────────────────────────────────────

def tokenize_for_code(text: str) -> list[str]:
    """
    SE-specific tokenizer — identical to the one used in diagnose_bm25.py
    after hyperparameter tuning on val_full.jsonl.

    Two-step normalisation:
      1. CamelCase splitting: insert a space at every lowercase→uppercase
         boundary so compound identifiers decompose into sub-tokens.
         "BlobStoreIndex" → "Blob Store Index"
         This is critical for matching Java class names against Jira
         descriptions, which typically describe behaviour using natural
         English words rather than raw identifiers.
      2. Non-alphanumeric substitution: replace all punctuation and symbols
         with spaces, then lowercase and split.

    Consistency with the tuned BM25 baseline (k1=1.5, b=0.5) ensures that
    the retrieval signal injected into RRF is directly comparable to the
    baseline reported in the thesis evaluation.
    """
    if not text:
        return []
    # Step 1 — split CamelCase
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    # Step 2 — normalise non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    return text.lower().split()


def build_bm25_index(texts: list[str]) -> BM25Okapi:
    """
    Build a BM25Okapi index with the hyperparameters tuned on val_full.jsonl:
        k1 = 1.5  — controls term-frequency saturation
        b  = 0.5  — controls document-length normalisation
    These parameters are held fixed to ensure reproducibility and comparability
    with the BM25 baseline figures reported in the evaluation section.
    """
    tokenized = [tokenize_for_code(t) for t in texts]
    return BM25Okapi(tokenized, k1=1.5, b=0.5)


def bm25_scores_for_query(index: BM25Okapi, query_text: str) -> np.ndarray:
    """Score all documents against query_text. Returns (N,) float32 array."""
    tokens = tokenize_for_code(query_text)
    if not tokens:
        return np.zeros(index.corpus_size, dtype=np.float32)
    return index.get_scores(tokens).astype(np.float32)


# ── Ranking + RRF ──────────────────────────────────────────────────────────────

def scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    """
    Convert scores to 1-based ranks (highest score = rank 1).
    Ties are broken by original index order (stable sort).
    """
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def rrf_combine(bi_ranks: np.ndarray, bm25_ranks: np.ndarray, k: int) -> np.ndarray:
    """
    Reciprocal Rank Fusion (Cormack et al. 2009).
    rrf(i) = 1/(k + bi_ranks[i]) + 1/(k + bm25_ranks[i])
    Standard k=60 balances early-rank precision with late-rank contribution.
    """
    return (1.0 / (k + bi_ranks) + 1.0 / (k + bm25_ranks)).astype(np.float64)


# ── Bi-encoder helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def encode_classes_batch(
    model, tokenizer, texts: list[str],
    device, max_class_tokens: int, batch_size: int = 128,
) -> torch.Tensor:
    """Encode class texts → (N, proj_dim) CPU tensor."""
    model.eval()
    embeds: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tokenizer(chunk, max_length=max_class_tokens,
                        truncation=True, padding="max_length",
                        return_tensors="pt")
        emb = model.encode_class(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        ).cpu()
        embeds.append(emb)
    return torch.cat(embeds, dim=0)


@torch.no_grad()
def encode_feature_single(
    model, tokenizer, chunks: list[dict], text: str,
    device, max_chunk_tokens: int, max_chunks: int,
) -> torch.Tensor:
    """Encode one issue's feature → (proj_dim,) CPU tensor."""
    model.eval()
    chunks = (chunks if chunks else [{"text": text}])[:max_chunks]
    enc    = tokenizer(
        [c.get("text", "") for c in chunks],
        max_length=max_chunk_tokens, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    ids   = enc["input_ids"].unsqueeze(0).to(device)
    mask  = enc["attention_mask"].unsqueeze(0).to(device)
    cmask = torch.ones(1, len(chunks), dtype=torch.long, device=device)
    return model.encode_feature(ids, mask, cmask).squeeze(0).cpu()


@torch.no_grad()
def score_feature_vs_classes(
    model, f_embed: torch.Tensor, c_embeds: torch.Tensor,
    device, batch_size: int = 4096,
) -> np.ndarray:
    """Bi-encoder sigmoid scores → (N,) numpy float32 array."""
    model.eval()
    scores: list[torch.Tensor] = []
    f = f_embed.unsqueeze(0).to(device)
    for start in range(0, c_embeds.size(0), batch_size):
        c_blk = c_embeds[start : start + batch_size].to(device)
        logit = model.interaction(f.expand(c_blk.size(0), -1), c_blk)
        scores.append(torch.sigmoid(logit).detach().cpu())
    return torch.cat(scores, dim=0).numpy().astype(np.float32)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_full_jsonl(path: Path) -> dict[str, list[dict]]:
    log.info("Loading %s ...", path)
    by_issue: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        total = sum(1 for _ in f)
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, total=total, desc="  Reading", unit=" lines"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            jid = rec.get("jira_id", "")
            if jid:
                by_issue[jid].append(rec)
    log.info("  %d issues  |  %d total records",
             len(by_issue), sum(len(v) for v in by_issue.values()))
    return dict(by_issue)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_ce_samples(
    input_path:  Path,
    checkpoint:  Path,
    config:      dict,
    output_path: Path,
    top_k:       int,
    rrf_k:       int,
    embed_batch: int,
):
    tcfg   = config["training"]
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    log.info("Device      : %s", device)
    log.info("Top-K       : %d  (RRF window)", top_k)
    log.info("RRF k param : %d", rrf_k)

    # ── Load bi-encoder ──────────────────────────────────────────────────
    model_name = config["model"]["model_name"]
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = build_model(config["model"]).to(device)
    state      = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.eval()
    log.info("Bi-encoder  : %s", checkpoint)

    # ── Load input JSONL ─────────────────────────────────────────────────
    by_issue = load_full_jsonl(input_path)

    # ── Group by sha_before ──────────────────────────────────────────────
    sha_to_classes: dict[str, list[dict]] = defaultdict(list)
    sha_seen:       dict[str, set[str]]   = defaultdict(set)
    issue_to_sha:   dict[str, str]        = {}
    issue_feature:  dict[str, tuple]      = {}

    for jid, records in by_issue.items():
        if not records:
            continue
        sha = records[0].get("sha_before", "")
        issue_to_sha[jid] = sha
        r0 = records[0]
        issue_feature[jid] = (r0.get("feature_text", ""),
                               r0.get("feature_chunks", []))
        for rec in records:
            cp = rec.get("class_path", "")
            if cp and cp not in sha_seen[sha]:
                sha_seen[sha].add(cp)
                sha_to_classes[sha].append(rec)

    log.info("Unique sha_before snapshots : %d", len(sha_to_classes))
    log.info("Total unique (sha, class)   : %d",
             sum(len(v) for v in sha_to_classes.values()))

    # ── Lookup tables ────────────────────────────────────────────────────
    label_lookup: dict[tuple, int] = {}
    for jid, records in by_issue.items():
        for rec in records:
            label_lookup[(jid, rec.get("class_path", ""))] = int(rec.get("label", 0))

    sha_class_text: dict[tuple, str] = {}
    for sha, class_records in sha_to_classes.items():
        for rec in class_records:
            sha_class_text[(sha, rec.get("class_path", ""))] = rec.get("class_text", "")

    # ── Phase 1: bi-encoder embeddings + BM25 index per snapshot ─────────
    log.info("=" * 60)
    log.info("Phase 1 / 2 — Encoding snapshots (bi-encoder + BM25)")
    log.info("=" * 60)

    sha_bi_embeds:   dict[str, torch.Tensor] = {}
    sha_bm25_index:  dict[str, BM25Okapi]    = {}
    sha_class_paths: dict[str, list[str]]    = {}

    for sha, class_records in tqdm(sha_to_classes.items(),
                                   desc="Encoding snapshots"):
        texts  = [r.get("class_text", "") for r in class_records]
        cpaths = [r.get("class_path", "")  for r in class_records]

        sha_bi_embeds[sha]   = encode_classes_batch(
            model, tokenizer, texts, device,
            max_class_tokens = tcfg.get("max_class_tokens", 512),
            batch_size       = embed_batch,
        )
        sha_bm25_index[sha]  = build_bm25_index(texts)
        sha_class_paths[sha] = cpaths

    # ── Phase 2: RRF fusion and sample writing ────────────────────────────
    log.info("=" * 60)
    log.info("Phase 2 / 2 — RRF fusion and sample writing")
    log.info("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written            = 0
    n_pos_written        = 0
    n_issues_done        = 0
    n_issues_no_pos_pool = 0
    n_issues_no_pos_topk = 0
    n_bm25_rescued       = 0
    ceiling_values:  list[float] = []
    bm25_only_counts: list[int]  = []

    with open(output_path, "w", encoding="utf-8") as out_f:
        for jid in tqdm(by_issue, desc="Scoring + fusing"):
            sha = issue_to_sha.get(jid)
            if sha is None or sha not in sha_bi_embeds:
                continue

            feat_text, feat_chunks = issue_feature[jid]
            class_paths = sha_class_paths[sha]
            N           = len(class_paths)

            all_labels  = [label_lookup.get((jid, cp), 0) for cp in class_paths]
            n_pos_total = sum(all_labels)
            if n_pos_total == 0:
                n_issues_no_pos_pool += 1
                continue

            # ── Bi-encoder ────────────────────────────────────────────────
            f_embed   = encode_feature_single(
                model, tokenizer, feat_chunks, feat_text, device,
                max_chunk_tokens = tcfg.get("max_chunk_tokens", 512),
                max_chunks       = tcfg.get("max_chunks", 8),
            )
            bi_scores = score_feature_vs_classes(
                model, f_embed, sha_bi_embeds[sha], device,
            )                                            # (N,) float32
            bi_ranks  = scores_to_ranks(bi_scores)      # (N,) int

            # ── BM25 ──────────────────────────────────────────────────────
            flat_feature = (
                " ".join(c.get("text", "") for c in feat_chunks)
                if feat_chunks else feat_text
            )
            bm25_scores = bm25_scores_for_query(sha_bm25_index[sha], flat_feature)
            bm25_ranks  = scores_to_ranks(bm25_scores)  # (N,) int

            # ── RRF ───────────────────────────────────────────────────────
            rrf_scores = rrf_combine(bi_ranks, bm25_ranks, k=rrf_k)
            top_idx    = np.argsort(-rrf_scores, kind="stable")[:min(top_k, N)]

            # Which candidates were outside the bi-encoder's own top-K
            bi_top_k_set = set(np.where(bi_ranks <= top_k)[0].tolist())

            topk_labels   = [all_labels[i] for i in top_idx]
            n_pos_in_topk = sum(topk_labels)
            if n_pos_in_topk == 0:
                n_issues_no_pos_topk += 1

            ceiling_values.append(n_pos_in_topk / n_pos_total)

            bm25_only = [i for i in top_idx if i not in bi_top_k_set]
            bm25_only_counts.append(len(bm25_only))
            if any(all_labels[i] == 1 and i not in bi_top_k_set for i in top_idx):
                n_bm25_rescued += 1

            for rrf_pos, idx in enumerate(top_idx, start=1):
                cp    = class_paths[idx]
                label = label_lookup.get((jid, cp), 0)
                out_f.write(json.dumps({
                    "jira_id":          jid,
                    "class_path":       cp,
                    "feature_text":     flat_feature,
                    "feature_chunks":   feat_chunks,
                    "class_text":       sha_class_text.get((sha, cp), ""),
                    "label":            label,
                    "biencoder_score":  round(float(bi_scores[idx]), 6),
                    "biencoder_rank":   int(bi_ranks[idx]),
                    "bm25_rank":        int(bm25_ranks[idx]),
                    "rrf_score":        round(float(rrf_scores[idx]), 8),
                    "rrf_rank":         rrf_pos,
                    "from_bm25_rescue": bool(idx not in bi_top_k_set),
                    "n_pos_total":      n_pos_total,
                    "n_pos_in_topk":    n_pos_in_topk,
                }, ensure_ascii=False) + "\n")
                n_written += 1
                if label == 1:
                    n_pos_written += 1

            n_issues_done += 1

    # ── Summary ────────────────────────────────────────────────────────────
    mean_ceiling   = sum(ceiling_values) / len(ceiling_values) if ceiling_values else 0.0
    n_perfect      = sum(1 for v in ceiling_values if v == 1.0)
    n_partial      = sum(1 for v in ceiling_values if 0.0 < v < 1.0)
    n_zero         = sum(1 for v in ceiling_values if v == 0.0)
    mean_bm25_only = sum(bm25_only_counts) / max(len(bm25_only_counts), 1)

    log.info("=" * 60)
    log.info("SAMPLE GENERATION COMPLETE")
    log.info("=" * 60)
    log.info("  Issues processed               : %d", n_issues_done)
    log.info("  Issues skipped (no pos in pool): %d", n_issues_no_pos_pool)
    log.info("  Records written                : %d  (avg %.1f per issue)",
             n_written, n_written / max(n_issues_done, 1))
    log.info("  Positives in output            : %d  (pos rate %.2f%%)",
             n_pos_written, 100 * n_pos_written / max(n_written, 1))
    log.info("")
    log.info("  PIPELINE RECALL CEILING (RRF top-%d)", top_k)
    log.info("    Mean ceiling                 : %.4f", mean_ceiling)
    log.info("    Issues: all pos in top-%d    : %d  (%.1f%%)",
             top_k, n_perfect, 100 * n_perfect / max(n_issues_done, 1))
    log.info("    Issues: some pos missed      : %d  (%.1f%%)",
             n_partial, 100 * n_partial / max(n_issues_done, 1))
    log.info("    Issues: all pos missed (zero): %d  (%.1f%%)",
             n_zero, 100 * n_zero / max(n_issues_done, 1))
    log.info("")
    log.info("  BM25 CONTRIBUTION")
    log.info("    Issues where BM25 rescued ≥1 positive : %d", n_bm25_rescued)
    log.info("    Mean BM25-only candidates per issue   : %.1f / %d",
             mean_bm25_only, top_k)
    log.info("    Total BM25-only records written       : %d",
             sum(bm25_only_counts))
    log.info("")
    log.info("  Output: %s", output_path)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build CE samples via RRF fusion of bi-encoder and BM25"
    )
    parser.add_argument("--input",       required=True,
                        help="Full-codebase JSONL (train_2000/val_full/test_full)")
    parser.add_argument("--checkpoint",  required=True,
                        help="Bi-encoder best_model.pt")
    parser.add_argument("--config",      required=True,
                        help="Bi-encoder config.json")
    parser.add_argument("--output",      required=True,
                        help="Output CE JSONL path")
    parser.add_argument("--top-k",       type=int, default=100,
                        help="RRF candidates to keep per issue (default 100)")
    parser.add_argument("--rrf-k",       type=int, default=60,
                        help="RRF k constant — Cormack et al. 2009 (default 60)")
    parser.add_argument("--embed-batch", type=int, default=256,
                        help="Bi-encoder class encoding batch size (default 256)")
    args = parser.parse_args()

    for p, name in [
        (args.input,      "--input"),
        (args.checkpoint, "--checkpoint"),
        (args.config,     "--config"),
    ]:
        if not Path(p).exists():
            import sys; sys.exit(f"{name} not found: {p}")

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    build_ce_samples(
        input_path  = Path(args.input),
        checkpoint  = Path(args.checkpoint),
        config      = config,
        output_path = Path(args.output),
        top_k       = args.top_k,
        rrf_k       = args.rrf_k,
        embed_batch = args.embed_batch,
    )


if __name__ == "__main__":
    main()
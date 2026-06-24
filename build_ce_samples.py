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

RRF can be disabled with --rrf false, in which case BM25 is skipped entirely
(no index built, no scoring) and candidates are selected by bi-encoder score
alone. Useful as an ablation baseline against the fused pipeline.

Pipeline per issue
------------------
1. Encode all candidate classes with bi-encoder   → bi_scores  (N,)
2. [--rrf true only] Score all candidates with BM25 → bm25_scores (N,)
3. Convert score(s) to 1-based rank(s) over N
4. [--rrf true] Compute RRF score for every candidate; else rank by bi-encoder alone
5. Take top-K by the selection score (default K=100)
6. Write one JSONL record per candidate, preserving component score(s)

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
    bm25_rank         1-based rank by BM25 over the full pool (null if --rrf false)
    rrf_score         Combined RRF score used for final selection (null if --rrf false)
    rrf_rank          1-based rank by the selection score (position in output, 1..top_k)
    from_bm25_rescue  True if NOT in bi-encoder top-K but rescued by BM25 (always False if --rrf false)
    fusion_mode       "rrf" or "biencoder_only" — which mode produced this record
    n_pos_total       Total ground-truth positives for this issue
    n_pos_in_topk     Positives that landed in the selected top-K window

Multi-GPU
---------
Bi-encoder encoding is embarrassingly parallel across sha_before snapshots —
each snapshot's class embeddings (and, if enabled, BM25 index) are independent
of every other snapshot. Rather than wrapping the model in
torch.nn.DataParallel (which only splits a single batch within one process and
adds host-side scatter/gather overhead for many small, variably-shaped calls
like the ones in this script), this script spawns one worker process per GPU,
each with its own full copy of the model. Snapshots are partitioned
round-robin across workers, so each GPU encodes a disjoint subset of
snapshots and scores a disjoint subset of issues at full independent
throughput. Outputs are merged at the end. With 4 GPUs this gives close to a
4x speedup on Phase 1 (encoding), which is normally the bottleneck.

Usage (requires: pip install rank-bm25)
---------------------------------------
    python build_ce_samples.py \
        --input      .../SAMPLES_V4/train_2000.jsonl \
        --checkpoint .../run_016/best_model.pt \
        --config     .../run_016/config.json \
        --output     .../CE_SAMPLES_V2/ce_train.jsonl \
        --top-k      100

    # Disable RRF (bi-encoder only, no BM25):
    python build_ce_samples.py --input ... --checkpoint ... --config ... \
        --output ce_train_biencoder_only.jsonl --rrf false

    # Use all visible GPUs (default) or restrict explicitly:
    python build_ce_samples.py --input ... --output ... --num-gpus 4
    python build_ce_samples.py --input ... --output ... --gpu-ids 0,1,2,3
    python build_ce_samples.py --input ... --output ... --num-gpus 1   # single GPU, old behaviour

    python build_ce_samples.py --input .../val_full.jsonl  ... --output ce_val.jsonl
    python build_ce_samples.py --input .../test_full.jsonl ... --output ce_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
import re
import ssl
import traceback
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

def load_full_jsonl(path: Path, quiet: bool = False) -> dict[str, list[dict]]:
    if not quiet:
        log.info("Loading %s ...", path)
    by_issue: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        total = sum(1 for _ in f)
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, total=total, desc="  Reading", unit=" lines", disable=quiet):
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
    if not quiet:
        log.info("  %d issues  |  %d total records",
                 len(by_issue), sum(len(v) for v in by_issue.values()))
    return dict(by_issue)


# ── Per-worker pipeline (Phase 1 + Phase 2 over an assigned partition) ─────────

def process_partition(
    rank:             int,
    world_size:       int,
    device:           torch.device,
    input_path:       Path,
    checkpoint:       Path,
    config:           dict,
    part_output_path: Path,
    top_k:            int,
    rrf_k:            int,
    embed_batch:      int,
    use_rrf:          bool,
) -> dict:
    """
    Runs the full encode → score → select → write pipeline restricted to a
    partition of sha_before snapshots (assigned round-robin by rank). With
    world_size=1 this is the entire dataset — identical to the original
    single-GPU behaviour.
    """
    tag  = f"[worker {rank}/{world_size} @ {device}]"
    tcfg = config["training"]
    log.info("%s Device : %s", tag, device)
    log.info("%s Top-K  : %d", tag, top_k)
    log.info("%s Fusion : %s", tag, f"RRF (k={rrf_k})" if use_rrf else "bi-encoder only (RRF disabled)")

    # ── Load bi-encoder ──────────────────────────────────────────────────
    model_name = config["model"]["model_name"]
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = build_model(config["model"]).to(device)
    state      = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.eval()
    log.info("%s Bi-encoder : %s", tag, checkpoint)

    # ── Load input JSONL (each worker reads independently — see module docstring) ─
    by_issue = load_full_jsonl(input_path, quiet=(rank != 0))

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

    # ── Partition snapshots round-robin across workers ────────────────────
    all_shas = sorted(sha_to_classes.keys())
    if world_size > 1:
        assigned_shas  = set(all_shas[rank::world_size])
        sha_to_classes = {s: v for s, v in sha_to_classes.items() if s in assigned_shas}
        by_issue       = {jid: recs for jid, recs in by_issue.items()
                           if issue_to_sha.get(jid) in assigned_shas}
        issue_to_sha   = {jid: s for jid, s in issue_to_sha.items() if s in assigned_shas}
        issue_feature  = {jid: f for jid, f in issue_feature.items() if jid in by_issue}
        log.info("%s Snapshots assigned : %d / %d", tag, len(sha_to_classes), len(all_shas))

    log.info("%s Snapshots to encode : %d", tag, len(sha_to_classes))
    log.info("%s Issues to score     : %d", tag, len(by_issue))

    # ── Lookup tables ────────────────────────────────────────────────────
    label_lookup: dict[tuple, int] = {}
    for jid, records in by_issue.items():
        for rec in records:
            label_lookup[(jid, rec.get("class_path", ""))] = int(rec.get("label", 0))

    sha_class_text: dict[tuple, str] = {}
    for sha, class_records in sha_to_classes.items():
        for rec in class_records:
            sha_class_text[(sha, rec.get("class_path", ""))] = rec.get("class_text", "")

    # ── Phase 1: bi-encoder embeddings (+ BM25 index, if enabled) ─────────
    log.info("%s Phase 1/2 — Encoding snapshots", tag)

    sha_bi_embeds:   dict[str, torch.Tensor] = {}
    sha_bm25_index:  dict[str, BM25Okapi]    = {}
    sha_class_paths: dict[str, list[str]]    = {}

    for sha, class_records in tqdm(sha_to_classes.items(),
                                    desc=f"{tag} Encoding",
                                    position=rank, leave=(rank == 0)):
        texts  = [r.get("class_text", "") for r in class_records]
        cpaths = [r.get("class_path", "")  for r in class_records]

        sha_bi_embeds[sha] = encode_classes_batch(
            model, tokenizer, texts, device,
            max_class_tokens = tcfg.get("max_class_tokens", 512),
            batch_size       = embed_batch,
        )
        if use_rrf:
            sha_bm25_index[sha] = build_bm25_index(texts)
        sha_class_paths[sha] = cpaths

    # ── Phase 2: candidate selection (RRF or bi-encoder-only) + writing ───
    log.info("%s Phase 2/2 — Selection and sample writing", tag)

    part_output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written, n_pos_written = 0, 0
    n_issues_done            = 0
    n_issues_no_pos_pool     = 0
    n_issues_no_pos_topk     = 0
    n_bm25_rescued           = 0
    ceiling_values:   list[float] = []
    bm25_only_counts: list[int]   = []

    with open(part_output_path, "w", encoding="utf-8") as out_f:
        for jid in tqdm(by_issue, desc=f"{tag} Scoring", position=rank, leave=(rank == 0)):
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

            flat_feature = (
                " ".join(c.get("text", "") for c in feat_chunks)
                if feat_chunks else feat_text
            )

            # Which candidates were inside the bi-encoder's own top-K
            bi_top_k_set = set(np.where(bi_ranks <= top_k)[0].tolist())

            if use_rrf:
                # ── BM25 ──────────────────────────────────────────────────
                bm25_scores = bm25_scores_for_query(sha_bm25_index[sha], flat_feature)
                bm25_ranks  = scores_to_ranks(bm25_scores)  # (N,) int

                # ── RRF ───────────────────────────────────────────────────
                rrf_scores = rrf_combine(bi_ranks, bm25_ranks, k=rrf_k)
                top_idx    = np.argsort(-rrf_scores, kind="stable")[:min(top_k, N)]
            else:
                bm25_ranks = None
                rrf_scores = None
                top_idx    = np.argsort(-bi_scores, kind="stable")[:min(top_k, N)]

            topk_labels   = [all_labels[i] for i in top_idx]
            n_pos_in_topk = sum(topk_labels)
            if n_pos_in_topk == 0:
                n_issues_no_pos_topk += 1

            ceiling_values.append(n_pos_in_topk / n_pos_total)

            if use_rrf:
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
                    "bm25_rank":        int(bm25_ranks[idx]) if use_rrf else None,
                    "rrf_score":        round(float(rrf_scores[idx]), 8) if use_rrf else None,
                    "rrf_rank":         rrf_pos,
                    "from_bm25_rescue": bool(idx not in bi_top_k_set) if use_rrf else False,
                    "fusion_mode":      "rrf" if use_rrf else "biencoder_only",
                    "n_pos_total":      n_pos_total,
                    "n_pos_in_topk":    n_pos_in_topk,
                }, ensure_ascii=False) + "\n")
                n_written += 1
                if label == 1:
                    n_pos_written += 1

            n_issues_done += 1

    log.info("%s Done — %d issues, %d records -> %s",
             tag, n_issues_done, n_written, part_output_path)

    return {
        "n_written":            n_written,
        "n_pos_written":        n_pos_written,
        "n_issues_done":        n_issues_done,
        "n_issues_no_pos_pool": n_issues_no_pos_pool,
        "n_issues_no_pos_topk": n_issues_no_pos_topk,
        "n_bm25_rescued":       n_bm25_rescued,
        "ceiling_values":       ceiling_values,
        "bm25_only_counts":     bm25_only_counts,
    }


# ── Multi-GPU orchestration ─────────────────────────────────────────────────────

def _worker_entry(rank, world_size, gpu_id, input_path, checkpoint, config,
                   part_output_path, top_k, rrf_k, embed_batch, use_rrf, result_queue):
    """Top-level (picklable) target run inside each spawned worker process."""
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        stats = process_partition(
            rank=rank, world_size=world_size, device=device,
            input_path=input_path, checkpoint=checkpoint, config=config,
            part_output_path=part_output_path, top_k=top_k, rrf_k=rrf_k,
            embed_batch=embed_batch, use_rrf=use_rrf,
        )
        result_queue.put((rank, "ok", stats))
    except Exception:
        result_queue.put((rank, "error", traceback.format_exc()))
        raise


def merge_parts(part_paths: list[Path], final_output_path: Path) -> None:
    log.info("Merging %d worker output file(s) -> %s", len(part_paths), final_output_path)
    with open(final_output_path, "w", encoding="utf-8") as out_f:
        for pp in part_paths:
            if not pp.exists():
                continue
            with open(pp, encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
            pp.unlink()


def aggregate_stats(stats_list: list[dict]) -> dict:
    merged = {
        "n_written": 0, "n_pos_written": 0, "n_issues_done": 0,
        "n_issues_no_pos_pool": 0, "n_issues_no_pos_topk": 0,
        "n_bm25_rescued": 0, "ceiling_values": [], "bm25_only_counts": [],
    }
    for s in stats_list:
        for k in ("n_written", "n_pos_written", "n_issues_done",
                  "n_issues_no_pos_pool", "n_issues_no_pos_topk", "n_bm25_rescued"):
            merged[k] += s.get(k, 0)
        merged["ceiling_values"].extend(s.get("ceiling_values", []))
        merged["bm25_only_counts"].extend(s.get("bm25_only_counts", []))
    return merged


def print_summary(stats: dict, output_path: Path, top_k: int, use_rrf: bool) -> None:
    ceiling_values   = stats["ceiling_values"]
    bm25_only_counts = stats["bm25_only_counts"]
    n_written        = stats["n_written"]
    n_pos_written    = stats["n_pos_written"]
    n_issues_done    = stats["n_issues_done"]

    mean_ceiling   = sum(ceiling_values) / len(ceiling_values) if ceiling_values else 0.0
    n_perfect      = sum(1 for v in ceiling_values if v == 1.0)
    n_partial      = sum(1 for v in ceiling_values if 0.0 < v < 1.0)
    n_zero         = sum(1 for v in ceiling_values if v == 0.0)
    mean_bm25_only = sum(bm25_only_counts) / max(len(bm25_only_counts), 1)

    log.info("=" * 60)
    log.info("SAMPLE GENERATION COMPLETE")
    log.info("=" * 60)
    log.info("  Issues processed               : %d", n_issues_done)
    log.info("  Issues skipped (no pos in pool): %d", stats["n_issues_no_pos_pool"])
    log.info("  Records written                : %d  (avg %.1f per issue)",
             n_written, n_written / max(n_issues_done, 1))
    log.info("  Positives in output            : %d  (pos rate %.2f%%)",
             n_pos_written, 100 * n_pos_written / max(n_written, 1))
    log.info("")
    log.info("  PIPELINE RECALL CEILING (%s top-%d)",
             "RRF" if use_rrf else "bi-encoder-only", top_k)
    log.info("    Mean ceiling                 : %.4f", mean_ceiling)
    log.info("    Issues: all pos in top-%d    : %d  (%.1f%%)",
             top_k, n_perfect, 100 * n_perfect / max(n_issues_done, 1))
    log.info("    Issues: some pos missed      : %d  (%.1f%%)",
             n_partial, 100 * n_partial / max(n_issues_done, 1))
    log.info("    Issues: all pos missed (zero): %d  (%.1f%%)",
             n_zero, 100 * n_zero / max(n_issues_done, 1))
    log.info("")
    if use_rrf:
        log.info("  BM25 CONTRIBUTION")
        log.info("    Issues where BM25 rescued ≥1 positive : %d", stats["n_bm25_rescued"])
        log.info("    Mean BM25-only candidates per issue   : %.1f / %d",
                 mean_bm25_only, top_k)
        log.info("    Total BM25-only records written       : %d",
                 sum(bm25_only_counts))
    else:
        log.info("  BM25 CONTRIBUTION : skipped (--rrf false — bi-encoder-only ranking)")
    log.info("")
    log.info("  Output: %s", output_path)


def resolve_gpu_ids(args) -> list[int]:
    """
    Decide which CUDA device indices to use.
    --gpu-ids takes priority; otherwise --num-gpus (default: all visible
    CUDA devices); returns [] if no CUDA is visible (CPU/MPS single-process
    fallback).
    """
    if args.gpu_ids:
        ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
        avail = torch.cuda.device_count()
        for i in ids:
            if i >= avail:
                import sys
                sys.exit(f"--gpu-ids requested cuda:{i} but only {avail} CUDA device(s) visible")
        return ids
    if torch.cuda.is_available():
        avail = torch.cuda.device_count()
        n = args.num_gpus if args.num_gpus is not None else avail
        n = max(1, min(n, avail))
        return list(range(n))
    return []


# ── Entry point ────────────────────────────────────────────────────────────────

def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in ("true", "t", "yes", "y", "1"):
        return True
    if v in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {v!r}")


def main():
    parser = argparse.ArgumentParser(
        description="Build CE samples via RRF fusion of bi-encoder and BM25 (or bi-encoder only)"
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
                        help="Candidates to keep per issue (default 100)")
    parser.add_argument("--rrf-k",       type=int, default=60,
                        help="RRF k constant — Cormack et al. 2009 (default 60); "
                             "ignored if --rrf false")
    parser.add_argument("--rrf",         type=str2bool, default=True,
                        help="true (default): fuse bi-encoder + BM25 via RRF. "
                             "false: rank by bi-encoder only and skip BM25 entirely.")
    parser.add_argument("--embed-batch", type=int, default=256,
                        help="Bi-encoder class encoding batch size, per GPU (default 256)")
    parser.add_argument("--num-gpus",    type=int, default=None,
                        help="Number of CUDA devices to use (default: all visible). "
                             "Each GPU runs an independent worker over a partition of "
                             "sha_before snapshots. Use 1 for single-GPU (legacy) behaviour.")
    parser.add_argument("--gpu-ids",     type=str, default=None,
                        help="Comma-separated CUDA device indices, e.g. '0,1,2,3'. "
                             "Overrides --num-gpus.")
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("RRF fusion : %s", "ENABLED (bi-encoder + BM25)" if args.rrf else "DISABLED (bi-encoder only)")

    gpu_ids = resolve_gpu_ids(args)

    # ── Single process: 1 GPU, or no CUDA visible (CPU/MPS) ───────────────
    if len(gpu_ids) <= 1:
        if gpu_ids:
            device = torch.device(f"cuda:{gpu_ids[0]}")
        else:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        log.info("Running single-process on device: %s", device)

        stats = process_partition(
            rank=0, world_size=1, device=device,
            input_path=Path(args.input), checkpoint=Path(args.checkpoint), config=config,
            part_output_path=output_path, top_k=args.top_k, rrf_k=args.rrf_k,
            embed_batch=args.embed_batch, use_rrf=args.rrf,
        )
        print_summary(stats, output_path, args.top_k, args.rrf)
        return

    # ── Multi-GPU: one worker process per GPU ──────────────────────────────
    world_size = len(gpu_ids)
    log.info("Multi-GPU run : %d workers on CUDA device(s) %s", world_size, gpu_ids)

    ctx          = mp.get_context("spawn")
    result_queue = ctx.Queue()
    part_paths   = [output_path.with_name(f"{output_path.stem}.part{r}{output_path.suffix}")
                    for r in range(world_size)]

    procs = []
    for rank, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(
            target=_worker_entry,
            args=(rank, world_size, gpu_id, Path(args.input), Path(args.checkpoint), config,
                  part_paths[rank], args.top_k, args.rrf_k, args.embed_batch, args.rrf,
                  result_queue),
        )
        p.start()
        procs.append(p)

    results  = {}
    errored  = []
    remaining = set(range(world_size))
    while remaining:
        try:
            rank, status, payload = result_queue.get(timeout=5)
        except queue.Empty:
            for r in list(remaining):
                if not procs[r].is_alive():
                    errored.append((r, f"worker process died unexpectedly "
                                        f"(exit code {procs[r].exitcode})"))
                    remaining.discard(r)
            continue
        if status == "error":
            errored.append((rank, payload))
        else:
            results[rank] = payload
        remaining.discard(rank)

    for p in procs:
        p.join()

    if errored:
        for rank, info in errored:
            log.error("Worker rank %d failed:\n%s", rank, info)
        import sys
        sys.exit(f"{len(errored)} worker(s) failed — see tracebacks above")

    merge_parts([part_paths[r] for r in range(world_size)], output_path)
    merged_stats = aggregate_stats([results[r] for r in range(world_size)])
    print_summary(merged_stats, output_path, args.top_k, args.rrf)


if __name__ == "__main__":
    main()
"""
flim_finetune.py — Stage 1: fine-tune CodeBERT on (issue, function) pairs
====
Replicates FLIM's CodeBERT fine-tuning (code-search objective) on our data:
  positive pair = (feature description, one method of a truly impacted class)
  negatives     = in-batch (other functions in the same batch)

Hyper-parameters follow the FLIM README:
  nl_length=128, code_length=256, batch=32, lr=2e-5, 10 epochs.

Validation = file-level ranking on a sample of val issues using max-pooled
function similarity (FLIM's semantic score), early-stopped on MRR.

Multi-GPU:
  Controlled via config_flim.json "multi_gpu" key:
    { "enabled": true, "device_ids": [0, 1, 2, 3] }
  DataParallel wraps ONLY the embed step, not the full forward, so that
  InfoNCE always sees the full batch of in-batch negatives (B negatives,
  not B/N). Optimizer is attached to the raw model before any wrapping.

Usage:
    python flim_finetune.py --config config_flim.json
"""
from __future__ import annotations
import argparse, json, logging, os, random, ssl, urllib3
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from flim_model  import build_flim_encoder
from flim_common import (extract_functions, load_jsonl_by_issue,
                    mean_reciprocal_rank, recall_at_k,
                    open_source_cache, lookup_source)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── Multi-GPU helpers ────

class EmbedWrapper(nn.Module):
    """
    Exposes model.embed(input_ids, attention_mask) as forward().
    Required for nn.DataParallel, which only distributes forward().
    Keeping the embed step separate means InfoNCE is always computed
    on the FULL batch after gathering, preserving the original objective.
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        return self.base.embed(input_ids, attention_mask)


def get_primary_device(multi_cfg: dict) -> torch.device:
    """
    Returns the primary (output-gathering) device.
    With multi_gpu, primary = cuda:<device_ids[0]>.
    Otherwise falls through cuda → mps → cpu.
    """
    if (multi_cfg.get("enabled", False)
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 0):
        ids = multi_cfg.get("device_ids", [0])
        return torch.device(f"cuda:{ids[0]}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_embed_parallel(
    model:       nn.Module,
    multi_cfg:   dict,
    primary_dev: torch.device,
) -> nn.Module:
    """
    Wraps model.embed() in DataParallel if multi_gpu.enabled and enough
    CUDA devices are available.  Returns a plain EmbedWrapper otherwise.

    The returned callable signature is:
        embed_par(input_ids, attention_mask) -> embeddings

    Inputs must be on primary_dev; DataParallel scatters to other devices
    automatically. Outputs are always gathered on primary_dev.
    """
    wrapper = EmbedWrapper(model)

    if not multi_cfg.get("enabled", False):
        log.info("multi_gpu disabled — single device: %s", primary_dev)
        return wrapper

    if not torch.cuda.is_available():
        log.warning("multi_gpu.enabled=true but CUDA unavailable — single device.")
        return wrapper

    n_avail    = torch.cuda.device_count()
    requested  = multi_cfg.get("device_ids", list(range(n_avail)))
    device_ids = [d for d in requested if d < n_avail]

    if len(device_ids) < 2:
        log.warning(
            "multi_gpu requested but only %d valid device(s) in device_ids=%s. "
            "Falling back to single device.", len(device_ids), requested)
        return wrapper

    dp = nn.DataParallel(wrapper, device_ids=device_ids)
    log.info("DataParallel active on CUDA devices: %s  (primary: cuda:%d)",
             device_ids, device_ids[0])
    for i in device_ids:
        props = torch.cuda.get_device_properties(i)
        log.info("  cuda:%d — %s  (%.1f GB)", i, props.name,
                 props.total_memory / 1e9)
    return dp


# ── InfoNCE loss (central, full-batch) ────

def infonce_loss(nl_emb: torch.Tensor,
                 code_emb: torch.Tensor,
                 scale: float = 20.0) -> torch.Tensor:
    """
    Symmetric InfoNCE with full in-batch negatives.
    nl_emb and code_emb must already be L2-normalised (done by model.embed).
    Both tensors are on the primary device (gathered by DataParallel).

    Computing this centrally — rather than inside model.forward() — is the
    key correctness guarantee: every query sees B-1 negatives regardless of
    how many GPUs split the forward pass.
    """
    sim    = torch.matmul(nl_emb, code_emb.t()) * scale   # (B, B)
    labels = torch.arange(sim.size(0), device=sim.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2.0


# ── Dataset ────

class PairDataset(Dataset):
    def __init__(self, pairs, tokenizer, nl_len, code_len):
        self.pairs, self.tok = pairs, tokenizer
        self.nl_len, self.code_len = nl_len, code_len

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        nl, code = self.pairs[idx]
        n = self.tok(nl,   max_length=self.nl_len,   truncation=True,
                    padding="max_length", return_tensors="pt")
        c = self.tok(code, max_length=self.code_len, truncation=True,
                    padding="max_length", return_tensors="pt")
        return {"nl_ids":   n["input_ids"].squeeze(0),
                "nl_mask":  n["attention_mask"].squeeze(0),
                "code_ids": c["input_ids"].squeeze(0),
                "code_mask":c["attention_mask"].squeeze(0)}


# ── Pair builder ────

def build_pairs(train_jsonl: Path, func_cfg: dict, max_pairs_per_file: int,
                cache_conn=None):
    """
    Build (feature_text, java_function) training pairs from the positive
    records in train.jsonl.

    cache_conn : sqlite3.Connection from open_source_cache(), or None.
                 When provided, raw Java source is looked up by (sha_before,
                 class_path) so the AST extractor receives actual method bodies
                 instead of class-summary text.  A per-run hit/miss tally is
                 logged at the end to track fidelity.
    """
    by_issue = load_jsonl_by_issue(train_jsonl)
    log.info("Loaded %d issues from %s", len(by_issue), train_jsonl)
    pairs, seen = [], set()
    cache_hits = cache_misses = 0

    for jid, recs in tqdm(by_issue.items(),
                    desc="Building (feature, function) pairs",
                    unit="issue", dynamic_ncols=True):
        feat = next((r.get("feature_text", "") for r in recs
                    if r.get("feature_text")), "")
        if not feat:
            continue
        for r in recs:
            if int(r.get("label", 0)) != 1:
                continue
            key = (jid, r.get("class_path", ""))
            if key in seen:
                continue
            seen.add(key)
            raw_src = lookup_source(cache_conn,
                                    r.get("sha_before", ""),
                                    r.get("class_path", ""))
            if raw_src is not None:
                cache_hits += 1
            else:
                cache_misses += 1
            funcs = extract_functions(r.get("class_text", ""), func_cfg,
                                      raw_source=raw_src)
            for fn in funcs[:max_pairs_per_file]:
                pairs.append((feat, fn))

    total_anchors = cache_hits + cache_misses
    log.info(
        "Built %d fine-tuning pairs  |  unique (issue, file) anchors: %d",
        len(pairs), len(seen),
    )
    log.info(
        "Source cache — hits: %d / %d  (%.1f%%)  misses: %d  "
        "[hits use real Java; misses fall back to class_text summary]",
        cache_hits, total_anchors,
        100 * cache_hits / max(total_anchors, 1),
        cache_misses,
    )
    if cache_conn is None:
        log.warning(
            "No source cache provided — all %d positive anchors use class_text "
            "summaries.  Run build_source_cache.py locally and pass --source-cache.",
            len(seen),
        )
    return pairs


# ── Embedding helper (inference / val) ────

@torch.no_grad()
def embed_texts(embed_par, tok, texts, primary_device, max_len, bs=256):
    """
    Encodes a list of texts through embed_par (EmbedWrapper or DataParallel).
    All inputs are sent to primary_device; DataParallel scatters internally.
    """
    out = []
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], max_length=max_len, truncation=True,
                  padding="max_length", return_tensors="pt")
        out.append(embed_par(enc["input_ids"].to(primary_device),
                    enc["attention_mask"].to(primary_device)).cpu())
    return torch.cat(out) if out else torch.empty(0)


# ── Validation ────

@torch.no_grad()
def file_level_eval(embed_par, tok, val_issues, cfg, primary_device,
                    cache_conn=None):
    """
    FLIM semantic score (max-pool over functions, max-pool over query chunks).

    cache_conn : sqlite3.Connection or None.  When provided, raw Java source
                 is used for function extraction; otherwise falls back to
                 class_text.  Same lookup pattern as build_pairs.
    """
    mcfg, fcfg = cfg["model"], cfg["functions"]
    ft         = cfg["finetune"]
    rng        = random.Random(ft.get("seed", 123456))

    items = list(val_issues.items())
    rng.shuffle(items)
    eval_items = items[: ft.get("val_max_issues", 50)]
    log.info("Running file-level eval on %d sampled val issues", len(eval_items))

    mrrs, r10s = [], []
    for jid, recs in tqdm(eval_items, desc="  Val eval",
                    unit="issue", leave=False, dynamic_ncols=True):
        pos = [r for r in recs if int(r.get("label", 0)) == 1]
        neg = [r for r in recs if int(r.get("label", 0)) == 0]
        if not pos:
            continue
        k_neg = max(0, ft.get("val_max_candidates", 100) - len(pos))
        cands = pos + rng.sample(neg, min(k_neg, len(neg)))

        r0     = recs[0]
        chunks = ([c.get("text", "") for c in r0.get("feature_chunks", [])]
                  or [r0.get("feature_text", "")])
        q = embed_texts(embed_par, tok,
                    chunks[: cfg["features"]["max_chunks"]],
                    primary_device, mcfg["nl_length"])

        scores, labels = [], []
        for r in cands:
            raw_src = lookup_source(cache_conn,
                                    r.get("sha_before", ""),
                                    r.get("class_path", ""))
            funcs = extract_functions(r.get("class_text", ""), fcfg,
                                      raw_source=raw_src)
            fe    = embed_texts(embed_par, tok, funcs,
                    primary_device, mcfg["code_length"])
            sims  = q @ fe.t()
            scores.append(sims.max().item())
            labels.append(int(r.get("label", 0)))
        mrrs.append(mean_reciprocal_rank(scores, labels))
        r10s.append(recall_at_k(scores, labels, 10))

    n = max(len(mrrs), 1)
    return sum(mrrs) / n, sum(r10s) / n


# ── Main ────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--source-cache", default=None,
        help=(
            "Path to source_cache.sqlite built by build_source_cache.py. "
            "When provided, raw Java source is used for function extraction "
            "instead of class_text summaries, making fine-tuning faithful to "
            "the original FLIM. Can also be set via config['source_cache']."
        ),
    )
    args = ap.parse_args()
    cfg  = json.load(open(args.config, encoding="utf-8"))
    mcfg, ft, fcfg = cfg["model"], cfg["finetune"], cfg["functions"]
    multi_cfg       = cfg.get("multi_gpu", {})

    # Source cache: CLI flag takes precedence over config key
    cache_path  = args.source_cache or cfg.get("source_cache")
    cache_conn  = open_source_cache(cache_path)
    if cache_conn:
        log.info("Source cache loaded: %s", cache_path)
    else:
        log.warning(
            "Source cache NOT loaded (path=%s). Fine-tuning will use class_text "
            "summaries for function extraction — results will be degraded. "
            "Run build_source_cache.py locally and pass --source-cache.",
            cache_path,
        )

    torch.manual_seed(ft.get("seed", 123456))
    random.seed(ft.get("seed", 123456))

    primary = get_primary_device(multi_cfg)
    log.info("Primary device : %s", primary)
    log.info("Model          : %s", mcfg["model_name"])
    if torch.cuda.is_available():
        log.info("CUDA devices available: %d", torch.cuda.device_count())

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(out_dir / "config_flim.json", "w"), indent=2)
    log.info("Output dir: %s", out_dir)

    tok   = AutoTokenizer.from_pretrained(mcfg["model_name"])

    # ── Model + optimizer BEFORE wrapping ──
    # Optimizer must be created on the raw model.  DataParallel does NOT
    # own parameters — it holds a reference to model via wrapper.base.
    # Gradients accumulate on the primary-device replica (model itself).
    model = build_flim_encoder(mcfg).to(primary)
    log.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    opt = torch.optim.AdamW(model.parameters(), lr=ft["lr"],
                    weight_decay=ft.get("weight_decay", 0.01))

    # ── Wrap for multi-GPU embedding (after optimizer) ──
    embed_par = setup_embed_parallel(model, multi_cfg, primary)

    log.info("Building fine-tuning pairs from %s", cfg["train_jsonl"])
    pairs  = build_pairs(Path(cfg["train_jsonl"]), fcfg,
                    ft.get("max_pairs_per_file", 30))
    ds     = PairDataset(pairs, tok, mcfg["nl_length"], mcfg["code_length"])
    loader = DataLoader(ds, batch_size=ft["batch_size"], shuffle=True,
                    num_workers=ft.get("num_workers", 4),
                    pin_memory=primary.type == "cuda", drop_last=True)
    log.info("DataLoader: %d batches/epoch  (batch_size=%d)",
             len(loader), ft["batch_size"])

    total = len(loader) * ft["epochs"]
    sched = get_linear_schedule_with_warmup(
        opt, int(total * ft.get("warmup_ratio", 0.1)), total)

    scale     = mcfg.get("scale", 20.0)
    val_issues = load_jsonl_by_issue(Path(cfg["val_jsonl"]))
    log.info("Validation issues: %d  (from %s)", len(val_issues), cfg["val_jsonl"])
    best_mrr = -1.0

    for epoch in range(1, ft["epochs"] + 1):
        # ── Training ──
        model.train()
        tot, n_batches = 0.0, 0
        pbar = tqdm(loader,
                    desc=f"Epoch {epoch:02d}/{ft['epochs']:02d} [train]",
                    unit="batch", dynamic_ncols=True)
        for batch in pbar:
            nl_ids   = batch["nl_ids"].to(primary)
            nl_mask  = batch["nl_mask"].to(primary)
            code_ids = batch["code_ids"].to(primary)
            code_mask= batch["code_mask"].to(primary)

            # embed_par distributes across GPUs; outputs gathered on primary
            nl_emb   = embed_par(nl_ids, nl_mask)       # (B, H)
            code_emb = embed_par(code_ids, code_mask)   # (B, H)

            # Loss computed centrally with FULL batch → all B in-batch negatives
            loss = infonce_loss(nl_emb, code_emb, scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                    ft.get("max_grad_norm", 1.0))
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item(); n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = tot / max(n_batches, 1)
        log.info("Epoch %02d/%02d | avg train loss %.4f", epoch, ft["epochs"], avg_loss)

        # ── Validation ──
        model.eval()
        mrr, r10 = file_level_eval(embed_par, tok, val_issues, cfg, primary)
        log.info("Epoch %02d/%02d | val MRR %.4f | val R@10 %.4f",
                 epoch, ft["epochs"], mrr, r10)

        if mrr > best_mrr:
            best_mrr = mrr
            # Always save from the raw model, not the wrapper
            torch.save({"model_state": model.state_dict(),
                    "epoch": epoch, "val_mrr": mrr},
                    out_dir / "best_flim_encoder.pt")
            log.info("  ↑ new best MRR %.4f at epoch %d — checkpoint saved",
                    mrr, epoch)

    log.info("Fine-tuning complete.  Best val MRR = %.4f", best_mrr)
    log.info("Best checkpoint: %s", out_dir / "best_flim_encoder.pt")


if __name__ == "__main__":
    main()
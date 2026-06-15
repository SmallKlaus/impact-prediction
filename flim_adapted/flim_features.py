"""
flim_features.py — Stage 2: compute FLIM features for a split
====
For every (issue, candidate class) pair in the input JSONL, computes:

Semantic (function-level interaction, FLIM core — rescaled to [0,1]):
    f_sem_max     max over functions of max over query chunks
    f_sem_mean    mean over functions
    f_sem_top3    mean of top-3 functions
    f_sem_header  similarity with the file header pseudo-function

Lexical (adapted subset of Ye et al. / Fejzer et al. IR features):
    f_lex_tfidf     TF-IDF cosine (report ↔ full file), fit per sha snapshot
    f_lex_id        TF-IDF cosine (report ↔ identifiers only)
    f_cls_name      1.0 if class simple name occurs in the report
    f_path_overlap  Jaccard overlap of path tokens vs report tokens

Multi-GPU:
    Controlled via config_flim.json "multi_gpu" key.
    embed_batch_size is split across GPUs by DataParallel.
    Tip: with N GPUs you can raise embed_batch_size to N×256 without
    increasing per-GPU memory to keep GPU utilisation high.

Run once per split:
    python flim_features.py \
        --input  .../train_1000.jsonl \
        --checkpoint .../flim_run_001/best_flim_encoder.pt \
        --config     config_flim.json \
        --output     .../FLIM/flim_train.features.jsonl
    (repeat for val_500.jsonl and test_full.jsonl)
"""
from __future__ import annotations
import argparse, json, logging, os, ssl, urllib3
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flim_model  import build_flim_encoder
from flim_common import (extract_functions, load_jsonl_by_issue,
                         normalize_tokens, tokens_as_text)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_ID_RE = __import__("re").compile(r"[A-Za-z_$][A-Za-z0-9_$]*")


def identifiers_text(code: str) -> str:
    return tokens_as_text(" ".join(_ID_RE.findall(code or "")))


# ── Multi-GPU helpers ────

class EmbedWrapper(nn.Module):
    """
    Exposes model.embed(input_ids, attention_mask) as forward().
    Required for nn.DataParallel compatibility (DataParallel only
    distributes the forward() call, not arbitrary methods).
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        return self.base.embed(input_ids, attention_mask)


def get_primary_device(multi_cfg: dict) -> torch.device:
    """
    Returns the primary (output-gathering) device based on multi_gpu config.
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
    Returns a callable: embed_par(input_ids, attention_mask) -> embeddings.
    Wraps with DataParallel when multi_gpu.enabled and enough devices exist.
    Falls back to a plain EmbedWrapper (single-device) otherwise.

    NOTE: model must already be on primary_dev before calling this.
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


# ── Embedding helper ────

@torch.no_grad()
def embed_texts(embed_par, tok, texts, primary_device, max_len, bs=256):
    """
    Batch-encodes a list of texts through embed_par.
    Inputs are sent to primary_device; DataParallel scatters to other devices.
    Outputs are always returned on CPU to keep VRAM pressure low between
    per-snapshot encode-and-score cycles.

    With N GPUs active, each micro-batch of size bs is split into N shards
    of bs/N internally by DataParallel — effective per-GPU batch = bs/N.
    """
    out = []
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], max_length=max_len, truncation=True,
                  padding="max_length", return_tensors="pt")
        out.append(embed_par(enc["input_ids"].to(primary_device),
                             enc["attention_mask"].to(primary_device)).cpu())
    return torch.cat(out) if out else torch.empty(0)


# ── Main ────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      required=True,
                    help="Input JSONL split (train_1000 / val_500 / test_full)")
    ap.add_argument("--checkpoint", required=True,
                    help="best_flim_encoder.pt from flim_finetune.py")
    ap.add_argument("--config",     required=True,
                    help="config_flim.json")
    ap.add_argument("--output",     required=True,
                    help="Output feature JSONL path")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    mcfg, fcfg, featcfg = cfg["model"], cfg["functions"], cfg["features"]
    multi_cfg = cfg.get("multi_gpu", {})

    primary = get_primary_device(multi_cfg)
    log.info("Primary device : %s", primary)
    if torch.cuda.is_available():
        log.info("CUDA devices available: %d", torch.cuda.device_count())

    tok   = AutoTokenizer.from_pretrained(mcfg["model_name"])

    # Load base model → move to primary → wrap for multi-GPU
    model = build_flim_encoder(mcfg).to(primary)
    state = torch.load(args.checkpoint, map_location=primary)
    model.load_state_dict(state.get("model_state", state))
    model.eval()
    log.info("FLIM encoder loaded from: %s", args.checkpoint)

    embed_par = setup_embed_parallel(model, multi_cfg, primary)

    log.info("Loading issues from: %s", args.input)
    by_issue = load_jsonl_by_issue(Path(args.input))
    log.info("Issues loaded: %d", len(by_issue))

    # ── Group candidates by sha_before snapshot ──
    sha_classes:  dict[str, dict[str, str]] = {}
    issue_to_sha: dict[str, str] = {}
    for jid, recs in by_issue.items():
        sha = recs[0].get("sha_before", "")
        issue_to_sha[jid] = sha
        d = sha_classes.setdefault(sha, {})
        for r in recs:
            cp = r.get("class_path", "")
            if cp and cp not in d:
                d[cp] = r.get("class_text", "")
    log.info("Unique sha_before snapshots : %d", len(sha_classes))
    log.info("Total candidate classes     : %d",
             sum(len(v) for v in sha_classes.values()))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    snapshot_bar = tqdm(sha_classes.items(),
                        desc="Snapshots", unit="snap",
                        total=len(sha_classes), dynamic_ncols=True)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for sha, classes in snapshot_bar:
            paths = list(classes.keys())
            snapshot_bar.set_postfix(sha=sha[:8], n_classes=len(paths))
            log.info("Snapshot %s — %d candidate classes", sha[:10], len(paths))

            # ── Function extraction ──
            func_texts, func_owner, n_funcs_per_class = [], [], []
            for cp in tqdm(paths,
                           desc=f"  [{sha[:6]}] Extracting functions",
                           unit="class", leave=False, dynamic_ncols=True):
                funcs = extract_functions(classes[cp], fcfg)
                n_funcs_per_class.append(len(funcs))
                for fn in funcs:
                    func_texts.append(fn)
                    func_owner.append(len(n_funcs_per_class) - 1)
            log.info("  %d function segments from %d classes",
                     len(func_texts), len(paths))

            # ── Encode all function segments shared across issues this snapshot ──
            # DataParallel splits embed_batch_size across GPUs here:
            #   effective per-GPU load = embed_batch_size / n_active_gpus
            ebs = featcfg.get("embed_batch_size", 256)
            log.info("  Encoding %d function segments  (embed_batch_size=%d) ...",
                     len(func_texts), ebs)
            f_emb = embed_texts(embed_par, tok, func_texts, primary,
                                mcfg["code_length"], bs=ebs)
            owner = np.asarray(func_owner)
            log.info("  Function embeddings: %s", tuple(f_emb.shape))

            # ── Per-snapshot TF-IDF (fit on snapshot files only — no leakage) ──
            vec_full = TfidfVectorizer(analyzer=normalize_tokens,
                                       sublinear_tf=True, min_df=1)
            M_full   = vec_full.fit_transform([classes[cp] for cp in paths])
            vec_id   = TfidfVectorizer(sublinear_tf=True, min_df=1)
            M_id     = vec_id.fit_transform(
                [identifiers_text(classes[cp]) or "x" for cp in paths])

            cls_simple = [Path(cp).stem.lower() for cp in paths]
            path_toks  = [set(normalize_tokens(cp)) for cp in paths]

            # ── Per-issue feature computation ──
            issues_here = [j for j, s in issue_to_sha.items() if s == sha]
            for jid in tqdm(issues_here,
                            desc=f"  [{sha[:6]}] Issues",
                            unit="issue", leave=False, dynamic_ncols=True):
                recs   = by_issue[jid]
                r0     = recs[0]
                chunks = ([c.get("text", "") for c in r0.get("feature_chunks", [])]
                          or [r0.get("feature_text", "")])
                chunks = chunks[: featcfg.get("max_chunks", 4)]

                # Query embeddings: small number of chunks, lightweight
                q_emb = embed_texts(embed_par, tok, chunks, primary,
                                    mcfg["nl_length"])

                # Function similarities rescaled to [0, 1]
                sims = ((q_emb @ f_emb.t()).max(dim=0).values.numpy() + 1.0) / 2.0

                feat_full = r0.get("feature_text", "") or " ".join(chunks)
                q_full    = vec_full.transform([feat_full])
                q_id      = vec_id.transform([tokens_as_text(feat_full)])
                lex_full  = cosine_similarity(q_full, M_full).ravel()
                lex_id    = cosine_similarity(q_id,   M_id).ravel()
                feat_low  = feat_full.lower()
                feat_toks = set(normalize_tokens(feat_full))

                label_of    = {r.get("class_path", ""): int(r.get("label", 0))
                               for r in recs}
                n_pos_total = sum(label_of.values())
                if n_pos_total == 0:
                    continue

                cand_idx = [i for i, cp in enumerate(paths) if cp in label_of]
                for ci in cand_idx:
                    s  = np.sort(sims[owner == ci])[::-1]
                    cp = paths[ci]
                    top3       = float(s[: featcfg.get("top_k_mean", 3)].mean())
                    header_sim = float(sims[np.flatnonzero(owner == ci)[0]])
                    feats = {
                        "f_sem_max":      float(s[0]),
                        "f_sem_mean":     float(s.mean()),
                        "f_sem_top3":     top3,
                        "f_sem_header":   header_sim,
                        "f_lex_tfidf":    float(lex_full[ci]),
                        "f_lex_id":       float(lex_id[ci]),
                        "f_cls_name":     1.0 if cls_simple[ci] in feat_low else 0.0,
                        "f_path_overlap": (
                            len(path_toks[ci] & feat_toks) /
                            len(path_toks[ci] | feat_toks)
                            if (path_toks[ci] | feat_toks) else 0.0),
                    }
                    out_f.write(json.dumps({
                        "jira_id":     jid,
                        "class_path":  cp,
                        "label":       label_of[cp],
                        "n_pos_total": n_pos_total,
                        "n_functions": int(n_funcs_per_class[ci]),
                        "features":    feats,
                    }, ensure_ascii=False) + "\n")
                    n_written += 1

            log.info("  Snapshot %s done — running total written: %d",
                     sha[:10], n_written)

    log.info("Feature extraction complete.")
    log.info("Total records written: %d  →  %s", n_written, out_path)


if __name__ == "__main__":
    main()
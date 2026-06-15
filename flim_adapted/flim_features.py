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

Run once per split:
    python flim_features.py --input  .../train_1000.jsonl \
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


@torch.no_grad()
def embed_texts(model, tok, texts, device, max_len, bs=256):
    out = []
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], max_length=max_len, truncation=True,
                  padding="max_length", return_tensors="pt")
        out.append(model.embed(enc["input_ids"].to(device),
                               enc["attention_mask"].to(device)).cpu())
    return torch.cat(out) if out else torch.empty(0)


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

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    log.info("Device: %s", device)

    tok   = AutoTokenizer.from_pretrained(mcfg["model_name"])
    model = build_flim_encoder(mcfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state", state))
    model.eval()
    log.info("FLIM encoder loaded from: %s", args.checkpoint)

    log.info("Loading issues from: %s", args.input)
    by_issue = load_jsonl_by_issue(Path(args.input))
    log.info("Issues loaded: %d", len(by_issue))

    # ── Group candidate classes by sha_before snapshot ──
    sha_classes:  dict[str, dict[str, str]] = {}   # sha -> {class_path: class_text}
    issue_to_sha: dict[str, str] = {}
    for jid, recs in by_issue.items():
        sha = recs[0].get("sha_before", "")
        issue_to_sha[jid] = sha
        d = sha_classes.setdefault(sha, {})
        for r in recs:
            cp = r.get("class_path", "")
            if cp and cp not in d:
                d[cp] = r.get("class_text", "")
    log.info("Unique sha_before snapshots: %d  |  total candidate classes: %d",
             len(sha_classes),
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
            for ci, cp in enumerate(tqdm(paths,
                                         desc=f"  [{sha[:6]}] Extracting functions",
                                         unit="class", leave=False,
                                         dynamic_ncols=True)):
                funcs = extract_functions(classes[cp], fcfg)
                n_funcs_per_class.append(len(funcs))
                for fn in funcs:
                    func_texts.append(fn)
                    func_owner.append(ci)
            log.info("  %d total function segments extracted from %d classes",
                     len(func_texts), len(paths))

            # ── Encode all function segments (shared across issues this snapshot) ──
            log.info("  Encoding %d function segments ...", len(func_texts))
            f_emb = embed_texts(model, tok, func_texts, device,
                                mcfg["code_length"],
                                featcfg.get("embed_batch_size", 256))
            owner = np.asarray(func_owner)
            log.info("  Function embeddings shape: %s", tuple(f_emb.shape))

            # ── Per-snapshot TF-IDF matrices ──
            vec_full = TfidfVectorizer(analyzer=normalize_tokens,
                                       sublinear_tf=True, min_df=1)
            M_full   = vec_full.fit_transform([classes[cp] for cp in paths])
            vec_id   = TfidfVectorizer(sublinear_tf=True, min_df=1)
            M_id     = vec_id.fit_transform(
                [identifiers_text(classes[cp]) or "x" for cp in paths])

            cls_simple = [Path(cp).stem.lower() for cp in paths]
            path_toks  = [set(normalize_tokens(cp)) for cp in paths]

            # ── Process each issue whose snapshot is this sha ──
            issues_here = [j for j, s in issue_to_sha.items() if s == sha]
            for jid in tqdm(issues_here,
                             desc=f"  [{sha[:6]}] Issues",
                             unit="issue", leave=False, dynamic_ncols=True):
                recs   = by_issue[jid]
                r0     = recs[0]
                chunks = ([c.get("text", "") for c in r0.get("feature_chunks", [])]
                          or [r0.get("feature_text", "")])
                chunks = chunks[: featcfg.get("max_chunks", 4)]

                q_emb = embed_texts(model, tok, chunks, device, mcfg["nl_length"])

                # (n_funcs,) — per-function max cosine over query chunks, mapped to [0,1]
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
                    top3 = float(s[: featcfg.get("top_k_mean", 3)].mean())
                    # f_sem_header = similarity of the first segment (file header chunk
                    # when include_file_header=True, else top function)
                    header_sim = float(sims[np.flatnonzero(owner == ci)[0]])
                    feats = {
                        "f_sem_max":     float(s[0]),
                        "f_sem_mean":    float(s.mean()),
                        "f_sem_top3":    top3,
                        "f_sem_header":  header_sim,
                        "f_lex_tfidf":   float(lex_full[ci]),
                        "f_lex_id":      float(lex_id[ci]),
                        "f_cls_name":    1.0 if cls_simple[ci] in feat_low else 0.0,
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
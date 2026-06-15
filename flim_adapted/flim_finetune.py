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

Usage:
    python flim_finetune.py --config config_flim.json
"""
from __future__ import annotations
import argparse, json, logging, os, random, ssl, urllib3
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from flim_model  import build_flim_encoder
from flim_common import (extract_functions, load_jsonl_by_issue,
                         mean_reciprocal_rank, recall_at_k)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


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
                "code_mask": c["attention_mask"].squeeze(0)}


def build_pairs(train_jsonl: Path, func_cfg: dict, max_pairs_per_file: int):
    by_issue = load_jsonl_by_issue(train_jsonl)
    pairs, seen = [], set()
    for jid, recs in by_issue.items():
        feat = next((r.get("feature_text", "") for r in recs
                     if r.get("feature_text")), "")
        if not feat: continue
        for r in recs:
            if int(r.get("label", 0)) != 1: continue
            key = (jid, r.get("class_path", ""))
            if key in seen: continue
            seen.add(key)
            funcs = extract_functions(r.get("class_text", ""), func_cfg)
            for fn in funcs[:max_pairs_per_file]:
                pairs.append((feat, fn))
    return pairs


@torch.no_grad()
def embed_texts(model, tok, texts, device, max_len, bs=256):
    out = []
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], max_length=max_len, truncation=True,
                  padding="max_length", return_tensors="pt")
        out.append(model.embed(enc["input_ids"].to(device),
                               enc["attention_mask"].to(device)).cpu())
    return torch.cat(out) if out else torch.empty(0)


@torch.no_grad()
def file_level_eval(model, tok, val_issues, cfg, device):
    """FLIM semantic score (max over functions of max over query chunks)."""
    mcfg, fcfg = cfg["model"], cfg["functions"]
    ft         = cfg["finetune"]
    rng        = random.Random(ft.get("seed", 123456))
    mrrs, r10s = [], []

    items = list(val_issues.items())
    rng.shuffle(items)
    for jid, recs in items[: ft.get("val_max_issues", 50)]:
        pos = [r for r in recs if int(r.get("label", 0)) == 1]
        neg = [r for r in recs if int(r.get("label", 0)) == 0]
        if not pos: continue
        k_neg = max(0, ft.get("val_max_candidates", 100) - len(pos))
        cands = pos + rng.sample(neg, min(k_neg, len(neg)))

        r0     = recs[0]
        chunks = [c.get("text", "") for c in r0.get("feature_chunks", [])] \
                 or [r0.get("feature_text", "")]
        q = embed_texts(model, tok, chunks[: cfg["features"]["max_chunks"]],
                        device, mcfg["nl_length"])

        scores, labels = [], []
        for r in cands:
            funcs = extract_functions(r.get("class_text", ""), fcfg)
            fe    = embed_texts(model, tok, funcs, device, mcfg["code_length"])
            sims  = (q @ fe.t())                      # (n_chunks, n_funcs)
            scores.append(sims.max().item())
            labels.append(int(r.get("label", 0)))
        mrrs.append(mean_reciprocal_rank(scores, labels))
        r10s.append(recall_at_k(scores, labels, 10))
    n = max(len(mrrs), 1)
    return sum(mrrs) / n, sum(r10s) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg  = json.load(open(args.config, encoding="utf-8"))
    mcfg, ft, fcfg = cfg["model"], cfg["finetune"], cfg["functions"]

    torch.manual_seed(ft.get("seed", 123456))
    random.seed(ft.get("seed", 123456))
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(out_dir / "config_flim.json", "w"), indent=2)

    tok   = AutoTokenizer.from_pretrained(mcfg["model_name"])
    model = build_flim_encoder(mcfg).to(device)

    log.info("Building fine-tuning pairs from %s", cfg["train_jsonl"])
    pairs = build_pairs(Path(cfg["train_jsonl"]), fcfg,
                        ft.get("max_pairs_per_file", 30))
    log.info("Fine-tuning pairs: %d", len(pairs))

    ds     = PairDataset(pairs, tok, mcfg["nl_length"], mcfg["code_length"])
    loader = DataLoader(ds, batch_size=ft["batch_size"], shuffle=True,
                        num_workers=ft.get("num_workers", 4),
                        pin_memory=True, drop_last=True)

    opt   = torch.optim.AdamW(model.parameters(), lr=ft["lr"],
                              weight_decay=ft.get("weight_decay", 0.01))
    total = len(loader) * ft["epochs"]
    sched = get_linear_schedule_with_warmup(
        opt, int(total * ft.get("warmup_ratio", 0.1)), total)

    val_issues = load_jsonl_by_issue(Path(cfg["val_jsonl"]))
    best_mrr   = -1.0

    for epoch in range(1, ft["epochs"] + 1):
        model.train(); tot = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            loss = model(batch["nl_ids"].to(device),  batch["nl_mask"].to(device),
                         batch["code_ids"].to(device), batch["code_mask"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           ft.get("max_grad_norm", 1.0))
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item()
        log.info("Epoch %d | train loss %.4f", epoch, tot / max(len(loader), 1))

        model.eval()
        mrr, r10 = file_level_eval(model, tok, val_issues, cfg, device)
        log.info("Epoch %d | val MRR %.4f | val R@10 %.4f", epoch, mrr, r10)
        if mrr > best_mrr:
            best_mrr = mrr
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch, "val_mrr": mrr},
                       out_dir / "best_flim_encoder.pt")
            log.info("  ↑ new best — checkpoint saved")

    log.info("Done. Best val MRR = %.4f", best_mrr)


if __name__ == "__main__":
    main()
"""
diagnose_flim.py — Per-Issue FLIM Baseline Diagnostic
====
Same report structure as diagnose_ce.py:
 1. GLOBAL METRICS       — FLIM-LTR vs FLIM-semantic-only (f_sem_max) + deltas
 2. MRR DISTRIBUTION
 3. PER-PROJECT SUMMARY
 4. WORST-N ISSUES
 5. per_issue (JSON)

Because FLIM ranks the FULL candidate pool (no retrieval stage),
flim_recall@k here is directly comparable to your pipeline_recall@k.

Usage:
    python diagnose_flim.py \
        --test-features .../flim_test.features.jsonl \
        --ltr-model     .../flim_run_001/flim_ltr.pkl \
        --output        .../flim_run_001/diagnosis_flim.json \
        [--worst-n 20]
"""
from __future__ import annotations
import argparse, json, math, pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from flim_common import (FEATURE_COLUMNS, aggregate, average_precision,
                         mean_reciprocal_rank, ndcg_at_k, project_of,
                         recall_at_k)

KS = [1, 3, 5, 10, 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-features", required=True)
    ap.add_argument("--ltr-model",     required=True)
    ap.add_argument("--output",        required=True)
    ap.add_argument("--worst-n", type=int, default=20)
    args = ap.parse_args()

    with open(args.ltr_model, "rb") as f:
        ltr = pickle.load(f)
    cols = ltr["feature_columns"]

    by_jid = defaultdict(lambda: {"X": [], "label": [], "sem": [],
                                  "n_pos_total": 0})
    with open(args.test_features, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r  = json.loads(line)
            e  = by_jid[r["jira_id"]]
            ft = r["features"]
            e["X"].append([max(float(ft.get(c, 0.0)), 0.0) for c in cols])
            e["label"].append(int(r["label"]))
            e["sem"].append(float(ft.get("f_sem_max", 0.0)))
            e["n_pos_total"] = r.get("n_pos_total", 0)

    per_issue = {}
    for jid, e in by_jid.items():
        lb = e["label"]
        if not any(lb): continue
        X = np.asarray(e["X"], dtype=float)
        if ltr["type"] == "regressor":
            sc = ltr["regressor"].predict(X).tolist()
        else:
            sc = (X @ np.asarray(ltr["weights"])).tolist()
        sem = e["sem"]

        ent = {"project": project_of(jid), "n_candidates": len(lb),
               "n_pos_total": e["n_pos_total"] or sum(lb)}
        for k in KS:
            ent[f"flim_recall@{k}"] = round(recall_at_k(sc, lb, k), 4)
            ent[f"sem_recall@{k}"]  = round(recall_at_k(sem, lb, k), 4)
            ent[f"delta_recall@{k}"] = round(
                ent[f"flim_recall@{k}"] - ent[f"sem_recall@{k}"], 4)
        for k in [1, 3, 5, 10]:
            ent[f"flim_ndcg@{k}"] = round(ndcg_at_k(sc, lb, k), 4)
        ent["flim_map"] = round(average_precision(sc, lb), 4)
        ent["flim_mrr"] = round(mean_reciprocal_rank(sc, lb), 4)
        ent["sem_map"]  = round(average_precision(sem, lb), 4)
        ent["sem_mrr"]  = round(mean_reciprocal_rank(sem, lb), 4)
        ent["delta_mrr"] = round(ent["flim_mrr"] - ent["sem_mrr"], 4)
        ent["delta_map"] = round(ent["flim_map"] - ent["sem_map"], 4)
        per_issue[jid] = ent

    N = len(per_issue)
    all_ents = list(per_issue.values())
    def gmean(key): return round(sum(e[key] for e in all_ents) / N, 4) if N else 0.0

    global_metrics = {"n_issues": N,
        **{f"flim_recall@{k}": gmean(f"flim_recall@{k}") for k in KS},
        **{f"sem_recall@{k}":  gmean(f"sem_recall@{k}")  for k in KS},
        **{f"delta_recall@{k}": gmean(f"delta_recall@{k}") for k in KS},
        "flim_ndcg@10": gmean("flim_ndcg@10"),
        "flim_map": gmean("flim_map"), "flim_mrr": gmean("flim_mrr"),
        "sem_map":  gmean("sem_map"),  "sem_mrr":  gmean("sem_mrr"),
        "delta_mrr": gmean("delta_mrr"), "delta_map": gmean("delta_map")}

    mrr_vals = [e["flim_mrr"] for e in all_ents]
    mu = sum(mrr_vals) / N if N else 0.0
    mrr_dist = {
        "mrr_eq_1":    sum(1 for v in mrr_vals if v == 1.0),
        "mrr_gte_0.5": sum(1 for v in mrr_vals if 0.5 <= v < 1.0),
        "mrr_lt_0.5":  sum(1 for v in mrr_vals if 0.0 < v < 0.5),
        "mrr_eq_0":    sum(1 for v in mrr_vals if v == 0.0),
        "mrr_mean":    round(mu, 4),
        "mrr_std":     round(math.sqrt(sum((v - mu) ** 2
                          for v in mrr_vals) / N), 4) if N else 0.0}

    proj_issues = defaultdict(list)
    for jid, e in per_issue.items():
        proj_issues[e["project"]].append(e)
    per_project = {}
    for proj, ents in sorted(proj_issues.items()):
        per_project[proj] = {"n_issues": len(ents),
            **{f"flim_recall@{k}": aggregate([e[f"flim_recall@{k}"]
                                              for e in ents]) for k in KS},
            "flim_ndcg@10": aggregate([e["flim_ndcg@10"] for e in ents]),
            "flim_map":  aggregate([e["flim_map"]  for e in ents]),
            "flim_mrr":  aggregate([e["flim_mrr"]  for e in ents]),
            "sem_mrr":   aggregate([e["sem_mrr"]   for e in ents]),
            "delta_mrr": aggregate([e["delta_mrr"] for e in ents])}

    worst = sorted(per_issue.items(), key=lambda x: x[1]["flim_mrr"])[:args.worst_n]
    worst_issues = [{"jira_id": jid, "project": e["project"],
                     "flim_mrr": e["flim_mrr"], "sem_mrr": e["sem_mrr"],
                     "delta_mrr": e["delta_mrr"],
                     "flim_recall@5": e["flim_recall@5"],
                     "n_candidates": e["n_candidates"],
                     "n_pos_total": e["n_pos_total"]} for jid, e in worst]

    report = {"ltr_model": str(args.ltr_model),
              "test_features": str(args.test_features),
              "ltr_type": ltr["type"], "global": global_metrics,
              "mrr_distribution": mrr_dist, "per_project": per_project,
              "worst_issues": worst_issues, "per_issue": per_issue}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # ── Printed summary (same layout as diagnose_ce.py) ────
    W = 72
    print("\n" + "=" * W)
    print("GLOBAL METRICS  (FLIM ranks the FULL candidate pool)")
    print("=" * W)
    print(f"  Issues evaluated      : {N}")
    print(f"  LTR model chosen      : {ltr['type']} "
          f"({ltr.get('regressor_name') if ltr['type']=='regressor' else ltr.get('weights_name')})")
    print()
    print(f"  {'Metric':<18}  {'FLIM-LTR':>9}  {'Sem-only':>9}  {'Delta':>9}")
    print("  " + "-" * 50)
    for k in KS:
        d = global_metrics[f"delta_recall@{k}"]
        arrow = "▲" if d > 0 else ("▼" if d < 0 else " ")
        print(f"  {'Recall@'+str(k):<18}  "
              f"{global_metrics[f'flim_recall@{k}']:>9.4f}  "
              f"{global_metrics[f'sem_recall@{k}']:>9.4f}  {arrow}{abs(d):>8.4f}")
    print(f"  {'NDCG@10':<18}  {global_metrics['flim_ndcg@10']:>9.4f}")
    print(f"  {'MAP':<18}  {global_metrics['flim_map']:>9.4f}  "
          f"{global_metrics['sem_map']:>9.4f}  {global_metrics['delta_map']:>+9.4f}")
    print(f"  {'MRR':<18}  {global_metrics['flim_mrr']:>9.4f}  "
          f"{global_metrics['sem_mrr']:>9.4f}  {global_metrics['delta_mrr']:>+9.4f}")

    print("\n" + "=" * W)
    print("MRR DISTRIBUTION  (FLIM-LTR)")
    print("=" * W)
    for key, lab in [("mrr_eq_1", "MRR = 1.0  (1st hit at rank 1)"),
                     ("mrr_gte_0.5", "MRR in [0.5, 1.0)            "),
                     ("mrr_lt_0.5", "MRR in (0, 0.5)              "),
                     ("mrr_eq_0", "MRR = 0.0  (no hit)          ")]:
        v = mrr_dist[key]
        print(f"  {lab} : {v:4d}  ({100*v/max(N,1):.1f}%)")
    print(f"  Mean / Std                    : "
          f"{mrr_dist['mrr_mean']:.4f} / {mrr_dist['mrr_std']:.4f}")

    print("\n" + "=" * W)
    print("PER-PROJECT SUMMARY")
    print("=" * W)
    print(f"  {'Project':<16} {'N':>4}  {'MRR':>7}  {'SemMRR':>7}  "
          f"{'ΔMRR':>7}  {'R@5':>7}  {'R@10':>7}  {'MAP':>7}")
    print("  " + "-" * 66)
    for proj, s in sorted(per_project.items()):
        print(f"  {proj:<16} {s['n_issues']:>4}  "
              f"{s['flim_mrr']['mean']:>7.4f}  {s['sem_mrr']['mean']:>7.4f}  "
              f"{s['delta_mrr']['mean']:>+7.4f}  "
              f"{s['flim_recall@5']['mean']:>7.4f}  "
              f"{s['flim_recall@10']['mean']:>7.4f}  "
              f"{s['flim_map']['mean']:>7.4f}")

    print("\n" + "=" * W)
    print(f"WORST {args.worst_n} ISSUES BY FLIM MRR")
    print("=" * W)
    print(f"  {'Jira ID':<22} {'Proj':<10} {'MRR':>7}  {'SemMRR':>7}  "
          f"{'R@5':>7}  {'Cands':>6}  {'Pos':>3}")
    print("  " + "-" * 68)
    for w in worst_issues:
        print(f"  {w['jira_id']:<22} {w['project']:<10} "
              f"{w['flim_mrr']:>7.4f}  {w['sem_mrr']:>7.4f}  "
              f"{w['flim_recall@5']:>7.4f}  {w['n_candidates']:>6}  "
              f"{w['n_pos_total']:>3}")
    print(f"\nFull report saved to: {out}\n")


if __name__ == "__main__":
    main()
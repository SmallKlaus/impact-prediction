"""
flim_ltr.py — Stage 3: Adaptive LTR (re-implementation of Fejzer et al.'s
Adaptive_Process used by FLIM's train_adaptive.py)
====
Two-phase adaptive selection, model chosen by validation MAP:
  Phase 1 (prescoring): try several feature-weighting methods (chi2,
    mutual information, ExtraTrees / GradientBoosting / AdaBoost importances,
    variance, constant); score = features · weights.
  Phase 2 (regression): for the best weights, try SGDRegressor variants
    (4 losses × 4 penalties) on a 'cut' training subset (positives + the
    lowest-scored slice), target = score + label·max(score).
Final model = whichever of {best prescoring weights, best regressor}
has the higher validation MAP.

Usage:
    python flim_ltr.py \
        --train-features .../flim_train.features.jsonl \
        --val-features   .../flim_val.features.jsonl \
        --config         config_flim.json \
        --output-dir     /data1/amine/checkpoints/flim_run_001
"""
from __future__ import annotations
import argparse, json, logging, pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingRegressor,
                              AdaBoostClassifier)
from sklearn.feature_selection import chi2, mutual_info_classif, VarianceThreshold
from sklearn.linear_model import SGDRegressor

from flim_common import FEATURE_COLUMNS, average_precision

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def load_features(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            rows.append({"jira_id": r["jira_id"], "label": int(r["label"]),
                         **r["features"]})
    df = pd.DataFrame(rows)
    for c in FEATURE_COLUMNS:
        if c not in df: df[c] = 0.0
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].clip(lower=0.0).fillna(0.0)
    return df


def macro_map(df: pd.DataFrame, scores: np.ndarray) -> float:
    d = df[["jira_id", "label"]].copy(); d["score"] = scores
    aps = [average_precision(g["score"].tolist(), g["label"].tolist())
           for _, g in d.groupby("jira_id") if g["label"].any()]
    return float(np.mean(aps)) if aps else 0.0


def _norm(w):
    w = np.clip(np.nan_to_num(np.asarray(w, dtype=float)), 0, None)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


def weight_methods(seed: int):
    def w_chi2(X, y):   return _norm(chi2(X, y)[0])
    def w_mi(X, y):     return _norm(mutual_info_classif(X, y, random_state=seed))
    def w_extra(X, y):
        m = ExtraTreesClassifier(n_estimators=100, random_state=seed).fit(X, y)
        return _norm(m.feature_importances_)
    def w_gbr(X, y):
        m = GradientBoostingRegressor(n_estimators=100, random_state=seed).fit(X, y)
        return _norm(m.feature_importances_)
    def w_ada(X, y):
        m = AdaBoostClassifier(n_estimators=100, random_state=seed).fit(X, y)
        return _norm(m.feature_importances_)
    def w_var(X, y):
        fs = VarianceThreshold().fit(X)
        return _norm(fs.variances_)
    def w_const(X, y):  return np.ones(X.shape[1]) / X.shape[1]
    return {"chi2": w_chi2, "mutual_info": w_mi, "extra_trees": w_extra,
            "gbr_importance": w_gbr, "adaboost": w_ada,
            "variance": w_var, "const": w_const}


def cut_set(labels: np.ndarray, score: np.ndarray, perc: float) -> np.ndarray:
    """Positives + lowest-scored slice among score>0 (Fejzer's size_selectf)."""
    keep = labels == 1
    pos_scores = score[score > 0]
    if len(pos_scores):
        k  = max(int(perc * keep.sum()), 1)
        tm = np.sort(pos_scores)[:k].max()
        keep = keep | (score <= tm)
    return keep & (score > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-features", required=True)
    ap.add_argument("--val-features",   required=True)
    ap.add_argument("--config",         required=True)
    ap.add_argument("--output-dir",     required=True)
    args = ap.parse_args()
    cfg  = json.load(open(args.config, encoding="utf-8"))["ltr"]
    seed = cfg.get("seed", 123456)

    tr  = load_features(args.train_features)
    va  = load_features(args.val_features)
    Xtr, ytr = tr[FEATURE_COLUMNS].values, tr["label"].values
    Xva      = va[FEATURE_COLUMNS].values
    log.info("Train rows %d (pos %d) | Val rows %d",
             len(tr), int(ytr.sum()), len(va))

    # ── Phase 1: prescoring weight selection ────
    presc_log, best = [], ("", None, -1.0)
    for name, fn in weight_methods(seed).items():
        try:
            w = fn(Xtr, ytr)
        except Exception as e:
            log.warning("weights %s failed: %s", name, e); continue
        m = macro_map(va, Xva @ w)
        presc_log.append((name, [round(x, 5) for x in w], round(m, 4)))
        log.info("  prescoring %-16s val MAP %.4f", name, m)
        if m > best[2]: best = (name, w, m)
    w_name, w_best, w_map = best
    log.info("Best prescoring: %s (val MAP %.4f)", w_name, w_map)

    # ── Phase 2: regression model selection ────
    score_tr  = Xtr @ w_best
    target    = score_tr + ytr * score_tr.max()
    losses    = ["squared_error", "huber", "epsilon_insensitive",
                 "squared_epsilon_insensitive"]
    penalties = ["l2", "l1", "elasticnet", None]
    reg_log, reg_best = [], (None, "", -1.0)
    for loss, pen, alpha, perc in product(
            losses, penalties, cfg.get("alphas", [1e-4]),
            cfg.get("cut_percs", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3])):
        mask = cut_set(ytr, score_tr, perc)
        if mask.sum() < 10: continue
        try:
            reg = SGDRegressor(max_iter=1000, shuffle=False, loss=loss,
                               penalty=pen, alpha=alpha, random_state=seed)
            reg.fit(Xtr[mask], target[mask])
            m = macro_map(va, reg.predict(Xva))
        except Exception:
            continue
        name = f"SGD_{loss}_{pen}_{alpha}_cut{perc}"
        reg_log.append((name, round(m, 4)))
        if m > reg_best[2]: reg_best = (reg, name, m)
    log.info("Best regressor : %s (val MAP %.4f)", reg_best[1], reg_best[2])

    # ── Final choice by val MAP ────
    use_reg = reg_best[2] >= w_map
    artifact = {
        "feature_columns": FEATURE_COLUMNS,
        "type": "regressor" if use_reg else "weights",
        "weights": w_best, "weights_name": w_name, "weights_val_map": w_map,
        "regressor": reg_best[0], "regressor_name": reg_best[1],
        "regressor_val_map": reg_best[2],
    }
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "flim_ltr.pkl", "wb") as f:
        pickle.dump(artifact, f)
    json.dump({"chosen": artifact["type"],
               "prescoring_log": presc_log, "regression_log": reg_log,
               "best_prescoring": [w_name, w_map],
               "best_regression": [reg_best[1], reg_best[2]]},
              open(out / "flim_ltr_report.json", "w"), indent=2)
    log.info("Saved %s (chosen: %s)", out / "flim_ltr.pkl", artifact["type"])


if __name__ == "__main__":
    main()
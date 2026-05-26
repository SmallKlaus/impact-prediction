
TECHNICAL DEBT PRIORITIZATION RESEARCH
/
Claude finished the response
This chat will be mainly about the first Phase of the research pipeline currently being conducted. For the sake of organization and containment, I'll keep any questions or help needed with the first phase limited to this one conversation, that way you can rely on it as well as your memory of course, when answering. I'm currently conducting a run with hard negative mining starting at batch 2, using train.jsonl, val_500.jsonl and train_1000.jsonl for the HNM, HNM is being conducted every 2 epochs. My intuition is telling me that using HNM at batch 2 is kind of useless. Without HNM, I saw improvement until around epoch 6-7, how about we start the HNM at those batches to challenge the model once it has learned sufficiently?

May 22
Your intuition is solid. Starting HNM at batch 2 is essentially wasting the miner on an undertrained model — the embeddings at that stage are barely better than random, so the "hard" negatives it selects are mostly noise rather than genuinely informative near-misses. You end up poisoning early training with examples the model has no business tackling yet.

The reasoning for delaying HNM is well-grounded:

Why early HNM hurts

The bi-encoder needs stable embedding geometry before hard negatives become meaningful. Before that, the top-K non-positives the miner selects are essentially arbitrary.
Introducing hard negatives too early can cause the model to overfit to spurious surface-level similarity signals, which may actually worsen final geometry compared to training on easy negatives longer.
Focal loss already handles class imbalance; let it do its job on clean easy negatives first.
Why epoch 6-7 is a sensible trigger point

Your best sampled-val run peaked at epoch 6 (R@10=0.625) without HNM, meaning that's roughly when the model saturates on easy negatives and marginal learning flattens. That's exactly the right moment to introduce harder signal.
At that point the embeddings are stable enough that "hard" actually means semantically confusable, not just randomly close in an untrained space.
Practical suggestion

Rather than a fixed epoch trigger, you could also condition HNM start on a metric threshold — e.g., start mining once val R@10 > 0.50 or stops improving for 1 epoch — which makes it robust across runs with different convergence speeds. But if you want simplicity, hardcoding epoch 6 as the hnm_start_epoch is perfectly reasonable given your observed curve.

One thing to watch after HNM kicks in: expect a temporary dip in val metrics for 1-2 epochs as the model adjusts to harder examples. That's normal and not a reason to stop. The question is whether it recovers and surpasses the pre-HNM plateau.

Do you want to add a hnm_start_epoch config parameter to train.py, or do you prefer the metric-threshold approach?


diagnose.py
py


train.py
py


config.json
48 lines

json

I've attached 3 files that are part of the impact-prediction model (there are 4 other files : dataset.py, hard-negative-miner.py, loss.py, model.py - but I don't think they're useful to the current task and would just take part of your precious memory right now). 

Your current task : 

Implement the more robust HNM on metric threshold and add a param in config.json hnm_threshold so I can change it as I please.
Early stopping should be changed to watch R@50 and not R@10. This would be more useful to the experimentation to see whether it's worth adding a cross-encoder or not.
Add R@80, R@100, and MRR as metrics while you're at it (it doesn't change much structurally)
May 22

Architected metric-threshold HNM triggering and restructured evaluation metrics comprehensively
Architected metric-threshold HNM triggering and restructured evaluation metrics comprehensively
Good, I have a complete picture of both files. Let me implement all three changes cleanly.


Copy uploaded files to working directory
Copy uploaded files to working directory
Starting with train.py — metrics first, then HNM, then early stopping:


Edited 2 files
Edited 2 files
Now update _maybe_mine_hard_negatives with the metric-threshold logic:


Replace _maybe_mine_hard_negatives with threshold-gated version
Replace _maybe_mine_hard_negatives with threshold-gated version
Now update the training state block — switch to R@50 early stopping, add hnm_state and current_r50:


Switch early-stopping tracking from R@10 to R@50, add hnm_state and current_r50
Switch early-stopping tracking from R@10 to R@50, add hnm_state and current_r50
Update the log line that announces training start and the HNM startup log:


Update HNM enabled log to mention threshold
Update HNM enabled log to mention threshold
Update the mining call site in the training loop to pass the new args:


Pass current_r50 and hnm_state to _maybe_mine_hard_negatives
Pass current_r50 and hnm_state to _maybe_mine_hard_negatives
Update the epoch log line and the best-model / early-stopping block:


Update epoch log to show R@50 and MRR, and switch early stopping to R@50
Update epoch log to show R@50 and MRR, and switch early stopping to R@50
Update the final training complete log and test_metrics.json:


Edited 3 files
Edited 3 files
Now update the docstring at the top to reflect the changes:


Update docstring to reflect new metrics and early stopping
Update docstring to reflect new metrics and early stopping
Now diagnose.py:


Edited 5 files
Edited 5 files
Now update the worst-issues table to include MRR:


Edited 2 files
Edited 2 files
Finally, config.json:


Add hnm_threshold to config.json hard_negative_mining block
Add hnm_threshold to config.json hard_negative_mining block
Quick sanity check — verify all old references are gone and new ones are consistent:


Verify no stale references remain and all new ones are present
Verify no stale references remain and all new ones are present
All clean — no stale references, everything wired correctly. Copy to outputs:


Ran a command, read a file
Ran a command, read a file
Here's a summary of every change made:

train.py

New metric — mean_reciprocal_rank: added as a standalone function mirroring average_precision in style. Ranks the list, returns 1/(rank of first positive).
compute_ranking_metrics: default ks expanded to [5, 10, 20, 50, 80, 100], MRR slot added to the metrics dict and populated per issue.
Epoch log line: now shows R@10 | R@50 | R@100 | NDCG@10 | MAP | MRR.
Early stopping: best_recall_at_10 → best_recall_at_50, all references updated including test_metrics.json's best_val_recall@50 key and the final training-complete log.
_maybe_mine_hard_negatives: new signature with current_r50: float and hnm_state: dict. Two-gate logic: floor guard on start_epoch (unchanged, acts as safety minimum), then threshold gate on R@50. Once the threshold is crossed, hnm_state["unlocked"] is set and first_epoch is recorded so mine_every_n_epochs cadence counts correctly from that point.
Training loop state: added current_r50 = 0.0 and hnm_state = {"unlocked": False, "first_epoch": None}. current_r50 is updated right after each evaluate() call, so the value passed to mining at the top of epoch N is always the R@50 from epoch N-1.
diagnose.py

mean_reciprocal_rank added, MRR computed per issue and aggregated globally and per-project.
KS = [5, 10, 20, 50, 80, 100] — all downstream loops pick it up automatically.
Print tables updated: global block adds MRR, per-project table now shows R@10 | R@50 | R@80 | R@100 | MAP | MRR, worst-issues table adds MRR column.
config.json

"hnm_threshold": 0.65 added to hard_negative_mining. Given your sampled-val R@50 peaked at ~0.89 at epoch 6, 0.65 gives a comfortable margin — HNM should unlock around epoch 5-6 in a typical run. Tune freely.

Train
PY 

Diagnose
PY 

Config
Code · JSON 
Let's add a few aesthetic changes, your task 

Add a progress bar for validation ( I want to be able to see the progress of validation between epochs)
Add a progress bar for testing as well ( I want to be able to see the progress of testing in the final run)
May 22

Check current evaluate() function
Check current evaluate() function
Two-line change — add a desc parameter and wrap the loop with tqdm:


Edited 2 files
Edited 2 files
Now pass desc="Testing" at the test evaluation call site:


Ran 2 commands, edited a file, read a file
Ran 2 commands, edited a file, read a file
Small but clean changes. Here's what was done:

evaluate() got a desc: str = "Validating" parameter, and the bare for batch in loader loop became for batch in tqdm(loader, desc=desc, leave=False). Default label covers every epoch validation call automatically with no changes to the call sites.
The test evaluation call at the end of training explicitly passes desc="Testing" so the two contexts are distinguishable in the terminal output.
leave=False on both keeps the terminal tidy — the bars clear once done and don't stack up with the epoch training bars.

Train
PY 
The one adjustment worth making is to reset the patience counter after each mining round, since a mining round is a distribution shift that legitimately resets what "no improvement" means. If we do not reset it, the model might early-stop right after a mining round before it has had time to learn from the new negatives. Do you objectively this is better or not?

May 23

Weighed technical merit of patience counter reset strategy
Weighed technical merit of patience counter reset strategy
Yes, objectively it is better, and the reasoning is sound. Here's why it's not just a preference:

When HNM fires, the training distribution changes — the hard negatives injected are semantically confusable examples the model hasn't been optimized against yet. The immediate effect is almost always a temporary drop or plateau in val R@50, because the model is suddenly being evaluated against a harder implicit task than what it trained on in the preceding epochs. If patience has been counting down through the pre-mining plateau, you could be one or two epochs away from early stopping right as the new signal starts paying off. You'd be aborting exactly when the training is doing something meaningful.

The counter-argument worth considering is: what if the model genuinely can't learn from the new negatives and the plateau persists? In that case you'd want early stopping to still fire eventually. Resetting doesn't prevent that — patience still runs its full count from the reset point, so you get a fair window. You're not disabling early stopping, you're just saying "this event constitutes a legitimate fresh start, evaluate from here."

The only scenario where the reset is mildly wasteful is if mining fires but introduces no new samples (e.g. the mined negatives are identical to what was already in the training set). That's worth a quick log check but is unlikely in practice given the pool size.

So yes — implement it. One-liner in the training loop right after the loader is rebuilt:


Edited 2 files, ran 2 commands, viewed a file
Edited 2 files, ran 2 commands, viewed a file
One implementation note worth flagging: the reset is triggered by checking whether the current epoch was a mining epoch (epochs_since % mine_every == 0), not by whether the dataset actually changed. This is intentional — even if mining happened to inject near-identical negatives, the distribution might have shifted, and the cost of a spurious reset is just a few extra epochs, which is fine. The alternative (keying off new_n_batches != prev_n_batches) would miss resets when dataset size stays the same but the hard negative content changed, which is the common case once mining is mature and the pool stabilizes.


Train
PY 
The results of the training / testing with the latest model implemented have been analyzed by Sonnet 4.6 and it gave the following analysis : 

Overall progress vs previous run
Comparing to your previous real-codebase diagnosis:
MetricPrevious runThis runChangeR@100.2420.306+0.064 (+26%)R@500.5000.583+0.083 (+17%)R@1000.6810.682+0.001AUC-ROC0.8940.939+0.045MRR—0.312—
The improved class descriptions and the first mining round are clearly working. The model is meaningfully better at surfacing the right classes in the top 10 and top 50.
The mining behaviour — what happened
First mining round (epoch 6): excellent. Val R@50 jumped from 0.6594 to 0.7226 in a single epoch — a +0.063 gain. This is exactly what mining is supposed to do. The mined hard negatives were the model's genuine failure modes at that stage, and training on them immediately pushed the representations apart.
Second mining round (epoch 9): counterproductive. Val R@50 dropped from 0.7272 to 0.6539 — a regression of -0.073. The model never recovered in the remaining 3 epochs and early stopping triggered at epoch 11.
Why did the second round hurt? Several things compound here.
First, by epoch 9 the cosine LR schedule has decayed to ~4.5e-5, roughly half of where it was at epoch 6 (~8.7e-5). When mining injects a new distribution of hard negatives, the model needs enough gradient signal to adapt to them quickly. With a decayed LR the adaptation is slow, and the patience window of 3 epochs is not long enough for recovery to manifest in val R@50 when starting from a low LR.
Second, and more fundamentally, the second mining round found harder negatives than the first — this is by design, the model had already learned the epoch-6 hard negatives. But harder negatives at lower LR means the model takes longer to learn them, potentially longer than the 3-epoch patience window allows.
Third, look at the NCE loss behaviour. At epoch 8 you see values like 0.0175 and 0.0262 — the contrastive objective is nearly saturated, the model has learned to separate positive and negative embeddings well in the sampled val distribution. After the second mining round, NCE jumps back up (0.41, 0.71, 0.74) — the new hard negatives are actively disrupting the projection space that took 8 epochs to build.
What the val→test gap tells you
Best val R@50 is 0.727 (500-cap, ~150K samples). Test R@50 is 0.583 (full codebase, 2.57M samples). The ~0.14 gap is real and expected — the 500-cap val set still gives the model a much easier ranking problem than the full codebase. This is consistent with what you saw before. The test AUC-ROC of 0.939 is the most meaningful number here: it means the model's probability outputs are well-ordered — it ranks positives above negatives 93.9% of the time in pairwise comparisons, on the full-codebase test set. That is genuinely strong for this task.
F1@0.5 of 0.014 is not a problem. The positive rate in the full test set is roughly 0.1%, so predicting all-negative gives F1=0. The model's sigmoid outputs are not calibrated around 0.5 for a distribution this imbalanced — and they don't need to be. F1@0.5 is meaningless here and you should stop reporting it.

I want to explore two options : 

Extend patience after mining : explanation - 3 epochs is not enough time at a decayed LR for the model to fully absorb new hard negatives. This requires a small config change: set patience to 5 and let early stopping run naturally.
Reset LR after mining : After each mining round, partially reset the cosine schedule so the model gets a brief higher-LR window to learn the new negatives. This is the most principled approach but requires a code change to train.py to reinitialise the scheduler after mining.
Can you update the train.py and the config.json to include option 2 (reset LR after mining), so I can control whether to reset the LR after mining using a flag that can be switched true/flase in the config.json file, this way I keep the model flexible and experiment with both options.

Show more
9:31 PM

Architected LR reset mechanism with configurable fraction control
Architected LR reset mechanism with configurable fraction control
The user wants to add an LR reset after mining rounds, controlled by a config flag. Let me think about the best implementation.

The idea: after a mining round fires, partially reset the cosine LR schedule so the model gets a higher LR window to learn the new negatives.

Key design decisions:

Config flag: "reset_lr_after_mining": true/false in the hard_negative_mining block
What "reset" means: reinitialise the cosine scheduler with a new T_max covering the remaining epochs, starting from lr_head (or a fraction of it).
The cleanest approach: after mining fires, rebuild the scheduler from scratch with T_max = remaining_steps and starting from the original LR values. This gives a fresh cosine decay from the current LR peak down to eta_min over the remaining training window.

But wait - the optimizer's param group LRs need to be reset too, not just the scheduler. The scheduler doesn't reset the LR on its own; you need to explicitly set the LR in the optimizer param groups back to their initial values, then create a new scheduler.

Implementation:

After _maybe_mine_hard_negatives detects that mining fired (i.e., hnm_state["unlocked"] and this epoch is a mining epoch), if reset_lr_after_mining is True:
Reset optimizer LRs to their initial values (lr_head for new layers, lr_backbone for backbone)
Create a new CosineAnnealingLR with T_max = remaining steps
I should also consider: should we reset to full lr_head/lr_backbone, or to a fraction? The analysis suggests a partial reset. Let me add a lr_reset_fraction parameter - a multiplier on the original LR. Default 1.0 (full reset). This gives even more flexibility.

Actually, let me keep it simple - just full reset to original LRs, controlled by the boolean flag. If they want fractional reset they can always add it later.

Where to place the LR reset in the training loop:

Right after _maybe_mine_hard_negatives returns AND we confirm mining actually fired …
Show less
Done
Clean design problem. Two config params make sense here: the boolean flag reset_lr_after_mining and a lr_reset_fraction multiplier (e.g. 0.5 = reset to 50% of the original LRs rather than fully back to epoch-1 levels, which would be too aggressive at epoch 9). That gives you full control to experiment with both the on/off and the magnitude of the reset.


Viewed a file, edited a file
Viewed a file, edited a file
Now add the two new params to config.json:


Add reset_lr_after_mining and lr_reset_fraction to config
Add reset_lr_after_mining and lr_reset_fraction to config
Quick sanity check:


Ran 2 commands, read a file
Ran 2 commands, read a file
Here's what was added and why each decision was made:

train.py — the LR reset sits inside the same if epochs_since % mine_every == 0 block as the patience reset, so both always fire together on a mining epoch. The logic:

Reads lr_reset_fraction from config (default 1.0 if absent, so the feature is safely off by default when the flag is false)
Directly mutates group["lr"] on both param groups by name — this is intentional, because the cosine scheduler tracks LR relative to the last value it set, so you must update the param groups explicitly before constructing the new scheduler, otherwise it inherits the decayed LR as its new "peak"
Rebuilds CosineAnnealingLR over the remaining steps, which decays cleanly from the reset peak down to eta_min=1e-6
Logs the exact LR values and step count so you can verify the reset in the training output
config.json — two new fields in hard_negative_mining:

"reset_lr_after_mining": true — the on/off switch, set to true to test option 2
"lr_reset_fraction": 0.5 — resets to 50% of initial LRs (5e-5 for the head, 5e-6 for the backbone). This is the recommended starting value: aggressive enough to meaningfully accelerate learning from new negatives, conservative enough to not destabilise the representations that took 8 epochs to build. To test option 1 instead, just flip reset_lr_after_mining to false and set patience to 5 in the training block — no code changes needed.

Train
PY 

Config
Code · JSON 






Claude is AI and can make mistakes. Please double-check responses.
Train · PY
"""
train.py — Training Loop with Frozen Warmup and Gradual Unfreeze
=================================================================
Training schedule:
  Phase 1 (warmup_epochs):     Backbone fully frozen.
                                Only projection heads and interaction MLP train.
                                Higher LR for new layers (lr_head).

  Phase 2 (unfreeze_epoch onwards): Top n_unfreeze_layers backbone layers thawed.
                                Backbone gets a much lower LR (lr_backbone)
                                to avoid catastrophic forgetting.
                                New layers keep lr_head.

Evaluation metrics (computed each epoch on validation set):
  - Recall@k for k in [5, 10, 20, 50, 80, 100]
  - NDCG@k   for k in [5, 10, 20]
  - MAP, MRR
  - AUC-ROC, F1@0.5 (for completeness)

Recall@50 is the primary early-stopping metric: it directly measures how
much headroom the bi-encoder leaves for a cross-encoder reranker operating
on a top-50 shortlist.

Usage:
    python train.py --config config.json

config.json example:
    {
      "train_jsonl":        "data/training_samples/train.jsonl",
      "val_jsonl":          "data/training_samples/val.jsonl",
      "output_dir":         "checkpoints/run_001",
      "model": {
        "model_name":         "microsoft/unixcoder-base",
        "proj_dim":           256,
        "interaction_hidden": 256,
        "dropout":            0.1,
        "freeze_backbone":    true,
        "n_unfreeze_layers":  0
      },
      "loss": {
        "focal_alpha":        0.75,
        "focal_gamma":        2.0,
        "lambda_contrastive": 0.1,
        "contrastive_temp":   0.07
      },
      "training": {
        "batch_size":         32,
        "warmup_epochs":      2,
        "total_epochs":       10,
        "unfreeze_epoch":     3,
        "n_unfreeze_layers":  2,
        "lr_head":            1e-4,
        "lr_backbone":        1e-5,
        "weight_decay":       0.01,
        "max_grad_norm":      1.0,
        "max_chunk_tokens":   512,
        "max_class_tokens":   512,
        "max_chunks":         8,
        "label_smoothing":    0.0,
        "num_workers":        4,
        "eval_every_n_steps": 500,
        "patience":           3
      }
    }
"""

from __future__ import annotations
from tqdm import tqdm
import os
import ssl
import requests
import urllib3

# Disable urllib3 SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Disable standard library SSL checks
ssl._create_default_https_context = ssl._create_unverified_context

# Force requests to ignore SSL verification
os.environ['CURL_CA_BUNDLE'] = ''


import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from model               import ImpactScoreModel, build_model
from dataset             import build_dataloaders, SampleBatch, collate_fn
from loss                import CombinedLoss, build_loss
from hard_negative_miner import build_miner_from_config, HardNegativeMiner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Evaluation metrics ────────────────────────────────────────────────────────

def recall_at_k(scores: list[float], labels: list[int], k: int) -> float:
    """Recall@k: fraction of positives appearing in top-k ranked items."""
    if not any(labels):
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    top_k_labels = [lbl for _, lbl in ranked[:k]]
    n_pos = sum(labels)
    return sum(top_k_labels) / n_pos


def ndcg_at_k(scores: list[float], labels: list[int], k: int) -> float:
    """NDCG@k (binary relevance)."""
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    dcg = sum(
        lbl / math.log2(i + 2)
        for i, (_, lbl) in enumerate(ranked[:k])
    )
    ideal = sorted(labels, reverse=True)
    idcg = sum(
        lbl / math.log2(i + 2)
        for i, lbl in enumerate(ideal[:k])
    )
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(scores: list[float], labels: list[int]) -> float:
    """Average Precision for a single query."""
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos, cum_pos, ap = 0, 0, 0.0
    for i, (_, lbl) in enumerate(ranked):
        if lbl:
            cum_pos += 1
            n_pos   += 1
            ap      += cum_pos / (i + 1)
    return ap / n_pos if n_pos else 0.0


def mean_reciprocal_rank(scores: list[float], labels: list[int]) -> float:
    """MRR: reciprocal rank of the first positive in the ranked list."""
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    for i, (_, lbl) in enumerate(ranked):
        if lbl:
            return 1.0 / (i + 1)
    return 0.0


def compute_ranking_metrics(
    all_scores: list[float],
    all_labels: list[int],
    all_jira_ids: list[str],
    ks: list[int] = [5, 10, 20, 50, 80, 100],
) -> dict:
    """
    Compute per-issue ranking metrics, then macro-average across issues.

    Groups predictions by jira_id so each issue is one query and its
    candidate classes are the ranked items.
    """
    from collections import defaultdict
    issue_scores: dict[str, list] = defaultdict(list)
    issue_labels: dict[str, list] = defaultdict(list)

    for score, label, jid in zip(all_scores, all_labels, all_jira_ids):
        issue_scores[jid].append(score)
        issue_labels[jid].append(label)

    metrics: dict[str, list] = {f"recall@{k}": [] for k in ks}
    metrics.update({f"ndcg@{k}": [] for k in ks if k <= 20})
    metrics["map"] = []
    metrics["mrr"] = []

    for jid in issue_scores:
        s = issue_scores[jid]
        l = issue_labels[jid]
        if not any(l):
            continue   # skip issues with no positives in this split
        for k in ks:
            metrics[f"recall@{k}"].append(recall_at_k(s, l, k))
        for k in ks:
            if k <= 20:
                metrics[f"ndcg@{k}"].append(ndcg_at_k(s, l, k))
        metrics["map"].append(average_precision(s, l))
        metrics["mrr"].append(mean_reciprocal_rank(s, l))

    return {k: (sum(v) / len(v) if v else 0.0) for k, v in metrics.items()}


def compute_binary_metrics(
    all_scores: list[float],
    all_labels: list[int],
) -> dict:
    """AUC-ROC and F1@0.5 threshold."""
    try:
        from sklearn.metrics import roc_auc_score, f1_score
        auc  = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.0
        preds = [1 if s >= 0.5 else 0 for s in all_scores]
        f1    = f1_score(all_labels, preds, zero_division=0)
        return {"auc_roc": auc, "f1_at_0.5": f1}
    except ImportError:
        return {}


# ── Evaluation pass ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:      ImpactScoreModel,
    loader:     DataLoader,
    loss_fn:    CombinedLoss,
    device:     torch.device,
    desc:       str = "Validating",
) -> dict:
    model.eval()

    all_scores, all_labels, all_jira_ids = [], [], []
    total_loss = 0.0
    n_batches  = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        batch = batch.to(device)

        logits, f_proj, c_proj = model(
            batch.feature_input_ids,
            batch.feature_attention_mask,
            batch.feature_chunk_mask,
            batch.class_input_ids,
            batch.class_attention_mask,
        )
        _, breakdown = loss_fn(logits, f_proj, c_proj, batch.labels)
        total_loss  += breakdown["loss_total"]
        n_batches   += 1

        probs = torch.sigmoid(logits).cpu().tolist()
        lbls  = batch.labels.int().cpu().tolist()
        all_scores.extend(probs)
        all_labels.extend(lbls)
        all_jira_ids.extend(batch.jira_ids)

    avg_loss     = total_loss / max(n_batches, 1)
    rank_metrics = compute_ranking_metrics(all_scores, all_labels, all_jira_ids)
    bin_metrics  = compute_binary_metrics(all_scores, all_labels)

    return {"val_loss": avg_loss, **rank_metrics, **bin_metrics}


# ── Optimizer construction ────────────────────────────────────────────────────

def build_optimizer(
    model:        ImpactScoreModel,
    lr_head:      float,
    lr_backbone:  float,
    weight_decay: float,
) -> AdamW:
    """
    Two parameter groups:
      - Backbone (encoder): low LR to avoid catastrophic forgetting
      - New layers (projection, interaction): high LR for fast adaptation

    Backbone params with requires_grad=False are excluded automatically
    by AdamW (no gradient, no update).
    """
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    new_params      = (
        list(model.projection.parameters()) +
        list(model.interaction.parameters()) +
        (list(model.attn_pool.parameters()) if model.attn_pool else [])
    )

    param_groups = [
        {"params": new_params,      "lr": lr_head,     "name": "new_layers"},
        {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
    ]
    # Filter out empty groups
    param_groups = [g for g in param_groups if g["params"]]

    return AdamW(param_groups, weight_decay=weight_decay)


# ── Dynamic hard-negative mining helpers ──────────────────────────────────────

def _rebuild_train_loader(train_loader: DataLoader) -> DataLoader:
    """Rebuild the train DataLoader after mutating the underlying dataset.

    Workers fork their own copy of the dataset, so once the dataset's `samples`
    list has been swapped we need a fresh loader for the new state to take
    effect under `num_workers > 0`.
    """
    return DataLoader(
        train_loader.dataset,
        batch_size  = train_loader.batch_size,
        shuffle     = True,
        collate_fn  = train_loader.collate_fn,
        num_workers = train_loader.num_workers,
        pin_memory  = train_loader.pin_memory,
    )


def _resolve_per_issue_top_k(
    jira_ids:     list[str],
    initial_cnts: dict[str, int],
    override_k:   Optional[int],
) -> dict[str, int]:
    """Pick the hard-neg budget per issue.

    If `top_k_per_issue` is set in config, use it for every issue.
    Otherwise fall back to the per-issue count observed in the original
    train.jsonl so the dataset's pos / hard-neg / easy-neg ratio stays the
    same as before.
    """
    if override_k is not None and override_k > 0:
        return {jid: override_k for jid in jira_ids}
    return {jid: initial_cnts.get(jid, 0) for jid in jira_ids}


def _maybe_mine_hard_negatives(
    epoch:        int,
    miner:        Optional[HardNegativeMiner],
    hnm_cfg:      dict,
    train_loader: DataLoader,
    model:        ImpactScoreModel,
    device:       torch.device,
    initial_hn_counts: dict[str, int],
    current_r50:  float,
    hnm_state:    dict,
) -> tuple[DataLoader, bool]:
    """Run one mining pass at the *start* of `epoch` if conditions are met.

    Two-gate design:
      1. Floor guard  — never mine before `start_epoch` regardless of metrics.
      2. Threshold gate — wait until val R@50 >= `hnm_threshold`.  Once crossed
         for the first time, subsequent cadence uses `mine_every_n_epochs`
         measured from that first-mine epoch.

    Returns (train_loader, mined) where `mined=True` signals that a mining
    round actually ran this epoch (caller should reset patience counter).
    """
    if miner is None:
        return train_loader, False

    start_epoch = int(hnm_cfg.get("start_epoch", 2))
    mine_every  = int(hnm_cfg.get("mine_every_n_epochs", 1))
    threshold   = float(hnm_cfg.get("hnm_threshold", 0.0))

    # ── Gate 1: floor guard ───────────────────────────────────────────────────
    if epoch < start_epoch:
        log.info(
            "Epoch %d: HNM floor guard — waiting for epoch %d.",
            epoch, start_epoch,
        )
        return train_loader, False

    # ── Gate 2: metric threshold ──────────────────────────────────────────────
    if not hnm_state["unlocked"]:
        if current_r50 < threshold:
            log.info(
                "Epoch %d: HNM threshold not met — val R@50=%.4f < %.4f (threshold). "
                "Continuing with current negatives.",
                epoch, current_r50, threshold,
            )
            return train_loader, False
        # Threshold crossed for the first time
        hnm_state["unlocked"]    = True
        hnm_state["first_epoch"] = epoch
        log.info(
            "Epoch %d: HNM threshold reached — val R@50=%.4f >= %.4f. "
            "Hard-negative mining unlocked.",
            epoch, current_r50, threshold,
        )

    # ── Cadence check ─────────────────────────────────────────────────────────
    epochs_since = epoch - hnm_state["first_epoch"]
    if epochs_since % mine_every != 0:
        next_mine = epoch + (mine_every - epochs_since % mine_every)
        log.info(
            "Epoch %d: skipping mining (only runs every %d epochs, next at epoch %d).",
            epoch, mine_every, next_mine,
        )
        return train_loader, False

    # ── Mine ──────────────────────────────────────────────────────────────────
    train_ds = train_loader.dataset
    jira_ids = train_ds.training_jira_ids()

    per_issue_top_k = _resolve_per_issue_top_k(
        jira_ids     = jira_ids,
        initial_cnts = initial_hn_counts,
        override_k   = hnm_cfg.get("top_k_per_issue"),
    )

    log.info(
        "Epoch %d: mining hard negatives for %d issues (sum top_k = %d) ...",
        epoch, len(jira_ids), sum(per_issue_top_k.values()),
    )

    mined = miner.mine(
        model            = model,
        device           = device,
        per_issue_top_k  = per_issue_top_k,
        jira_ids         = jira_ids,
    )

    n_hn = train_ds.replace_hard_negatives(mined)
    log.info(
        "Epoch %d: hard-neg slice replaced — %d HN samples now in training set (total %d).",
        epoch, n_hn, len(train_ds),
    )

    return _rebuild_train_loader(train_loader), True


# ── Training loop ─────────────────────────────────────────────────────────────

def train(config: dict):
    tcfg = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    log.info("Device: %s", device)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Tokenizer and data ────────────────────────────────────────────────────
    model_name = config["model"]["model_name"]
    log.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    log.info("Building dataloaders ...")
    train_loader, val_loader = build_dataloaders(
        train_path       = config["train_jsonl"],
        val_path         = config["val_jsonl"],
        tokenizer        = tokenizer,
        batch_size       = tcfg["batch_size"],
        max_chunk_tokens = tcfg.get("max_chunk_tokens", 512),
        max_class_tokens = tcfg.get("max_class_tokens", 512),
        max_chunks       = tcfg.get("max_chunks", 8),
        label_smoothing  = tcfg.get("label_smoothing", 0.0),
        num_workers      = tcfg.get("num_workers", 4),
    )
    log.info("Train samples: %d  |  Val samples: %d",
             len(train_loader.dataset), len(val_loader.dataset))

    # ── Dynamic hard-negative miner (optional) ────────────────────────────────
    hnm_cfg = config.get("hard_negative_mining", {}) or {}
    initial_hn_counts = train_loader.dataset.per_issue_hard_neg_counts()
    miner: Optional[HardNegativeMiner] = None
    if hnm_cfg.get("enabled", False):
        train_jids = set(train_loader.dataset.training_jira_ids())
        log.info(
            "Hard-negative mining ENABLED — start_epoch=%d, hnm_threshold=%.4f, "
            "top_k_per_issue=%s, embed_batch_size=%d",
            int(hnm_cfg.get("start_epoch", 2)),
            float(hnm_cfg.get("hnm_threshold", 0.0)),
            hnm_cfg.get("top_k_per_issue"),
            int(hnm_cfg.get("embed_batch_size", 64)),
        )
        miner = build_miner_from_config(
            config,
            tokenizer            = tokenizer,
            restrict_to_jira_ids = train_jids,
        )
    else:
        log.info("Hard-negative mining DISABLED — using static dependency-seeded HNs.")

    # ── Model, loss, optimizer ────────────────────────────────────────────────
    log.info("Building model ...")
    model   = build_model(config["model"]).to(device)
    loss_fn = build_loss(config["loss"])

    frozen, total = model.frozen_param_count()
    log.info("Backbone params: %d frozen / %d total", frozen, total)

    optimizer = build_optimizer(
        model,
        lr_head      = tcfg["lr_head"],
        lr_backbone  = tcfg["lr_backbone"],
        weight_decay = tcfg["weight_decay"],
    )

    total_steps = len(train_loader) * tcfg["total_epochs"]
    scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # ── Training state ────────────────────────────────────────────────────────
    best_recall_at_50 = 0.0
    patience_counter  = 0
    global_step       = 0
    unfrozen          = False
    current_r50       = 0.0          # val R@50 from previous epoch (gates HNM)
    hnm_state         = {"unlocked": False, "first_epoch": None}

    log.info("Starting training for %d epochs (warmup=%d, unfreeze_epoch=%d) ...",
             tcfg["total_epochs"], tcfg["warmup_epochs"], tcfg["unfreeze_epoch"])

    for epoch in range(1, tcfg["total_epochs"] + 1):

        # ── Dynamic hard-negative mining (start of epoch) ─────────────────────
        # Epoch 1 keeps the dependency-seeded HNs from train.jsonl; from
        # `start_epoch` onwards we replace them with model-mined ones.
        if miner is not None:
            prev_n_batches = len(train_loader)
            train_loader, mined_this_epoch = _maybe_mine_hard_negatives(
                epoch              = epoch,
                miner              = miner,
                hnm_cfg            = hnm_cfg,
                train_loader       = train_loader,
                model              = model,
                device             = device,
                initial_hn_counts  = initial_hn_counts,
                current_r50        = current_r50,
                hnm_state          = hnm_state,
            )
            if mined_this_epoch:
                # Mining is a distribution shift — reset patience so the model
                # gets a full window to adapt to the new negatives before
                # early stopping can fire.
                patience_counter = 0
                log.info("Epoch %d: patience counter reset after mining round.", epoch)
            new_n_batches = len(train_loader)
            if new_n_batches != prev_n_batches:
                # Dataset size can shift slightly if some issues lose candidates.
                # Rebuild the scheduler for the remaining epochs so cosine
                # annealing still terminates correctly.
                remaining_steps = new_n_batches * (tcfg["total_epochs"] - epoch + 1)
                scheduler = CosineAnnealingLR(
                    optimizer, T_max=max(remaining_steps, 1), eta_min=1e-6
                )

            if hnm_state["unlocked"] and hnm_state["first_epoch"] is not None:
                epochs_since = epoch - hnm_state["first_epoch"]
                mine_every   = int(hnm_cfg.get("mine_every_n_epochs", 1))
                if epochs_since % mine_every == 0:
                    # ── Patience reset ────────────────────────────────────────
                    patience_counter = 0
                    log.info(
                        "Epoch %d: patience counter reset after mining round "
                        "— model needs a fair window to learn new negatives.",
                        epoch,
                    )

                    # ── Optional LR reset ─────────────────────────────────────
                    # Controlled by hard_negative_mining.reset_lr_after_mining.
                    # lr_reset_fraction scales how far back toward the original
                    # LR we go — 1.0 = full reset to initial LRs, 0.5 = halfway.
                    # A fresh cosine schedule then decays from that peak over
                    # the remaining epochs, giving the model a higher-LR window
                    # to absorb the new hard negatives.
                    if hnm_cfg.get("reset_lr_after_mining", False):
                        frac = float(hnm_cfg.get("lr_reset_fraction", 1.0))
                        reset_lr_head     = tcfg["lr_head"]     * frac
                        reset_lr_backbone = tcfg["lr_backbone"] * frac
                        for group in optimizer.param_groups:
                            if group.get("name") == "new_layers":
                                group["lr"] = reset_lr_head
                            elif group.get("name") == "backbone":
                                group["lr"] = reset_lr_backbone
                        remaining_steps = len(train_loader) * (tcfg["total_epochs"] - epoch + 1)
                        scheduler = CosineAnnealingLR(
                            optimizer,
                            T_max   = max(remaining_steps, 1),
                            eta_min = 1e-6,
                        )
                        log.info(
                            "Epoch %d: LR reset to %.0f%% of initial — "
                            "head=%.2e  backbone=%.2e — "
                            "new cosine schedule over %d steps (%d remaining epochs).",
                            epoch, frac * 100,
                            reset_lr_head, reset_lr_backbone,
                            remaining_steps, tcfg["total_epochs"] - epoch + 1,
                        )

        # ── Unfreeze backbone after warmup ────────────────────────────────────
        if epoch == tcfg["unfreeze_epoch"] and not unfrozen:
            n = tcfg.get("n_unfreeze_layers", 2)
            log.info("Epoch %d: unfreezing top %d backbone layers.", epoch, n)
            model.unfreeze_top_layers(n)
            unfrozen = True

            # Rebuild optimizer to include newly unfrozen params
            optimizer = build_optimizer(
                model,
                lr_head      = tcfg["lr_head"],
                lr_backbone  = tcfg["lr_backbone"],
                weight_decay = tcfg["weight_decay"],
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max  = len(train_loader) * (tcfg["total_epochs"] - epoch + 1),
                eta_min = 1e-6,
            )
            frozen, total = model.frozen_param_count()
            log.info("After unfreeze: %d frozen / %d backbone params", frozen, total)

        # ── Train epoch ───────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{tcfg['total_epochs']:02d}", leave=False)

        for batch in pbar:
            batch = batch.to(device)

            logits, f_proj, c_proj = model(
                batch.feature_input_ids,
                batch.feature_attention_mask,
                batch.feature_chunk_mask,
                batch.class_input_ids,
                batch.class_attention_mask,
            )

            loss, breakdown = loss_fn(logits, f_proj, c_proj, batch.labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()

            epoch_loss  += breakdown["loss_total"]
            n_batches   += 1
            global_step += 1

            pbar.set_postfix({
                "loss": f"{breakdown['loss_total']:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.1e}"
            })

            if global_step % tcfg.get("eval_every_n_steps", 500) == 0:
                log.info(
                    "  step=%d  focal=%.4f  nce=%.4f  total=%.4f",
                    global_step,
                    breakdown["loss_focal"],
                    breakdown["loss_contrastive"],
                    breakdown["loss_total"],
                )

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ────────────────────────────────────────────────────────
        metrics = evaluate(model, val_loader, loss_fn, device)

        log.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  "
            "R@10=%.4f  R@50=%.4f  R@100=%.4f  NDCG@10=%.4f  MAP=%.4f  MRR=%.4f",
            epoch, tcfg["total_epochs"],
            avg_train_loss, metrics["val_loss"],
            metrics.get("recall@10",  0),
            metrics.get("recall@50",  0),
            metrics.get("recall@100", 0),
            metrics.get("ndcg@10",    0),
            metrics.get("map",        0),
            metrics.get("mrr",        0),
        )

        # Save checkpoint
        ckpt_path = output_dir / f"checkpoint_epoch{epoch:02d}.pt"
        torch.save({
            "epoch":          epoch,
            "global_step":    global_step,
            "model_state":    model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics":        metrics,
        }, ckpt_path)

        # Best model tracking (primary metric: Recall@50 — directly relevant to
        # the cross-encoder reranker decision: how much headroom does top-50 leave?)
        current_r50 = metrics.get("recall@50", 0.0)
        if current_r50 > best_recall_at_50:
            best_recall_at_50 = current_r50
            patience_counter  = 0
            best_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            log.info("  -> New best Recall@50=%.4f  saved to %s", current_r50, best_path)
        else:
            patience_counter += 1
            log.info(
                "  -> No improvement in R@50 (patience %d/%d)",
                patience_counter, tcfg["patience"],
            )
            if patience_counter >= tcfg["patience"]:
                log.info("Early stopping triggered.")
                break

        # Save metrics log
        metrics_log_path = output_dir / "metrics_log.jsonl"
        with open(metrics_log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, "train_loss": avg_train_loss,
                                **metrics}) + "\n")

    log.info("Training complete. Best Recall@50: %.4f", best_recall_at_50)

    # ── Final test evaluation ─────────────────────────────────────────────────
    # Runs automatically if "test_jsonl" is present in config.
    # Always uses the best checkpoint, never the last epoch's weights.
    test_path = config.get("test_jsonl")
    if test_path and Path(test_path).exists():
        best_path = output_dir / "best_model.pt"
        if not best_path.exists():
            log.warning("test_jsonl is set but best_model.pt not found — skipping test evaluation.")
        else:
            log.info("=" * 60)
            log.info("FINAL TEST EVALUATION")
            log.info("Checkpoint : %s", best_path)
            log.info("Test set   : %s", test_path)
            log.info("=" * 60)
 
            # Load best weights into the model (already on device)
            model.load_state_dict(torch.load(best_path, map_location=device))
 
            # Build test dataloader — same settings as val
            from dataset import ImpactSampleDataset
            test_ds = ImpactSampleDataset(
                jsonl_path       = test_path,
                tokenizer        = tokenizer,
                max_chunk_tokens = tcfg.get("max_chunk_tokens", 512),
                max_class_tokens = tcfg.get("max_class_tokens", 512),
                max_chunks       = tcfg.get("max_chunks", 8),
                label_smoothing  = 0.0,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size  = tcfg["batch_size"] * 2,
                shuffle     = False,
                collate_fn  = collate_fn,
                num_workers = tcfg.get("num_workers", 4),
                pin_memory  = True,
            )
            log.info("Test samples: %d", len(test_ds))
 
            test_metrics = evaluate(model, test_loader, loss_fn, device, desc="Testing")
 
            log.info(
                "TEST  val_loss=%.4f  R@5=%.4f  R@10=%.4f  R@20=%.4f  R@50=%.4f  "
                "R@80=%.4f  R@100=%.4f  NDCG@10=%.4f  MAP=%.4f  MRR=%.4f  AUC=%.4f",
                test_metrics["val_loss"],
                test_metrics.get("recall@5",   0),
                test_metrics.get("recall@10",  0),
                test_metrics.get("recall@20",  0),
                test_metrics.get("recall@50",  0),
                test_metrics.get("recall@80",  0),
                test_metrics.get("recall@100", 0),
                test_metrics.get("ndcg@10",    0),
                test_metrics.get("map",        0),
                test_metrics.get("mrr",        0),
                test_metrics.get("auc_roc",    0),
            )
 
            # Persist test results alongside training metrics
            test_results_path = output_dir / "test_metrics.json"
            with open(test_results_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "checkpoint": str(best_path),
                        "test_jsonl": test_path,
                        "best_val_recall@50": best_recall_at_50,
                        **{f"test_{k}": v for k, v in test_metrics.items()},
                    },
                    f, indent=2,
                )
            log.info("Test metrics saved to %s", test_results_path)
    elif test_path:
        log.warning("test_jsonl is set to '%s' but the file was not found — skipping.", test_path)
    else:
        log.info("No test_jsonl in config — skipping final test evaluation.")

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train impact score BiEncoder")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    train(config)


if __name__ == "__main__":
    main()
 









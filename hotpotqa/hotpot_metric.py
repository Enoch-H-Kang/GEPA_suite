# hotpot_metric.py
from __future__ import annotations

import re
import string
from typing import Optional

import dspy


# -------------------------
# Normalization helpers
# -------------------------
def normalize_title(t: str) -> str:
    # normalize underscores + whitespace + case
    t = t.replace("_", " ").strip()
    t = " ".join(t.split())
    return t.lower()


def _normalize_answer(s: str) -> str:
    """
    SQuAD/Hotpot-style normalization (lower, strip punctuation, remove articles, fix whitespace).
    HotpotQA reports EM and F1 under similar normalization conventions. :contentReference[oaicite:14]{index=14}
    """
    if s is None:
        return ""
    s = s.lower()

    # remove punctuation
    s = "".join(ch for ch in s if ch not in set(string.punctuation))

    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)

    # white space fix
    s = " ".join(s.split())
    return s


# -------------------------
# Scores
# -------------------------
def hotpot_em_score(gold, pred, trace=None) -> float:
    gold_a = _normalize_answer(getattr(gold, "answer", "") or "")
    pred_a = _normalize_answer(getattr(pred, "answer", "") or "")
    return 1.0 if gold_a == pred_a and gold_a != "" else 0.0


def hotpot_doc_recall(gold, pred) -> float:
    gold_titles = {normalize_title(t) for t in getattr(gold, "titles", [])}
    got_titles = {normalize_title(t) for t in getattr(pred, "titles", [])}
    if not gold_titles:
        return 0.0
    return len(gold_titles & got_titles) / len(gold_titles)


# -------------------------
# Feedback (stage-aware)
# -------------------------
def _stage_from_pred_name(pred_name: Optional[str]) -> str:
    """
    Decide which retrieval stage the feedback should reference.

    - hop1 stage: summarize_hop1, create_query_hop2
    - hop2 stage: summarize_hop2, final_answer, or program-level feedback
    """
    if not pred_name:
        return "hop2"
    name = pred_name.lower()
    if "summarize_hop1" in name or "create_query_hop2" in name:
        return "hop1"
    return "hop2"


def hotpot_feedback_text(gold, pred, pred_name: Optional[str] = None) -> str:
    """
    GEPA HotpotQA feedback described in the paper:
    "identifies the set of relevant documents remaining to be retrieved at each stage" :contentReference[oaicite:15]{index=15}
    """
    gold_list = [normalize_title(t) for t in getattr(gold, "titles", [])]

    hop1_titles = {normalize_title(t) for t in getattr(pred, "titles_hop1", [])}
    hop2_titles = {normalize_title(t) for t in getattr(pred, "titles_hop2", [])}
    stage = _stage_from_pred_name(pred_name)

    if stage == "hop1":
        retrieved = hop1_titles
        stage_label = "After hop1 retrieval"
    else:
        retrieved = hop1_titles | hop2_titles
        stage_label = "After hop2 retrieval"

    missing = [t for t in gold_list if t not in retrieved]

    # Keep it simple & aligned with the paper’s description.
    return (
        f"{stage_label}, remaining gold supporting docs to retrieve: {missing}\n"
        f"Gold supporting docs (all): {gold_list}\n"
        f"Retrieved titles so far: {sorted(retrieved)}"
    )


# -------------------------
# GEPA metric (must accept 5 args)
# -------------------------
def hotpot_metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    DSPy GEPA requires metric(gold, pred, trace, pred_name, pred_trace). :contentReference[oaicite:16]{index=16}
    Return score + feedback; score must remain consistent across pred_name requests.
    """
    score = hotpot_em_score(gold, pred, trace=None)
    feedback = hotpot_feedback_text(gold, pred, pred_name=pred_name)
    return dspy.Prediction(score=score, feedback=feedback)


# -------------------------
# Learning curve helpers
# -------------------------
def build_best_so_far_curve_from_detailed_results(detailed_results, baseline_score: float | None = None):
    """
    Extract a best-so-far curve vs "rollouts" (= metric calls).
    DSPy exposes `detailed_results` when track_stats=True. :contentReference[oaicite:17]{index=17}

    Different DSPy versions may expose different attribute names; we use fallbacks.
    """
    scores = list(getattr(detailed_results, "val_aggregate_scores", []) or [])
    eval_counts = getattr(detailed_results, "discovery_eval_counts", None)

    if eval_counts is None:
        # Fallback: candidates in discovery order.
        eval_counts = list(range(1, len(scores) + 1))
    else:
        eval_counts = list(eval_counts)

    triples = [(i, int(eval_counts[i]), float(scores[i])) for i in range(len(scores))]
    triples.sort(key=lambda x: x[1])

    curve = []
    best = float("-inf")

    if baseline_score is not None:
        best = float(baseline_score)
        curve.append(dict(rollouts=0, candidate_idx=None, candidate_val_score=baseline_score, best_val_score=best))

    for cand_idx, rollouts, cand_score in triples:
        if cand_score > best:
            best = cand_score
        curve.append(
            dict(
                rollouts=int(rollouts),
                candidate_idx=int(cand_idx),
                candidate_val_score=float(cand_score),
                best_val_score=float(best),
            )
        )
    return curve

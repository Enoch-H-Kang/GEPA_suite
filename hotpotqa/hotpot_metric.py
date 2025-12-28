# hotpot_metric.py
from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization."""
    s = s.lower()

    # remove punctuation
    s = "".join(ch for ch in s if ch not in set(string.punctuation))

    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)

    # white space fix
    s = " ".join(s.split())
    return s


def hotpot_exact_match(example, pred, trace=None):
    gold = normalize_answer(example.answer)
    got = normalize_answer(getattr(pred, "answer", ""))

    em = 1.0 if gold == got else 0.0
    if trace is not None:
        return em >= 1.0
    return em


def hotpot_f1(example, pred, trace=None):
    gold = normalize_answer(example.answer)
    got = normalize_answer(getattr(pred, "answer", ""))

    gold_toks = gold.split() if gold else []
    got_toks = got.split() if got else []

    if len(gold_toks) == 0 and len(got_toks) == 0:
        f1 = 1.0
    elif len(gold_toks) == 0 or len(got_toks) == 0:
        f1 = 0.0
    else:
        common = Counter(gold_toks) & Counter(got_toks)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(got_toks)
            recall = num_same / len(gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)

    if trace is not None:
        return f1 >= 1.0
    return f1


# --- Retrieval-side helpers (for debugging + GEPA-style feedback) ---

def normalize_title(t: str) -> str:
    return t.strip()


def hotpot_doc_recall(example, pred, trace=None):
    gold = {normalize_title(t) for t in example.titles}
    got = {normalize_title(t) for t in getattr(pred, "titles", [])}
    recall = len(gold & got) / max(1, len(gold))
    if trace is not None:
        return recall >= 1.0
    return recall


def _retrieval_feedback(gold_titles: list[str], retrieved_titles: list[str]) -> str:
    gold = [normalize_title(t) for t in gold_titles]
    got = {normalize_title(t) for t in retrieved_titles}

    correct = [t for t in gold if t in got]
    missing = [t for t in gold if t not in got]
    return (
        f"Correct gold docs retrieved so far: {correct}\n"
        f"Gold docs still missing: {missing}"
    )


def hotpot_feedback_by_stage(example, pred) -> dict[str, str]:
    """
    Stage-wise feedback that matches GEPA’s description for HotpotQA:
    'identify the set of relevant documents remaining to be retrieved at each stage'
    and provide as feedback to modules at that stage. 
    """
    hop1 = getattr(pred, "titles_hop1", []) or []
    hop2 = getattr(pred, "titles_hop2", []) or []
    union_2 = list(dict.fromkeys(list(hop1) + list(hop2)))  # keep order, unique

    fb_hop1 = _retrieval_feedback(example.titles, hop1)
    fb_hop2 = _retrieval_feedback(example.titles, union_2)

    # Answer feedback is useful for the final module
    em = hotpot_exact_match(example, pred)
    f1 = hotpot_f1(example, pred)
    fb_answer = (
        f"Gold answer: {example.answer}\n"
        f"Predicted answer: {getattr(pred, 'answer', '')}\n"
        f"ExactMatch: {em:.3f}  F1: {f1:.3f}"
    )

    # Map feedback to your module names (these match hotpot_program.py below)
    return {
        "summarize_hop1": fb_hop1,
        "create_query_hop2": fb_hop1,
        "summarize_hop2": fb_hop2,
        "answer_question": fb_answer,
    }

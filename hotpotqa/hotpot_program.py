# hotpot_program.py
from __future__ import annotations

from typing import Dict, List, Tuple

import dspy


def _title_from_doc(doc: str) -> str:
    # "Title | Abstract"
    return doc.split(" | ", 1)[0].strip()


def _format_passages(passages: List[str], max_chars: int = 14000) -> str:
    """
    Soft input guardrail: keep retrieved passages bounded so you don't
    blow past the paper's 16,384 token context window. :contentReference[oaicite:9]{index=9}

    This is NOT "capping your reasoning"; it's input-budget hygiene.
    """
    out = []
    total = 0
    for i, p in enumerate(passages, 1):
        chunk = f"[{i}] {p}"
        if total + len(chunk) > max_chars:
            break
        out.append(chunk)
        total += len(chunk)
    return "\n".join(out)


def _dedupe_keep_best_score(title_scores: List[Tuple[str, float]]) -> List[str]:
    best: Dict[str, float] = {}
    for t, s in title_scores:
        if t not in best or s > best[t]:
            best[t] = s
    return [t for t, _ in sorted(best.items(), key=lambda x: x[1], reverse=True)]


class SummarizeDocsSig(dspy.Signature):
    """Summarize retrieved wiki abstracts for multi-hop QA."""
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc="Prior hop summary or empty string.")
    passages: str = dspy.InputField(desc="Numbered 'Title | abstract' passages.")
    summary: str = dspy.OutputField(desc="Concise summary emphasizing facts useful for the next hop.")


class QueryHop2Sig(dspy.Signature):
    """Write the second-hop query given the question and first-hop summary."""
    question: str = dspy.InputField()
    summary_1: str = dspy.InputField()
    query: str = dspy.OutputField(desc="A short search query string (not an explanation).")


class FinalAnswerSig(dspy.Signature):
    """Answer the question from the two hop summaries."""
    question: str = dspy.InputField()
    summary_1: str = dspy.InputField()
    summary_2: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Final answer only (yes/no or short span).")


class HotpotMultiHopQA(dspy.Module):
    """
    HotpotQA version of HoVerMultiHop:
      - hop1 retrieve + summarize
      - hop2 query + retrieve + summarize
      - final: answer (instead of hop3 query) :contentReference[oaicite:10]{index=10}
    """
    def __init__(self, search_fn, k_per_hop: int = 5):
        super().__init__()
        self.search_fn = search_fn
        self.k = k_per_hop

        self.summarize_hop1 = dspy.ChainOfThought(SummarizeDocsSig)
        self.create_query_hop2 = dspy.Predict(QueryHop2Sig)
        self.summarize_hop2 = dspy.ChainOfThought(SummarizeDocsSig)

        self.final_answer = dspy.ChainOfThought(FinalAnswerSig)

    def forward(self, question: str):
        title_scores: List[Tuple[str, float]] = []

        # Hop 1
        docs1, scores1 = self.search_fn(question, k=self.k)
        titles1 = [_title_from_doc(d) for d in docs1]
        title_scores += list(zip(titles1, scores1))

        summary_1 = self.summarize_hop1(
            question=question,
            context="",
            passages=_format_passages(docs1),
        ).summary

        # Hop 2
        query2 = self.create_query_hop2(question=question, summary_1=summary_1).query
        docs2, scores2 = self.search_fn(query2, k=self.k)
        titles2 = [_title_from_doc(d) for d in docs2]
        title_scores += list(zip(titles2, scores2))

        summary_2 = self.summarize_hop2(
            question=question,
            context=summary_1,
            passages=_format_passages(docs2),
        ).summary

        # Final answer (replaces hop 3 query)
        answer = self.final_answer(
            question=question,
            summary_1=summary_1,
            summary_2=summary_2,
        ).answer

        titles_ranked = _dedupe_keep_best_score(title_scores)

        return dspy.Prediction(
            answer=answer,
            titles=titles_ranked,
            titles_hop1=titles1,
            titles_hop2=titles2,
            summary_1=summary_1,
            summary_2=summary_2,
            query2=query2,
        )

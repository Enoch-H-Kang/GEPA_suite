# hotpot_program.py
from __future__ import annotations

from typing import Dict, List, Tuple
import dspy


def _format_passages(passages: List[str], max_chars: int = 6000) -> str:
    """
    Keep bounded to avoid blowing context. Raise max_chars if you want.
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


def _title_from_doc(doc: str) -> str:
    # "Title | Abstract"
    return doc.split(" | ", 1)[0].strip()


def _dedupe_keep_best_score(title_scores: List[Tuple[str, float]]) -> List[str]:
    best: Dict[str, float] = {}
    for t, s in title_scores:
        if t not in best or s > best[t]:
            best[t] = s
    return [t for t, _ in sorted(best.items(), key=lambda x: x[1], reverse=True)]


class SummarizeDocsSig(dspy.Signature):
    """Summarize retrieved wiki abstracts into a short evidence summary useful for the next step."""
    question: str = dspy.InputField()
    passages: str = dspy.InputField(desc="Numbered 'Title | abstract' passages from Wikipedia.")
    summary: str = dspy.OutputField(desc="Concise summary of what we learned and what is still missing.")


class QueryHop2Sig(dspy.Signature):
    """Write the second-hop query given the question and first-hop summary."""
    question: str = dspy.InputField()
    summary_1: str = dspy.InputField()
    query: str = dspy.OutputField(desc="A focused search query for the missing document(s).")


class AnswerSig(dspy.Signature):
    """Answer the question given the summaries from hop1 and hop2."""
    question: str = dspy.InputField()
    summary_1: str = dspy.InputField()
    summary_2: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Final answer. Prefer short, direct answers.")


class HotpotMultiHopQA(dspy.Module):
    """
    2-hop retrieval + answer:
      Hop1: query = question -> retrieve docs -> summarize
      Hop2: generate query2 from (question, summary1) -> retrieve -> summarize
      Final: answer from (question, summary1, summary2)

    This matches GEPA’s description: modify last hop of HoVerMultiHop to answer instead of generating another query. 
    """
    def __init__(self, search_fn, k_per_hop: int = 5, max_passage_chars: int = 6000):
        super().__init__()
        self.search_fn = search_fn
        self.k = k_per_hop
        self.max_passage_chars = max_passage_chars

        # 2 doc summary modules
        self.summarize_hop1 = dspy.ChainOfThought(SummarizeDocsSig)
        self.summarize_hop2 = dspy.ChainOfThought(SummarizeDocsSig)

        # 1 query writer module (hop2)
        self.create_query_hop2 = dspy.Predict(QueryHop2Sig)

        # final answer module
        self.answer_question = dspy.ChainOfThought(AnswerSig)

    def forward(self, question: str):
        title_scores: List[Tuple[str, float]] = []

        # Hop 1: query = question
        docs1, scores1 = self.search_fn(question, k=self.k)
        titles_hop1 = [_title_from_doc(d) for d in docs1]
        title_scores += list(zip(titles_hop1, scores1))

        summary_1 = self.summarize_hop1(
            question=question,
            passages=_format_passages(docs1, max_chars=self.max_passage_chars),
        ).summary

        # Hop 2: query2 from (question, summary_1)
        query2 = self.create_query_hop2(question=question, summary_1=summary_1).query
        docs2, scores2 = self.search_fn(query2, k=self.k)
        titles_hop2 = [_title_from_doc(d) for d in docs2]
        title_scores += list(zip(titles_hop2, scores2))

        summary_2 = self.summarize_hop2(
            question=question,
            passages=_format_passages(docs2, max_chars=self.max_passage_chars),
        ).summary

        # Final: answer
        ans = self.answer_question(question=question, summary_1=summary_1, summary_2=summary_2).answer

        titles_ranked = _dedupe_keep_best_score(title_scores)

        return dspy.Prediction(
            answer=ans,
            titles=titles_ranked,
            titles_hop1=titles_hop1,
            titles_hop2=titles_hop2,
            summary_1=summary_1,
            summary_2=summary_2,
            query2=query2,
        )

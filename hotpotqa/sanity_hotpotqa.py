# sanity_hotpotqa.py
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import dspy

from hotpot_data import load_hotpotqa_splits
from hotpot_metric import hotpot_em_score, hotpot_doc_recall, hotpot_feedback_text
from hotpot_program import HotpotMultiHopQA
from wiki_retriever import build_or_load_bm25, make_search_fn


def configure_dspy_lm_from_vllm(
    *,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int | None = None,
    max_tokens: int = 8192,
    cache: bool = False,
    num_retries: int = 3,
):
    """
    vLLM OpenAI-compatible endpoint configuration.
    Defaults match the GEPA paper decoding settings for Qwen3-8B (except top_k, which is optional
    because some client stacks reject it).
    """
    lm_kwargs = dict(
        api_base=api_base,
        api_key=api_key,
        model_type="chat",
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        cache=cache,
        num_retries=num_retries,
    )
    if top_k is not None:
        # Only include if your DSPy/LiteLLM/vLLM stack supports it.
        lm_kwargs["top_k"] = top_k

    lm = dspy.LM(f"openai/{model}", **lm_kwargs)
    dspy.configure(lm=lm)


def main():
    ap = argparse.ArgumentParser()

    # vLLM endpoint
    ap.add_argument("--api_base", type=str, default=os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api_key", type=str, default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--model", type=str, default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B"))

    # Decoding (defaults align with your GEPA config)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=None, help="Only set if your stack accepts top_k (e.g., 20).")
    ap.add_argument("--max_tokens", type=int, default=8192)

    # Data + retrieval
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--n_examples", type=int, default=3)
    ap.add_argument("--k_per_hop", type=int, default=5)
    ap.add_argument("--retriever_threads", type=int, default=2)

    ap.add_argument("--work_dir", type=str, default=os.environ.get("WORK", "/tmp/hotpot_workdir"))
    ap.add_argument("--dump_jsonl", type=str, default=None, help="Optional path to write per-example results.jsonl")

    args = ap.parse_args()

    configure_dspy_lm_from_vllm(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        cache=False,
        num_retries=3,
    )

    work = Path(args.work_dir)
    data_dir = work / "data"
    wiki_dir = data_dir / "wiki17"
    index_dir = data_dir / "wiki17_bm25"

    corpus, retriever, stemmer = build_or_load_bm25(wiki_dir=wiki_dir, index_dir=index_dir)
    search_fn = make_search_fn(corpus, retriever, stemmer, n_threads=args.retriever_threads)

    train, dev, test = load_hotpotqa_splits(seed=args.seed, n_train=150, n_dev=300, n_test=300)
    split_map = {"train": train, "dev": dev, "test": test}
    examples = split_map[args.split][: args.n_examples]

    prog = HotpotMultiHopQA(search_fn=search_fn, k_per_hop=args.k_per_hop)

    dump_path = Path(args.dump_jsonl) if args.dump_jsonl else None
    if dump_path:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        f_dump = dump_path.open("w", encoding="utf-8")
    else:
        f_dump = None

    try:
        for i, ex in enumerate(examples, 1):
            t0 = time.time()
            pred = prog(question=ex.question)
            dt = time.time() - t0

            em = hotpot_em_score(ex, pred)
            doc_rec = hotpot_doc_recall(ex, pred)

            fb_hop1 = hotpot_feedback_text(ex, pred, pred_name="summarize_hop1")
            fb_hop2 = hotpot_feedback_text(ex, pred, pred_name="summarize_hop2")
            fb_query2 = hotpot_feedback_text(ex, pred, pred_name="create_query_hop2")
            fb_final = hotpot_feedback_text(ex, pred, pred_name="final_answer")

            print(f"\n=== EXAMPLE {i}/{len(examples)} ===")
            print("Q:", ex.question)
            print("GOLD ANSWER:", ex.answer)
            print("PRED  ANSWER:", pred.answer)
            print(f"EM: {em:.3f} | DOC-RECALL: {doc_rec:.3f} | LATENCY: {dt:.2f}s")

            print("\nGOLD TITLES:", ex.titles)
            print("HOP1 TITLES:", pred.titles_hop1)
            print("HOP2 TITLES:", pred.titles_hop2)
            print("RANKED TITLES (top 10):", pred.titles[:10])

            print("\nQUERY2:", pred.query2)

            # Feedback at different stages (this is what GEPA will “see” depending on pred_name)
            print("\n[FEEDBACK] after hop1 (summarize_hop1):\n", fb_hop1)
            print("\n[FEEDBACK] for hop2 query writer (create_query_hop2):\n", fb_query2)
            print("\n[FEEDBACK] after hop2 (summarize_hop2):\n", fb_hop2)
            print("\n[FEEDBACK] for final answer stage (final_answer):\n", fb_final)

            if f_dump:
                rec = {
                    "idx": i,
                    "question": ex.question,
                    "gold_answer": ex.answer,
                    "pred_answer": pred.answer,
                    "em": float(em),
                    "doc_recall": float(doc_rec),
                    "gold_titles": list(ex.titles),
                    "titles_hop1": list(pred.titles_hop1),
                    "titles_hop2": list(pred.titles_hop2),
                    "titles_ranked": list(pred.titles),
                    "query2": pred.query2,
                    "summary_1": pred.summary_1,
                    "summary_2": pred.summary_2,
                    "feedback_hop1": fb_hop1,
                    "feedback_query2": fb_query2,
                    "feedback_hop2": fb_hop2,
                    "feedback_final": fb_final,
                    "latency_sec": float(dt),
                }
                f_dump.write(json.dumps(rec, ensure_ascii=False) + "\n")

    finally:
        if f_dump:
            f_dump.close()
            print(f"\nWrote JSONL dump to: {dump_path}")


if __name__ == "__main__":
    main()

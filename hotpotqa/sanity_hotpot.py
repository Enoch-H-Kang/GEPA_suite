# sanity_hotpot.py
import os
import time
import dspy

from hotpot_data import load_hotpot_splits
from hotpot_metric import hotpot_exact_match, hotpot_f1, hotpot_doc_recall, hotpot_feedback_by_stage
from hotpot_program import HotpotMultiHopQA
from wiki_retriever import build_or_load_bm25, make_search_fn


def configure_dspy_lm_from_vllm():
    """
    Configure DSPy to use a vLLM OpenAI-compatible server.

    GEPA uses Qwen3-8B with temp=0.6, top_p=0.95, top_k=20. 
    """
    api_base = os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B")

    temperature = float(os.environ.get("VLLM_TEMPERATURE", "0.6"))
    top_p = float(os.environ.get("VLLM_TOP_P", "0.95"))
    top_k = int(os.environ.get("VLLM_TOP_K", "20"))
    max_tokens = int(os.environ.get("VLLM_MAX_TOKENS", "10000"))

    lm = dspy.LM(
        f"openai/{model}",
        api_base=api_base,
        api_key=api_key,
        model_type="chat",
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,        # remove if your DSPy/LiteLLM build complains
        max_tokens=max_tokens,
        cache=False,
        num_retries=3,
    )
    dspy.configure(lm=lm)


def main():
    configure_dspy_lm_from_vllm()

    WORK = os.environ.get("WORK", "/tmp/hotpot_workdir")
    wiki_dir = os.path.join(WORK, "wiki17")
    index_dir = os.path.join(WORK, "wiki17_bm25")

    corpus, retriever, stemmer = build_or_load_bm25(wiki_dir=wiki_dir, index_dir=index_dir)
    search_fn = make_search_fn(corpus, retriever, stemmer, n_threads=1)

    # HotpotQA pool (we slice 150/300/300 like GEPA) 
    train, dev, test = load_hotpot_splits(
        seed=0,
        n_train=150, n_dev=300, n_test=300,
        subset=os.environ.get("HOTPOT_SUBSET", "fullwiki"),
        pool_split=os.environ.get("HOTPOT_POOL_SPLIT", "train"),
    )

    prog = HotpotMultiHopQA(search_fn, k_per_hop=5, max_passage_chars=6000)

    for ex in dev[:3]:
        t0 = time.time()
        pred = prog(question=ex.question)
        dt = time.time() - t0

        em = hotpot_exact_match(ex, pred)
        f1 = hotpot_f1(ex, pred)
        docrec = hotpot_doc_recall(ex, pred)

        print("QUESTION:", ex.question)
        print("GOLD ANSWER:", ex.answer)
        print("PRED ANSWER:", pred.answer)
        print(f"EM: {em:.3f}  F1: {f1:.3f}  DOC-RECALL: {docrec:.3f}")
        print("GOLD TITLES:", ex.titles)
        print("PRED TITLES (top 10):", pred.titles[:10])
        print("QUERY2:", pred.query2)
        print(f"LATENCY: {dt:.2f}s")

        fbs = hotpot_feedback_by_stage(ex, pred)
        print("FEEDBACK summarize_hop1:\n", fbs["summarize_hop1"])
        print("FEEDBACK create_query_hop2:\n", fbs["create_query_hop2"])
        print("FEEDBACK summarize_hop2:\n", fbs["summarize_hop2"])
        print("FEEDBACK answer_question:\n", fbs["answer_question"])
        print("=" * 100)


if __name__ == "__main__":
    main()

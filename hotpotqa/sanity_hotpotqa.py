# sanity_hotpotqa.py
import os
import time

import dspy

from hotpot_data import load_hotpotqa_splits
from hotpot_metric import hotpot_em_score, hotpot_feedback_text
from hotpot_program import HotpotMultiHopQA
from wiki_retriever import build_or_load_bm25, make_search_fn


def configure_dspy_lm_from_vllm():
    """
    Paper settings for Qwen3-8B: temp=0.6, top-p=0.95, top-k=20; ctx up to 16384. :contentReference[oaicite:18]{index=18}
    """
    api_base = os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B")

    lm = dspy.LM(
        f"openai/{model}",
        api_base=api_base,
        api_key=api_key,
        model_type="chat",
        temperature=0.6,
        top_p=0.95,
        # top_k=20,  # enable only if your client/server supports passing it through
        max_tokens=8192,    # not tiny; still bounded by model context window
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
    search_fn = make_search_fn(corpus, retriever, stemmer, n_threads=2)

    train, dev, test = load_hotpotqa_splits(seed=0, n_train=150, n_dev=300, n_test=300)

    prog = HotpotMultiHopQA(search_fn=search_fn, k_per_hop=5)

    for ex in dev[:3]:
        t0 = time.time()
        pred = prog(question=ex.question)
        dt = time.time() - t0

        print("Q:", ex.question)
        print("GOLD ANSWER:", ex.answer)
        print("PRED ANSWER:", pred.answer)
        print("EM:", hotpot_em_score(ex, pred))
        print("GOLD TITLES:", ex.titles)
        print("HOP1 TITLES:", pred.titles_hop1)
        print("HOP2 TITLES:", pred.titles_hop2)
        print("FEEDBACK (for create_query_hop2):\n", hotpot_feedback_text(ex, pred, pred_name="create_query_hop2"))
        print(f"LATENCY: {dt:.2f}s")
        print("=" * 100)


if __name__ == "__main__":
    main()

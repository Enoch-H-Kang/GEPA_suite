# sanity_hover.py
import os
import time

import dspy

from hover_data import load_hover_splits
from hover_metric import hover_recall_score, hover_feedback_text
from hover_program import HoverMultiHop
from wiki_retriever import build_or_load_bm25, make_search_fn


def configure_dspy_lm_from_vllm():
    """
    Paper settings for Qwen3-8B: temp=0.6, top-p=0.95, top-k=20; ctx up to 16384.

    NOTE:
    - top_k is not part of the official OpenAI schema; vLLM supports it, but some clients won't forward it.
      If you get schema errors, set top_k on the vLLM server side and leave it commented out here.
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

    # Put these under $WORK on cluster; local testing can use anything.
    WORK = os.environ.get("WORK", "/tmp/hover_workdir")
    wiki_dir = os.path.join(WORK, "wiki17")
    index_dir = os.path.join(WORK, "wiki17_bm25")

    # BM25 over wiki abstracts (DSPy tutorial style)
    corpus, retriever, stemmer = build_or_load_bm25(wiki_dir=wiki_dir, index_dir=index_dir)
    search_fn = make_search_fn(corpus, retriever, stemmer, n_threads=2)

    # GEPA paper uses 150/300/300 for HoVer
    # If you updated your loader to support max_num_hops, this will use it; otherwise it falls back.
    try:
        train, dev, test = load_hover_splits(seed=0, n_train=150, n_dev=300, n_test=300, max_num_hops=3)
    except TypeError:
        train, dev, test = load_hover_splits(seed=0, n_train=150, n_dev=300, n_test=300)

    prog = HoverMultiHop(search_fn=search_fn, k_per_hop=5)

    for ex in dev[:3]:
        t0 = time.time()
        pred = prog(claim=ex.claim)
        dt = time.time() - t0

        print("CLAIM:", ex.claim)
        print("GOLD TITLES:", ex.titles)

        # Program outputs (top-level ranked titles)
        print("PRED TITLES (top 10):", getattr(pred, "titles", [])[:10])

        # Hop-by-hop debug info (if your program returns them)
        if hasattr(pred, "titles_hop1"):
            print("HOP1 TITLES:", pred.titles_hop1)
        if hasattr(pred, "titles_hop2"):
            print("HOP2 TITLES:", pred.titles_hop2)
        if hasattr(pred, "titles_hop3"):
            print("HOP3 TITLES:", pred.titles_hop3)

        # Queries/summaries (useful for debugging query writers + summaries)
        if hasattr(pred, "query2"):
            print("QUERY2:", pred.query2)
        if hasattr(pred, "query3"):
            print("QUERY3:", pred.query3)
        if hasattr(pred, "summary_1"):
            print("SUMMARY1:", pred.summary_1)
        if hasattr(pred, "summary_2"):
            print("SUMMARY2:", pred.summary_2)

        # Metric + feedback
        print("RECALL:", hover_recall_score(ex, pred))

        # Try to request feedback "for" a module (hotpot-style); fall back if signature doesn't support it.
        try:
            fb = hover_feedback_text(ex, pred, pred_name="create_query_hop2")
            print("FEEDBACK (for create_query_hop2):\n", fb)
        except TypeError:
            fb = hover_feedback_text(ex, pred)
            print("FEEDBACK:\n", fb)

        print(f"LATENCY: {dt:.2f}s")
        print("=" * 100)


if __name__ == "__main__":
    main()

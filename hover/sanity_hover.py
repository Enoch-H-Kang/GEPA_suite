# sanity_hover.py
import os
import time
import dspy

from hover_data import load_hover_splits
from wiki_retriever import build_or_load_bm25, make_search_fn
from hover_program import HoverMultiHop
from hover_metric import hover_recall, hover_feedback_text


def configure_dspy_lm_from_vllm():
    """
    Configure DSPy to use a vLLM OpenAI-compatible server.

    Start server separately:
      vllm serve Qwen/Qwen3-8B --port 8000 --api-key EMPTY --generation-config vllm ...
    """
    api_base = os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B")

    # NOTE:
    # - model_type="chat" is usually correct for OpenAI-compatible chat completions.
    # - top_k is not part of OpenAI's official schema, but vLLM supports it; whether it is forwarded
    #   depends on your DSPy/LiteLLM version. If it errors, remove top_k from here and control it
    #   at the server side or via extra-body kwargs.
    lm = dspy.LM(
        f"openai/{model}",
        api_base=api_base,
        api_key=api_key,
        model_type="chat",
        temperature=0.6,
        top_p=0.95,
        top_k=20,          # remove if your client complains
        max_tokens=10000,    # keep small for sanity test
        cache=False,       # sanity: avoid confusing cache interactions
        num_retries=3,
    )

    # Your DSPy error message suggested this exact function:
    dspy.configure(lm=lm)


def make_search_with_scores(search_only):
    """
    Your HoverMultiHop expects: search(query, k) -> (docs, scores).
    If your make_search_fn already returns (docs, scores), delete this wrapper.
    """
    def search_with_scores(query: str, k: int = 5):
        out = search_only(query, k)
        # If your search_only returns docs only:
        if isinstance(out, list):
            docs = out
            scores = [float(k - i) for i in range(len(docs))]
            return docs, scores
        # If your search_only already returns (docs, scores):
        return out
    return search_with_scores


def main():
    # 0) Configure LM (fixes "No LM is loaded" error)
    configure_dspy_lm_from_vllm()

    # -----------------------
    # Paths: put these under $WORK on the cluster; for local testing any path is fine
    # -----------------------
    WORK = os.environ.get("WORK", "/tmp/hover_workdir")
    wiki_dir = os.path.join(WORK, "wiki17")
    index_dir = os.path.join(WORK, "wiki17_bm25")

    # 1) Retriever (DSPy tutorial style)
    corpus, retriever, stemmer = build_or_load_bm25(
        wiki_dir=wiki_dir,
        index_dir=index_dir,
    )

    search_only = make_search_fn(corpus, retriever, stemmer, n_threads=1)
    search_fn = make_search_with_scores(search_only)

    # 2) Data splits (GEPA paper uses 150/300/300)
    train, dev, test = load_hover_splits(seed=0, n_train=150, n_dev=300, n_test=300)

    # 3) Program
    prog = HoverMultiHop(search_fn, k_per_hop=5)

    # 4) Run a few examples
    for ex in dev[:3]:
        t0 = time.time()
        pred = prog(claim=ex.claim)
        dt = time.time() - t0

        print("CLAIM:", ex.claim)
        print("GOLD TITLES:", ex.titles)
        print("PRED TITLES (top 10):", pred.titles[:10])
        print("RECALL:", hover_recall(ex, pred))
        print(f"LATENCY: {dt:.2f}s")
        print("FEEDBACK:\n", hover_feedback_text(ex, pred))
        print("=" * 100)


if __name__ == "__main__":
    main()

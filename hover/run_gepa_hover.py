'''
# Example concept (adjust for your environment / ROCm):
vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 --port 8000 \
  --api-key EMPTY \
  --max-model-len 16384

#In another terminal:

export VLLM_API_BASE="http://127.0.0.1:8000/v1"
export VLLM_API_KEY="EMPTY"
export VLLM_MODEL="Qwen/Qwen3-8B"

python run_gepa_hover.py \
  --work_dir "$WORK/gepa_hover/data" \
  --log_dir "$WORK/gepa_hover/logs" \
  --num_threads 32 \
  --retriever_threads 8 \
  --max_metric_calls 10000


'''

# run_gepa_hover.py
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import dspy
from dspy.evaluate import Evaluate

from hover_data import load_hover_splits
from hover_metric import (
    hover_recall_score,
    hover_metric_with_feedback,
    build_best_so_far_curve_from_detailed_results,
)
from hover_program import HoverMultiHop
from wiki_retriever import build_or_load_bm25, make_search_fn


def configure_dspy_lm_from_vllm():
    """
    Matches the GEPA paper's Qwen3-8B decoding settings:
    temperature=0.6, top_p=0.95, top_k=20; context up to 16384. :contentReference[oaicite:15]{index=15}

    NOTE: top_k is not part of the official OpenAI schema; if your stack rejects it,
    set top_k at the vLLM server-side generation config instead.
    """
    api_base = os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B")

    lm_kwargs = dict(
        api_base=api_base,
        api_key=api_key,
        model_type="chat",
        temperature=0.6,
        top_p=0.95,
        # top_k=20,   # uncomment only if your client/server accepts it
        # No max_tokens limit - let vLLM dynamically allocate based on available context window
        cache=False,
        num_retries=10,  # Increased retries for connection stability
        timeout=300,  # 5 minute timeout for long generations
    )

    lm = dspy.LM(f"openai/{model}", **lm_kwargs)
    dspy.configure(lm=lm)
    return lm


def write_curve_csv(curve, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rollouts", "candidate_idx", "candidate_val_score", "best_val_score"])
        w.writeheader()
        for row in curve:
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k_per_hop", type=int, default=5)

    # Rollout budget: set this to the GEPA paper's rollout budget for HoVer if you want strict matching.
    # In DSPy GEPA, rollouts ~= metric calls, controlled by max_metric_calls. :contentReference[oaicite:16]{index=16}
    ap.add_argument("--max_metric_calls", type=int, default=6858)

    # Parallelism
    ap.add_argument("--num_threads", type=int, default=32)
    ap.add_argument("--retriever_threads", type=int, default=4)

    # Data paths
    ap.add_argument("--work_dir", type=str, default=os.environ.get("WORK", "/tmp/hover_workdir"))
    ap.add_argument("--log_dir", type=str, default=None)

    args = ap.parse_args()

    # 1) Configure LM
    lm = configure_dspy_lm_from_vllm()

    # 2) Retriever
    work = Path(args.work_dir)
    wiki_dir = work / "wiki17"
    index_dir = work / "wiki17_bm25"

    corpus, retriever, stemmer = build_or_load_bm25(wiki_dir=wiki_dir, index_dir=index_dir)
    search_fn = make_search_fn(corpus, retriever, stemmer, n_threads=args.retriever_threads)

    # 3) Data
    train, dev, test = load_hover_splits(seed=args.seed, n_train=150, n_dev=300, n_test=300)

    # 4) Program
    student = HoverMultiHop(search_fn=search_fn, k_per_hop=args.k_per_hop)

    # 5) Baseline dev eval (float score)
    evaluator_dev = Evaluate(devset=dev, metric=hover_recall_score, num_threads=args.num_threads, display_progress=True)
    baseline_dev = evaluator_dev(student).score
    print(f"[BASELINE] dev score: {baseline_dev:.2f}")

    # 6) GEPA (GEPA only => use_merge=False)
    # - metric must accept 5 args. :contentReference[oaicite:17]{index=17}
    # - log_dir enables resume. :contentReference[oaicite:18]{index=18}
    # - track_stats provides detailed_results for learning curve extraction. :contentReference[oaicite:19]{index=19}
    gepa = dspy.GEPA(
        metric=hover_metric_with_feedback,
        reflection_lm=lm,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=3,
        candidate_selection_strategy="pareto",
        use_merge=False,  # GEPA-only
        num_threads=args.num_threads,
        log_dir=args.log_dir,
        track_stats=True,
        seed=args.seed,
    )

    optimized = gepa.compile(student, trainset=train, valset=dev)

    # 7) Evaluate optimized
    opt_dev = evaluator_dev(optimized).score
    print(f"[OPTIMIZED] dev score: {opt_dev:.2f}")

    evaluator_test = Evaluate(devset=test, metric=hover_recall_score, num_threads=args.num_threads, display_progress=True)
    opt_test = evaluator_test(optimized).score
    print(f"[OPTIMIZED] test score: {opt_test:.2f}")

    # 8) Learning curve: score vs rollouts
    dr = optimized.detailed_results
    curve = build_best_so_far_curve_from_detailed_results(dr, baseline_score=baseline_dev)

    out_dir = Path(args.log_dir) if args.log_dir else (work / "gepa_hover_logs")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(
        json.dumps(
            dict(
                baseline_dev=baseline_dev,
                optimized_dev=opt_dev,
                optimized_test=opt_test,
                total_metric_calls=dr.total_metric_calls,
                num_full_val_evals=dr.num_full_val_evals,
                log_dir=str(dr.log_dir),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
    write_curve_csv(curve, out_dir / "curve.csv")

    print(f"Saved curve + summary to: {out_dir}")


if __name__ == "__main__":
    main()

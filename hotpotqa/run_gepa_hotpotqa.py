'''
In one terminal, run 
vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 --port 8000 \
  --api-key EMPTY \
  --max-model-len 16384

In another terminal, run
export VLLM_API_BASE="http://127.0.0.1:8000/v1"
export VLLM_API_KEY="EMPTY"
export VLLM_MODEL="Qwen/Qwen3-8B"

python run_gepa_hotpotqa.py \
  --work_dir "/tmp/gepa_hotpot" \
  --log_dir  "/tmp/gepa_hotpot/logs" \
  --num_threads 32 \
  --retriever_threads 8 \
  --max_metric_calls 10000

'''

# run_gepa_hotpotqa.py
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import dspy
from dspy.evaluate import Evaluate

from hotpot_data import load_hotpotqa_splits
from hotpot_metric import (
    hotpot_em_score,
    hotpot_metric_with_feedback,
    build_best_so_far_curve_from_detailed_results,
)
from hotpot_program import HotpotMultiHopQA
from wiki_retriever import build_or_load_bm25, make_search_fn


def configure_dspy_lm_from_vllm():
    """
    GEPA paper uses Qwen3-8B with temp=0.6, top-p=0.95, top-k=20, ctx up to 16384. :contentReference[oaicite:20]{index=20}
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
        # top_k=20,  # pass only if supported; otherwise configure server-side
        max_tokens=8192,
        cache=False,
        num_retries=3,
    )
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

    # Rollout budget in DSPy GEPA = max_metric_calls (metric evaluations). :contentReference[oaicite:21]{index=21}
    ap.add_argument("--max_metric_calls", type=int, default=5000)

    ap.add_argument("--num_threads", type=int, default=32)
    ap.add_argument("--retriever_threads", type=int, default=4)

    ap.add_argument("--work_dir", type=str, default=os.environ.get("WORK", "/tmp/hotpot_workdir"))
    ap.add_argument("--log_dir", type=str, default=None)

    args = ap.parse_args()

    lm = configure_dspy_lm_from_vllm()

    work = Path(args.work_dir)
    wiki_dir = work / "wiki17"
    index_dir = work / "wiki17_bm25"

    corpus, retriever, stemmer = build_or_load_bm25(wiki_dir=wiki_dir, index_dir=index_dir)
    search_fn = make_search_fn(corpus, retriever, stemmer, n_threads=args.retriever_threads)

    train, dev, test = load_hotpotqa_splits(seed=args.seed, n_train=150, n_dev=300, n_test=300)

    student = HotpotMultiHopQA(search_fn=search_fn, k_per_hop=args.k_per_hop)

    evaluator_dev = Evaluate(devset=dev, metric=hotpot_em_score, num_threads=args.num_threads, display_progress=True)
    baseline_dev = evaluator_dev(student).score
    print(f"[BASELINE] dev EM: {baseline_dev * 100:.2f}")

    gepa = dspy.GEPA(
        metric=hotpot_metric_with_feedback,  # must accept 5 args :contentReference[oaicite:22]{index=22}
        reflection_lm=lm,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=3,
        candidate_selection_strategy="pareto",
        use_merge=False,   # GEPA-only (no Merge)
        num_threads=args.num_threads,
        log_dir=args.log_dir,
        track_stats=True,
        seed=args.seed,
    )

    optimized = gepa.compile(student, trainset=train, valset=dev)

    opt_dev = evaluator_dev(optimized).score
    print(f"[OPTIMIZED] dev EM: {opt_dev * 100:.2f}")

    evaluator_test = Evaluate(devset=test, metric=hotpot_em_score, num_threads=args.num_threads, display_progress=True)
    opt_test = evaluator_test(optimized).score
    print(f"[OPTIMIZED] test EM: {opt_test * 100:.2f}")

    dr = optimized.detailed_results
    curve = build_best_so_far_curve_from_detailed_results(dr, baseline_score=baseline_dev)

    out_dir = Path(args.log_dir) if args.log_dir else (work / "gepa_hotpot_logs")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(
        json.dumps(
            dict(
                baseline_dev_em=baseline_dev,
                optimized_dev_em=opt_dev,
                optimized_test_em=opt_test,
                total_metric_calls=getattr(dr, "total_metric_calls", None),
                num_full_val_evals=getattr(dr, "num_full_val_evals", None),
                log_dir=str(getattr(dr, "log_dir", out_dir)),
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

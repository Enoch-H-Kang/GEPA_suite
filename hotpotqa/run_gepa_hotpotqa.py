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

export VLLM_API_BASE="http://127.0.0.1:8000/v1"
export VLLM_API_KEY="EMPTY"
export VLLM_MODEL="Qwen/Qwen3-8B"

python run_gepa_hotpotqa.py \
  --run_dir "$WORK/gepa_Qwen/hotpotqa/runs/single_gepa" \
  --work_dir "$WORK/gepa_Qwen/hotpotqa/data" \
  --num_threads 12 \
  --retriever_threads 8 \
  --max_metric_calls 10000

'''

# run_gepa_hotpotqa.py
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
    Qwen3-8B via vLLM OpenAI-compatible endpoint.
    Paper-like decoding: temperature=0.6, top_p=0.95, ctx up to 16384 (server-side). 
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
        # top_k=20,  # pass only if supported; otherwise set server-side
        # keep outputs generous; actual limit is still bounded by model context
        max_tokens=8192,
        cache=False,
        num_retries=3,
    )
    dspy.configure(lm=lm)
    return lm


def write_curve_csv(curve, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["rollouts", "candidate_idx", "candidate_val_score", "best_val_score"],
        )
        w.writeheader()
        for row in curve:
            w.writerow(row)


def _safe_int(x, default=None):
    if x is None:
        return default
    try:
        return int(x)
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k_per_hop", type=int, default=5)

    # Required by the compare driver: each variant writes into its own run_dir.
    ap.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Output dir for this single run (curve.csv, summary.json, GEPA logs).",
    )

    # Variant knobs
    ap.add_argument("--use_merge", type=int, default=0, choices=[0, 1])
    ap.add_argument("--bon", type=int, default=1)
    ap.add_argument("--itr", type=int, default=1)

    # Live-updating: run GEPA in stages of this many metric calls, reusing same log_dir.
    ap.add_argument(
        "--stage_step",
        type=int,
        default=0,
        help="If >0, run GEPA in stages of this many metric calls and write curve.csv after each stage.",
    )

    # Budget
    ap.add_argument(
        "--max_metric_calls",
        type=int,
        default=5000,
        help="Total metric calls budget (rollouts) for GEPA.",
    )

    # Parallelism
    ap.add_argument("--num_threads", type=int, default=32)
    ap.add_argument("--retriever_threads", type=int, default=4)

    # Data/cache dirs
    ap.add_argument(
        "--work_dir",
        type=str,
        default=os.environ.get("WORK", "/tmp/hotpot_workdir"),
        help="Directory used to store wiki abstracts + bm25 index.",
    )

    # Optional explicit GEPA log dir (defaults under run_dir)
    ap.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="GEPA checkpoint/log directory. If unset, uses <run_dir>/gepa_logs for resume.",
    )

    args = ap.parse_args()

    # -----------------------
    # Setup output dirs (resume-friendly)
    # -----------------------
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    gepa_log_dir = Path(args.log_dir) if args.log_dir else (run_dir / "gepa_logs")
    gepa_log_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    (run_dir / "config.json").write_text(
        json.dumps(
            dict(
                seed=args.seed,
                k_per_hop=args.k_per_hop,
                use_merge=bool(args.use_merge),
                bon=args.bon,
                itr=args.itr,
                stage_step=args.stage_step,
                max_metric_calls=args.max_metric_calls,
                num_threads=args.num_threads,
                retriever_threads=args.retriever_threads,
                work_dir=str(args.work_dir),
                gepa_log_dir=str(gepa_log_dir),
                vllm_api_base=os.environ.get("VLLM_API_BASE", None),
                vllm_model=os.environ.get("VLLM_MODEL", None),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------
    # Configure LM
    # -----------------------
    lm = configure_dspy_lm_from_vllm()

    # -----------------------
    # Retriever (BM25 over wiki abstracts 2017)
    # -----------------------
    work = Path(args.work_dir)
    wiki_dir = work / "wiki17"
    index_dir = work / "wiki17_bm25"

    corpus, retriever, stemmer = build_or_load_bm25(wiki_dir=wiki_dir, index_dir=index_dir)
    search_fn = make_search_fn(corpus, retriever, stemmer, n_threads=args.retriever_threads)

    # -----------------------
    # Data
    # -----------------------
    train, dev, test = load_hotpotqa_splits(seed=args.seed, n_train=150, n_dev=300, n_test=300)

    # -----------------------
    # Program
    # -----------------------
    student = HotpotMultiHopQA(search_fn=search_fn, k_per_hop=args.k_per_hop)

    # -----------------------
    # Baseline
    # -----------------------
    evaluator_dev = Evaluate(devset=dev, metric=hotpot_em_score, num_threads=args.num_threads, display_progress=True)
    baseline_dev = evaluator_dev(student).score
    print(f"[BASELINE] dev EM: {baseline_dev * 100:.2f}")

    # -----------------------
    # GEPA factory
    # -----------------------
    extra_gepa = {}
    # Your custom DSPy build accepts bon/itr. If it doesn't, this will error; in that case you can
    # adapt to pass via gepa_kwargs (but you said you added these args to GEPA).
    extra_gepa["bon"] = _safe_int(args.bon, 1)
    extra_gepa["itr"] = _safe_int(args.itr, 1)

    def make_gepa(max_calls: int):
        return dspy.GEPA(
            metric=hotpot_metric_with_feedback,  # must accept 5 args
            reflection_lm=lm,
            max_metric_calls=int(max_calls),
            reflection_minibatch_size=3,
            candidate_selection_strategy="pareto",
            use_merge=bool(args.use_merge),
            num_threads=args.num_threads,
            log_dir=str(gepa_log_dir),
            track_stats=True,
            seed=args.seed,
            **extra_gepa,
        )

    def evaluate_and_write(optimized):
        # Evaluate optimized on dev/test
        opt_dev = evaluator_dev(optimized).score
        evaluator_test = Evaluate(
            devset=test,
            metric=hotpot_em_score,
            num_threads=args.num_threads,
            display_progress=True,
        )
        opt_test = evaluator_test(optimized).score

        dr = optimized.detailed_results
        curve = build_best_so_far_curve_from_detailed_results(dr, baseline_score=baseline_dev)

        # Save outputs into run_dir (so compare script can read run_dir/curve.csv)
        (run_dir / "summary.json").write_text(
            json.dumps(
                dict(
                    baseline_dev_em=baseline_dev,
                    optimized_dev_em=opt_dev,
                    optimized_test_em=opt_test,
                    total_metric_calls=getattr(dr, "total_metric_calls", None),
                    num_full_val_evals=getattr(dr, "num_full_val_evals", None),
                    log_dir=str(getattr(dr, "log_dir", str(gepa_log_dir))),
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_dir / "curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
        write_curve_csv(curve, run_dir / "curve.csv")

        print(f"[OPTIMIZED] dev EM:  {opt_dev * 100:.2f}")
        print(f"[OPTIMIZED] test EM: {opt_test * 100:.2f}")

    # -----------------------
    # Run GEPA (staged or single-shot)
    # -----------------------
    optimized = None

    if args.stage_step and args.stage_step > 0:
        step = int(args.stage_step)
        total = int(args.max_metric_calls)
        if step <= 0:
            raise ValueError("--stage_step must be > 0 when provided.")
        if total <= 0:
            raise ValueError("--max_metric_calls must be > 0.")

        budgets = list(range(step, total + step, step))
        budgets[-1] = total

        for b in budgets:
            print(f"[GEPA] compiling up to max_metric_calls={b} (resume log_dir={gepa_log_dir})")
            gepa = make_gepa(b)
            optimized = gepa.compile(student, trainset=train, valset=dev)
            evaluate_and_write(optimized)
            print(f"[GEPA] wrote curve to: {run_dir/'curve.csv'}")

    else:
        print(f"[GEPA] compiling max_metric_calls={args.max_metric_calls} (log_dir={gepa_log_dir})")
        gepa = make_gepa(args.max_metric_calls)
        optimized = gepa.compile(student, trainset=train, valset=dev)
        evaluate_and_write(optimized)

    print(f"Saved run artifacts to: {run_dir}")


if __name__ == "__main__":
    main()

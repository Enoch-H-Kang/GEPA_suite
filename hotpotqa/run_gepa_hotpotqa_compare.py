'''

export VLLM_API_BASE="http://127.0.0.1:8000/v1"
export VLLM_API_KEY="EMPTY"
export VLLM_MODEL="Qwen/Qwen3-8B"

python run_gepa_hotpotqa_compare.py \
  --out_root "logs/run1" \
  --refresh_sec 20 \
  --stage_step 500 \
  --seed 0 \
  --max_metric_calls 10000 \
  --num_threads 16 \
  --retriever_threads 8



'''


# run_gepa_hotpotqa_compare.py
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import subprocess
from pathlib import Path

def read_curve_csv(path: Path):
    if not path.exists():
        return None
    xs, ys = [], []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(int(row["rollouts"]))
            ys.append(float(row["best_val_score"]))
    if not xs:
        return None
    return xs, ys

def merge_step_curves(curves: dict[str, tuple[list[int], list[float]]]):
    # union x-axis, forward-fill each curve
    all_x = sorted({x for xs, _ in curves.values() for x in xs})
    out_rows = []

    # precompute pointers
    ptr = {k: 0 for k in curves}
    last = {k: None for k in curves}

    for x in all_x:
        row = {"rollouts": x}
        for name, (xs, ys) in curves.items():
            i = ptr[name]
            while i < len(xs) and xs[i] <= x:
                last[name] = ys[i]
                i += 1
            ptr[name] = i
            row[name] = last[name]
        out_rows.append(row)

    return all_x, out_rows

def write_merged_csv(out_path: Path, variants: list[str], rows: list[dict]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rollouts"] + variants)
        w.writeheader()
        for row in rows:
            w.writerow(row)

def plot_curves_png(out_png: Path, curves: dict[str, tuple[list[int], list[float]]], title: str):
    # headless-safe
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    for name, (xs, ys) in curves.items():
        # step-like curve (best-so-far)
        plt.step(xs, ys, where="post", label=name)

    plt.xlabel("Metric calls (rollouts)")
    plt.ylabel("Best validation score so far")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()

def spawn_run(worker_py: Path, run_dir: Path, common_args: list[str], env_overrides: dict[str,str], log_file: Path):
    env = os.environ.copy()
    env.update(env_overrides)

    cmd = [sys.executable, str(worker_py)] + common_args + ["--run_dir", str(run_dir)]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fout = log_file.open("w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=fout, stderr=subprocess.STDOUT, env=env)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=str, default="run_gepa_hotpotqa.py",
                    help="Your single-run worker script.")
    ap.add_argument("--out_root", type=str, required=True,
                    help="Root dir to store all three runs + comparison outputs.")
    ap.add_argument("--refresh_sec", type=int, default=30,
                    help="How often to refresh comparison_live.png while runs are active.")
    ap.add_argument("--stage_step", type=int, default=0,
                    help="Forwarded to worker; enables live curve updates if >0.")

    # Optional: route each run to a different vLLM endpoint for TRUE parallelism
    ap.add_argument("--api_bases", type=str, default="",
                    help="Comma-separated VLLM_API_BASE list (len 1 or 3).")

    # Everything else: pass-through to your worker
    # (Add your usual flags here; we keep it open-ended with "unknown args")
    args, unknown = ap.parse_known_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    worker_py = Path(args.worker)

    # Define the three variants you asked for
    variants = [
        {"name": "GEPA", "subdir": "gepa", "use_merge": 0, "bon": 1, "itr": 1},
        {"name": "GEPA+merge", "subdir": "gepa_merge", "use_merge": 1, "bon": 1, "itr": 1},
        {"name": "GEPA bon=5 itr=5", "subdir": "gepa_bon5_itr5", "use_merge": 0, "bon": 5, "itr": 5},
    ]

    # Assign API bases (optional)
    api_bases = [s.strip() for s in args.api_bases.split(",") if s.strip()]
    if len(api_bases) not in (0, 1, 3):
        raise ValueError("--api_bases must be empty, length 1, or length 3.")

    procs = []
    for i, v in enumerate(variants):
        run_dir = out_root / v["subdir"]
        run_dir.mkdir(parents=True, exist_ok=True)

        # build args for THIS variant
        var_args = list(unknown)
        var_args += ["--use_merge", str(v["use_merge"])]
        if v["bon"] is not None:
            var_args += ["--bon", str(v["bon"])]
        if v["itr"] is not None:
            var_args += ["--itr", str(v["itr"])]
        if args.stage_step:
            var_args += ["--stage_step", str(args.stage_step)]

        env_overrides = {}
        if api_bases:
            env_overrides["VLLM_API_BASE"] = api_bases[i] if len(api_bases) == 3 else api_bases[0]

        log_file = run_dir / "stdout.log"
        p = spawn_run(worker_py, run_dir, var_args, env_overrides, log_file)
        procs.append((v["name"], run_dir, p))

    # Live plot loop (updates a PNG on disk)
    live_png = out_root / "comparison_live.png"
    final_png = out_root / "comparison.png"
    merged_csv = out_root / "comparison_curves.csv"

    while True:
        alive = any(p.poll() is None for _, _, p in procs)

        curves = {}
        for name, run_dir, _p in procs:
            c = read_curve_csv(run_dir / "curve.csv")
            if c is not None:
                curves[name] = c

        if curves:
            plot_curves_png(
                live_png,
                curves,
                title="HotpotQA: GEPA variant comparison (live)"
            )
            # also write merged csv (best-effort while running)
            _, rows = merge_step_curves(curves)
            write_merged_csv(merged_csv, list(curves.keys()), rows)

        if not alive:
            break
        time.sleep(args.refresh_sec)

    # Final render (same as live, but final filename)
    curves = {}
    for name, run_dir, _p in procs:
        c = read_curve_csv(run_dir / "curve.csv")
        if c is not None:
            curves[name] = c

    if curves:
        plot_curves_png(final_png, curves, title="HotpotQA: GEPA variant comparison (final)")
        _, rows = merge_step_curves(curves)
        write_merged_csv(merged_csv, list(curves.keys()), rows)

    # return non-zero if any failed
    rc = 0
    for name, run_dir, p in procs:
        if p.returncode != 0:
            rc = 1
            print(f"[WARN] {name} failed. See: {run_dir/'stdout.log'}", file=sys.stderr)

    print(f"Saved live plot:  {live_png}")
    print(f"Saved final plot: {final_png}")
    print(f"Saved merged CSV: {merged_csv}")
    sys.exit(rc)

if __name__ == "__main__":
    main()

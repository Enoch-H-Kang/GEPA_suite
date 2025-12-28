#!/bin/bash
#SBATCH -J hotpot_gepa_cmp
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH -p mi2104x
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 12

set -euo pipefail

module purge
module load hpcfund
module list

source $WORK/gepa_hotpotqa/venv/gepa/bin/activate

# --- caches in $WORK (avoid $HOME quota) ---
export HF_HOME="$WORK/gepa_hotpotqa/cache/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$WORK/gepa_hotpotqa/cache/xdg"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"

# --- resume-friendly output root ---
OUT_ROOT="$WORK/gepa_hotpotqa/runs/latest"
mkdir -p "$OUT_ROOT"

# Stamp a small "run marker" so you can see last job time
date > "$OUT_ROOT/last_submit_time.txt"
echo "OUT_ROOT=$OUT_ROOT"

# --- model config ---
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

# Optional ROCm/vLLM toggles (uncomment if needed)
# export VLLM_TARGET_DEVICE="rocm"
# export VLLM_USE_TRITON_FLASH_ATTN=0

echo "==== rocm-smi (sanity) ===="
rocm-smi || true
echo "==========================="

# --- Choose ports that won't collide with other jobs on same node ---
# Keep them in a high range; derived from job ID.
BASE_PORT=$(( 18000 + (SLURM_JOB_ID % 1000) * 10 ))
PORT0=$((BASE_PORT + 0))
PORT1=$((BASE_PORT + 1))
PORT2=$((BASE_PORT + 2))

echo "Using ports: $PORT0 $PORT1 $PORT2" | tee "$OUT_ROOT/ports.txt"

# Save API bases so you can quickly reuse them for debugging
API0="http://127.0.0.1:${PORT0}/v1"
API1="http://127.0.0.1:${PORT1}/v1"
API2="http://127.0.0.1:${PORT2}/v1"
echo "$API0,$API1,$API2" > "$OUT_ROOT/api_bases.txt"

# --- log rotation (append-friendly) ---
mkdir -p "$OUT_ROOT/server_logs"
mkdir -p "$OUT_ROOT/driver_logs"

# If you want a clean restart sometimes, set CLEAN=1 when submitting:
#   CLEAN=1 sbatch job.hotpot_compare_resume.sbatch
if [[ "${CLEAN:-0}" == "1" ]]; then
  echo "[CLEAN=1] Removing previous curves/plots (keeping caches) ..."
  rm -rf "$OUT_ROOT/logs" || true
  rm -f "$OUT_ROOT/comparison_live.png" "$OUT_ROOT/comparison.png" "$OUT_ROOT/comparison_curves.csv" || true
fi

# --- start 3 vLLM servers on 3 GPUs ---
# Note: cluster docs use HIP_VISIBLE_DEVICES with srun job steps. Keep that. 
# We redirect to per-job server logs under OUT_ROOT/server_logs.
export HIP_VISIBLE_DEVICES=0
srun -n 1 --exact \
  bash -lc "vllm serve \"$VLLM_MODEL\" --host 127.0.0.1 --port ${PORT0} --api-key \"$VLLM_API_KEY\" --max-model-len 16384" \
  >> "$OUT_ROOT/server_logs/vllm_${PORT0}.log" 2>&1 &

export HIP_VISIBLE_DEVICES=1
srun -n 1 --exact \
  bash -lc "vllm serve \"$VLLM_MODEL\" --host 127.0.0.1 --port ${PORT1} --api-key \"$VLLM_API_KEY\" --max-model-len 16384" \
  >> "$OUT_ROOT/server_logs/vllm_${PORT1}.log" 2>&1 &

export HIP_VISIBLE_DEVICES=2
srun -n 1 --exact \
  bash -lc "vllm serve \"$VLLM_MODEL\" --host 127.0.0.1 --port ${PORT2} --api-key \"$VLLM_API_KEY\" --max-model-len 16384" \
  >> "$OUT_ROOT/server_logs/vllm_${PORT2}.log" 2>&1 &

# --- wait for servers ---
echo "Waiting for vLLM servers..."
for port in "$PORT0" "$PORT1" "$PORT2"; do
  for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "vLLM on ${port} is up."
      break
    fi
    sleep 2
  done
done

# --- run compare driver ---
# IMPORTANT: we reuse OUT_ROOT/logs every time, so runs resume *within each variant*
# as long as your worker uses run_dir for its GEPA log_dir and supports staged resume.
cd $WORK/gepa_hotpotqa/repo

# Optional: store the exact command line for reproducibility
cat > "$OUT_ROOT/driver_logs/last_command.txt" <<EOF
python run_gepa_hotpotqa_compare.py \
  --out_root "$OUT_ROOT/logs" \
  --refresh_sec 20 \
  --stage_step 500 \
  --api_bases "$API0,$API1,$API2" \
  --seed 0 \
  --max_metric_calls 10000 \
  --num_threads 12 \
  --retriever_threads 8 \
  >> "$OUT_ROOT/driver_logs/compare_driver.log" 2>&1
EOF

# Run and append output to a stable driver log
python run_gepa_hotpotqa_compare.py \
--out_root "$OUT_ROOT/logs" \
  --run_dir "$OUT_ROOT/logs/run" \
  --stage_step 500 \
  --api_bases "$API0,$API1,$API2" \
  --seed 0 \
  --max_metric_calls 10000 \
  --num_threads 12 \
  --retriever_threads 8 \
  >> "$OUT_ROOT/driver_logs/compare_driver.log" 2>&1

echo "Done. Outputs in: $OUT_ROOT/logs"

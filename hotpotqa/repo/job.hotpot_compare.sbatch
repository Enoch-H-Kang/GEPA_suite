#!/bin/bash
#SBATCH -J hotpot_gepa_cmp
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH -p mi2104x
#SBATCH -t 4-00:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 12

set -euo pipefail

module purge
module load hpcfund
module list

source $WORK/gepa_hotpotqa/venv/gepa/bin/activate

# Caches in $WORK
export HF_HOME="$WORK/gepa_hotpotqa/cache/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$WORK/gepa_hotpotqa/cache/xdg"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="$WORK/gepa_hotpotqa/runs/$RUN_ID"
mkdir -p "$OUT_ROOT"
echo "OUT_ROOT=$OUT_ROOT"

export VLLM_MODEL="Qwen/Qwen3-8B"
export VLLM_API_KEY="EMPTY"

echo "==== rocm-smi (sanity) ===="
rocm-smi || true
echo "==========================="

# --- start 3 vLLM servers on 3 GPUs (job-step style from HPC Fund docs) ---
export HIP_VISIBLE_DEVICES=0
srun -n 1 -o "$OUT_ROOT/step_vllm_8000.%J.log" --exact \
  vllm serve "$VLLM_MODEL" --host 127.0.0.1 --port 8000 --api-key "$VLLM_API_KEY" --max-model-len 16384 \
  > "$OUT_ROOT/vllm_8000.log" 2>&1 &

export HIP_VISIBLE_DEVICES=1
srun -n 1 -o "$OUT_ROOT/step_vllm_8001.%J.log" --exact \
  vllm serve "$VLLM_MODEL" --host 127.0.0.1 --port 8001 --api-key "$VLLM_API_KEY" --max-model-len 16384 \
  > "$OUT_ROOT/vllm_8001.log" 2>&1 &

export HIP_VISIBLE_DEVICES=2
srun -n 1 -o "$OUT_ROOT/step_vllm_8002.%J.log" --exact \
  vllm serve "$VLLM_MODEL" --host 127.0.0.1 --port 8002 --api-key "$VLLM_API_KEY" --max-model-len 16384 \
  > "$OUT_ROOT/vllm_8002.log" 2>&1 &

# Wait until all three are reachable
echo "Waiting for vLLM servers..."
for port in 8000 8001 8002; do
  for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "vLLM on ${port} is up."
      break
    fi
    sleep 2
  done
done

# Run your compare driver (NOTE: your compare script does NOT define --run_dir, so don't pass it)
cd $WORK/gepa_hotpotqa/repo

python run_gepa_hotpotqa_compare.py \
  --out_root "$OUT_ROOT/logs" \
  --run_dir "$OUT_ROOT/logs/run" \
  --refresh_sec 20 \
  --stage_step 500 \
  --api_bases "http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1" \
  --seed 0 \
  --max_metric_calls 10000 \
  --num_threads 12 \
  --retriever_threads 8

echo "Done. Outputs in: $OUT_ROOT/logs"

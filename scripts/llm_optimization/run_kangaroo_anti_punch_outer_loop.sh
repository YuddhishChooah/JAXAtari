#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

SEED="${SEED:-20260501}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)_anti_punch_overcorrection_fix_s${SEED}}"
BASE_RESULTS="${BASE_RESULTS:-scripts/llm_optimization/runs/single_game/kangaroo_resume3_20260430_144102/optimization_results.json}"
OUTPUT_DIR="${OUTPUT_DIR:-scripts/llm_optimization/runs/single_game/kangaroo_${STAMP}}"
MAX_ITERS="${MAX_ITERS:-2}"

if [[ ! -f "$BASE_RESULTS" ]]; then
  echo "Missing resume seed results: $BASE_RESULTS" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
if [[ ! -f "$OUTPUT_DIR/optimization_results.json" ]]; then
  cp "$BASE_RESULTS" "$OUTPUT_DIR/optimization_results.json"
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  printf "ANTHROPIC_API_KEY: " >&2
  read -rs ANTHROPIC_API_KEY
  printf "\n" >&2
  export ANTHROPIC_API_KEY
fi

.venv/bin/python scripts/llm_optimization/llm_optimization_loop.py \
  --game kangaroo \
  --provider anthropic \
  --model claude-opus-4-7 \
  --resume-from-results \
  --max-iters "$MAX_ITERS" \
  --num-envs 8 \
  --num-episodes 8 \
  --max-steps 4000 \
  --search-max-steps 2500 \
  --param-samples 8 \
  --cma-generations 4 \
  --target-score 1000000000 \
  --no-stop-on-strong-best \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR"

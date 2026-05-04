#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run one seeded RQ1 LeGPS rerun suite.

This launcher prompts for the provider API key if it is not already present in
the environment. The key is exported only for this process and is not written to
repo files or shell history.

Environment overrides:
  PROVIDER                 anthropic or openai (default: anthropic)
  MODEL                    model name (default: claude-opus-4-7)
  RUN_ID                   suite run id (default: rq1_seeded_<timestamp>)
  GAMES                    space-separated game list (default: current five canonical games)
  SEED                     base seed (default: 20260480)
  MAX_ITERS                outer-loop iterations (default: 5)
  NUM_ENVS                 parallel envs (default: 8)
  NUM_EPISODES             eval episodes (default: 16)
  MAX_STEPS                eval horizon (default: 10000)
  SEARCH_MAX_STEPS         search horizon (default: 10000)
  PARAM_SAMPLES            CMA-ES population/samples (default: 16)
  CMA_GENERATIONS          CMA-ES generations (default: 5)
  COMPLETION_GRACE_SECONDS result-file shutdown grace (default: 60)

Any extra arguments are appended to run_unified_suite.py.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON:-${repo_root}/.venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="${PYTHON:-python}"
fi

provider="${PROVIDER:-anthropic}"
model="${MODEL:-claude-opus-4-7}"
run_id="${RUN_ID:-rq1_seeded_$(date +%Y%m%d_%H%M%S)}"
games_text="${GAMES:-pong freeway asterix breakout skiing}"
seed="${SEED:-20260480}"
max_iters="${MAX_ITERS:-5}"
num_envs="${NUM_ENVS:-8}"
num_episodes="${NUM_EPISODES:-16}"
max_steps="${MAX_STEPS:-10000}"
search_max_steps="${SEARCH_MAX_STEPS:-10000}"
param_samples="${PARAM_SAMPLES:-16}"
cma_generations="${CMA_GENERATIONS:-5}"
completion_grace_seconds="${COMPLETION_GRACE_SECONDS:-60}"
read -r -a games <<< "${games_text}"

case "${provider}" in
  anthropic)
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
      read -r -s -p "Anthropic API key: " ANTHROPIC_API_KEY
      export ANTHROPIC_API_KEY
      echo
    fi
    ;;
  openai)
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      read -r -s -p "OpenAI API key: " OPENAI_API_KEY
      export OPENAI_API_KEY
      echo
    fi
    ;;
  *)
    echo "Unsupported PROVIDER=${provider}; expected anthropic or openai." >&2
    exit 2
    ;;
esac

exec "${python_bin}" scripts/llm_optimization/run_unified_suite.py \
  --games "${games[@]}" \
  --run-id "${run_id}" \
  --provider "${provider}" \
  --model "${model}" \
  --max-iters "${max_iters}" \
  --target-score 1000000000 \
  --num-envs "${num_envs}" \
  --num-episodes "${num_episodes}" \
  --max-steps "${max_steps}" \
  --search-max-steps "${search_max_steps}" \
  --optimizer cma-es \
  --param-samples "${param_samples}" \
  --cma-generations "${cma_generations}" \
  --seed "${seed}" \
  --completion-grace-seconds "${completion_grace_seconds}" \
  --continue-on-error \
  "$@"

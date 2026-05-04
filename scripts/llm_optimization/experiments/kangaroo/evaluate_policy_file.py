#!/usr/bin/env python
"""Evaluate and optionally retune an existing Kangaroo policy file.

This is API-free. It is meant for testing manual/local policy patches without
starting another LLM outer-loop run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.llm_optimization.llm_optimization_loop import (
    OptimizationConfig,
    ParallelEvaluator,
    ParameterSearcher,
)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_policy(policy_path: Path):
    spec = importlib.util.spec_from_file_location(policy_path.stem, policy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load policy module from {policy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def summarize_rewards(rewards: list[float]) -> dict[str, Any]:
    arr = np.asarray(rewards, dtype=float)
    histogram = Counter(str(int(value)) if float(value).is_integer() else str(value) for value in rewards)
    if arr.size == 0:
        stats = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    else:
        stats = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
        }
    stats["histogram"] = dict(sorted(histogram.items()))
    stats["episodes_above_200"] = int(sum(value > 200 for value in rewards))
    return stats


def as_float_params(params: dict[str, Any]) -> dict[str, jax.Array]:
    return {key: jnp.asarray(value, dtype=jnp.float32) for key, value in params.items()}


def evaluate(policy_fn, params: dict[str, jax.Array], *, episodes: int, steps: int, seed: int, frame_stack: int):
    config = OptimizationConfig(
        num_parallel_envs=episodes,
        num_eval_episodes=episodes,
        max_steps_per_episode=steps,
        search_max_steps=steps,
        frame_stack_size=frame_stack,
        verbose=False,
        seed=seed,
    )
    evaluator = ParallelEvaluator(config, game="kangaroo")
    metrics = evaluator.evaluate_policy(
        policy_fn,
        params,
        jrandom.PRNGKey(seed),
        num_episodes=episodes,
        max_steps=steps,
    )
    rewards = [float(value) for value in metrics.get("total_rewards", [])]
    return metrics, summarize_rewards(rewards)


def retune(policy_fn, base_params: dict[str, jax.Array], args: argparse.Namespace, frame_stack: int):
    config = OptimizationConfig(
        num_parallel_envs=args.search_episodes,
        num_eval_episodes=args.eval_episodes,
        max_steps_per_episode=args.eval_steps,
        search_max_steps=args.search_steps,
        frame_stack_size=frame_stack,
        optimizer="cma-es",
        num_param_samples=args.param_samples,
        cma_es_generations=args.cma_generations,
        param_perturbation_scale=args.param_scale,
        verbose=True,
        seed=args.seed,
    )
    evaluator = ParallelEvaluator(config, game="kangaroo")

    def init_params():
        return base_params

    searcher = ParameterSearcher(config, evaluator)
    search_params, search_metrics = searcher.search(policy_fn, init_params, jrandom.PRNGKey(args.seed))
    eval_metrics = evaluator.evaluate_policy(
        policy_fn,
        search_params,
        jrandom.PRNGKey(args.seed + 1),
        num_episodes=args.eval_episodes,
        max_steps=args.eval_steps,
    )
    return {
        "protocol": {
            "search_episodes": args.search_episodes,
            "search_steps": args.search_steps,
            "param_samples": args.param_samples,
            "cma_generations": args.cma_generations,
            "param_scale": args.param_scale,
            "eval_episodes": args.eval_episodes,
            "eval_steps": args.eval_steps,
        },
        "search_metrics": to_jsonable(search_metrics),
        "retuned_metrics": to_jsonable(eval_metrics),
        "retuned_params": to_jsonable(search_params),
        "retuned_reward_summary": summarize_rewards([float(value) for value in eval_metrics.get("total_rewards", [])]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Kangaroo policy file")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260480)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--retune", action="store_true")
    parser.add_argument("--search-episodes", type=int, default=4)
    parser.add_argument("--search-steps", type=int, default=5000)
    parser.add_argument("--param-samples", type=int, default=8)
    parser.add_argument("--cma-generations", type=int, default=4)
    parser.add_argument("--param-scale", type=float, default=0.3)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--eval-steps", type=int, default=10000)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    policy_path = resolve_path(args.policy_path)
    output_path = resolve_path(args.output)
    module = load_policy(policy_path)
    params = as_float_params(module.init_params())

    print(f"Evaluating {policy_path}", flush=True)
    print(f"episodes={args.episodes}, steps={args.steps}, seed={args.seed}", flush=True)
    metrics, reward_summary = evaluate(
        module.policy,
        params,
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        frame_stack=args.frame_stack,
    )
    print(
        "Evaluation summary: "
        f"mean={reward_summary['mean']:.2f}, min={reward_summary['min']:.2f}, "
        f"max={reward_summary['max']:.2f}, >200={reward_summary['episodes_above_200']}",
        flush=True,
    )

    retune_result = retune(module.policy, params, args, args.frame_stack) if args.retune else None
    if retune_result is not None:
        summary = retune_result["retuned_reward_summary"]
        print(
            "Retune summary: "
            f"mean={summary['mean']:.2f}, min={summary['min']:.2f}, "
            f"max={summary['max']:.2f}, >200={summary['episodes_above_200']}",
            flush=True,
        )

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policy_path": str(policy_path.relative_to(PROJECT_ROOT)),
        "runtime": {
            "elapsed_seconds": time.time() - started,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "jax_version": jax.__version__,
            "numpy_version": np.__version__,
        },
        "evaluation_protocol": {
            "episodes": args.episodes,
            "steps": args.steps,
            "seed": args.seed,
            "frame_stack": args.frame_stack,
        },
        "metrics": to_jsonable(metrics),
        "reward_summary": reward_summary,
        "base_params": to_jsonable(params),
        "parameter_retune": retune_result,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

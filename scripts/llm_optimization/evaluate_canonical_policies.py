#!/usr/bin/env python
"""Re-evaluate canonical best LeGPS policies without calling any LLM API."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from scripts.llm_optimization.llm_optimization_loop import OptimizationConfig, ParallelEvaluator


DEFAULT_MANIFEST = PROJECT_ROOT / "scripts/llm_optimization/runs/best_10000_steps/manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "scripts/llm_optimization/analysis/evaluations/canonical_reproducibility/latest_canonical_evaluation.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text.replace("\\", "/"))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_policy_module(policy_path: Path):
    spec = importlib.util.spec_from_file_location(policy_path.stem, policy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load policy module from {policy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def as_jax_params(params: dict[str, Any]) -> dict[str, jax.Array]:
    return {key: jnp.asarray(value, dtype=jnp.float32) for key, value in params.items()}


def summarize_rewards(rewards: list[float]) -> dict[str, float]:
    arr = np.asarray(rewards, dtype=float)
    return {
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "std": float(np.std(arr)) if arr.size else float("nan"),
        "min": float(np.min(arr)) if arr.size else float("nan"),
        "max": float(np.max(arr)) if arr.size else float("nan"),
    }


def evaluate_game(
    game: str,
    entry: dict[str, Any],
    *,
    num_episodes: int,
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    policy_path = resolve_repo_path(entry["canonical_policy_path"])
    module = load_policy_module(policy_path)
    params = as_jax_params(entry.get("best_params", {}))

    config = OptimizationConfig(
        num_parallel_envs=min(8, num_episodes),
        num_eval_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        search_max_steps=max_steps,
        frame_stack_size=0,
        seed=seed,
        verbose=False,
    )
    evaluator = ParallelEvaluator(config, game=game)
    metrics = evaluator.evaluate_policy(
        module.policy,
        params,
        jax.random.PRNGKey(seed),
        num_episodes=num_episodes,
        max_steps=max_steps,
    )

    rewards = [float(value) for value in metrics.get("total_rewards", [])]
    stored_score = float(entry.get("avg_return_10000", float("nan")))
    observed_score = float(metrics["avg_return"])
    delta = observed_score - stored_score if math.isfinite(stored_score) else float("nan")

    return {
        "game": game,
        "policy_path": str(policy_path),
        "seed": seed,
        "num_episodes": num_episodes,
        "max_steps": max_steps,
        "stored_avg_return": stored_score,
        "observed_avg_return": observed_score,
        "delta_vs_stored": delta,
        "avg_player_score": float(metrics.get("avg_player_score", observed_score)),
        "avg_enemy_score": float(metrics.get("avg_enemy_score", 0.0)),
        "win_rate": float(metrics.get("win_rate", 0.0)),
        "reward_summary": summarize_rewards(rewards),
        "total_rewards": rewards,
        "objective_note": entry.get("objective_note", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-evaluate canonical saved LeGPS policies")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--games", nargs="+", default=None)
    parser.add_argument("--num-episodes", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260450)
    parser.add_argument(
        "--use-manifest-seeds",
        action="store_true",
        help="Use each game's saved canonical seed instead of --seed + index.",
    )
    args = parser.parse_args()

    manifest_path = resolve_repo_path(args.manifest)
    manifest = read_json(manifest_path)
    available_games = list(manifest.get("games", {}).keys())
    games = args.games or available_games

    started = time.time()
    results = []
    for index, game in enumerate(games):
        if game not in manifest.get("games", {}):
            raise KeyError(f"Game {game!r} is not present in {manifest_path}")
        entry = manifest["games"][game]
        seed = int(entry.get("seed", args.seed + index)) if args.use_manifest_seeds else args.seed + index
        print(f"Evaluating {game} with seed={seed}, episodes={args.num_episodes}, steps={args.max_steps}", flush=True)
        results.append(
            evaluate_game(
                game,
                entry,
                num_episodes=args.num_episodes,
                max_steps=args.max_steps,
                seed=seed,
            )
        )

    output = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest_path": str(manifest_path),
        "protocol": {
            "num_episodes": args.num_episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "use_manifest_seeds": bool(args.use_manifest_seeds),
        },
        "runtime": {
            "elapsed_seconds": time.time() - started,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "jax_version": jax.__version__,
            "numpy_version": np.__version__,
        },
        "results": results,
    }
    output_path = resolve_repo_path(args.output)
    write_json(output_path, output)
    print(f"Wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Run a unified LeGPS outer-loop suite in an isolated run directory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GAMES = ["pong", "freeway", "asterix", "breakout", "skiing"]


def json_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def json_read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_result(output_dir: Path) -> dict[str, Any]:
    results_path = output_dir / "optimization_results.json"
    if not results_path.exists():
        return {}
    return json_read(results_path)


def result_entry(
    *,
    game: str,
    status: str,
    returncode: int | None,
    elapsed: float | None,
    seed: int,
    output_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    result_data: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "seed": seed,
        "output_dir": str(output_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "best_score": result_data.get("best_score"),
        "best_policy_path": result_data.get("best_policy_path"),
    }


def run_game_process(
    command: list[str],
    output_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    completion_grace_seconds: int,
) -> tuple[int, float, dict[str, Any], bool]:
    """Run one game and tolerate JAX shutdown hangs after results are written."""
    start = time.time()
    result_seen_at: float | None = None
    terminated_after_results = False

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=stdout, stderr=stderr)

        while True:
            returncode = process.poll()
            result_data = load_result(output_dir)
            if returncode is not None:
                return returncode, time.time() - start, result_data, terminated_after_results

            if result_data:
                if result_seen_at is None:
                    result_seen_at = time.time()
                elif time.time() - result_seen_at >= completion_grace_seconds:
                    process.terminate()
                    try:
                        returncode = process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        returncode = process.wait(timeout=10)
                    terminated_after_results = True
                    return returncode, time.time() - start, result_data, terminated_after_results

            time.sleep(5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the unified LeGPS suite")
    parser.add_argument("--games", nargs="+", default=DEFAULT_GAMES, choices=DEFAULT_GAMES)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--base-dir", type=str, default="scripts/llm_optimization/runs/unified_suite")
    parser.add_argument("--provider", type=str, default="anthropic")
    parser.add_argument("--model", type=str, default="claude-opus-4-7")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--target-score", type=float, default=1_000_000_000.0)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--search-max-steps", type=int, default=2000)
    parser.add_argument("--optimizer", type=str, default="cma-es", choices=["none", "random", "cma-es", "bayes"])
    parser.add_argument("--param-samples", type=int, default=16)
    parser.add_argument("--cma-generations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--completion-grace-seconds",
        type=int,
        default=60,
        help="If optimization_results.json exists but the child process stays alive, terminate it after this grace period.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Reuse existing optimization_results.json files inside the run directory.",
    )
    args = parser.parse_args()

    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment")
    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    suite_dir = PROJECT_ROOT / args.base_dir / run_id
    suite_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = suite_dir / "manifest.json"
    if manifest_path.exists() and args.skip_completed:
        manifest = json_read(manifest_path)
        manifest["games"] = args.games
        manifest.setdefault("results", {})
    else:
        manifest: dict[str, Any] = {
            "run_id": run_id,
            "suite_dir": str(suite_dir),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "games": args.games,
            "protocol": {
                "provider": args.provider,
                "model": args.model,
                "temperature": args.temperature,
                "max_iters": args.max_iters,
                "target_score": args.target_score,
                "num_envs": args.num_envs,
                "num_episodes": args.num_episodes,
                "max_steps": args.max_steps,
                "search_max_steps": args.search_max_steps,
                "optimizer": args.optimizer,
                "param_samples": args.param_samples,
                "cma_generations": args.cma_generations,
                "seed": args.seed,
                "resume_from_results": False,
                "stop_on_strong_best": False,
                "completion_grace_seconds": args.completion_grace_seconds,
            },
            "results": {},
        }
    json_write(manifest_path, manifest)

    for index, game in enumerate(args.games):
        game_seed = args.seed + index
        output_dir = suite_dir / game
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = output_dir / "outerloop.out.log"
        stderr_path = output_dir / "outerloop.err.log"
        existing_result = load_result(output_dir)
        if args.skip_completed and existing_result:
            manifest["results"][game] = result_entry(
                game=game,
                status="completed",
                returncode=0,
                elapsed=manifest.get("results", {}).get(game, {}).get("elapsed_seconds"),
                seed=game_seed,
                output_dir=output_dir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                result_data=existing_result,
            )
            json_write(manifest_path, manifest)
            print(
                f"\n=== Skipping {game} ({index + 1}/{len(args.games)}): "
                f"existing best_score={existing_result.get('best_score')} ===",
                flush=True,
            )
            continue

        command = [
            sys.executable,
            "-u",
            "scripts/llm_optimization/llm_optimization_loop.py",
            "--game",
            game,
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--max-iters",
            str(args.max_iters),
            "--target-score",
            str(args.target_score),
            "--num-envs",
            str(args.num_envs),
            "--num-episodes",
            str(args.num_episodes),
            "--max-steps",
            str(args.max_steps),
            "--search-max-steps",
            str(args.search_max_steps),
            "--optimizer",
            args.optimizer,
            "--param-samples",
            str(args.param_samples),
            "--cma-generations",
            str(args.cma_generations),
            "--seed",
            str(game_seed),
            "--output-dir",
            str(output_dir.relative_to(PROJECT_ROOT)),
            "--no-stop-on-strong-best",
        ]
        if args.temperature is not None:
            command.extend(["--temperature", str(args.temperature)])

        print(f"\n=== Running {game} ({index + 1}/{len(args.games)}) ===", flush=True)
        print(f"Output: {output_dir}", flush=True)
        returncode, elapsed, result_data, terminated_after_results = run_game_process(
            command,
            output_dir,
            stdout_path,
            stderr_path,
            args.completion_grace_seconds,
        )
        status = "completed" if result_data else "failed"
        manifest["results"][game] = result_entry(
            game=game,
            status=status,
            returncode=returncode,
            elapsed=elapsed,
            seed=game_seed,
            output_dir=output_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            result_data=result_data,
        )
        manifest["results"][game]["terminated_after_results"] = terminated_after_results
        json_write(manifest_path, manifest)

        print(
            f"{game}: {status}, best_score={manifest['results'][game]['best_score']}, "
            f"elapsed={elapsed / 60:.1f} min",
            flush=True,
        )
        if status != "completed" and not args.continue_on_error:
            return returncode or 1

    print(f"\nSuite complete: {suite_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

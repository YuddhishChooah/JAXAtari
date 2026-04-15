#!/usr/bin/env python
"""
Render a Pong policy rollout to MP4.

Defaults to the best current unified Pong run:
- best policy path from optimization_results.json
- best_params from optimization_results.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import jaxatari
from jaxatari.games.mods.pong_mods import LazyEnemyWrapper, RandomizedEnemyWrapper
from jaxatari.wrappers import AtariWrapper, ObjectCentricWrapper, FlattenObservationWrapper


def load_policy_module(policy_path: Path):
    spec = importlib.util.spec_from_file_location(policy_path.stem, policy_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_run_artifacts(policy_dir: Path, use_best_params: bool) -> tuple[Path, dict[str, float], int]:
    results_path = policy_dir / "optimization_results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))
    best_policy_rel = data.get("best_policy_path")
    if best_policy_rel:
        policy_path = PROJECT_ROOT / Path(best_policy_rel)
    else:
        policy_path = policy_dir / "policy_v1.py"
    if use_best_params:
        params = {k: float(v) for k, v in data["best_params"].items()}
    else:
        policy_module = load_policy_module(policy_path)
        params = {k: float(v) for k, v in policy_module.init_params().items()}
    frame_stack = int(data.get("config", {}).get("frame_stack_size", 2) or 2)
    return policy_path, params, frame_stack


def upscale_frame(frame: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return frame
    height, width = frame.shape[:2]
    return cv2.resize(frame, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)


def unwrap_render_state(state):
    current = state
    while hasattr(current, "env_state"):
        current = current.env_state
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description="Render best Pong policy to MP4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--no-best-params", action="store_true",
                        help="Use policy init_params() instead of the unified run's best_params")
    parser.add_argument("--mod", type=str, default="none",
                        choices=["none", "lazy_enemy", "randomized_enemy"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    policy_dir = PROJECT_ROOT / "scripts" / "llm_optimization" / "unified_prompt_main" / "pong"
    policy_path, params, frame_stack = load_run_artifacts(policy_dir, use_best_params=not args.no_best_params)
    policy_module = load_policy_module(policy_path)
    params = {k: jnp.float32(v) for k, v in params.items()}

    out_dir = PROJECT_ROOT / "scripts" / "llm_optimization" / "analysis" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = "" if args.mod == "none" else f"_{args.mod}"
        output_path = out_dir / f"unified_pong_best_policy{suffix}.mp4"

    base_env = jaxatari.make("pong")
    if args.mod == "lazy_enemy":
        base_env = LazyEnemyWrapper(base_env)
    elif args.mod == "randomized_enemy":
        base_env = RandomizedEnemyWrapper(base_env)
    wrapped_env = FlattenObservationWrapper(
        ObjectCentricWrapper(
            AtariWrapper(base_env, episodic_life=True),
            frame_stack_size=frame_stack,
            frame_skip=1,
            clip_reward=False,
        )
    )

    key = jrandom.PRNGKey(args.seed)
    obs, state = wrapped_env.reset(key)
    obs_size = obs.shape[-1]
    obs_flat = jnp.asarray(obs[-obs_size:] if obs.shape[0] > obs_size else obs, dtype=jnp.float32)

    initial_frame = np.asarray(base_env.render(unwrap_render_state(state)), dtype=np.uint8)
    initial_frame = upscale_frame(initial_frame, args.scale)
    height, width = initial_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    total_reward = 0.0
    player_points = 0
    enemy_points = 0
    steps = 0
    done = False

    try:
        while steps < args.max_steps and not done:
            frame = np.asarray(base_env.render(unwrap_render_state(state)), dtype=np.uint8)
            frame = upscale_frame(frame, args.scale)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            action = policy_module.policy(obs_flat, params)
            action = int(jnp.clip(action, 0, 12))
            next_obs, next_state, reward, terminated, truncated, _ = wrapped_env.step(state, action)

            reward_float = float(reward)
            total_reward += reward_float
            if reward_float > 0:
                player_points += 1
            elif reward_float < 0:
                enemy_points += 1

            obs_flat = jnp.asarray(
                next_obs[-obs_size:] if next_obs.shape[0] > obs_size else next_obs,
                dtype=jnp.float32,
            )
            state = next_state
            done = bool(jnp.logical_or(terminated, truncated))
            steps += 1

        # Write final frame a few times for pause at end.
        final_frame = np.asarray(base_env.render(unwrap_render_state(state)), dtype=np.uint8)
        final_frame = upscale_frame(final_frame, args.scale)
        final_bgr = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
        for _ in range(args.fps):
            writer.write(final_bgr)
    finally:
        writer.release()

    print(f"Saved video: {output_path}")
    print(
        f"Summary: mod={args.mod}, steps={steps}, total_reward={total_reward:.1f}, "
        f"player_points={player_points}, enemy_points={enemy_points}, done={done}"
    )


if __name__ == "__main__":
    main()

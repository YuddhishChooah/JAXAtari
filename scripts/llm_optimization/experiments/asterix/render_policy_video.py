#!/usr/bin/env python
"""
Render the best Asterix policy rollout to MP4.

Defaults to the current unified Asterix run:
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
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import jaxatari
from jaxatari.wrappers import AtariWrapper, FlattenObservationWrapper, ObjectCentricWrapper


def load_policy_module(policy_path: Path):
    spec = importlib.util.spec_from_file_location(policy_path.stem, policy_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_run_artifacts(policy_dir: Path) -> tuple[Path, dict[str, float], int]:
    results_path = policy_dir / "optimization_results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))
    best_policy_rel = data.get("best_policy_path")
    if best_policy_rel:
        best_policy_path = PROJECT_ROOT / Path(best_policy_rel)
    else:
        best_policy_path = policy_dir / "policy_v1.py"
    params = {k: float(v) for k, v in data["best_params"].items()}
    frame_stack = int(data.get("config", {}).get("frame_stack_size", 2) or 2)
    return best_policy_path, params, frame_stack


def upscale_frame(frame: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def unwrap_render_state(state):
    current = state
    while hasattr(current, "env_state"):
        current = current.env_state
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description="Render best Asterix policy to MP4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    policy_dir = PROJECT_ROOT / "scripts" / "llm_optimization" / "unified_prompt_main" / "asterix"
    policy_path, params, frame_stack = load_run_artifacts(policy_dir)
    policy_module = load_policy_module(policy_path)
    params = {k: jnp.float32(v) for k, v in params.items()}

    out_dir = PROJECT_ROOT / "scripts" / "llm_optimization" / "analysis" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else out_dir / "unified_asterix_best_policy.mp4"

    base_env = jaxatari.make("asterix")
    wrapped_env = FlattenObservationWrapper(
        ObjectCentricWrapper(
            AtariWrapper(base_env, frame_stack_size=frame_stack, frame_skip=1, episodic_life=False)
        )
    )

    key = jrandom.PRNGKey(args.seed)
    obs, state = wrapped_env.reset(key)
    obs_size = obs.shape[-1]
    obs_flat = jnp.asarray(obs[-obs_size:] if obs.shape[0] > obs_size else obs, dtype=jnp.float32)

    initial_frame = np.asarray(base_env.render(unwrap_render_state(state)), dtype=np.uint8)
    initial_frame = upscale_frame(initial_frame, args.scale)
    height, width = initial_frame.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    total_reward = 0.0
    item_pickups = 0
    steps = 0
    done = False

    try:
        while steps < args.max_steps and not done:
            frame = np.asarray(base_env.render(unwrap_render_state(state)), dtype=np.uint8)
            frame = upscale_frame(frame, args.scale)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            action = policy_module.policy(obs_flat, params)
            action = int(jnp.clip(action, 0, wrapped_env.action_space().n - 1))
            next_obs, next_state, reward, done_flag, _ = wrapped_env.step(state, action)

            reward_float = float(reward)
            total_reward += reward_float
            if reward_float > 0:
                item_pickups += 1

            obs_flat = jnp.asarray(
                next_obs[-obs_size:] if next_obs.shape[0] > obs_size else next_obs,
                dtype=jnp.float32,
            )
            state = next_state
            done = bool(done_flag)
            steps += 1

        final_frame = np.asarray(base_env.render(unwrap_render_state(state)), dtype=np.uint8)
        final_frame = upscale_frame(final_frame, args.scale)
        final_bgr = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
        for _ in range(args.fps):
            writer.write(final_bgr)
    finally:
        writer.release()

    final_state = unwrap_render_state(state)
    lives_remaining = int(getattr(final_state, "lives", 0))
    print(f"Saved video: {output_path}")
    print(
        f"Summary: steps={steps}, total_reward={total_reward:.1f}, "
        f"item_pickups={item_pickups}, lives_remaining={lives_remaining}, done={done}"
    )


if __name__ == "__main__":
    main()

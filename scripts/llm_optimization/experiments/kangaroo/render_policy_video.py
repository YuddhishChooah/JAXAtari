#!/usr/bin/env python
"""
Render a Kangaroo LeGPS policy rollout to MP4 or GIF.

Kangaroo is still exploratory in the thesis state, so this script defaults to
the latest Kangaroo run directory with optimization_results.json instead of a
canonical best-policy folder.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local optional packages
    cv2 = None

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import jaxatari
from jaxatari.wrappers import AtariWrapper, FlattenObservationWrapper, ObjectCentricWrapper


ACTION_NAMES = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}


def load_policy_module(policy_path: Path):
    spec = importlib.util.spec_from_file_location(policy_path.stem, policy_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def find_latest_kangaroo_run() -> Path:
    candidates: list[Path] = []
    search_roots = [
        PROJECT_ROOT / "scripts" / "llm_optimization" / "runs" / "single_game",
        PROJECT_ROOT / "scripts" / "llm_optimization" / "runs" / "unified_suite",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        candidates.extend(root.glob("**/kangaroo/optimization_results.json"))
    if not candidates:
        raise FileNotFoundError(
            "No Kangaroo optimization_results.json found. Pass --policy-dir explicitly."
        )
    latest_results = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest_results.parent


def load_run_artifacts(
    policy_dir: Path,
    *,
    policy_path_override: Path | None = None,
    history_index: int | None = None,
) -> tuple[Path, dict[str, float], int]:
    results_path = policy_dir / "optimization_results.json"
    data = json.loads(results_path.read_text(encoding="utf-8-sig"))
    if history_index is not None:
        history = data.get("history", [])
        if history_index < 0 or history_index >= len(history):
            raise IndexError(f"--history-index {history_index} outside history length {len(history)}")
        entry = history[history_index]
        params = {key: float(value) for key, value in entry["best_params"].items()}
        policy_path = PROJECT_ROOT / Path(entry.get("filepath", ""))
    elif policy_path_override is not None:
        policy_path = policy_path_override if policy_path_override.is_absolute() else PROJECT_ROOT / policy_path_override
        policy_module = load_policy_module(policy_path)
        params = {key: float(value) for key, value in policy_module.init_params().items()}
    else:
        best_policy_rel = data.get("best_policy_path")
        if best_policy_rel:
            policy_path = PROJECT_ROOT / Path(best_policy_rel)
        else:
            policy_path = policy_dir / "policy_v1.py"
        params = {key: float(value) for key, value in data["best_params"].items()}
    frame_stack = int(data.get("config", {}).get("frame_stack_size", 1) or 1)
    return policy_path, params, frame_stack


def upscale_frame(frame: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return frame
    return np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)


def unwrap_render_state(state):
    current = state
    while True:
        if hasattr(current, "atari_state"):
            current = current.atari_state
        elif hasattr(current, "env_state"):
            current = current.env_state
        else:
            return current


def jax_scalar_to_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except TypeError:
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Kangaroo LeGPS policy to MP4")
    parser.add_argument("--seed", type=int, default=20260480)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--policy-dir", type=str, default=None)
    parser.add_argument("--policy-path", type=str, default=None)
    parser.add_argument(
        "--history-index",
        type=int,
        default=None,
        help="Use policy/params from optimization_results.json history entry, zero-based.",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    policy_dir = Path(args.policy_dir) if args.policy_dir else find_latest_kangaroo_run()
    if not policy_dir.is_absolute():
        policy_dir = PROJECT_ROOT / policy_dir

    policy_path_override = Path(args.policy_path) if args.policy_path else None
    policy_path, params, frame_stack = load_run_artifacts(
        policy_dir,
        policy_path_override=policy_path_override,
        history_index=args.history_index,
    )
    policy_module = load_policy_module(policy_path)
    params = {key: jnp.float32(value) for key, value in params.items()}

    out_dir = PROJECT_ROOT / "scripts" / "llm_optimization" / "analysis" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else out_dir / "kangaroo_policy_rollout.gif"
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_suffix = output_path.suffix.lower()
    if output_suffix not in {".mp4", ".gif"}:
        raise ValueError("Output path must end in .mp4 or .gif")
    if output_suffix == ".mp4" and cv2 is None:
        raise RuntimeError("OpenCV is not installed in this environment; use a .gif output path instead.")

    base_env = jaxatari.make("kangaroo")
    wrapped_env = FlattenObservationWrapper(
        ObjectCentricWrapper(
            AtariWrapper(base_env, episodic_life=True),
            frame_stack_size=frame_stack,
            frame_skip=1,
            clip_reward=False,
        )
    )

    obs, state = wrapped_env.reset(jrandom.PRNGKey(args.seed))
    obs_size = obs.shape[-1]
    obs_flat = jnp.asarray(obs[-obs_size:] if obs.shape[0] > obs_size else obs, dtype=jnp.float32)

    initial_frame = np.asarray(base_env.render(unwrap_render_state(state)), dtype=np.uint8)
    initial_frame = upscale_frame(initial_frame, args.scale)
    height, width = initial_frame.shape[:2]
    writer = None
    gif_frames: list[Image.Image] = []
    if output_suffix == ".mp4":
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")

    def write_frame(frame: np.ndarray) -> None:
        if output_suffix == ".mp4":
            assert writer is not None
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            gif_frames.append(Image.fromarray(frame))

    total_reward = 0.0
    reward_events = 0
    action_counts: Counter[int] = Counter()
    y_values: list[float] = []
    x_values: list[float] = []
    done = False
    steps = 0

    try:
        while steps < args.max_steps and not done:
            if steps % max(1, args.frame_stride) == 0:
                frame = np.asarray(base_env.render(unwrap_render_state(state)), dtype=np.uint8)
                frame = upscale_frame(frame, args.scale)
                write_frame(frame)

            x_values.append(float(obs_flat[0]))
            y_values.append(float(obs_flat[1]))

            action = int(jnp.clip(policy_module.policy(obs_flat, params), 0, wrapped_env.action_space().n - 1))
            action_counts[action] += 1
            next_obs, next_state, reward, terminated, truncated, _ = wrapped_env.step(state, action)

            reward_float = float(reward)
            total_reward += reward_float
            if reward_float != 0.0:
                reward_events += 1

            obs_flat = jnp.asarray(
                next_obs[-obs_size:] if next_obs.shape[0] > obs_size else next_obs,
                dtype=jnp.float32,
            )
            state = next_state
            done = bool(jnp.logical_or(terminated, truncated))
            steps += 1

        final_frame = np.asarray(base_env.render(unwrap_render_state(state)), dtype=np.uint8)
        final_frame = upscale_frame(final_frame, args.scale)
        for _ in range(args.fps):
            write_frame(final_frame)
    finally:
        if writer is not None:
            writer.release()

    if output_suffix == ".gif":
        if not gif_frames:
            raise RuntimeError("No GIF frames were recorded")
        gif_frames[0].save(
            output_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=max(1, int(1000 / max(1, args.fps))),
            loop=0,
        )

    final_state = unwrap_render_state(state)
    readable_actions = {
        f"{action}:{ACTION_NAMES.get(action, 'UNKNOWN')}": count
        for action, count in sorted(action_counts.items())
    }
    print(f"Saved video: {output_path}")
    print(f"Policy dir: {policy_dir}")
    print(f"Policy: {policy_path}")
    print(
        f"Summary: steps={steps}, total_reward={total_reward:.1f}, "
        f"reward_events={reward_events}, done={done}"
    )
    print(f"Action counts: {readable_actions}")
    if x_values and y_values:
        print(
            "Position ranges: "
            f"x=({min(x_values):.1f}, {max(x_values):.1f}), "
            f"y=({min(y_values):.1f}, {max(y_values):.1f}), "
            f"final=({x_values[-1]:.1f}, {y_values[-1]:.1f})"
        )
    print(
        "Final raw state: "
        f"score={jax_scalar_to_int(getattr(final_state, 'score', None))}, "
        f"lives={jax_scalar_to_int(getattr(final_state, 'lives', None))}, "
        f"level={jax_scalar_to_int(getattr(final_state, 'current_level', None))}"
    )


if __name__ == "__main__":
    main()

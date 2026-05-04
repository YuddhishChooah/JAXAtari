#!/usr/bin/env python
"""Probe short action blocks at Kangaroo's first post-200 ladder top.

The shaped-v2 controller reliably reaches the first 200-point reward, then
climbs into a ladder-top trap. This script replays to the known post-reward
state at y=140 and tries short forced-action blocks before returning control to
the policy. It is API-free and writes a compact JSON artifact.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jrandom

import jaxatari
from jaxatari.wrappers import AtariWrapper, FlattenObservationWrapper, ObjectCentricWrapper

from render_policy_video import ACTION_NAMES, PROJECT_ROOT, jax_scalar_to_int, unwrap_render_state


DEFAULT_POLICY = (
    "scripts/llm_optimization/experiments/kangaroo/"
    "policy_shaped_v2_post200_guard_dodge.py"
)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_policy(policy_path: Path):
    spec = importlib.util.spec_from_file_location(policy_path.stem, policy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import policy from {policy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_env(frame_stack: int):
    base_env = jaxatari.make("kangaroo")
    env = FlattenObservationWrapper(
        ObjectCentricWrapper(
            AtariWrapper(base_env, episodic_life=True),
            frame_stack_size=frame_stack,
            frame_skip=1,
            clip_reward=False,
        )
    )
    return env


def flatten_obs(obs):
    obs_size = obs.shape[-1]
    return jnp.asarray(obs[-obs_size:] if obs.shape[0] > obs_size else obs, dtype=jnp.float32)


def state_summary(state, obs_flat, *, total_reward: float, step: int) -> dict[str, Any]:
    raw = unwrap_render_state(state)
    return {
        "step": step,
        "score": jax_scalar_to_int(getattr(raw, "score", None)),
        "lives": jax_scalar_to_int(getattr(raw, "lives", None)),
        "total_reward": float(total_reward),
        "player_x": float(obs_flat[0]),
        "player_y": float(obs_flat[1]),
        "player_w": float(obs_flat[2]),
        "player_h": float(obs_flat[3]),
        "falling_x": float(obs_flat[368]),
        "falling_y": float(obs_flat[369]),
        "falling_active": float(obs_flat[372]),
    }


def replay_to_probe_state(policy_module, params, *, seed: int, frame_stack: int, target_step: int):
    env = make_env(frame_stack)
    obs, state = env.reset(jrandom.PRNGKey(seed))
    obs_flat = flatten_obs(obs)
    total_reward = 0.0

    for step in range(target_step):
        action = int(jnp.clip(policy_module.policy(obs_flat, params), 0, env.action_space().n - 1))
        obs, state, reward, terminated, truncated, _ = env.step(state, action)
        total_reward += float(reward)
        obs_flat = flatten_obs(obs)
        if bool(terminated) or bool(truncated):
            raise RuntimeError(f"Policy terminated before probe state at step {step}.")

    return env, obs, state, obs_flat, total_reward


def rollout_candidate(
    env,
    obs,
    state,
    obs_flat,
    policy_module,
    params,
    *,
    start_step: int,
    start_reward: float,
    action_blocks: list[tuple[int, int]],
    continue_steps: int,
) -> dict[str, Any]:
    total_reward = float(start_reward)
    min_y = float(obs_flat[1])
    max_x = float(obs_flat[0])
    min_x = float(obs_flat[0])
    first_life_loss_step = None
    done = False
    step = start_step
    action_counts: dict[str, int] = {}

    forced: list[int] = []
    for action, hold in action_blocks:
        forced.extend([action] * hold)

    for local_step in range(len(forced) + continue_steps):
        if local_step < len(forced):
            action = forced[local_step]
        else:
            action = int(jnp.clip(policy_module.policy(obs_flat, params), 0, env.action_space().n - 1))
        action_name = ACTION_NAMES.get(action, "UNKNOWN")
        action_counts[action_name] = action_counts.get(action_name, 0) + 1

        obs, state, reward, terminated, truncated, _ = env.step(state, action)
        total_reward += float(reward)
        obs_flat = flatten_obs(obs)
        step += 1

        raw = unwrap_render_state(state)
        lives = jax_scalar_to_int(getattr(raw, "lives", None))
        if lives < 3 and first_life_loss_step is None:
            first_life_loss_step = step
        min_y = min(min_y, float(obs_flat[1]))
        max_x = max(max_x, float(obs_flat[0]))
        min_x = min(min_x, float(obs_flat[0]))

        if bool(terminated) or bool(truncated):
            done = True
            break

    raw = unwrap_render_state(state)
    final_lives = jax_scalar_to_int(getattr(raw, "lives", None))
    final_score = jax_scalar_to_int(getattr(raw, "score", None))
    final_x = float(obs_flat[0])
    final_y = float(obs_flat[1])

    # This ranking is intentionally mechanics-oriented. Real reward remains the
    # final metric, but here we need to discover nonterminal route transitions.
    objective = (
        total_reward
        + (500.0 if not done and final_lives >= 3 else 0.0)
        + max(0.0, 140.0 - min_y)
        + 0.1 * (max_x - min_x)
        - (0.5 * max(0.0, final_y - 140.0))
    )

    return {
        "blocks": [
            {"action": action, "action_name": ACTION_NAMES.get(action, "UNKNOWN"), "hold": hold}
            for action, hold in action_blocks
        ],
        "done": done,
        "first_life_loss_step": first_life_loss_step,
        "total_reward": float(total_reward),
        "score": final_score,
        "lives": final_lives,
        "final_x": final_x,
        "final_y": final_y,
        "min_y": min_y,
        "x_range": max_x - min_x,
        "action_counts": action_counts,
        "objective": objective,
    }


def candidate_key(row: dict[str, Any]) -> tuple:
    return (
        row["total_reward"],
        row["lives"],
        0 if row["done"] else 1,
        row["min_y"] * -1,
        row["x_range"],
        row["objective"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Kangaroo top-ladder transition mechanics")
    parser.add_argument("--policy-path", default=DEFAULT_POLICY)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--target-step", type=int, default=610)
    parser.add_argument("--continue-steps", type=int, default=140)
    parser.add_argument("--beam-size", type=int, default=24)
    parser.add_argument("--output", default="scripts/llm_optimization/analysis/traces/kangaroo_top_ladder_transition_probe.json")
    args = parser.parse_args()

    started = time.time()
    policy_path = resolve_path(args.policy_path)
    policy_module = load_policy(policy_path)
    params = {key: jnp.float32(float(value)) for key, value in policy_module.init_params().items()}

    base_env, base_obs, base_state, base_obs_flat, base_reward = replay_to_probe_state(
        policy_module,
        params,
        seed=args.seed,
        frame_stack=args.frame_stack,
        target_step=args.target_step,
    )
    start = state_summary(base_state, base_obs_flat, total_reward=base_reward, step=args.target_step)

    action_set = [0, 2, 3, 4, 5, 8, 9, 11, 12, 13, 16, 17]
    hold_set = [2, 4, 8, 12, 16, 24]

    single_rows = [
        rollout_candidate(
            base_env,
            base_obs,
            base_state,
            base_obs_flat,
            policy_module,
            params,
            start_step=args.target_step,
            start_reward=base_reward,
            action_blocks=[(action, hold)],
            continue_steps=args.continue_steps,
        )
        for action in action_set
        for hold in hold_set
    ]
    best_single = sorted(single_rows, key=candidate_key, reverse=True)[: args.beam_size]

    second_rows = []
    second_actions = [2, 3, 4, 5, 8, 9, 13, 16, 17]
    second_holds = [4, 8, 12, 16]
    for row in best_single:
        prefix = [(block["action"], block["hold"]) for block in row["blocks"]]
        for action in second_actions:
            for hold in second_holds:
                second_rows.append(
                    rollout_candidate(
                        base_env,
                        base_obs,
                        base_state,
                        base_obs_flat,
                        policy_module,
                        params,
                        start_step=args.target_step,
                        start_reward=base_reward,
                        action_blocks=prefix + [(action, hold)],
                        continue_steps=args.continue_steps,
                    )
                )

    all_rows = single_rows + second_rows
    top_rows = sorted(all_rows, key=candidate_key, reverse=True)
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policy": str(policy_path.relative_to(PROJECT_ROOT)),
        "seed": args.seed,
        "start": start,
        "search": {
            "action_set": [ACTION_NAMES.get(action, str(action)) for action in action_set],
            "hold_set": hold_set,
            "second_action_set": [ACTION_NAMES.get(action, str(action)) for action in second_actions],
            "second_hold_set": second_holds,
            "continue_steps": args.continue_steps,
            "beam_size": args.beam_size,
            "elapsed_seconds": time.time() - started,
        },
        "summary": {
            "single_candidates": len(single_rows),
            "second_candidates": len(second_rows),
            "total_candidates": len(all_rows),
            "above_200_count": sum(1 for row in all_rows if row["total_reward"] > 200.0),
            "alive_count": sum(1 for row in all_rows if not row["done"] and row["lives"] >= 3),
            "best_total_reward": max(row["total_reward"] for row in all_rows),
            "best_score": max(row["score"] for row in all_rows),
        },
        "top_by_objective": top_rows[:60],
        "above_200": [row for row in top_rows if row["total_reward"] > 200.0][:60],
        "alive": [row for row in top_rows if not row["done"] and row["lives"] >= 3][:60],
    }

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {output_path}")
    print(f"Start: {start}")
    print(f"Summary: {payload['summary']}")
    print("Top candidates:")
    for row in payload["top_by_objective"][:8]:
        blocks = " + ".join(f"{block['action_name']}x{block['hold']}" for block in row["blocks"])
        print(
            f"  {blocks}: reward={row['total_reward']:.1f}, score={row['score']}, "
            f"lives={row['lives']}, done={row['done']}, final=({row['final_x']:.1f},{row['final_y']:.1f}), "
            f"min_y={row['min_y']:.1f}, x_range={row['x_range']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

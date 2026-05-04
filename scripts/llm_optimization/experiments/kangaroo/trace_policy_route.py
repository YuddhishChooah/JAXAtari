#!/usr/bin/env python
"""Trace a Kangaroo LeGPS policy around route/reward failure points."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import jaxatari
from jaxatari.wrappers import AtariWrapper, FlattenObservationWrapper, ObjectCentricWrapper

from render_policy_video import (
    ACTION_NAMES,
    PROJECT_ROOT,
    find_latest_kangaroo_run,
    jax_scalar_to_int,
    load_policy_module,
    load_run_artifacts,
    unwrap_render_state,
)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def object_flag(value: Any, default: bool = False) -> bool:
    try:
        return bool(value)
    except (TypeError, ValueError):
        return default


def player_from_state(raw_state: Any, obs_flat: jnp.ndarray) -> dict[str, Any]:
    player = getattr(raw_state, "player", None)
    return {
        "x": to_float(getattr(player, "x", obs_flat[0])),
        "y": to_float(getattr(player, "y", obs_flat[1])),
        "w": to_float(getattr(player, "w", obs_flat[2])),
        "h": to_float(getattr(player, "h", obs_flat[3])),
        "bottom_y": to_float(getattr(player, "y", obs_flat[1])) + to_float(getattr(player, "h", obs_flat[3])),
        "orientation": to_float(getattr(player, "orientation", obs_flat[7])),
        "is_climbing": object_flag(getattr(player, "is_climbing", False)),
        "is_crashing": object_flag(getattr(player, "is_crashing", False)),
    }


def active_ladders(obs_flat: jnp.ndarray, player: dict[str, Any], limit: int) -> list[dict[str, float]]:
    lx = np.asarray(obs_flat[168:188], dtype=float)
    ly = np.asarray(obs_flat[188:208], dtype=float)
    lw = np.asarray(obs_flat[208:228], dtype=float)
    lh = np.asarray(obs_flat[228:248], dtype=float)
    la = np.asarray(obs_flat[248:268], dtype=float) > 0.5

    pcx = player["x"] + player["w"] * 0.5
    pby = player["bottom_y"]
    py = player["y"]
    ladders: list[dict[str, float]] = []
    for index in range(20):
        if not la[index]:
            continue
        center_x = lx[index] + lw[index] * 0.5
        top_y = ly[index]
        bottom_y = ly[index] + lh[index]
        ladders.append(
            {
                "index": float(index),
                "x": float(lx[index]),
                "center_x": float(center_x),
                "top_y": float(top_y),
                "bottom_y": float(bottom_y),
                "w": float(lw[index]),
                "h": float(lh[index]),
                "dx": float(center_x - pcx),
                "abs_dx": float(abs(center_x - pcx)),
                "bottom_diff": float(bottom_y - pby),
                "abs_bottom_diff": float(abs(bottom_y - pby)),
                "top_above_player": float(top_y < py),
            }
        )

    ladders.sort(key=lambda item: (item["abs_bottom_diff"], item["abs_dx"]))
    return ladders[:limit]


def active_hazards(obs_flat: jnp.ndarray, player: dict[str, Any], limit: int) -> dict[str, list[dict[str, float]]]:
    px = player["x"] + player["w"] * 0.5
    py = player["y"]

    def collect_group(
        xs: np.ndarray,
        ys: np.ndarray,
        active: np.ndarray,
        states: np.ndarray | None = None,
    ) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for index, is_active in enumerate(active > 0.5):
            if not is_active:
                continue
            dx = float(xs[index] - px)
            dy = float(ys[index] - py)
            row = {
                "index": float(index),
                "x": float(xs[index]),
                "y": float(ys[index]),
                "dx": dx,
                "dy": dy,
                "distance": float((dx * dx + dy * dy) ** 0.5),
            }
            if states is not None:
                row["state"] = float(states[index])
            rows.append(row)
        rows.sort(key=lambda item: item["distance"])
        return rows[:limit]

    monkeys = collect_group(
        np.asarray(obs_flat[376:380], dtype=float),
        np.asarray(obs_flat[380:384], dtype=float),
        np.asarray(obs_flat[392:396], dtype=float),
        np.asarray(obs_flat[400:404], dtype=float),
    )
    thrown = collect_group(
        np.asarray(obs_flat[408:412], dtype=float),
        np.asarray(obs_flat[412:416], dtype=float),
        np.asarray(obs_flat[424:428], dtype=float),
        np.asarray(obs_flat[432:436], dtype=float),
    )

    falling_active = float(obs_flat[372]) > 0.5
    falling: list[dict[str, float]] = []
    if falling_active:
        dx = float(obs_flat[368] - px)
        dy = float(obs_flat[369] - py)
        falling.append(
            {
                "index": 0.0,
                "x": float(obs_flat[368]),
                "y": float(obs_flat[369]),
                "dx": dx,
                "dy": dy,
                "distance": float((dx * dx + dy * dy) ** 0.5),
            }
        )

    return {
        "monkeys": monkeys,
        "thrown_coconuts": thrown,
        "falling_coconut": falling,
    }


def compact_record(
    *,
    step: int,
    obs_flat: jnp.ndarray,
    raw_state: Any,
    action: int,
    reward: float,
    next_obs_flat: jnp.ndarray,
    next_raw_state: Any,
    terminated: bool,
    truncated: bool,
    ladder_limit: int,
    hazard_limit: int,
) -> dict[str, Any]:
    player = player_from_state(raw_state, obs_flat)
    next_player = player_from_state(next_raw_state, next_obs_flat)
    lives = jax_scalar_to_int(getattr(raw_state, "lives", None))
    next_lives = jax_scalar_to_int(getattr(next_raw_state, "lives", None))
    score = jax_scalar_to_int(getattr(raw_state, "score", None))
    next_score = jax_scalar_to_int(getattr(next_raw_state, "score", None))

    return {
        "step": step,
        "action": action,
        "action_name": ACTION_NAMES.get(action, "UNKNOWN"),
        "reward": reward,
        "score_before": score,
        "score_after": next_score,
        "lives_before": lives,
        "lives_after": next_lives,
        "level_before": jax_scalar_to_int(getattr(raw_state, "current_level", None)),
        "level_after": jax_scalar_to_int(getattr(next_raw_state, "current_level", None)),
        "terminated": terminated,
        "truncated": truncated,
        "player": player,
        "next_player": next_player,
        "nearest_ladders": active_ladders(obs_flat, player, ladder_limit),
        "nearest_hazards": active_hazards(obs_flat, player, hazard_limit),
    }


def summarize_trace(records: list[dict[str, Any]], all_steps: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [row for row in all_steps if row["reward"] != 0.0]
    life_losses = [row for row in all_steps if row["lives_after"] < row["lives_before"]]
    actions = Counter(row["action_name"] for row in all_steps)

    first_reward_step = rewards[0]["step"] if rewards else None
    post_reward_rows = [row for row in all_steps if first_reward_step is not None and row["step"] > first_reward_step]
    post_reward_min_y = None
    post_reward_max_upward_progress = None
    if post_reward_rows:
        reward_y = rewards[0]["player"]["y"]
        post_reward_min_y = min(row["player"]["y"] for row in post_reward_rows)
        post_reward_max_upward_progress = reward_y - post_reward_min_y

    return {
        "steps_recorded": len(records),
        "total_steps": len(all_steps),
        "total_reward": float(sum(row["reward"] for row in all_steps)),
        "reward_events": rewards,
        "life_loss_events": life_losses,
        "action_counts": dict(actions),
        "first_reward_step": first_reward_step,
        "post_reward_min_y": post_reward_min_y,
        "post_reward_max_upward_progress": post_reward_max_upward_progress,
        "final_step": all_steps[-1] if all_steps else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace Kangaroo LeGPS route behavior")
    parser.add_argument("--seed", type=int, default=20260480)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--window-start", type=int, default=560)
    parser.add_argument("--window-end", type=int, default=840)
    parser.add_argument("--policy-dir", type=str, default=None)
    parser.add_argument("--policy-path", type=str, default=None)
    parser.add_argument("--history-index", type=int, default=None)
    parser.add_argument("--ladder-limit", type=int, default=6)
    parser.add_argument("--hazard-limit", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    base_env = jaxatari.make("kangaroo")
    env = FlattenObservationWrapper(
        ObjectCentricWrapper(
            AtariWrapper(base_env, episodic_life=True),
            frame_stack_size=frame_stack,
            frame_skip=1,
            clip_reward=False,
        )
    )

    obs, state = env.reset(jrandom.PRNGKey(args.seed))
    obs_size = obs.shape[-1]
    obs_flat = jnp.asarray(obs[-obs_size:] if obs.shape[0] > obs_size else obs, dtype=jnp.float32)

    window_records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    all_compact: list[dict[str, Any]] = []

    for step in range(args.max_steps):
        raw_state = unwrap_render_state(state)
        action = int(jnp.clip(policy_module.policy(obs_flat, params), 0, env.action_space().n - 1))
        next_obs, next_state, reward, terminated, truncated, _ = env.step(state, action)
        next_obs_flat = jnp.asarray(
            next_obs[-obs_size:] if next_obs.shape[0] > obs_size else next_obs,
            dtype=jnp.float32,
        )
        next_raw_state = unwrap_render_state(next_state)
        reward_float = float(reward)
        record = compact_record(
            step=step,
            obs_flat=obs_flat,
            raw_state=raw_state,
            action=action,
            reward=reward_float,
            next_obs_flat=next_obs_flat,
            next_raw_state=next_raw_state,
            terminated=bool(terminated),
            truncated=bool(truncated),
            ladder_limit=args.ladder_limit,
            hazard_limit=args.hazard_limit,
        )
        all_compact.append(
            {
                "step": record["step"],
                "action": record["action"],
                "action_name": record["action_name"],
                "reward": record["reward"],
                "score_after": record["score_after"],
                "lives_before": record["lives_before"],
                "lives_after": record["lives_after"],
                "terminated": record["terminated"],
                "truncated": record["truncated"],
                "player": record["player"],
            }
        )
        if args.window_start <= step <= args.window_end:
            window_records.append(record)
        if reward_float != 0.0 or record["lives_after"] < record["lives_before"] or bool(terminated) or bool(truncated):
            event_records.append(record)

        obs_flat = next_obs_flat
        state = next_state
        if bool(terminated) or bool(truncated):
            break

    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT
        / "scripts"
        / "llm_optimization"
        / "analysis"
        / "traces"
        / f"{policy_dir.name}_{policy_path.stem}_route_trace.json"
    )
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_trace(window_records, all_compact)
    payload = {
        "policy_dir": str(policy_dir.relative_to(PROJECT_ROOT)),
        "policy_path": str(policy_path.relative_to(PROJECT_ROOT)),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "window": {
            "start": args.window_start,
            "end": args.window_end,
        },
        "summary": summary,
        "event_records": event_records,
        "window_records": window_records,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote trace: {output_path}")
    print(
        "Summary: "
        f"steps={summary['total_steps']}, reward={summary['total_reward']:.1f}, "
        f"reward_events={len(summary['reward_events'])}, life_losses={len(summary['life_loss_events'])}"
    )
    if summary["reward_events"]:
        event = summary["reward_events"][0]
        print(
            "First reward: "
            f"step={event['step']}, action={event['action']}:{event['action_name']}, "
            f"x={event['player']['x']:.1f}, y={event['player']['y']:.1f}, "
            f"score_after={event['score_after']}"
        )
    if summary["life_loss_events"]:
        event = summary["life_loss_events"][0]
        print(
            "First life loss: "
            f"step={event['step']}, action={event['action']}:{event['action_name']}, "
            f"x={event['player']['x']:.1f}, y={event['player']['y']:.1f}, "
            f"score_after={event['score_after']}, lives_after={event['lives_after']}"
        )


if __name__ == "__main__":
    main()

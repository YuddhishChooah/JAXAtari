"""Kangaroo shaped-v2 patch with post-200 punch guard and top-state dodge.

This extends policy_shaped_v2_post200_guard with one extra priority change:
when the player is near the ladder top and a coconut is dangerous, dodge before
trying to dismount. This tests whether the post-200 death is mainly caused by
near-top dismount overriding hazard avoidance.
"""

import jax.numpy as jnp

from scripts.llm_optimization.experiments.kangaroo.policy_shaped_v2_post200_guard import (
    LEFT,
    LEFTFIRE,
    P_H,
    P_W,
    P_X,
    P_Y,
    RIGHT,
    RIGHTFIRE,
    UP,
    _coconut_danger,
    _move_toward_x,
    _nearest_monkey,
    _on_column_ladder,
    _select_next_ladder_from_top,
    _select_reachable_ladder,
    init_params,
)


def policy(obs_flat, params):
    obs = obs_flat.astype(jnp.float32)

    px = obs[P_X]
    py = obs[P_Y]
    pw = obs[P_W]
    ph = obs[P_H]
    pcx = px + pw * 0.5
    pby = py + ph
    pcy = py + ph * 0.5

    reach_tol = params["reach_tol"]
    align_tol = params["align_tol"]
    top_tol = params["top_tol"]
    punch_dx = params["punch_dx"]
    punch_dy = params["punch_dy"]
    danger_r = params["danger_r"]
    col_frac = params["col_frac"]

    any_reach, r_lcx, r_lty, r_lby, _ = _select_reachable_ladder(obs, pcx, pby, reach_tol)
    any_on, c_lcx, c_lty, c_lby, _ = _on_column_ladder(obs, pcx, pby, py, col_frac)

    tgt_lcx = jnp.where(any_on, c_lcx, r_lcx)
    tgt_lty = jnp.where(any_on, c_lty, r_lty)
    tgt_lby = jnp.where(any_on, c_lby, r_lby)

    align_dx = tgt_lcx - pcx
    near_top = any_on & (py < tgt_lty + top_tol)
    above_bottom = py < tgt_lby - 2.0
    climbing = any_on & above_bottom & ~near_top

    has_mon, mdx, mdy = _nearest_monkey(obs, pcx, pcy)
    monkey_in_punch = has_mon & (jnp.abs(mdx) < punch_dx) & (jnp.abs(mdy) < punch_dy)
    monkey_in_punch = monkey_in_punch & ~(any_on & near_top)
    punch_action = jnp.where(mdx > 0, RIGHTFIRE, LEFTFIRE)

    coco_danger, threat_x = _coconut_danger(obs, pcx, pcy, danger_r)
    dodge_action = jnp.where(threat_x > pcx, LEFT, RIGHT)

    has_next, next_lcx = _select_next_ladder_from_top(obs, pcx, tgt_lty, reach_tol)
    dismount_dx = next_lcx - pcx
    dismount_action = jnp.where(has_next, jnp.where(dismount_dx > 0, RIGHT, LEFT), RIGHT)

    approach_action = _move_toward_x(align_dx, align_tol)
    approach_action = jnp.where(
        (jnp.abs(align_dx) < align_tol) & (any_reach | any_on),
        UP,
        approach_action,
    )

    action = jnp.where(
        monkey_in_punch,
        punch_action,
        jnp.where(
            climbing,
            UP,
            jnp.where(
                coco_danger,
                dodge_action,
                jnp.where(
                    near_top,
                    dismount_action,
                    jnp.where(any_reach | any_on, approach_action, RIGHT),
                ),
            ),
        ),
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)

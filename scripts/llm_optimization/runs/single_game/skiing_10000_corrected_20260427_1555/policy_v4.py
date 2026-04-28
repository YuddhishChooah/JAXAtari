"""
Auto-generated policy v4
Generated at: 2026-04-27 16:16:48
"""

import jax
import jax.numpy as jnp

# Action constants
NOOP = 0
RIGHT = 1
LEFT = 2
FIRE = 3
DOWN = 4

FRAME_SIZE = 73

# Frame indices
SKIER_X = 0
SKIER_Y = 1
SKIER_ORI = 7
FLAG_X = 8
FLAG_Y = 10
FLAG_ACT = 16
TREE_X = 24
TREE_Y = 28
TREE_ACT = 40

GATE_HALF_WIDTH = 16.0


def init_params():
    return {
        "steer_gain": jnp.array(0.11),       # dx -> desired heading offset
        "ori_dead": jnp.array(0.55),         # heading dead zone for tucking
        "turn_commit": jnp.array(0.95),      # larger threshold to start a turn
        "dead_zone": jnp.array(6.0),         # px tolerance to switch to next gate
        "tree_danger_y": jnp.array(22.0),
        "tree_danger_x": jnp.array(20.0),
    }


def _orientation_to_index(ori):
    angles = jnp.array([270.0, 292.5, 315.0, 337.5, 22.5, 45.0, 67.5, 90.0])
    diffs = jnp.abs(angles - ori)
    return jnp.argmin(diffs)


def _select_gate(curr, dead_zone):
    fx = jax.lax.dynamic_slice(curr, (FLAG_X,), (2,))
    fy = jax.lax.dynamic_slice(curr, (FLAG_Y,), (2,))
    fa = jax.lax.dynamic_slice(curr, (FLAG_ACT,), (2,))
    sx = curr[SKIER_X]
    sy = curr[SKIER_Y]

    valid = (fa > 0.5) & (fy >= sy - 4.0)
    score = jnp.where(valid, fy, 1e6)
    idx0 = jnp.argmin(score)

    # primary target
    primary_x = fx[idx0]
    primary_y = fy[idx0]
    primary_active = fa[idx0]

    # if primary already centered, pre-aim at the other active gate
    other_idx = 1 - idx0
    other_active = fa[other_idx] > 0.5
    primary_dx = (primary_x + GATE_HALF_WIDTH) - sx
    centered = jnp.abs(primary_dx) < dead_zone

    use_other = centered & other_active
    out_x = jnp.where(use_other, fx[other_idx], primary_x)
    out_y = jnp.where(use_other, fy[other_idx], primary_y)

    # fallback if nothing valid
    any_valid = jnp.any(valid)
    fallback_idx = jnp.argmax(fa)
    out_x = jnp.where(any_valid, out_x, fx[fallback_idx])
    out_y = jnp.where(any_valid, out_y, fy[fallback_idx])
    return out_x, out_y, primary_active


def _tree_threat(curr, params):
    sx = curr[SKIER_X]
    sy = curr[SKIER_Y]
    tx = jax.lax.dynamic_slice(curr, (TREE_X,), (4,))
    ty = jax.lax.dynamic_slice(curr, (TREE_Y,), (4,))
    ta = jax.lax.dynamic_slice(curr, (TREE_ACT,), (4,))
    dy = ty - sy
    dx = tx - sx
    threat = (ta > 0.5) & (dy > -4.0) & (dy < params["tree_danger_y"]) & (jnp.abs(dx) < params["tree_danger_x"])
    any_threat = jnp.any(threat)
    score = jnp.where(threat, jnp.abs(dy), 1e6)
    idx = jnp.argmin(score)
    return any_threat, tx[idx] - sx


def policy(obs_flat, params):
    curr = jax.lax.dynamic_slice(obs_flat, (FRAME_SIZE,), (FRAME_SIZE,))

    sx = curr[SKIER_X]
    ori = curr[SKIER_ORI]
    ori_idx = _orientation_to_index(ori).astype(jnp.float32)

    flag_x, _, _ = _select_gate(curr, params["dead_zone"])
    target_x = flag_x + GATE_HALF_WIDTH
    dx = target_x - sx

    # Tree avoidance override
    threat, tree_dx = _tree_threat(curr, params)
    avoid_dx = -jnp.sign(tree_dx) * 30.0
    dx = jnp.where(threat, avoid_dx, dx)

    # Desired heading index: 3.5 is straight
    desired_idx = 3.5 + params["steer_gain"] * dx
    desired_idx = jnp.clip(desired_idx, 2.0, 5.0)

    idx_err = desired_idx - ori_idx

    # Heading is near-straight: tuck regardless of dx
    heading_straight = jnp.abs(ori_idx - 3.5) < (params["ori_dead"] + 0.5)

    # Turn commit thresholds (asymmetric: harder to start a turn)
    need_right = idx_err > params["turn_commit"]
    need_left = idx_err < -params["turn_commit"]

    # When already near desired heading, just tuck
    action = jnp.where(
        need_right, RIGHT,
        jnp.where(need_left, LEFT,
                  jnp.where(heading_straight, DOWN, DOWN))
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
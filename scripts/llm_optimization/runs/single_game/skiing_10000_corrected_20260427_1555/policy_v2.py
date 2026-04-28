"""
Auto-generated policy v2
Generated at: 2026-04-27 16:03:21
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
        "steer_gain": jnp.array(0.10),     # maps dx to desired heading offset
        "ori_dead": jnp.array(0.55),       # heading-index dead zone
        "urgency_y": jnp.array(25.0),      # y-distance below which steering is urgent
        "tree_danger_y": jnp.array(20.0),  # tree threat y range
        "tree_danger_x": jnp.array(22.0),  # tree threat x range
        "far_dx_tol": jnp.array(14.0),     # dx tolerance when gate is far -> just tuck
    }


def _orientation_to_index(ori):
    angles = jnp.array([270.0, 292.5, 315.0, 337.5, 22.5, 45.0, 67.5, 90.0])
    diffs = jnp.abs(angles - ori)
    return jnp.argmin(diffs)


def _select_gate(curr):
    fx = jax.lax.dynamic_slice(curr, (FLAG_X,), (2,))
    fy = jax.lax.dynamic_slice(curr, (FLAG_Y,), (2,))
    fa = jax.lax.dynamic_slice(curr, (FLAG_ACT,), (2,))
    sy = curr[SKIER_Y]
    valid = (fa > 0.5) & (fy >= sy - 4.0)
    score = jnp.where(valid, fy, 1e6)
    idx = jnp.argmin(score)
    any_valid = jnp.any(valid)
    fallback_idx = jnp.argmax(fa)
    idx = jnp.where(any_valid, idx, fallback_idx)
    return fx[idx], fy[idx], fa[idx]


def _tree_threat(curr, params):
    sx = curr[SKIER_X]
    sy = curr[SKIER_Y]
    tx = jax.lax.dynamic_slice(curr, (TREE_X,), (4,))
    ty = jax.lax.dynamic_slice(curr, (TREE_Y,), (4,))
    ta = jax.lax.dynamic_slice(curr, (TREE_ACT,), (4,))
    dy = ty - sy
    dx = tx - sx
    threatening = (
        (ta > 0.5)
        & (dy > -4.0)
        & (dy < params["tree_danger_y"])
        & (jnp.abs(dx) < params["tree_danger_x"])
    )
    any_threat = jnp.any(threatening)
    score = jnp.where(threatening, jnp.abs(dy), 1e6)
    idx = jnp.argmin(score)
    return any_threat, tx[idx] - sx


def policy(obs_flat, params):
    curr = jax.lax.dynamic_slice(obs_flat, (FRAME_SIZE,), (FRAME_SIZE,))

    sx = curr[SKIER_X]
    sy = curr[SKIER_Y]
    ori = curr[SKIER_ORI]
    ori_idx = _orientation_to_index(ori).astype(jnp.float32)

    flag_x, flag_y, _ = _select_gate(curr)
    target_x = flag_x + GATE_HALF_WIDTH
    dx = target_x - sx

    # Tree avoidance overrides gate target when imminent
    threat, tree_dx = _tree_threat(curr, params)
    avoid_dx = -jnp.sign(tree_dx) * 30.0
    dx = jnp.where(threat, avoid_dx, dx)

    # Urgency: how close is the gate vertically?
    gate_dist_y = jnp.maximum(flag_y - sy, 0.0)
    # urgency in [0,1]: 1 if gate is at skier, ~0 if far
    urgency = jnp.clip(1.0 - gate_dist_y / jnp.maximum(params["urgency_y"], 1.0), 0.0, 1.0)
    # When threat, force urgency high to steer immediately
    urgency = jnp.where(threat, 1.0, urgency)

    # Desired heading index, scaled by urgency so we don't fight when far
    desired_idx = 3.5 + params["steer_gain"] * dx * urgency
    desired_idx = jnp.clip(desired_idx, 2.0, 5.0)
    idx_err = desired_idx - ori_idx

    # Decide steering: only when error is meaningful
    need_right = idx_err > params["ori_dead"]
    need_left = idx_err < -params["ori_dead"]

    # When gate is far and dx is moderate, ignore steering -> tuck
    far_and_ok = (gate_dist_y > params["urgency_y"]) & (jnp.abs(dx) < params["far_dx_tol"])
    need_right = need_right & (~far_and_ok)
    need_left = need_left & (~far_and_ok)

    # Default action is DOWN (tuck) to maximize downhill speed
    action = jnp.where(
        need_right,
        RIGHT,
        jnp.where(need_left, LEFT, DOWN),
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
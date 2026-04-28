"""
Auto-generated policy v5
Generated at: 2026-04-27 16:23:44
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
        "dead_zone": jnp.array(10.0),       # px tolerance to gate center; below this, go straight
        "steer_gain": jnp.array(0.06),      # maps dx to desired heading offset
        "ori_dead": jnp.array(0.7),         # heading-index deadband for steering decision
        "tree_danger_y": jnp.array(22.0),   # y-distance below which a tree is threatening
        "tree_danger_x": jnp.array(20.0),   # x-distance considered in path
        "far_gate_y": jnp.array(25.0),      # if gate is farther than this in y, just tuck
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


def _nearest_tree_threat(curr, params):
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


def _compute_dx(curr, params):
    sx = curr[SKIER_X]
    sy = curr[SKIER_Y]
    flag_x, flag_y, _ = _select_gate(curr)
    target_x = flag_x + GATE_HALF_WIDTH
    dx_gate = target_x - sx
    gate_dy = flag_y - sy

    threat, tree_dx = _nearest_tree_threat(curr, params)
    avoid_dx = -jnp.sign(tree_dx) * 40.0

    # If gate is far away in y, don't bother steering; just tuck.
    far = gate_dy > params["far_gate_y"]
    dx = jnp.where(far & (~threat), 0.0, dx_gate)
    dx = jnp.where(threat, avoid_dx, dx)
    return dx, threat


def _desired_heading_idx(dx, dead_zone, steer_gain, threat):
    # 3.5 is straightest. Snap to 3.5 inside dead zone.
    inside = jnp.abs(dx) < dead_zone
    raw = 3.5 + steer_gain * dx
    desired = jnp.where(inside, 3.5, raw)
    # Normal clip keeps to moderate notches; tree threat allows wider range.
    lo_normal, hi_normal = 2.5, 4.5
    lo_threat, hi_threat = 1.5, 5.5
    lo = jnp.where(threat, lo_threat, lo_normal)
    hi = jnp.where(threat, hi_threat, hi_normal)
    return jnp.clip(desired, lo, hi)


def policy(obs_flat, params):
    curr = jax.lax.dynamic_slice(obs_flat, (FRAME_SIZE,), (FRAME_SIZE,))

    ori = curr[SKIER_ORI]
    ori_idx = _orientation_to_index(ori).astype(jnp.float32)

    dx, threat = _compute_dx(curr, params)
    desired_idx = _desired_heading_idx(
        dx, params["dead_zone"], params["steer_gain"], threat
    )
    idx_err = desired_idx - ori_idx

    need_right = idx_err > params["ori_dead"]
    need_left = idx_err < -params["ori_dead"]

    # Default to DOWN (tuck) whenever heading is acceptable.
    action = jnp.where(
        need_right,
        RIGHT,
        jnp.where(need_left, LEFT, DOWN),
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
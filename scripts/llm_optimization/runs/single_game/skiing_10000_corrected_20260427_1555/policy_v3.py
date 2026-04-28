"""
Auto-generated policy v3
Generated at: 2026-04-27 16:09:56
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
        "dead_zone": jnp.array(6.0),         # px tolerance to gate center
        "steer_gain": jnp.array(0.09),       # dx -> heading offset
        "ori_dead": jnp.array(0.55),         # heading-index dead zone
        "tree_danger_y": jnp.array(20.0),
        "tree_danger_x": jnp.array(20.0),
        "far_gate_dy": jnp.array(30.0),      # gate considered far if dy above this
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


def policy(obs_flat, params):
    curr = jax.lax.dynamic_slice(obs_flat, (FRAME_SIZE,), (FRAME_SIZE,))

    sx = curr[SKIER_X]
    sy = curr[SKIER_Y]
    ori = curr[SKIER_ORI]
    ori_idx = _orientation_to_index(ori).astype(jnp.float32)

    flag_x, flag_y, _ = _select_gate(curr)
    target_x = flag_x + GATE_HALF_WIDTH
    dx = target_x - sx
    gate_dy = flag_y - sy

    # Tree avoidance override
    threat, tree_dx = _nearest_tree_threat(curr, params)
    avoid_dx = -jnp.sign(tree_dx) * 30.0
    dx = jnp.where(threat, avoid_dx, dx)

    # Desired heading index: 3.5 is straight; clip to fast/moderate notches
    desired_idx = 3.5 + params["steer_gain"] * dx
    desired_idx = jnp.clip(desired_idx, 2.0, 5.0)
    idx_err = desired_idx - ori_idx

    # Steering needed only when both lateral error and heading error are significant.
    # If gate is far, ignore small lateral error and just tuck.
    far_gate = gate_dy > params["far_gate_dy"]
    big_dx = jnp.abs(dx) > params["dead_zone"]
    big_ori = jnp.abs(idx_err) > params["ori_dead"]

    need_steer = big_ori & (big_dx | threat) & (~far_gate | threat)
    need_right = need_steer & (idx_err > 0.0)
    need_left = need_steer & (idx_err < 0.0)

    # Default action when not steering is DOWN (tuck) to maximize downhill speed.
    action = jnp.where(
        need_right, RIGHT,
        jnp.where(need_left, LEFT, DOWN),
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
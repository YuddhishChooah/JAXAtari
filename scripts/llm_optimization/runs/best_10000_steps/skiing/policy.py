"""
Auto-generated policy v1
Generated at: 2026-04-27 15:56:46
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
FLAG_X = 8       # 8:10
FLAG_Y = 10      # 10:12
FLAG_ACT = 16    # 16:18
TREE_X = 24      # 24:28
TREE_Y = 28      # 28:32
TREE_ACT = 40    # 40:44

GATE_HALF_WIDTH = 16.0  # center is flag_x + 16


def init_params():
    return {
        "dead_zone": jnp.array(8.0),       # px tolerance to gate center before steering
        "steer_gain": jnp.array(0.08),     # maps dx to desired heading index offset
        "ori_dead": jnp.array(0.6),        # dead zone in heading-index space
        "tree_danger_y": jnp.array(20.0),  # y-distance below which a tree is threatening
        "tree_danger_x": jnp.array(20.0),  # x-distance considered in path
        "tuck_thresh": jnp.array(4.0),     # if aligned within this, tuck (DOWN)
    }


def _orientation_to_index(ori):
    # angles: 270, 292.5, 315, 337.5, 22.5, 45, 67.5, 90
    # indices 0..7 with 3,4 being straightest
    angles = jnp.array([270.0, 292.5, 315.0, 337.5, 22.5, 45.0, 67.5, 90.0])
    diffs = jnp.abs(angles - ori)
    return jnp.argmin(diffs)


def _select_gate(curr):
    fx = jax.lax.dynamic_slice(curr, (FLAG_X,), (2,))
    fy = jax.lax.dynamic_slice(curr, (FLAG_Y,), (2,))
    fa = jax.lax.dynamic_slice(curr, (FLAG_ACT,), (2,))
    sy = curr[SKIER_Y]
    # prefer active gates with y >= skier.y (below/at skier)
    valid = (fa > 0.5) & (fy >= sy - 4.0)
    # score: smaller (fy - sy) is better but must be >=0; use fy as proxy
    score = jnp.where(valid, fy, 1e6)
    idx = jnp.argmin(score)
    any_valid = jnp.any(valid)
    # fallback: first active gate
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
    threatening = (ta > 0.5) & (dy > -4.0) & (dy < params["tree_danger_y"]) & (jnp.abs(dx) < params["tree_danger_x"])
    any_threat = jnp.any(threatening)
    # pick the nearest threatening tree
    score = jnp.where(threatening, jnp.abs(dy), 1e6)
    idx = jnp.argmin(score)
    return any_threat, tx[idx] - sx


def policy(obs_flat, params):
    curr = jax.lax.dynamic_slice(obs_flat, (FRAME_SIZE,), (FRAME_SIZE,))

    sx = curr[SKIER_X]
    ori = curr[SKIER_ORI]
    ori_idx = _orientation_to_index(ori).astype(jnp.float32)

    flag_x, _, _ = _select_gate(curr)
    target_x = flag_x + GATE_HALF_WIDTH
    dx = target_x - sx

    # Tree avoidance: if a tree is on our line, push target away from it
    threat, tree_dx = _nearest_tree_threat(curr, params)
    # if tree to the right (tree_dx > 0), we want to go left -> negative dx
    avoid_dx = -jnp.sign(tree_dx) * 30.0
    dx = jnp.where(threat, avoid_dx, dx)

    # Desired heading index: 3.5 is straight, +right, -left
    desired_idx = 3.5 + params["steer_gain"] * dx
    desired_idx = jnp.clip(desired_idx, 2.0, 5.0)  # never go to extreme angles

    idx_err = desired_idx - ori_idx

    # Decide action
    aligned = jnp.abs(dx) < params["tuck_thresh"]
    need_right = idx_err > params["ori_dead"]
    need_left = idx_err < -params["ori_dead"]

    action = jnp.where(
        need_right, RIGHT,
        jnp.where(need_left, LEFT,
                  jnp.where(aligned, DOWN, NOOP))
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
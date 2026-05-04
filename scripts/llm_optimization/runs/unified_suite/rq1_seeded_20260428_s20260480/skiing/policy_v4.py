"""
Auto-generated policy v4
Generated at: 2026-04-28 19:56:08
"""

import jax
import jax.numpy as jnp

# Action constants
NOOP = 0
RIGHT = 1
LEFT = 2
FIRE = 3
DOWN = 4

# Frame layout
FRAME_SIZE = 73
SKIER_X = 0
SKIER_Y = 1
SKIER_ORIENT = 7
FLAG_X = 8
FLAG_Y = 10
FLAG_ACTIVE = 16
TREE_X = 24
TREE_Y = 28
TREE_ACTIVE = 40

GATE_HALF = 16.0


def init_params():
    return {
        "heading_gain": jnp.array(0.7),     # px-error -> desired signed angle
        "max_heading": jnp.array(22.5),     # cap on desired signed heading (fast notch)
        "heading_dead": jnp.array(12.0),    # tolerance in heading-tracking (deg)
        "down_angle": jnp.array(25.0),      # |signed| <= this => DOWN
        "tree_shift": jnp.array(12.0),      # lateral push around dangerous trees
        "tree_danger_y": jnp.array(30.0),   # vertical proximity for tree threat
        "err_dead": jnp.array(5.0),         # lateral error tolerance (px)
    }


def _signed_orient(orient):
    return jnp.where(orient > 180.0, orient - 360.0, orient)


def _pick_gate(curr):
    fx = jax.lax.dynamic_slice(curr, (FLAG_X,), (2,))
    fy = jax.lax.dynamic_slice(curr, (FLAG_Y,), (2,))
    fa = jax.lax.dynamic_slice(curr, (FLAG_ACTIVE,), (2,))
    skier_y = curr[SKIER_Y]
    ahead = (fa > 0.5) & (fy >= skier_y - 4.0)
    d_ahead = jnp.where(ahead, fy - skier_y, 1e6)
    d_any = jnp.where(fa > 0.5, jnp.abs(fy - skier_y) + 500.0, 1e6)
    use_ahead = jnp.any(ahead)
    d = jnp.where(use_ahead, d_ahead, d_any)
    idx = jnp.argmin(d)
    return fx[idx] + GATE_HALF, fy[idx]


def _tree_threat(curr, gate_cx, danger_y):
    tx = jax.lax.dynamic_slice(curr, (TREE_X,), (4,))
    ty = jax.lax.dynamic_slice(curr, (TREE_Y,), (4,))
    ta = jax.lax.dynamic_slice(curr, (TREE_ACTIVE,), (4,))
    skier_x = curr[SKIER_X]
    skier_y = curr[SKIER_Y]
    tcx = tx + 8.0
    dy = ty - skier_y
    lo = jnp.minimum(skier_x, gate_cx) - 10.0
    hi = jnp.maximum(skier_x, gate_cx) + 10.0
    in_path = (tcx > lo) & (tcx < hi)
    near_y = (dy > -4.0) & (dy < danger_y)
    threat = (ta > 0.5) & in_path & near_y
    score = jnp.where(threat, dy, 1e6)
    idx = jnp.argmin(score)
    has = jnp.any(threat)
    side = jnp.sign(tcx[idx] - skier_x)
    return has, side


def _select_target(curr, gate_cx, tree_shift, danger_y):
    has_threat, tree_side = _tree_threat(curr, gate_cx, danger_y)
    avoid = jnp.where(has_threat, -tree_side * tree_shift, 0.0)
    # Clamp avoidance so target stays inside the gate (gate spans ~32 px around cx)
    avoid = jnp.clip(avoid, -12.0, 12.0)
    return gate_cx + avoid


def _decide_turn(signed, desired, dead):
    err = desired - signed
    turn_right = err > dead
    turn_left = err < -dead
    return turn_right, turn_left


def policy(obs_flat, params):
    curr = jax.lax.dynamic_slice(obs_flat, (FRAME_SIZE,), (FRAME_SIZE,))

    skier_x = curr[SKIER_X]
    orient = curr[SKIER_ORIENT]
    signed = _signed_orient(orient)

    gate_cx, _gate_y = _pick_gate(curr)
    target_x = _select_target(curr, gate_cx, params["tree_shift"], params["tree_danger_y"])

    err_x = target_x - skier_x

    # Suppress turning when lateral error is small
    err_small = jnp.abs(err_x) < params["err_dead"]
    desired_raw = params["heading_gain"] * err_x
    desired = jnp.clip(desired_raw, -params["max_heading"], params["max_heading"])
    desired = jnp.where(err_small, 0.0, desired)

    turn_right, turn_left = _decide_turn(signed, desired, params["heading_dead"])

    # DOWN whenever the heading is in the fast set (|signed| <= down_angle)
    fast_heading = jnp.abs(signed) <= params["down_angle"]
    forward = jnp.where(fast_heading, DOWN, NOOP)

    action = jnp.where(turn_right, RIGHT,
              jnp.where(turn_left, LEFT, forward))

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
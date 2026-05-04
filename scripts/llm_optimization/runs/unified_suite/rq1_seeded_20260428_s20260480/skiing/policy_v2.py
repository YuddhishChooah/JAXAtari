"""
Auto-generated policy v2
Generated at: 2026-04-28 18:35:09
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
        "heading_gain": jnp.array(1.2),     # px-error -> desired signed angle
        "max_heading": jnp.array(28.0),     # cap on desired signed heading
        "heading_dead": jnp.array(6.0),     # tolerance in heading-tracking (deg)
        "tree_shift": jnp.array(28.0),      # lateral push around dangerous trees
        "tree_danger_y": jnp.array(28.0),   # vertical proximity for tree threat
        "tuck_angle": jnp.array(10.0),      # |signed| below this -> DOWN, else NOOP
    }


def _signed_orient(orient):
    return jnp.where(orient > 180.0, orient - 360.0, orient)


def _pick_gate(curr):
    fx = jax.lax.dynamic_slice(curr, (FLAG_X,), (2,))
    fy = jax.lax.dynamic_slice(curr, (FLAG_Y,), (2,))
    fa = jax.lax.dynamic_slice(curr, (FLAG_ACTIVE,), (2,))
    skier_y = curr[SKIER_Y]
    # Prefer gates ahead (below or near skier) with smallest positive distance
    ahead = (fa > 0.5) & (fy >= skier_y - 8.0)
    d_ahead = jnp.where(ahead, fy - skier_y + 1000.0 * (fy < skier_y - 8.0), 1e6)
    # Fallback: any active gate by absolute distance
    d_any = jnp.where(fa > 0.5, jnp.abs(fy - skier_y), 1e6)
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
    # tree is in path: between skier_x and gate_cx (with margin) and y close
    lo = jnp.minimum(skier_x, gate_cx) - 12.0
    hi = jnp.maximum(skier_x, gate_cx) + 12.0
    in_path = (tcx > lo) & (tcx < hi)
    near_y = (dy > -4.0) & (dy < danger_y)
    threat = (ta > 0.5) & in_path & near_y
    # nearest threat by y
    score = jnp.where(threat, dy, 1e6)
    idx = jnp.argmin(score)
    has = jnp.any(threat)
    side = jnp.sign(tcx[idx] - skier_x)  # tree side relative to skier
    return has, side


def _track_heading(signed, desired, dead):
    err = desired - signed
    turn_right = err > dead
    turn_left = err < -dead
    return turn_right, turn_left


def policy(obs_flat, params):
    curr = jax.lax.dynamic_slice(obs_flat, (FRAME_SIZE,), (FRAME_SIZE,))
    prev = jax.lax.dynamic_slice(obs_flat, (0,), (FRAME_SIZE,))

    skier_x = curr[SKIER_X]
    orient = curr[SKIER_ORIENT]
    signed = _signed_orient(orient)

    gate_cx, gate_y = _pick_gate(curr)

    # Tree-aware lateral target adjustment
    has_threat, tree_side = _tree_threat(curr, gate_cx, params["tree_danger_y"])
    avoid = jnp.where(has_threat, -tree_side * params["tree_shift"], 0.0)
    target_x = gate_cx + avoid

    # Lateral error -> desired signed heading (proportional)
    err_x = target_x - skier_x
    desired = jnp.clip(
        params["heading_gain"] * err_x,
        -params["max_heading"],
        params["max_heading"],
    )

    # Track desired heading with LEFT/RIGHT; release otherwise
    turn_right, turn_left = _track_heading(signed, desired, params["heading_dead"])

    # Forward action when not turning: DOWN if heading is near-straight, else NOOP
    near_straight = jnp.abs(signed) < params["tuck_angle"]
    forward = jnp.where(near_straight, DOWN, NOOP)

    action = jnp.where(turn_right, RIGHT,
              jnp.where(turn_left, LEFT, forward))

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
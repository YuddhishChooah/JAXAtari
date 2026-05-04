"""
Auto-generated policy v5
Generated at: 2026-04-28 20:07:46
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
        "heading_gain": jnp.array(0.45),     # px-error -> desired signed angle
        "engage_dead": jnp.array(15.0),      # large: only engage turn if far from target
        "release_dead": jnp.array(3.0),      # small: release once close
        "tuck_angle": jnp.array(28.0),       # |signed| <= this -> DOWN allowed
        "tree_shift": jnp.array(12.0),       # small lateral push around trees
        "tree_danger_y": jnp.array(24.0),    # vertical proximity for tree threat
    }


def _signed_orient(orient):
    return jnp.where(orient > 180.0, orient - 360.0, orient)


def _pick_gate(curr):
    fx = jax.lax.dynamic_slice(curr, (FLAG_X,), (2,))
    fy = jax.lax.dynamic_slice(curr, (FLAG_Y,), (2,))
    fa = jax.lax.dynamic_slice(curr, (FLAG_ACTIVE,), (2,))
    skier_y = curr[SKIER_Y]
    ahead = (fa > 0.5) & (fy >= skier_y - 8.0)
    d_ahead = jnp.where(ahead, fy - skier_y + 1000.0 * (fy < skier_y - 8.0), 1e6)
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


def _decide_turn(signed, desired, engage, release):
    err = desired - signed
    abs_err = jnp.abs(err)
    # Engage when far from desired; once engaged we still only steer while
    # outside the small release deadband.
    active = abs_err > release
    far = abs_err > engage
    # Use the larger (engage) threshold as a single gate: only turn when really off.
    do_turn = far | (active & (abs_err > release) & (abs_err > engage * 0.5))
    turn_right = do_turn & (err > 0)
    turn_left = do_turn & (err < 0)
    return turn_right, turn_left


def policy(obs_flat, params):
    curr = jax.lax.dynamic_slice(obs_flat, (FRAME_SIZE,), (FRAME_SIZE,))

    skier_x = curr[SKIER_X]
    orient = curr[SKIER_ORIENT]
    signed = _signed_orient(orient)

    gate_cx, gate_y = _pick_gate(curr)

    has_threat, tree_side = _tree_threat(curr, gate_cx, params["tree_danger_y"])
    avoid = jnp.where(has_threat, -tree_side * params["tree_shift"], 0.0)
    target_x = gate_cx + avoid

    err_x = target_x - skier_x
    # Cap desired heading to the fast notch (~22.5 deg) so we don't command
    # the slow ±45/±67.5 regimes.
    desired = jnp.clip(params["heading_gain"] * err_x, -22.5, 22.5)

    turn_right, turn_left = _decide_turn(
        signed, desired, params["engage_dead"], params["release_dead"]
    )

    # When not turning, prefer DOWN if heading is within tuck range.
    near_straight = jnp.abs(signed) <= params["tuck_angle"]
    forward = jnp.where(near_straight, DOWN, NOOP)

    action = jnp.where(turn_right, RIGHT,
              jnp.where(turn_left, LEFT, forward))

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
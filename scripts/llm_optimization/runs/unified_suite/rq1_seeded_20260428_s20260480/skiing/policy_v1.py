"""
Auto-generated policy v1
Generated at: 2026-04-28 18:23:39
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
FLAG_X = 8       # 8:10
FLAG_Y = 10      # 10:12
FLAG_ACTIVE = 16 # 16:18
TREE_X = 24      # 24:28
TREE_Y = 28      # 28:32
TREE_ACTIVE = 40 # 40:44

GATE_HALF_WIDTH = 16.0  # center is flag_x + 16


def init_params():
    return {
        "dead_zone": jnp.array(6.0),       # px tolerance to gate center
        "lookahead": jnp.array(0.6),       # how strongly orientation predicts future x
        "tree_danger_y": jnp.array(20.0),  # vertical distance considered dangerous
        "tree_danger_x": jnp.array(20.0),  # lateral distance considered dangerous
        "tuck_bias": jnp.array(0.5),       # >0 favors DOWN over NOOP when aligned
        "turn_release": jnp.array(1.5),    # release turn when heading already strong
    }


def _select_target_gate(curr):
    fx = jax.lax.dynamic_slice(curr, (FLAG_X,), (2,))
    fy = jax.lax.dynamic_slice(curr, (FLAG_Y,), (2,))
    fa = jax.lax.dynamic_slice(curr, (FLAG_ACTIVE,), (2,))
    skier_y = curr[SKIER_Y]
    # Prefer active gates at or below skier; pick the closest such gate.
    below = (fy >= skier_y - 2.0) & (fa > 0.5)
    dist = jnp.where(below, fy - skier_y, 1e6)
    # Fallback: any active gate
    any_active = jnp.where(fa > 0.5, jnp.abs(fy - skier_y), 1e6)
    use_below = jnp.any(below)
    dist = jnp.where(use_below, dist, any_active)
    idx = jnp.argmin(dist)
    return fx[idx] + GATE_HALF_WIDTH, fy[idx], fa[idx]


def _nearest_tree_offset(curr, danger_y):
    tx = jax.lax.dynamic_slice(curr, (TREE_X,), (4,))
    ty = jax.lax.dynamic_slice(curr, (TREE_Y,), (4,))
    ta = jax.lax.dynamic_slice(curr, (TREE_ACTIVE,), (4,))
    skier_x = curr[SKIER_X]
    skier_y = curr[SKIER_Y]
    dy = ty - skier_y
    near = (ta > 0.5) & (dy > -4.0) & (dy < danger_y)
    dx = tx + 8.0 - skier_x  # tree center approx
    big = jnp.full_like(dx, 1e6)
    abs_dx = jnp.where(near, jnp.abs(dx), big)
    idx = jnp.argmin(abs_dx)
    return dx[idx], abs_dx[idx]


def _orientation_signed(orient):
    # Map 270..337.5 -> negative (left), 22.5..90 -> positive (right)
    # Convert: if orient > 180, signed = orient - 360 (gives -90..-22.5)
    return jnp.where(orient > 180.0, orient - 360.0, orient)


def policy(obs_flat, params):
    curr = jax.lax.dynamic_slice(obs_flat, (FRAME_SIZE,), (FRAME_SIZE,))

    skier_x = curr[SKIER_X]
    orient = curr[SKIER_ORIENT]
    signed = _orientation_signed(orient)  # negative=left-facing, positive=right-facing

    gate_cx, _, _ = _select_target_gate(curr)

    # Predicted x adjustment from current heading: lookahead * signed_angle
    predicted_x = skier_x + params["lookahead"] * signed

    # Tree avoidance: if a tree is dangerously close laterally, push away
    tree_dx, tree_abs = _nearest_tree_offset(curr, params["tree_danger_y"])
    tree_threat = tree_abs < params["tree_danger_x"]
    # If tree is to the right (dx>0), we want to go left -> target shifts left
    avoid_shift = jnp.where(tree_threat, -jnp.sign(tree_dx) * 24.0, 0.0)

    target_x = gate_cx + avoid_shift
    error = target_x - predicted_x

    dead = params["dead_zone"]
    release = params["turn_release"]

    # If heading already strongly biased in the desired direction, release.
    want_right = error > dead
    want_left = error < -dead
    heading_already_right = signed > release
    heading_already_left = signed < -release

    # Aligned: small error -> NOOP or DOWN
    aligned = jnp.abs(error) <= dead

    # Choose turn direction, but release if heading already strong that way
    do_right = want_right & (~heading_already_right)
    do_left = want_left & (~heading_already_left)

    # Counter-steer if heading too extreme even when error small
    extreme_left = signed < -45.0
    extreme_right = signed > 45.0
    counter_right = aligned & extreme_left
    counter_left = aligned & extreme_right

    aligned_action = jnp.where(params["tuck_bias"] > 0.0, DOWN, NOOP)

    action = jnp.where(
        do_right | counter_right, RIGHT,
        jnp.where(
            do_left | counter_left, LEFT,
            aligned_action
        )
    )

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
"""
Auto-generated policy v2
Generated at: 2026-04-27 17:49:52
"""

import jax
import jax.numpy as jnp

# Actions
NOOP, RIGHT, LEFT, FIRE, DOWN = 0, 1, 2, 3, 4

FRAME_SIZE = 73
SKIER_X, SKIER_Y, SKIER_ORI = 0, 1, 7
FLAG_X0, FLAG_Y0 = 8, 10
FLAG_ACTIVE0 = 16
TREE_X0, TREE_Y0 = 24, 28
TREE_ACTIVE0 = 40

GATE_HALF_WIDTH = 16.0


def init_params():
    return {
        "dead_zone": 3.0,            # px tolerance around gate center for active steering
        "tuck_margin": 11.0,         # tuck if |dx| < gate_half - margin gap, i.e. inside gate
        "ori_release": 1.0,          # stop adding turn at this many notches off straight
        "ori_target": 0.6,           # actively recenter heading if |ori_n| > this and aligned
        "tree_danger_y": 22.0,
        "tree_danger_x": 14.0,
        "far_gate_y": 40.0,          # if gate is farther than this, prioritize speed
    }


def signed_orientation(ori_deg):
    signed = jnp.where(ori_deg > 180.0, ori_deg - 360.0, ori_deg)
    return signed / 22.5  # notches: -4..+4, 0 = straight down


def select_gate(curr):
    skier_y = curr[SKIER_Y]
    fx = jax.lax.dynamic_slice(curr, (FLAG_X0,), (2,))
    fy = jax.lax.dynamic_slice(curr, (FLAG_Y0,), (2,))
    fa = jax.lax.dynamic_slice(curr, (FLAG_ACTIVE0,), (2,))
    ahead = (fy >= skier_y - 4.0) & (fa > 0.5)
    dist = jnp.where(ahead, fy - skier_y, 1e6)
    idx = jnp.argmin(dist)
    return fx[idx], fy[idx], jnp.any(ahead)


def nearest_tree_dx(curr, skier_x, skier_y, danger_y):
    tx = jax.lax.dynamic_slice(curr, (TREE_X0,), (4,))
    ty = jax.lax.dynamic_slice(curr, (TREE_Y0,), (4,))
    ta = jax.lax.dynamic_slice(curr, (TREE_ACTIVE0,), (4,))
    dy = ty - skier_y
    near = (ta > 0.5) & (dy > -6.0) & (dy < danger_y)
    dx = tx + 8.0 - skier_x
    big = jnp.full_like(dx, 1e6)
    dx_near = jnp.where(near, dx, big)
    idx = jnp.argmin(jnp.abs(dx_near))
    return dx_near[idx], jnp.any(near)


def policy(obs_flat, params):
    curr = jax.lax.dynamic_slice(obs_flat, (FRAME_SIZE,), (FRAME_SIZE,))

    skier_x = curr[SKIER_X]
    skier_y = curr[SKIER_Y]
    ori_n = signed_orientation(curr[SKIER_ORI])

    gx, gy, gate_valid = select_gate(curr)
    gate_center = gx + GATE_HALF_WIDTH
    dx_gate = gate_center - skier_x
    target_dx = jnp.where(gate_valid, dx_gate, 0.0)
    gate_dy = jnp.where(gate_valid, gy - skier_y, 0.0)

    # Tree avoidance overrides steering target
    tree_dx, has_tree = nearest_tree_dx(curr, skier_x, skier_y, params["tree_danger_y"])
    tree_close = has_tree & (jnp.abs(tree_dx) < params["tree_danger_x"])
    avoid_dx = -jnp.sign(tree_dx) * 30.0
    target_dx = jnp.where(tree_close, avoid_dx, target_dx)

    dead = params["dead_zone"]
    ori_release = params["ori_release"]
    ori_target = params["ori_target"]
    tuck_margin = params["tuck_margin"]
    far_gate_y = params["far_gate_y"]

    abs_dx = jnp.abs(target_dx)
    abs_ori = jnp.abs(ori_n)

    # Inside-gate test: anywhere within gate (with safety margin) is good enough to tuck
    inside_gate = abs_dx < (GATE_HALF_WIDTH - tuck_margin)
    # Far gate + roughly aligned -> commit to speed
    far_and_ok = (gate_dy > far_gate_y) & (abs_dx < GATE_HALF_WIDTH)

    # Active steering toward gate when clearly off
    want_right = target_dx > dead
    want_left = target_dx < -dead
    do_right = want_right & (ori_n < ori_release)
    do_left = want_left & (ori_n > -ori_release)

    # Heading recovery: when aligned (or inside gate), drive heading back toward straight
    aligned = abs_dx < dead * 2.0
    recover_zone = aligned | inside_gate
    recover_right = recover_zone & (ori_n < -ori_target)  # facing left -> press RIGHT
    recover_left = recover_zone & (ori_n > ori_target)

    # Build action with priority: avoid/steer > recovery > tuck > noop
    action = jnp.int32(NOOP)
    action = jnp.where(do_right, jnp.int32(RIGHT), action)
    action = jnp.where(do_left, jnp.int32(LEFT), action)
    is_noop = action == NOOP
    action = jnp.where(is_noop & recover_right, jnp.int32(RIGHT), action)
    action = jnp.where(is_noop & recover_left, jnp.int32(LEFT), action)

    # Tuck whenever idle and heading is near-straight (notches 3 or 4 territory)
    heading_fast = abs_ori < 1.1
    do_tuck = (action == NOOP) & heading_fast & (inside_gate | far_and_ok | aligned)
    action = jnp.where(do_tuck, jnp.int32(DOWN), action)

    # Default-prefer DOWN over NOOP when heading is straight and no steering needed
    default_down = (action == NOOP) & (abs_ori < 0.6)
    action = jnp.where(default_down, jnp.int32(DOWN), action)

    return action


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
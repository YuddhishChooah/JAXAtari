"""
Auto-generated policy v5
Generated at: 2026-04-27 17:53:41
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
        "dead_zone": 6.0,         # px tolerance around gate center for steering
        "lookahead_y": 30.0,      # if nearest gate farther than this, just tuck
        "tree_danger_y": 20.0,    # tree y proximity threshold
        "tree_danger_x": 12.0,    # tree x proximity threshold
        "ori_release": 1.6,       # |signed heading notches| above which we stop turning
        "tuck_ori": 1.6,          # tuck allowed when |ori_n| <= this
    }


def signed_orientation(ori_deg):
    signed = jnp.where(ori_deg > 180.0, ori_deg - 360.0, ori_deg)
    return signed / 22.5  # in notches; -4..+4


def select_gate(curr):
    skier_y = curr[SKIER_Y]
    fx = jax.lax.dynamic_slice(curr, (FLAG_X0,), (2,))
    fy = jax.lax.dynamic_slice(curr, (FLAG_Y0,), (2,))
    fa = jax.lax.dynamic_slice(curr, (FLAG_ACTIVE0,), (2,))

    # Drop gates that are at or above the skier (already passed)
    ahead = (fy > skier_y + 2.0) & (fa > 0.5)
    dist = jnp.where(ahead, fy - skier_y, 1e6)
    idx = jnp.argmin(dist)
    return fx[idx], fy[idx], jnp.any(ahead)


def nearest_tree_dx(curr, skier_x, skier_y, danger_y, danger_x):
    tx = jax.lax.dynamic_slice(curr, (TREE_X0,), (4,))
    ty = jax.lax.dynamic_slice(curr, (TREE_Y0,), (4,))
    ta = jax.lax.dynamic_slice(curr, (TREE_ACTIVE0,), (4,))
    dy = ty - skier_y
    dx = tx + 8.0 - skier_x
    near = (ta > 0.5) & (dy > -4.0) & (dy < danger_y) & (jnp.abs(dx) < danger_x)
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
    gate_far = (gy - skier_y) > params["lookahead_y"]

    # Tree avoidance (only if a tree truly blocks us)
    tree_dx, has_tree = nearest_tree_dx(
        curr, skier_x, skier_y, params["tree_danger_y"], params["tree_danger_x"]
    )
    avoid_dx = -jnp.sign(tree_dx) * 25.0

    # Final lateral target
    target_dx = jnp.where(gate_valid, dx_gate, 0.0)
    target_dx = jnp.where(has_tree, avoid_dx, target_dx)

    dead = params["dead_zone"]
    ori_release = params["ori_release"]
    tuck_ori = params["tuck_ori"]

    want_right = (target_dx > dead) & (~gate_far | has_tree)
    want_left = (target_dx < -dead) & (~gate_far | has_tree)

    do_right = want_right & (ori_n < ori_release)
    do_left = want_left & (ori_n > -ori_release)

    # Recovery if heading is off and we are not actively steering toward target
    facing_too_left = ori_n < -ori_release
    facing_too_right = ori_n > ori_release
    recover_right = (~do_right) & (~do_left) & facing_too_left
    recover_left = (~do_right) & (~do_left) & facing_too_right

    action = jnp.int32(NOOP)
    action = jnp.where(do_right, jnp.int32(RIGHT), action)
    action = jnp.where(do_left, jnp.int32(LEFT), action)
    action = jnp.where((action == NOOP) & recover_right, jnp.int32(RIGHT), action)
    action = jnp.where((action == NOOP) & recover_left, jnp.int32(LEFT), action)

    # Default: tuck whenever not steering and heading is reasonable
    heading_ok = jnp.abs(ori_n) <= tuck_ori
    do_tuck = (action == NOOP) & heading_ok
    action = jnp.where(do_tuck, jnp.int32(DOWN), action)

    return action


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
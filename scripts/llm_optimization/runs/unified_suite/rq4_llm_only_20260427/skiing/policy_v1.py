"""
Auto-generated policy v1
Generated at: 2026-04-27 17:48:28
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

GATE_HALF_WIDTH = 16.0  # center is flag_x + 16


def init_params():
    return {
        "dead_zone": 6.0,        # px tolerance around gate center
        "lookahead_y": 55.0,     # only consider gates within this y-distance ahead
        "tree_danger_y": 22.0,   # tree y proximity threshold
        "tree_danger_x": 14.0,   # tree x proximity threshold
        "ori_release": 1.6,      # |signed orientation| above which we stop turning further
        "tuck_dx": 3.0,          # |dx| below which we tuck (DOWN)
    }


def signed_orientation(ori_deg):
    # Map 270..360 -> -90..0, 0..90 -> 0..90 (in units of 22.5 notches)
    # Convert degrees to signed degrees relative to straight-down (0 deg means straight, +right, -left)
    signed = jnp.where(ori_deg > 180.0, ori_deg - 360.0, ori_deg)
    # Now: -90 (hard left) ... 0 (straight) ... 90 (hard right)
    # Express in notches of 22.5 for easier thresholding
    return signed / 22.5


def select_gate(curr):
    skier_y = curr[SKIER_Y]
    fx = jax.lax.dynamic_slice(curr, (FLAG_X0,), (2,))
    fy = jax.lax.dynamic_slice(curr, (FLAG_Y0,), (2,))
    fa = jax.lax.dynamic_slice(curr, (FLAG_ACTIVE0,), (2,))

    # We want the gate that is ahead (fy >= skier_y - small) and active, with smallest fy >= skier_y
    ahead = (fy >= skier_y - 4.0) & (fa > 0.5)
    # Distance metric: prefer closest ahead; push non-ahead far away
    dist = jnp.where(ahead, fy - skier_y, 1e6)
    idx = jnp.argmin(dist)
    gx = fx[idx]
    gy = fy[idx]
    valid = jnp.any(ahead)
    return gx, gy, valid


def nearest_tree_dx(curr, skier_x, skier_y, danger_y):
    tx = jax.lax.dynamic_slice(curr, (TREE_X0,), (4,))
    ty = jax.lax.dynamic_slice(curr, (TREE_Y0,), (4,))
    ta = jax.lax.dynamic_slice(curr, (TREE_ACTIVE0,), (4,))
    dy = ty - skier_y
    near = (ta > 0.5) & (dy > -6.0) & (dy < danger_y)
    dx = tx + 8.0 - skier_x  # tree center approx
    # If no near tree, return a large dx
    big = jnp.full_like(dx, 1e6)
    dx_near = jnp.where(near, dx, big)
    # Find tree with minimal |dx|
    idx = jnp.argmin(jnp.abs(dx_near))
    chosen_dx = dx_near[idx]
    has_tree = jnp.any(near)
    return chosen_dx, has_tree


def policy(obs_flat, params):
    curr = jax.lax.dynamic_slice(obs_flat, (FRAME_SIZE,), (FRAME_SIZE,))

    skier_x = curr[SKIER_X]
    skier_y = curr[SKIER_Y]
    ori = curr[SKIER_ORI]
    ori_n = signed_orientation(ori)  # negative=left-facing, positive=right-facing

    gx, gy, gate_valid = select_gate(curr)
    gate_center = gx + GATE_HALF_WIDTH
    dx_gate = gate_center - skier_x  # positive => need to go right

    # If no valid gate, target straight ahead
    target_dx = jnp.where(gate_valid, dx_gate, 0.0)

    # Tree avoidance: if tree is dangerously close, override target to steer away
    tree_dx, has_tree = nearest_tree_dx(curr, skier_x, skier_y, params["tree_danger_y"])
    tree_close_x = jnp.abs(tree_dx) < params["tree_danger_x"]
    # Steer opposite to tree: if tree is to the right (tree_dx>0), go left (target negative)
    avoid_dx = -jnp.sign(tree_dx) * 30.0
    use_avoid = has_tree & tree_close_x
    target_dx = jnp.where(use_avoid, avoid_dx, target_dx)

    dead = params["dead_zone"]
    ori_release = params["ori_release"]
    tuck_dx = params["tuck_dx"]

    # Decide steering with heading feedback:
    # Want to go right (target_dx > dead): press RIGHT only if ori_n < ori_release
    # Want to go left (target_dx < -dead): press LEFT only if ori_n > -ori_release
    # Otherwise NOOP/DOWN to recover heading.
    want_right = target_dx > dead
    want_left = target_dx < -dead

    # If currently facing far the wrong way, definitely correct
    facing_too_left = ori_n < -ori_release   # ori_n very negative => facing left
    facing_too_right = ori_n > ori_release

    # Press RIGHT if want_right AND not already pointing strongly right
    do_right = want_right & (ori_n < ori_release)
    do_left = want_left & (ori_n > -ori_release)

    # If we're not wanting to turn but heading is skewed, recover
    recover_right = (~want_right) & (~want_left) & facing_too_left
    recover_left = (~want_right) & (~want_left) & facing_too_right

    # Action priority: avoidance/gate steering > recovery > tuck/noop
    action = jnp.int32(NOOP)
    action = jnp.where(do_right, jnp.int32(RIGHT), action)
    action = jnp.where(do_left, jnp.int32(LEFT), action)
    action = jnp.where((action == NOOP) & recover_right, jnp.int32(RIGHT), action)
    action = jnp.where((action == NOOP) & recover_left, jnp.int32(LEFT), action)

    # Tuck (DOWN) when well-aligned and heading roughly straight
    aligned = jnp.abs(target_dx) < tuck_dx
    heading_ok = jnp.abs(ori_n) < 1.1  # within +/- 1 notch of straight
    do_tuck = (action == NOOP) & aligned & heading_ok
    action = jnp.where(do_tuck, jnp.int32(DOWN), action)

    return action


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
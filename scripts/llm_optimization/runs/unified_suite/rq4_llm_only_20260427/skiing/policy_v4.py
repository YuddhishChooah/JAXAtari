"""
Auto-generated policy v4
Generated at: 2026-04-27 17:52:30
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
        "dead_zone": 3.5,        # px tolerance around gate center (tighter)
        "ori_release": 0.8,      # stop turning once heading exceeds this many notches toward target
        "ori_recover": 1.3,      # recover heading once it drifts beyond this (speed-kill threshold)
        "tree_danger_y": 24.0,
        "tree_danger_x": 14.0,
        "preaim_dy": 35.0,       # within this dy, lock onto the gate aggressively
    }


def signed_orientation(ori_deg):
    # Map degrees to signed notches: -4 (hard left) ... 0 (straight) ... +4 (hard right)
    signed = jnp.where(ori_deg > 180.0, ori_deg - 360.0, ori_deg)
    return signed / 22.5


def select_gate(curr):
    skier_y = curr[SKIER_Y]
    fx = jax.lax.dynamic_slice(curr, (FLAG_X0,), (2,))
    fy = jax.lax.dynamic_slice(curr, (FLAG_Y0,), (2,))
    fa = jax.lax.dynamic_slice(curr, (FLAG_ACTIVE0,), (2,))

    ahead = (fy >= skier_y - 4.0) & (fa > 0.5)
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
    dx = tx + 8.0 - skier_x
    big = jnp.full_like(dx, 1e6)
    dx_near = jnp.where(near, dx, big)
    idx = jnp.argmin(jnp.abs(dx_near))
    chosen_dx = dx_near[idx]
    has_tree = jnp.any(near)
    return chosen_dx, has_tree


def policy(obs_flat, params):
    curr = jax.lax.dynamic_slice(obs_flat, (FRAME_SIZE,), (FRAME_SIZE,))

    skier_x = curr[SKIER_X]
    skier_y = curr[SKIER_Y]
    ori_n = signed_orientation(curr[SKIER_ORI])

    gx, gy, gate_valid = select_gate(curr)
    gate_center = gx + GATE_HALF_WIDTH
    dx_gate = gate_center - skier_x

    # If no valid gate, target straight down (dx=0)
    target_dx = jnp.where(gate_valid, dx_gate, 0.0)

    # When gate is far below, shrink target to encourage going straight & fast.
    # When close, use full correction.
    dy = jnp.where(gate_valid, gy - skier_y, 0.0)
    close = dy < params["preaim_dy"]
    # In far regime, only correct large lateral errors (>8px) to avoid steering early
    far_target = jnp.where(jnp.abs(target_dx) > 8.0, target_dx, 0.0)
    eff_target = jnp.where(close, target_dx, far_target)

    # Tree avoidance override
    tree_dx, has_tree = nearest_tree_dx(curr, skier_x, skier_y, params["tree_danger_y"])
    tree_close_x = jnp.abs(tree_dx) < params["tree_danger_x"]
    avoid_dx = -jnp.sign(tree_dx) * 30.0
    use_avoid = has_tree & tree_close_x
    eff_target = jnp.where(use_avoid, avoid_dx, eff_target)

    dead = params["dead_zone"]
    ori_release = params["ori_release"]
    ori_recover = params["ori_recover"]

    want_right = eff_target > dead
    want_left = eff_target < -dead

    # Steer toward target only if heading not yet committed in that direction
    do_right = want_right & (ori_n < ori_release)
    do_left = want_left & (ori_n > -ori_release)

    # Heading recovery when not actively steering and heading is past speed-kill threshold
    not_steering = (~want_right) & (~want_left)
    recover_right = not_steering & (ori_n < -ori_recover)  # facing left, push right
    recover_left = not_steering & (ori_n > ori_recover)    # facing right, push left

    # Also recover gently when current heading is opposite of target's sign
    # (e.g., want slight right but facing strongly left). This complements the
    # steering branch but only triggers if we drifted past recover threshold.
    drift_right = want_right & (ori_n < -ori_recover)
    drift_left = want_left & (ori_n > ori_recover)

    action = jnp.int32(DOWN)  # default: tuck and accelerate
    action = jnp.where(do_right, jnp.int32(RIGHT), action)
    action = jnp.where(do_left, jnp.int32(LEFT), action)
    action = jnp.where(recover_right | drift_right, jnp.int32(RIGHT), action)
    action = jnp.where(recover_left | drift_left, jnp.int32(LEFT), action)

    return action


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
"""
Auto-generated policy v3
Generated at: 2026-04-27 17:51:15
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
        "dead_zone": 3.0,        # px alignment tolerance for gate center
        "lookahead_y": 80.0,     # consider gates within this y-distance
        "ori_release": 0.9,      # |signed heading notches| above which we stop turning further
        "ori_recenter": 0.6,     # recenter heading toward straight when |ori_n| above this
        "tree_danger_x": 16.0,   # tree x proximity threshold
        "tree_danger_y": 24.0,   # tree y proximity threshold
    }


def signed_orientation_notches(ori_deg):
    # Map degrees to signed notches of 22.5: -4 (hard left) ... 0 (straight) ... +4 (hard right)
    signed = jnp.where(ori_deg > 180.0, ori_deg - 360.0, ori_deg)
    return signed / 22.5


def select_gate(curr, lookahead_y):
    skier_y = curr[SKIER_Y]
    fx = jax.lax.dynamic_slice(curr, (FLAG_X0,), (2,))
    fy = jax.lax.dynamic_slice(curr, (FLAG_Y0,), (2,))
    fa = jax.lax.dynamic_slice(curr, (FLAG_ACTIVE0,), (2,))
    ahead = (fy >= skier_y - 4.0) & (fa > 0.5) & (fy - skier_y < lookahead_y + 50.0)
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
    near = (ta > 0.5) & (dy > -4.0) & (dy < danger_y)
    dx = tx + 8.0 - skier_x
    big = jnp.full_like(dx, 1e6)
    dx_near = jnp.where(near, dx, big)
    idx = jnp.argmin(jnp.abs(dx_near))
    chosen_dx = dx_near[idx]
    has_tree = jnp.any(near)
    return chosen_dx, has_tree


def policy(obs_flat, params):
    prev = jax.lax.dynamic_slice(obs_flat, (0,), (FRAME_SIZE,))
    curr = jax.lax.dynamic_slice(obs_flat, (FRAME_SIZE,), (FRAME_SIZE,))

    skier_x = curr[SKIER_X]
    skier_y = curr[SKIER_Y]
    ori_n = signed_orientation_notches(curr[SKIER_ORI])

    gx, gy, gate_valid = select_gate(curr, params["lookahead_y"])
    gate_center = gx + GATE_HALF_WIDTH
    dx_gate = gate_center - skier_x

    # Frames-to-gate estimate using flag motion (positive y-velocity expected)
    prev_fy = jax.lax.dynamic_slice(prev, (FLAG_Y0,), (2,))
    curr_fy = jax.lax.dynamic_slice(curr, (FLAG_Y0,), (2,))
    flag_vy = jnp.maximum(jnp.mean(curr_fy - prev_fy), 0.5)
    # negative dy => gate moving up toward skier; we want time until gy reaches skier_y
    # In this game, gy increases as it approaches skier (objects move up screen but coords?)
    # Use |gy - skier_y| / flag_vy as rough horizon
    horizon = jnp.abs(gy - skier_y) / flag_vy

    target_dx = jnp.where(gate_valid, dx_gate, 0.0)

    # Tree avoidance: graded magnitude
    tree_dx, has_tree = nearest_tree_dx(curr, skier_x, skier_y, params["tree_danger_y"])
    tree_close = has_tree & (jnp.abs(tree_dx) < params["tree_danger_x"])
    # Steer opposite to tree, magnitude scaled by closeness (closer => stronger)
    closeness = (params["tree_danger_x"] - jnp.abs(tree_dx)) / params["tree_danger_x"]
    closeness = jnp.clip(closeness, 0.0, 1.0)
    avoid_dx = -jnp.sign(tree_dx) * (8.0 + 14.0 * closeness)
    target_dx = jnp.where(tree_close, avoid_dx, target_dx)

    dead = params["dead_zone"]
    ori_release = params["ori_release"]
    ori_recenter = params["ori_recenter"]

    # Desired notch heading derived from target_dx and horizon.
    # Need bigger heading magnitude when little time remains.
    # urgency in [0,1]: high when horizon small
    urgency = jnp.clip(1.0 - horizon / 30.0, 0.0, 1.0)
    # desired heading notch in roughly [-1.5, 1.5]
    desired_ori = jnp.sign(target_dx) * jnp.minimum(jnp.abs(target_dx) / 8.0, 1.0) * (0.6 + 0.9 * urgency)
    desired_ori = jnp.where(jnp.abs(target_dx) < dead, 0.0, desired_ori)

    heading_err = desired_ori - ori_n  # positive => need to turn more right

    # Steering decision based on heading error, not raw dx.
    want_right = heading_err > 0.4
    want_left = heading_err < -0.4

    # Cap turning so we don't over-rotate beyond ori_release in either direction
    do_right = want_right & (ori_n < ori_release)
    do_left = want_left & (ori_n > -ori_release)

    # Recentering: if no active steering needed and heading is off-center, gently recenter
    no_steer = (~do_right) & (~do_left)
    recenter_right = no_steer & (ori_n < -ori_recenter)
    recenter_left = no_steer & (ori_n > ori_recenter)

    action = jnp.int32(DOWN)  # default: tuck for speed
    action = jnp.where(recenter_left, jnp.int32(LEFT), action)
    action = jnp.where(recenter_right, jnp.int32(RIGHT), action)
    action = jnp.where(do_left, jnp.int32(LEFT), action)
    action = jnp.where(do_right, jnp.int32(RIGHT), action)

    return action


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
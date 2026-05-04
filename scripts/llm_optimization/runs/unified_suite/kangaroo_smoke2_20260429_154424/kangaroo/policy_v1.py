"""
Auto-generated policy v1
Generated at: 2026-04-29 15:45:01
"""

import jax
import jax.numpy as jnp


# Action constants
NOOP = 0
FIRE = 1
UP = 2
RIGHT = 3
LEFT = 4
DOWN = 5
UPRIGHT = 6
UPLEFT = 7
RIGHTFIRE = 11
LEFTFIRE = 12


def init_params():
    return {
        "ladder_align_tol": jnp.float32(8.0),   # px tolerance to consider aligned with ladder
        "danger_dx": jnp.float32(20.0),         # x-distance to consider monkey/coconut dangerous
        "danger_dy": jnp.float32(16.0),         # y-distance for danger
        "climb_bias": jnp.float32(0.5),         # >0 prefers climbing once aligned
        "target_y_weight": jnp.float32(1.0),    # weight on vertical distance for ladder choice
        "punch_range": jnp.float32(24.0),       # range to use FIRE on monkey
    }


def _nearest_active_ladder(px, py, lx, ly, lactive, w_y):
    # cost = |dx| + w_y * max(0, py - ly)  (prefer ladders above us / at our level)
    dx = lx - px
    dy = py - ly  # positive if ladder is above player
    cost = jnp.abs(dx) + w_y * jnp.maximum(dy, 0.0)
    # mask inactive
    big = jnp.float32(1e6)
    cost = jnp.where(lactive > 0.5, cost, big)
    idx = jnp.argmin(cost)
    return lx[idx], ly[idx], lactive[idx]


def _danger_score(px, py, ex, ey, eactive, dx_th, dy_th):
    dx = ex - px
    dy = ey - py
    near = (jnp.abs(dx) < dx_th) & (jnp.abs(dy) < dy_th) & (eactive > 0.5)
    # signed direction: +1 enemy to right, -1 to left
    sign = jnp.sign(dx)
    any_near = jnp.any(near)
    # pick the closest dangerous one's sign
    dist = jnp.where(near, jnp.abs(dx) + jnp.abs(dy), jnp.float32(1e6))
    idx = jnp.argmin(dist)
    return any_near, sign[idx], jnp.abs(dx[idx])


def policy(obs_flat, params):
    px = obs_flat[0]
    py = obs_flat[1]

    ladder_x = obs_flat[168:188]
    ladder_y = obs_flat[188:208]
    ladder_w = obs_flat[208:228]
    ladder_active = obs_flat[248:268]

    child_y = obs_flat[361]

    fc_x = obs_flat[368:369]
    fc_y = obs_flat[369:370]
    fc_active = obs_flat[372:373]

    monkey_x = obs_flat[376:380]
    monkey_y = obs_flat[380:384]
    monkey_active = obs_flat[392:396]

    coconut_x = obs_flat[408:412]
    coconut_y = obs_flat[412:416]
    coconut_active = obs_flat[424:428]

    align_tol = params["ladder_align_tol"]
    dx_th = params["danger_dx"]
    dy_th = params["danger_dy"]
    w_y = params["target_y_weight"]
    punch_range = params["punch_range"]

    # ladder centers (x is left edge in object format; use x + w/2)
    lcx = ladder_x + ladder_w * 0.5
    tx, ty, tactive = _nearest_active_ladder(px, py, lcx, ladder_y, ladder_active, w_y)

    have_ladder = tactive > 0.5
    dx_l = tx - px
    aligned = jnp.abs(dx_l) < align_tol
    ladder_above = ty < py  # ladder is higher (smaller y)

    # default: move horizontally toward ladder; if aligned, climb up
    move_dir_x = jnp.sign(dx_l)  # -1 left, +1 right, 0 none
    horiz_action = jnp.where(move_dir_x > 0, RIGHT, LEFT)
    horiz_action = jnp.where(jnp.abs(move_dir_x) < 0.5, NOOP, horiz_action)

    climb_action = jnp.where(ladder_above, UP, DOWN)

    base_action = jnp.where(have_ladder & aligned, climb_action, horiz_action)
    # if no ladder available, try to move toward child x via... just use NOOP fallback
    base_action = jnp.where(have_ladder, base_action, NOOP)

    # Danger handling: monkeys and coconuts
    m_near, m_sign, m_dist = _danger_score(
        px, py, monkey_x, monkey_y, monkey_active, dx_th, dy_th
    )
    c_near, c_sign, _ = _danger_score(
        px, py, coconut_x, coconut_y, coconut_active, dx_th, dy_th
    )
    fc_near, fc_sign, _ = _danger_score(
        px, py, fc_x, fc_y, fc_active, dx_th, dy_th
    )

    # If monkey is in punch range, FIRE toward it
    punch_now = m_near & (m_dist < punch_range)
    punch_action = jnp.where(m_sign > 0, RIGHTFIRE, LEFTFIRE)

    # If coconut/falling coconut nearby, dodge opposite direction
    dodge_any = c_near | fc_near
    dodge_sign = jnp.where(c_near, c_sign, fc_sign)
    dodge_action = jnp.where(dodge_sign > 0, LEFT, RIGHT)

    action = base_action
    action = jnp.where(dodge_any, dodge_action, action)
    action = jnp.where(punch_now, punch_action, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
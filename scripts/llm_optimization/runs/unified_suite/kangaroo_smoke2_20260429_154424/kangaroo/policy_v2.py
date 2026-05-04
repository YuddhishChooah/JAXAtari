"""
Auto-generated policy v2
Generated at: 2026-04-29 15:47:12
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

BIG = jnp.float32(1e6)


def init_params():
    return {
        "align_tol": jnp.float32(6.0),       # x-alignment tolerance for ladder
        "reach_tol": jnp.float32(20.0),      # how close ladder bottom must be to player y
        "top_margin": jnp.float32(4.0),      # ignore ladder if py is already above its top
        "danger_dx": jnp.float32(20.0),      # x-range for hazard
        "danger_dy": jnp.float32(18.0),      # y-range for hazard
        "punch_range": jnp.float32(26.0),    # FIRE range vs monkey
        "w_child": jnp.float32(0.4),         # bias ladder choice toward child_x
        "w_reach": jnp.float32(2.0),         # penalty on ladder reachability
    }


def _select_ladder(px, py, lcx, ly, lh, lactive, child_x, p):
    # ladder bottom and top (y grows downward)
    l_top = ly
    l_bot = ly + lh
    # reachable: bottom near or below player, top above player
    bot_dy = jnp.abs(l_bot - py)               # how far ladder bottom is from player
    top_above = (l_top + p["top_margin"]) < py # ladder top is meaningfully above player
    reachable = (bot_dy < p["reach_tol"] + 30.0) & top_above & (lactive > 0.5)

    cost = jnp.abs(lcx - px) \
        + p["w_reach"] * bot_dy \
        + p["w_child"] * jnp.abs(lcx - child_x)
    cost = jnp.where(reachable, cost, BIG)
    idx = jnp.argmin(cost)
    found = cost[idx] < BIG * 0.5
    return lcx[idx], l_top[idx], l_bot[idx], found


def _hazard_dir(px, py, ex, ey, eactive, dx_th, dy_th):
    dx = ex - px
    dy = ey - py
    near = (jnp.abs(dx) < dx_th) & (jnp.abs(dy) < dy_th) & (eactive > 0.5)
    dist = jnp.where(near, jnp.abs(dx) + jnp.abs(dy), BIG)
    idx = jnp.argmin(dist)
    any_near = jnp.any(near)
    return any_near, jnp.sign(dx[idx]), jnp.abs(dx[idx])


def _move_toward_x(dx, align_tol):
    # returns LEFT/RIGHT, defaulting to RIGHT if perfectly aligned (anti-stall)
    go_right = dx > 0.0
    aligned = jnp.abs(dx) < align_tol
    act = jnp.where(go_right, RIGHT, LEFT)
    return act, aligned


def policy(obs_flat, params):
    px = obs_flat[0]
    py = obs_flat[1]
    orient = obs_flat[7]  # 90 right, 270 left

    ladder_x = obs_flat[168:188]
    ladder_y = obs_flat[188:208]
    ladder_w = obs_flat[208:228]
    ladder_h = obs_flat[228:248]
    ladder_active = obs_flat[248:268]

    child_x = obs_flat[360]

    fc_x = obs_flat[368:369]
    fc_y = obs_flat[369:370]
    fc_active = obs_flat[372:373]

    monkey_x = obs_flat[376:380]
    monkey_y = obs_flat[380:384]
    monkey_active = obs_flat[392:396]

    coconut_x = obs_flat[408:412]
    coconut_y = obs_flat[412:416]
    coconut_active = obs_flat[424:428]

    p = params
    align_tol = p["align_tol"]
    dx_th = p["danger_dx"]
    dy_th = p["danger_dy"]

    lcx = ladder_x + ladder_w * 0.5
    tx, t_top, t_bot, found = _select_ladder(
        px, py, lcx, ladder_y, ladder_h, ladder_active, child_x, p
    )

    dx_l = tx - px
    aligned_x = jnp.abs(dx_l) < align_tol
    # in vertical climb band: player is between ladder top and bottom (with slack)
    in_band = (py <= t_bot + p["reach_tol"]) & (py >= t_top - p["top_margin"])

    # mode 1: aligned and in band -> climb UP
    # mode 2: aligned but below ladder bottom (shouldn't happen often) -> UP too
    # mode 3: not aligned -> traverse toward ladder x
    horiz_act, _ = _move_toward_x(dx_l, align_tol)
    # if perfectly aligned but not yet in band, fall through to climb (UP)
    climb_or_traverse = jnp.where(aligned_x & in_band, UP, horiz_act)

    # if no ladder found, traverse toward child_x to look for one
    fallback_act, _ = _move_toward_x(child_x - px, align_tol)
    base_action = jnp.where(found, climb_or_traverse, fallback_act)

    # ---- Hazards ----
    m_near, m_sign, m_dist = _hazard_dir(
        px, py, monkey_x, monkey_y, monkey_active, dx_th, dy_th
    )
    c_near, c_sign, _ = _hazard_dir(
        px, py, coconut_x, coconut_y, coconut_active, dx_th, dy_th
    )
    fc_near, fc_sign, _ = _hazard_dir(
        px, py, fc_x, fc_y, fc_active, dx_th, dy_th
    )

    # Punch monkey if very close horizontally
    punch_now = m_near & (m_dist < p["punch_range"])
    # Orient punch using monkey side; fall back to player orientation
    face_right = orient < 180.0
    punch_dir_right = jnp.where(jnp.abs(m_sign) > 0.5, m_sign > 0, face_right)
    punch_action = jnp.where(punch_dir_right, RIGHTFIRE, LEFTFIRE)

    # Dodge thrown / falling coconuts: step opposite
    dodge_any = c_near | fc_near
    dodge_sign = jnp.where(fc_near, fc_sign, c_sign)
    dodge_action = jnp.where(dodge_sign > 0, LEFT, RIGHT)

    action = base_action
    action = jnp.where(dodge_any, dodge_action, action)
    action = jnp.where(punch_now, punch_action, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
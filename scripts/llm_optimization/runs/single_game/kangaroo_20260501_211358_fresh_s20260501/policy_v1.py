"""
Auto-generated policy v1
Generated at: 2026-05-01 21:14:51
"""

import jax
import jax.numpy as jnp


# ---- observation index aliases ----
PLAYER_X, PLAYER_Y, PLAYER_W, PLAYER_H = 0, 1, 2, 3
PLAYER_ORI = 7

LAD_X0, LAD_X1 = 168, 188
LAD_Y0, LAD_Y1 = 188, 208
LAD_W0, LAD_W1 = 208, 228
LAD_H0, LAD_H1 = 228, 248
LAD_A0, LAD_A1 = 248, 268

CHILD_X, CHILD_Y = 360, 361

FCOCO_X, FCOCO_Y, FCOCO_A = 368, 369, 372

MON_X0, MON_X1 = 376, 380
MON_Y0, MON_Y1 = 380, 384
MON_A0, MON_A1 = 392, 396

COCO_X0, COCO_X1 = 408, 412
COCO_Y0, COCO_Y1 = 412, 416
COCO_A0, COCO_A1 = 424, 428


# ---- actions ----
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT = 6, 7
DOWNRIGHT, DOWNLEFT = 8, 9
RIGHTFIRE, LEFTFIRE = 11, 12


def init_params():
    return {
        "ladder_reach": jnp.float32(14.0),     # |ladder_bottom - player_bottom| tolerance
        "ladder_align": jnp.float32(4.0),      # |player_center - ladder_center| to enter climb
        "top_dismount": jnp.float32(8.0),      # near top -> step off
        "punch_dx": jnp.float32(14.0),         # horizontal punch range
        "danger_r": jnp.float32(20.0),         # avoid distance for falling/thrown coconuts
        "climb_hold": jnp.float32(6.0),        # x-band tolerance to keep climbing
    }


def _select_ladder(obs, p_cx, p_by):
    lx = jax.lax.dynamic_slice(obs, (LAD_X0,), (20,))
    ly = jax.lax.dynamic_slice(obs, (LAD_Y0,), (20,))
    lw = jax.lax.dynamic_slice(obs, (LAD_W0,), (20,))
    lh = jax.lax.dynamic_slice(obs, (LAD_H0,), (20,))
    la = jax.lax.dynamic_slice(obs, (LAD_A0,), (20,))

    lcx = lx + lw * 0.5
    lby = ly + lh
    lty = ly

    # reachable from current platform: bottom near player feet, top above player
    reach_err = jnp.abs(lby - p_by)
    upward = lty < (p_by - 4.0)
    active = la > 0.5

    # cost: reach error + small horizontal pull, only valid where reachable+upward+active
    valid = active & upward & (reach_err < 30.0)
    cost = reach_err + 0.05 * jnp.abs(lcx - p_cx)
    cost = jnp.where(valid, cost, 1e6)

    idx = jnp.argmin(cost)
    return lcx[idx], lty[idx], lby[idx], cost[idx] < 1e5


def _nearest_monkey(obs, px, py):
    mx = jax.lax.dynamic_slice(obs, (MON_X0,), (4,))
    my = jax.lax.dynamic_slice(obs, (MON_Y0,), (4,))
    ma = jax.lax.dynamic_slice(obs, (MON_A0,), (4,))
    dx = mx - px
    dy = my - py
    d = jnp.abs(dx) + jnp.abs(dy)
    d = jnp.where(ma > 0.5, d, 1e6)
    i = jnp.argmin(d)
    return mx[i], my[i], d[i] < 1e5


def _coconut_danger(obs, px, py, r):
    fx = obs[FCOCO_X]
    fy = obs[FCOCO_Y]
    fa = obs[FCOCO_A] > 0.5
    near_f = fa & (jnp.abs(fx - px) < r) & (jnp.abs(fy - py) < r * 1.5)

    cx = jax.lax.dynamic_slice(obs, (COCO_X0,), (4,))
    cy = jax.lax.dynamic_slice(obs, (COCO_Y0,), (4,))
    ca = jax.lax.dynamic_slice(obs, (COCO_A0,), (4,))
    near_t = jnp.any((ca > 0.5) & (jnp.abs(cx - px) < r) & (jnp.abs(cy - py) < r))

    # sign: +1 means coconut to the right of player
    cdx = jnp.where(fa, fx - px, 0.0)
    return near_f | near_t, cdx


def policy(obs_flat, params):
    obs = obs_flat.astype(jnp.float32)

    px = obs[PLAYER_X]
    py = obs[PLAYER_Y]
    pw = obs[PLAYER_W]
    ph = obs[PLAYER_H]
    p_cx = px + pw * 0.5
    p_by = py + ph

    lcx, lty, lby, lvalid = _select_ladder(obs, p_cx, p_by)

    align_err = p_cx - lcx  # >0 means player right of ladder
    abs_align = jnp.abs(align_err)

    on_column = lvalid & (abs_align < params["ladder_align"])
    in_climb_band = lvalid & (abs_align < params["climb_hold"])
    near_top = py < (lty + params["top_dismount"])

    # monkey punch decision (preserve early-route punch behavior)
    mx, my, m_ok = _nearest_monkey(obs, px, py)
    mdx = mx - px
    same_row = jnp.abs(my - py) < (ph + 6.0)
    in_punch = m_ok & same_row & (jnp.abs(mdx) < params["punch_dx"])
    punch_right = in_punch & (mdx >= 0.0)
    punch_left = in_punch & (mdx < 0.0)

    # coconut avoidance
    in_danger, cdx = _coconut_danger(obs, p_cx, py, params["danger_r"])
    dodge_left = in_danger & (cdx >= 0.0)
    dodge_right = in_danger & (cdx < 0.0)

    # Default route: head toward selected ladder, climb when aligned.
    # Horizontal step toward ladder
    go_right_to_lad = lvalid & (align_err < -1.0)
    go_left_to_lad = lvalid & (align_err > 1.0)

    # Climbing logic: keep going up while inside climb band and not at top
    climbing = in_climb_band & (~near_top)

    # Action priority
    a = jnp.int32(NOOP)

    # 1) climb if in band
    a = jnp.where(climbing, jnp.int32(UP), a)

    # 2) horizontal traversal toward ladder if not yet on column
    a = jnp.where((~climbing) & go_right_to_lad, jnp.int32(RIGHT), a)
    a = jnp.where((~climbing) & go_left_to_lad, jnp.int32(LEFT), a)

    # 3) punch monkey when in range (preserve first-route punch)
    a = jnp.where(punch_right, jnp.int32(RIGHTFIRE), a)
    a = jnp.where(punch_left, jnp.int32(LEFTFIRE), a)

    # 4) dodge coconut overrides
    a = jnp.where(dodge_left & (~climbing), jnp.int32(LEFT), a)
    a = jnp.where(dodge_right & (~climbing), jnp.int32(RIGHT), a)

    # 5) if at top of ladder, step off in direction of child
    child_dx = obs[CHILD_X] - p_cx
    dismount = on_column & near_top
    a = jnp.where(dismount & (child_dx >= 0.0), jnp.int32(RIGHT), a)
    a = jnp.where(dismount & (child_dx < 0.0), jnp.int32(LEFT), a)

    return a


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
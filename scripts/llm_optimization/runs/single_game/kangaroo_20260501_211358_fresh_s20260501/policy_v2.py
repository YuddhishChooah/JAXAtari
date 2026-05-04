"""
Auto-generated policy v2
Generated at: 2026-05-01 21:38:36
"""

import jax
import jax.numpy as jnp


# ---- observation index aliases ----
PLAYER_X, PLAYER_Y, PLAYER_W, PLAYER_H = 0, 1, 2, 3
PLAYER_ORI = 7

LAD_X0 = 168
LAD_Y0 = 188
LAD_W0 = 208
LAD_H0 = 228
LAD_A0 = 248

CHILD_X, CHILD_Y = 360, 361

FCOCO_X, FCOCO_Y, FCOCO_A = 368, 369, 372

MON_X0 = 376
MON_Y0 = 380
MON_A0 = 392

COCO_X0 = 408
COCO_Y0 = 412
COCO_A0 = 424


# ---- actions ----
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT = 6, 7
DOWNRIGHT, DOWNLEFT = 8, 9
RIGHTFIRE, LEFTFIRE = 11, 12


def init_params():
    return {
        "ladder_reach": jnp.float32(14.0),
        "ladder_align": jnp.float32(4.0),
        "top_dismount": jnp.float32(8.0),
        "punch_dx": jnp.float32(14.0),
        "danger_r": jnp.float32(20.0),
        "climb_hold": jnp.float32(8.0),
    }


def _select_ladder(obs, p_cx, p_by, reach_tol):
    lx = jax.lax.dynamic_slice(obs, (LAD_X0,), (20,))
    ly = jax.lax.dynamic_slice(obs, (LAD_Y0,), (20,))
    lw = jax.lax.dynamic_slice(obs, (LAD_W0,), (20,))
    lh = jax.lax.dynamic_slice(obs, (LAD_H0,), (20,))
    la = jax.lax.dynamic_slice(obs, (LAD_A0,), (20,))

    lcx = lx + lw * 0.5
    lby = ly + lh
    lty = ly

    reach_err = jnp.abs(lby - p_by)
    upward = lty < (p_by - 4.0)
    active = la > 0.5
    reachable = reach_err < (reach_tol + 12.0)

    valid = active & upward & reachable
    cost = reach_err + 0.05 * jnp.abs(lcx - p_cx)
    cost = jnp.where(valid, cost, 1e6)

    idx = jnp.argmin(cost)
    return lcx[idx], lty[idx], lby[idx], cost[idx] < 1e5


def _select_next_ladder(obs, p_cx, new_p_by, reach_tol):
    # Used to pick the next ladder above, from the new (higher) platform.
    return _select_ladder(obs, p_cx, new_p_by, reach_tol)


def _nearest_monkey(obs, px, py, ph):
    mx = jax.lax.dynamic_slice(obs, (MON_X0,), (4,))
    my = jax.lax.dynamic_slice(obs, (MON_Y0,), (4,))
    ma = jax.lax.dynamic_slice(obs, (MON_A0,), (4,))
    same_row = jnp.abs(my - py) < (ph + 6.0)
    d = jnp.abs(mx - px)
    d = jnp.where((ma > 0.5) & same_row, d, 1e6)
    i = jnp.argmin(d)
    return mx[i], my[i], d[i] < 1e5, d[i]


def _coconut_danger(obs, px, py, r):
    fx = obs[FCOCO_X]
    fy = obs[FCOCO_Y]
    fa = obs[FCOCO_A] > 0.5
    near_f = fa & (jnp.abs(fx - px) < r) & (jnp.abs(fy - py) < r * 1.5)

    cx = jax.lax.dynamic_slice(obs, (COCO_X0,), (4,))
    cy = jax.lax.dynamic_slice(obs, (COCO_Y0,), (4,))
    ca = jax.lax.dynamic_slice(obs, (COCO_A0,), (4,))
    near_t = jnp.any((ca > 0.5) & (jnp.abs(cx - px) < r) & (jnp.abs(cy - py) < r))

    cdx_f = jnp.where(fa & near_f, fx - px, 0.0)
    return near_f | near_t, cdx_f


def policy(obs_flat, params):
    obs = obs_flat.astype(jnp.float32)

    px = obs[PLAYER_X]
    py = obs[PLAYER_Y]
    pw = obs[PLAYER_W]
    ph = obs[PLAYER_H]
    p_cx = px + pw * 0.5
    p_by = py + ph

    reach = params["ladder_reach"]
    align_tol = params["ladder_align"]
    climb_hold = params["climb_hold"]
    top_dis = params["top_dismount"]
    punch_dx = params["punch_dx"]
    danger_r = params["danger_r"]

    # Primary ladder from current platform
    lcx, lty, lby, lvalid = _select_ladder(obs, p_cx, p_by, reach)

    align_err = p_cx - lcx  # >0 player right of ladder
    abs_align = jnp.abs(align_err)

    # On-column: real climb predicate, requires player to be vertically within ladder span.
    in_span = (py > lty - 6.0) & (p_by < lby + 6.0)
    on_column = lvalid & (abs_align < climb_hold) & in_span
    near_top = on_column & (py < (lty + top_dis))

    # Next ladder selection (from approximate next platform y = lty)
    nxt_lcx, _, _, nxt_valid = _select_next_ladder(obs, p_cx, lty + 4.0, reach)

    # Monkey punch (preserve first-route punch, independent of on_column)
    mx, my, m_ok, mdist = _nearest_monkey(obs, px, py, ph)
    mdx = mx - px
    in_punch = m_ok & (mdist < punch_dx)
    punch_right = in_punch & (mdx >= 0.0)
    punch_left = in_punch & (mdx < 0.0)

    # Coconut danger
    in_danger, cdx = _coconut_danger(obs, p_cx, py, danger_r)
    dodge_left = in_danger & (cdx >= 0.0)
    dodge_right = in_danger & (cdx < 0.0)

    # Horizontal traversal toward selected ladder
    go_right_lad = lvalid & (align_err < -align_tol)
    go_left_lad = lvalid & (align_err > align_tol)

    # Dismount direction: toward next ladder cx if available, else toward child
    child_dx = obs[CHILD_X] - p_cx
    target_dx = jnp.where(nxt_valid, nxt_lcx - p_cx, child_dx)
    dismount_right = near_top & (target_dx >= 0.0)
    dismount_left = near_top & (target_dx < 0.0)

    # Climb when on column and not at top
    climb = on_column & (~near_top)

    # ---- Priority via jnp.select (highest first) ----
    # Priority order:
    # 1) Punch monkey (preserve stable-200 first reward)
    # 2) Dismount at top of ladder toward next route
    # 3) Climb when on column
    # 4) Dodge coconut when not climbing
    # 5) Horizontal traversal toward ladder
    # 6) NOOP
    conds = [
        punch_right,
        punch_left,
        dismount_right,
        dismount_left,
        climb,
        dodge_left & (~on_column),
        dodge_right & (~on_column),
        go_right_lad,
        go_left_lad,
    ]
    actions = [
        jnp.int32(RIGHTFIRE),
        jnp.int32(LEFTFIRE),
        jnp.int32(RIGHT),
        jnp.int32(LEFT),
        jnp.int32(UP),
        jnp.int32(LEFT),
        jnp.int32(RIGHT),
        jnp.int32(RIGHT),
        jnp.int32(LEFT),
    ]
    a = jnp.select(conds, actions, default=jnp.int32(UP))
    return a


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
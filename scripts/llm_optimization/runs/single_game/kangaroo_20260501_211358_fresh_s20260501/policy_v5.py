"""
Auto-generated policy v5
Generated at: 2026-05-01 21:57:58
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
        "ladder_reach": jnp.float32(13.52),
        "ladder_align": jnp.float32(4.20),
        "top_dismount": jnp.float32(7.99),
        "punch_dx": jnp.float32(14.52),
        "danger_r": jnp.float32(20.31),
        "climb_hold": jnp.float32(6.39),
        "dismount_band": jnp.float32(12.0),
        "same_lad_pen": jnp.float32(40.0),
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


def _select_next_ladder(obs, p_cx, new_p_by, reach_tol, exclude_cx, exclude_tol, child_x, same_pen):
    lx = jax.lax.dynamic_slice(obs, (LAD_X0,), (20,))
    ly = jax.lax.dynamic_slice(obs, (LAD_Y0,), (20,))
    lw = jax.lax.dynamic_slice(obs, (LAD_W0,), (20,))
    lh = jax.lax.dynamic_slice(obs, (LAD_H0,), (20,))
    la = jax.lax.dynamic_slice(obs, (LAD_A0,), (20,))

    lcx = lx + lw * 0.5
    lby = ly + lh
    lty = ly

    reach_err = jnp.abs(lby - new_p_by)
    upward = lty < (new_p_by - 4.0)
    active = la > 0.5
    reachable = reach_err < (reach_tol + 14.0)

    same_col = jnp.abs(lcx - exclude_cx) < exclude_tol
    valid = active & upward & reachable

    # Tie-break: prefer ladders on the side of the child
    side_bias = -0.02 * jnp.sign(child_x - lcx) * jnp.sign(lcx - exclude_cx)
    cost = reach_err + 0.05 * jnp.abs(lcx - exclude_cx) + side_bias
    cost = cost + jnp.where(same_col, same_pen, 0.0)
    cost = jnp.where(valid, cost, 1e6)

    idx = jnp.argmin(cost)
    return lcx[idx], lty[idx], cost[idx] < 1e5


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
    dismount_band = params["dismount_band"]
    same_pen = params["same_lad_pen"]

    child_x = obs[CHILD_X]

    # Primary ladder from current platform
    lcx, lty, lby, lvalid = _select_ladder(obs, p_cx, p_by, reach)

    align_err = p_cx - lcx  # >0 player right of ladder
    abs_align = jnp.abs(align_err)

    in_span = (py > lty - 6.0) & (p_by < lby + 6.0)
    on_column = lvalid & (abs_align < climb_hold) & in_span
    near_top = on_column & (py < (lty + top_dis))

    # Post-climb lateral commit: geometry-only band above the just-climbed ladder top.
    # Triggers even if player has stepped slightly off the column.
    post_climb_zone = lvalid & (py < lty + top_dis) & (py > lty - dismount_band) \
                      & (jnp.abs(p_cx - lcx) < (climb_hold + dismount_band))

    # Next ladder selection (from approximate next platform y = lty + ph),
    # excluding the just-climbed column.
    new_p_by = lty + ph
    nxt_lcx, _, nxt_valid = _select_next_ladder(
        obs, p_cx, new_p_by, reach, lcx, climb_hold + 2.0, child_x, same_pen
    )

    # Monkey punch — preserve first-route punch unconditionally.
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

    # Dismount / post-climb commit direction: prefer next ladder cx, fallback child.
    target_dx = jnp.where(nxt_valid, nxt_lcx - p_cx, child_x - p_cx)
    dismount_right = post_climb_zone & (target_dx >= 0.0)
    dismount_left = post_climb_zone & (target_dx < 0.0)

    # Climb when on column and not at top
    climb = on_column & (~near_top)

    # In post-climb zone, allow coconut dodge to override (vertical column hazard).
    danger_in_zone = in_danger & post_climb_zone

    # ---- Priority via jnp.select (highest first) ----
    conds = [
        punch_right,                               # 1: preserve first reward
        punch_left,
        danger_in_zone & (cdx >= 0.0),             # 2: dodge falling coconut at top
        danger_in_zone & (cdx < 0.0),
        dismount_right,                            # 3: post-climb lateral commit
        dismount_left,
        climb,                                     # 4: climb
        dodge_left & (~on_column),                 # 5: ground-level dodge
        dodge_right & (~on_column),
        go_right_lad,                              # 6: traverse to ladder
        go_left_lad,
    ]
    actions = [
        jnp.int32(RIGHTFIRE),
        jnp.int32(LEFTFIRE),
        jnp.int32(LEFT),
        jnp.int32(RIGHT),
        jnp.int32(RIGHT),
        jnp.int32(LEFT),
        jnp.int32(UP),
        jnp.int32(LEFT),
        jnp.int32(RIGHT),
        jnp.int32(RIGHT),
        jnp.int32(LEFT),
    ]
    # Default is NOOP (not UP) to avoid silent stall-and-die at the top.
    a = jnp.select(conds, actions, default=jnp.int32(NOOP))
    return a


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
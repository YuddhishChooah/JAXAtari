"""
Auto-generated policy v3
Generated at: 2026-05-01 21:45:01
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
        "ladder_reach": jnp.float32(13.5),
        "ladder_align": jnp.float32(4.2),
        "top_dismount": jnp.float32(8.0),
        "punch_dx": jnp.float32(14.5),
        "danger_r": jnp.float32(20.3),
        "climb_hold": jnp.float32(6.4),
        "next_skip_dx": jnp.float32(8.0),
        "dismount_band": jnp.float32(10.0),
    }


def _select_ladder(obs, p_cx, p_by, reach_tol, skip_cx, skip_active, skip_dx):
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

    # Skip the just-used ladder column when requested
    skip_mask = skip_active & (jnp.abs(lcx - skip_cx) < skip_dx)
    cost = cost + jnp.where(skip_mask, 1e5, 0.0)
    cost = jnp.where(valid, cost, 1e6)

    idx = jnp.argmin(cost)
    return lcx[idx], lty[idx], lby[idx], cost[idx] < 1e5


def _nearest_monkey(obs, px, py, ph):
    mx = jax.lax.dynamic_slice(obs, (MON_X0,), (4,))
    my = jax.lax.dynamic_slice(obs, (MON_Y0,), (4,))
    ma = jax.lax.dynamic_slice(obs, (MON_A0,), (4,))
    same_row = jnp.abs(my - py) < (ph + 6.0)
    d = jnp.abs(mx - px)
    d = jnp.where((ma > 0.5) & same_row, d, 1e6)
    i = jnp.argmin(d)
    return mx[i], my[i], d[i] < 1e5, d[i]


def _falling_coconut_above(obs, p_cx, py, r):
    fx = obs[FCOCO_X]
    fy = obs[FCOCO_Y]
    fa = obs[FCOCO_A] > 0.5
    above = fy < py + 4.0
    near_x = jnp.abs(fx - p_cx) < r
    danger = fa & above & near_x
    sign = fx - p_cx  # >0 means coconut to right of player
    return danger, sign


def _thrown_coconut_danger(obs, p_cx, py, r):
    cx = jax.lax.dynamic_slice(obs, (COCO_X0,), (4,))
    cy = jax.lax.dynamic_slice(obs, (COCO_Y0,), (4,))
    ca = jax.lax.dynamic_slice(obs, (COCO_A0,), (4,))
    near = (ca > 0.5) & (jnp.abs(cx - p_cx) < r) & (jnp.abs(cy - py) < r)
    return jnp.any(near)


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
    next_skip_dx = params["next_skip_dx"]
    dismount_band = params["dismount_band"]

    # Primary ladder from current platform (no skip)
    lcx, lty, lby, lvalid = _select_ladder(
        obs, p_cx, p_by, reach,
        jnp.float32(0.0), jnp.bool_(False), next_skip_dx,
    )

    align_err = p_cx - lcx
    abs_align = jnp.abs(align_err)

    # Climb predicate: tighter alignment, asymmetric vertical slack
    in_span_climb = (py > lty - top_dis * 2.0) & (p_by < lby + 6.0)
    on_column = lvalid & (abs_align < climb_hold) & in_span_climb

    # Dismount band: decoupled from in_span. True when player's top is near ladder top.
    at_ladder_top = (
        lvalid
        & (abs_align < climb_hold + 2.0)
        & (py < lty + dismount_band)
        & (py > lty - dismount_band)
    )
    near_top = on_column & (py < (lty + top_dis))

    # Look-ahead ladder from upper platform, skipping just-used column
    nxt_lcx, nxt_lty, nxt_lby, nxt_valid = _select_ladder(
        obs, p_cx, lty + 4.0, reach,
        lcx, lvalid, next_skip_dx,
    )

    # Monkey punch (PRESERVED first-reward branch)
    mx, my, m_ok, mdist = _nearest_monkey(obs, px, py, ph)
    mdx = mx - px
    in_punch = m_ok & (mdist < punch_dx)
    punch_right = in_punch & (mdx >= 0.0)
    punch_left = in_punch & (mdx < 0.0)

    # Falling coconut hazard (especially while climbing)
    fcoco_danger, fcoco_sign = _falling_coconut_above(obs, p_cx, py, danger_r)
    thrown_danger = _thrown_coconut_danger(obs, p_cx, py, danger_r)

    # Escape off ladder when falling coconut is overhead and player is climbing
    climb_escape_left = on_column & fcoco_danger & (fcoco_sign >= 0.0)
    climb_escape_right = on_column & fcoco_danger & (fcoco_sign < 0.0)

    # Horizontal traversal toward selected ladder
    go_right_lad = lvalid & (align_err < -align_tol)
    go_left_lad = lvalid & (align_err > align_tol)

    # Dismount direction: toward next-ladder x if known, else toward child
    child_dx = obs[CHILD_X] - p_cx
    target_dx = jnp.where(nxt_valid, nxt_lcx - p_cx, child_dx)
    dismount_right = at_ladder_top & (target_dx >= 0.0) & (~near_top | (py < lty + 2.0))
    dismount_left = at_ladder_top & (target_dx < 0.0) & (~near_top | (py < lty + 2.0))

    # Climb when on column and not at top
    climb = on_column & (~near_top) & (~fcoco_danger)

    # Dodge thrown coconut when not climbing
    thrown_dodge_left = thrown_danger & (~on_column) & (p_cx > 80.0)
    thrown_dodge_right = thrown_danger & (~on_column) & (p_cx <= 80.0)

    # ---- Priority via jnp.select (highest first) ----
    conds = [
        punch_right,
        punch_left,
        climb_escape_left,
        climb_escape_right,
        dismount_right,
        dismount_left,
        climb,
        thrown_dodge_left,
        thrown_dodge_right,
        go_right_lad,
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
    a = jnp.select(conds, actions, default=jnp.int32(UP))
    return a


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
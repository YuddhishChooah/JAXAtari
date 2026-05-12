"""
Auto-generated policy v2
Generated at: 2026-05-08 15:58:55
"""

import jax
import jax.numpy as jnp


# ---- observation aliases ----
PLAYER_X, PLAYER_Y, PLAYER_W, PLAYER_H = 0, 1, 2, 3
PLAYER_ORIENT = 7

LADDER_X0, LADDER_X1 = 168, 188
LADDER_Y0, LADDER_Y1 = 188, 208
LADDER_W0, LADDER_W1 = 208, 228
LADDER_H0, LADDER_H1 = 228, 248
LADDER_ACT0, LADDER_ACT1 = 248, 268

CHILD_X, CHILD_Y, CHILD_ACT = 360, 361, 364
FCOCO_X, FCOCO_Y, FCOCO_ACT = 368, 369, 372

MONK_X0, MONK_X1 = 376, 380
MONK_Y0, MONK_Y1 = 380, 384
MONK_ACT0, MONK_ACT1 = 392, 396

COCO_X0, COCO_X1 = 408, 412
COCO_Y0, COCO_Y1 = 412, 416
COCO_ACT0, COCO_ACT1 = 424, 428

# Action ids
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT = 6, 7
DOWNRIGHT, DOWNLEFT = 8, 9
RIGHTFIRE, LEFTFIRE = 11, 12


def init_params():
    return {
        "reach_tol": jnp.float32(10.0),
        "align_tol": jnp.float32(4.0),
        "climb_top_tol": jnp.float32(6.0),
        "monkey_punch_dx": jnp.float32(26.0),
        "monkey_row_tol": jnp.float32(20.0),
        "coco_danger_r": jnp.float32(14.0),
        "overlap_frac": jnp.float32(0.40),
    }


# ---- perception helpers ----

def _ladder_arrays(obs):
    lx = jax.lax.dynamic_slice(obs, (LADDER_X0,), (20,))
    ly = jax.lax.dynamic_slice(obs, (LADDER_Y0,), (20,))
    lw = jax.lax.dynamic_slice(obs, (LADDER_W0,), (20,))
    lh = jax.lax.dynamic_slice(obs, (LADDER_H0,), (20,))
    la = jax.lax.dynamic_slice(obs, (LADDER_ACT0,), (20,))
    return lx, ly, lw, lh, la


def _select_reachable_ladder(obs, p_cx, p_by, reach_tol):
    lx, ly, lw, lh, la = _ladder_arrays(obs)
    lcx = lx + lw * 0.5
    lby = ly + lh
    lty = ly
    reach_ok = (jnp.abs(lby - p_by) < reach_tol) & (la > 0.5) & (lty < p_by - 8.0)
    big = jnp.float32(1e6)
    cost = jnp.where(reach_ok, jnp.abs(lcx - p_cx), big)
    idx = jnp.argmin(cost)
    has_ladder = jnp.any(reach_ok)
    return idx, has_ladder, lcx, lty, lby, lw


def _select_next_ladder_from_y(obs, p_cx, ref_by, reach_tol):
    # Used to pick a dismount direction by looking at ladders reachable
    # from a higher-platform player_bottom_y estimate.
    lx, ly, lw, lh, la = _ladder_arrays(obs)
    lcx = lx + lw * 0.5
    lby = ly + lh
    lty = ly
    reach_ok = (jnp.abs(lby - ref_by) < reach_tol * 1.5) & (la > 0.5) & (lty < ref_by - 8.0)
    big = jnp.float32(1e6)
    cost = jnp.where(reach_ok, jnp.abs(lcx - p_cx), big)
    idx = jnp.argmin(cost)
    has_next = jnp.any(reach_ok)
    return has_next, lcx[idx]


def _column_overlap_frac(p_x, p_w, lcx, lw):
    overlap_left = jnp.maximum(p_x, lcx - lw * 0.5)
    overlap_right = jnp.minimum(p_x + p_w, lcx + lw * 0.5)
    overlap = jnp.maximum(0.0, overlap_right - overlap_left)
    return overlap / jnp.maximum(p_w, 1.0)


def _nearest_monkey_threat(obs, p_x, p_y, row_tol):
    mx = jax.lax.dynamic_slice(obs, (MONK_X0,), (4,))
    my = jax.lax.dynamic_slice(obs, (MONK_Y0,), (4,))
    ma = jax.lax.dynamic_slice(obs, (MONK_ACT0,), (4,))
    dx = mx - p_x
    dy = my - p_y
    same_row = jnp.abs(dy) < row_tol
    active = ma > 0.5
    valid = active & same_row
    big = jnp.float32(1e6)
    abs_dx = jnp.where(valid, jnp.abs(dx), big)
    idx = jnp.argmin(abs_dx)
    near_dx = abs_dx[idx]
    near_sx = jnp.sign(dx[idx])
    has = jnp.any(valid)
    return has, near_dx, near_sx


def _falling_coco_danger(obs, p_x, p_y, p_w, r):
    fx = obs[FCOCO_X]
    fy = obs[FCOCO_Y]
    fa = obs[FCOCO_ACT]
    above = (fa > 0.5) & (jnp.abs(fx - (p_x + p_w * 0.5)) < r * 0.7) & (fy < p_y) & (fy > p_y - 60.0)
    return above, jnp.sign(fx - (p_x + p_w * 0.5))


def _thrown_coco_danger(obs, p_x, p_y, r):
    cx = jax.lax.dynamic_slice(obs, (COCO_X0,), (4,))
    cy = jax.lax.dynamic_slice(obs, (COCO_Y0,), (4,))
    ca = jax.lax.dynamic_slice(obs, (COCO_ACT0,), (4,))
    d2 = (cx - p_x) ** 2 + (cy - p_y) ** 2
    near = (ca > 0.5) & (d2 < r * r)
    return jnp.any(near)


# ---- policy ----

def policy(obs_flat, params):
    obs = obs_flat.astype(jnp.float32)

    p_x = obs[PLAYER_X]
    p_y = obs[PLAYER_Y]
    p_w = obs[PLAYER_W]
    p_h = obs[PLAYER_H]
    p_cx = p_x + p_w * 0.5
    p_by = p_y + p_h

    reach_tol = params["reach_tol"]
    align_tol = params["align_tol"]
    climb_top_tol = params["climb_top_tol"]
    punch_dx = params["monkey_punch_dx"]
    row_tol = params["monkey_row_tol"]
    coco_r = params["coco_danger_r"]
    overlap_frac = params["overlap_frac"]

    idx, has_ladder, lcx, lty, lby, lw = _select_reachable_ladder(
        obs, p_cx, p_by, reach_tol
    )
    tgt_cx = lcx[idx]
    tgt_top = lty[idx]
    tgt_bot = lby[idx]
    tgt_w = lw[idx]

    dx_to_lad = tgt_cx - p_cx
    ovf = _column_overlap_frac(p_x, p_w, tgt_cx, tgt_w)
    on_column = ovf > overlap_frac
    aligned = jnp.abs(dx_to_lad) < align_tol

    # climb gate: must be on column AND aligned AND ladder still has height above feet
    has_height = p_by > tgt_top + climb_top_tol
    in_band = (p_by <= tgt_bot + 4.0)
    can_climb = has_ladder & on_column & aligned & has_height & in_band

    near_top = has_ladder & on_column & ~has_height

    # dismount selection: pick next reachable ladder from upper platform (project p_by up)
    # Estimate upper-platform feet y as just below ladder top.
    upper_by = tgt_top + p_h + 2.0
    has_next, next_cx = _select_next_ladder_from_y(obs, p_cx, upper_by, reach_tol)
    # if no next ladder found, fall back to child x or simply RIGHT
    child_x = obs[CHILD_X]
    child_act = obs[CHILD_ACT]
    fallback_x = jnp.where(child_act > 0.5, child_x, p_cx + 30.0)
    dismount_target_x = jnp.where(has_next, next_cx, fallback_x)
    dismount_right = dismount_target_x > p_cx
    dismount_action = jnp.where(dismount_right, RIGHT, LEFT)

    # monkey punch
    m_has, m_dx, m_sx = _nearest_monkey_threat(obs, p_x, p_y, row_tol)
    punch_in_range = m_has & (m_dx < punch_dx)
    punch_action = jnp.where(m_sx >= 0, RIGHTFIRE, LEFTFIRE)

    # coconut hazards
    fc_above, fc_sx = _falling_coco_danger(obs, p_x, p_y, p_w, coco_r)
    thrown_near = _thrown_coco_danger(obs, p_x, p_y, coco_r)
    dodge_action = jnp.where(fc_sx >= 0, LEFT, RIGHT)

    # navigation: approach the selected ladder horizontally
    move_right = dx_to_lad > 0
    approach_action = jnp.where(move_right, RIGHT, LEFT)

    # if no reachable ladder, walk toward child_x as a coarse fallback
    fb_dx = jnp.where(child_act > 0.5, child_x - p_cx, jnp.float32(1.0))
    fallback_action = jnp.where(fb_dx > 0, RIGHT, LEFT)

    nav_action = jnp.where(has_ladder, approach_action, fallback_action)

    # compose by priority
    action = nav_action
    # 1) climb only when on-column + aligned + ladder has height
    action = jnp.where(can_climb, UP, action)
    # 2) dismount when at top of ladder
    action = jnp.where(near_top, dismount_action, action)
    # 3) punch in-range monkey (preserve first-route punch). Allow even if near_top,
    #    but not while actively climbing mid-ladder.
    action = jnp.where(punch_in_range & ~can_climb, punch_action, action)
    # 4) dodge falling coconut directly above
    action = jnp.where(fc_above & ~punch_in_range, dodge_action, action)
    # 5) if a thrown coconut is very close and we are not punching, briefly punch toward it
    action = jnp.where(thrown_near & ~punch_in_range & ~can_climb,
                       jnp.where(obs[PLAYER_ORIENT] >= 180.0, LEFTFIRE, RIGHTFIRE),
                       action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)


def dense_reward(obs_history, actions, rewards, total_reward, active_mask):
    obs = obs_history.astype(jnp.float32)
    mask = active_mask.astype(jnp.float32)
    T = obs.shape[0]
    msum = jnp.maximum(jnp.sum(mask), 1.0)

    p_x = obs[:, PLAYER_X]
    p_y = obs[:, PLAYER_Y]
    p_h = obs[:, PLAYER_H]
    p_w = obs[:, PLAYER_W]
    p_by = p_y + p_h
    p_cx = p_x + p_w * 0.5

    # ladders
    lcx_all = obs[:, LADDER_X0:LADDER_X1] + obs[:, LADDER_W0:LADDER_W1] * 0.5
    lw_all = obs[:, LADDER_W0:LADDER_W1]
    lby_all = obs[:, LADDER_Y0:LADDER_Y1] + obs[:, LADDER_H0:LADDER_H1]
    la_all = obs[:, LADDER_ACT0:LADDER_ACT1]

    # column-overlap fraction per ladder
    ov_left = jnp.maximum(p_x[:, None], lcx_all - lw_all * 0.5)
    ov_right = jnp.minimum((p_x + p_w)[:, None], lcx_all + lw_all * 0.5)
    ov = jnp.maximum(0.0, ov_right - ov_left) / jnp.maximum(p_w[:, None], 1.0)
    on_col_any = jnp.any((ov > 0.4) & (la_all > 0.5), axis=1)

    # 1) Best (lowest) y reached -> upward bonus
    y0 = p_y[0]
    masked_y = jnp.where(mask > 0.5, p_y, 1e6)
    min_y = jnp.min(masked_y)
    best_up = jnp.clip((y0 - min_y) / 80.0, 0.0, 2.0)

    # 2) Average upward progress over time
    upward = jnp.clip((y0 - p_y) / 80.0, 0.0, 2.0) * mask
    upward_term = jnp.sum(upward) / msum

    # 3) Real climbing: UP/UPLEFT/UPRIGHT issued while on a ladder column
    is_up = ((actions == UP) | (actions == UPRIGHT) | (actions == UPLEFT)).astype(jnp.float32)
    real_climb = is_up * on_col_any.astype(jnp.float32) * mask
    climb_term = jnp.sum(real_climb) / msum  # bounded ~1

    # 4) Wasted UP penalty: UP off-column
    wasted_up = is_up * (1.0 - on_col_any.astype(jnp.float32)) * mask
    wasted_up_frac = jnp.sum(wasted_up) / msum
    wasted_up_pen = jnp.minimum(wasted_up_frac, 0.4)

    # 5) Ladder approach signal: distance to nearest reachable ladder center
    reach_ok = (jnp.abs(lby_all - p_by[:, None]) < 12.0) & (la_all > 0.5)
    big = jnp.float32(1e6)
    dxs = jnp.where(reach_ok, jnp.abs(lcx_all - p_cx[:, None]), big)
    min_dx = jnp.min(dxs, axis=1)
    has_reach = jnp.any(reach_ok, axis=1).astype(jnp.float32)
    approach_score = jnp.clip(1.0 - min_dx / 80.0, 0.0, 1.0) * has_reach * mask
    approach_term = jnp.sum(approach_score) / msum

    # 6) Goal approach by y
    child_y = obs[:, CHILD_Y]
    child_act = obs[:, CHILD_ACT]
    have_child = (child_act > 0.5).astype(jnp.float32) * mask
    init_dy = jnp.where(have_child[0] > 0.5, jnp.abs(p_y[0] - child_y[0]), 150.0)
    dy_child = jnp.abs(p_y - child_y)
    closeness = jnp.clip((init_dy - dy_child) / 150.0, 0.0, 1.5) * have_child
    goal_term = jnp.sum(closeness) / jnp.maximum(jnp.sum(have_child), 1.0)

    # 7) Punch-farming penalty: many FIRE without upward progress
    is_fire = ((actions == FIRE) | (actions == RIGHTFIRE) | (actions == LEFTFIRE)).astype(jnp.float32) * mask
    fire_frac = jnp.sum(is_fire) / msum
    no_climb = (best_up < 0.25).astype(jnp.float32)
    punch_farm_pen = jnp.minimum(fire_frac, 0.5) * no_climb

    # 8) Idle penalty
    is_noop = (actions == NOOP).astype(jnp.float32) * mask
    noop_frac = jnp.sum(is_noop) / msum
    idle_pen = jnp.minimum(noop_frac, 0.5)

    # 9) Survival
    surv = jnp.sum(mask) / jnp.maximum(jnp.float32(T), 1.0)

    shaped = (
        1.5 * best_up
        + 0.6 * upward_term
        + 1.2 * climb_term
        + 0.7 * approach_term
        + 1.0 * goal_term
        + 0.3 * surv
        - 1.0 * wasted_up_pen
        - 1.2 * punch_farm_pen
        - 0.3 * idle_pen
    )
    return shaped
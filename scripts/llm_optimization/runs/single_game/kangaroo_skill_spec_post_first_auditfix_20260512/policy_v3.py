"""
Auto-generated policy v3
Generated at: 2026-05-12 10:03:31
"""

"""
Auto-generated policy v3
Conservative post-200 patch: preserves first-reward route, fixes dismount/next-ladder.
"""

import jax
import jax.numpy as jnp

# ---- Observation index aliases ----
PLAYER_X, PLAYER_Y, PLAYER_W, PLAYER_H = 0, 1, 2, 3
PLAYER_ACTIVE, PLAYER_ORIENT = 4, 7

LADDER_X_S, LADDER_X_E = 168, 188
LADDER_Y_S, LADDER_Y_E = 188, 208
LADDER_W_S, LADDER_W_E = 208, 228
LADDER_H_S, LADDER_H_E = 228, 248
LADDER_A_S, LADDER_A_E = 248, 268

CHILD_X, CHILD_Y, CHILD_ACTIVE = 360, 361, 364

FCOC_X, FCOC_Y, FCOC_A = 368, 369, 372

MONKEY_X_S, MONKEY_X_E = 376, 380
MONKEY_Y_S, MONKEY_Y_E = 380, 384
MONKEY_A_S, MONKEY_A_E = 392, 396

COCONUT_X_S, COCONUT_X_E = 408, 412
COCONUT_Y_S, COCONUT_Y_E = 412, 416
COCONUT_A_S, COCONUT_A_E = 424, 428

# Actions
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = 6, 7, 8, 9
UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE = 10, 11, 12, 13


def init_params():
    return {
        "reach_y_tol":      jnp.array(10.0),
        "overlap_frac":     jnp.array(0.5),
        "x_align_tol":      jnp.array(5.0),
        "monkey_punch_dx":  jnp.array(18.0),
        "monkey_punch_dy":  jnp.array(12.0),
        "coconut_dx":       jnp.array(12.0),
        "min_top_above":    jnp.array(6.0),
        "dismount_tol":     jnp.array(8.0),
    }


def _clip_params(p):
    return {
        "reach_y_tol":     jnp.clip(p["reach_y_tol"],     2.0,  30.0),
        "overlap_frac":    jnp.clip(p["overlap_frac"],     0.25,  0.9),
        "x_align_tol":     jnp.clip(p["x_align_tol"],     1.0,  12.0),
        "monkey_punch_dx": jnp.clip(p["monkey_punch_dx"],  6.0,  40.0),
        "monkey_punch_dy": jnp.clip(p["monkey_punch_dy"],  4.0,  24.0),
        "coconut_dx":      jnp.clip(p["coconut_dx"],       4.0,  30.0),
        "min_top_above":   jnp.clip(p["min_top_above"],    2.0,  20.0),
        "dismount_tol":    jnp.clip(p["dismount_tol"],     4.0,  20.0),
    }


def _x_overlap_frac(px, pw, lx, lw):
    inter = jnp.maximum(0.0, jnp.minimum(px + pw, lx + lw) - jnp.maximum(px, lx))
    denom = jnp.maximum(1.0, jnp.minimum(pw, lw))
    return inter / denom


def _x_overlap_frac_arr(px, pw, lx, lw):
    inter = jnp.maximum(0.0, jnp.minimum(px + pw, lx + lw) - jnp.maximum(px, lx))
    denom = jnp.maximum(1.0, jnp.minimum(pw, lw))
    return inter / denom


def _select_target_ladder(obs, p):
    """Select reachable ladder from current platform: bottom near feet, top above."""
    px = obs[PLAYER_X]; py = obs[PLAYER_Y]; pw = obs[PLAYER_W]; ph = obs[PLAYER_H]
    feet_y = py + ph
    pcx = px + pw * 0.5

    lx  = obs[LADDER_X_S:LADDER_X_E]
    ly  = obs[LADDER_Y_S:LADDER_Y_E]
    lw  = obs[LADDER_W_S:LADDER_W_E]
    lh  = obs[LADDER_H_S:LADDER_H_E]
    la  = obs[LADDER_A_S:LADDER_A_E]

    lcx  = lx + lw * 0.5
    lby  = ly + lh
    ltop = ly

    reach_ok  = jnp.abs(lby - feet_y) < p["reach_y_tol"]
    above_ok  = ltop < (py - p["min_top_above"])
    active_ok = la > 0.5
    valid     = reach_ok & above_ok & active_ok

    dx    = jnp.abs(lcx - pcx)
    score = jnp.where(valid, -dx, -1e6)
    idx   = jnp.argmax(score)
    any_v = jnp.any(valid)
    return idx, any_v, lcx, ltop, lby, lx, lw, la


def _on_any_climbable_ladder(obs, p):
    """Is the player x-overlapping ANY active ladder column?"""
    px = obs[PLAYER_X]; py = obs[PLAYER_Y]; pw = obs[PLAYER_W]; ph = obs[PLAYER_H]
    feet_y = py + ph
    lx = obs[LADDER_X_S:LADDER_X_E]
    ly = obs[LADDER_Y_S:LADDER_Y_E]
    lw = obs[LADDER_W_S:LADDER_W_E]
    lh = obs[LADDER_H_S:LADDER_H_E]
    la = obs[LADDER_A_S:LADDER_A_E]

    ov     = _x_overlap_frac_arr(px, pw, lx, lw)
    active = la > 0.5
    # ladder column intersects player height range
    has_height = (ly + lh) > py
    valid  = active & has_height
    score  = jnp.where(valid, ov, -1.0)
    idx    = jnp.argmax(score)
    best_ov  = ov[idx]
    best_top = ly[idx]
    best_lx  = lx[idx]
    best_lw  = lw[idx]
    on_col   = (best_ov >= p["overlap_frac"]) & valid[idx]
    return on_col, best_ov, best_top, best_lx, best_lw, idx


def _select_ladder_from_feet(obs, p, feet_y_override):
    """Run the same ladder selection logic but with a given feet_y value.
    Used for post-dismount next-ladder search with the actual current feet y."""
    px = obs[PLAYER_X]; py = obs[PLAYER_Y]; pw = obs[PLAYER_W]
    pcx = px + pw * 0.5

    lx  = obs[LADDER_X_S:LADDER_X_E]
    ly  = obs[LADDER_Y_S:LADDER_Y_E]
    lw  = obs[LADDER_W_S:LADDER_W_E]
    lh  = obs[LADDER_H_S:LADDER_H_E]
    la  = obs[LADDER_A_S:LADDER_A_E]

    lcx  = lx + lw * 0.5
    lby  = ly + lh

    reach_ok  = jnp.abs(lby - feet_y_override) < p["reach_y_tol"]
    above_ok  = ly < (py - p["min_top_above"])
    active_ok = la > 0.5
    valid     = reach_ok & above_ok & active_ok

    dx    = jnp.abs(lcx - pcx)
    score = jnp.where(valid, -dx, -1e6)
    idx   = jnp.argmax(score)
    any_v = jnp.any(valid)
    return lcx[idx], any_v


def _nearest_monkey(obs):
    px = obs[PLAYER_X]; py = obs[PLAYER_Y]; pw = obs[PLAYER_W]
    mx = obs[MONKEY_X_S:MONKEY_X_E]
    my = obs[MONKEY_Y_S:MONKEY_Y_E]
    ma = obs[MONKEY_A_S:MONKEY_A_E]
    pcx = px + pw * 0.5
    dx  = mx - pcx
    dy  = my - py
    active = ma > 0.5
    dist   = jnp.where(active, jnp.abs(dx) + jnp.abs(dy), 1e6)
    idx    = jnp.argmin(dist)
    return dx[idx], dy[idx], active[idx]


def _coconut_threat(obs, p):
    px  = obs[PLAYER_X]; py = obs[PLAYER_Y]; pw = obs[PLAYER_W]
    pcx = px + pw * 0.5

    fx = obs[FCOC_X]; fy = obs[FCOC_Y]; fa = obs[FCOC_A] > 0.5
    f_threat = fa & (jnp.abs(fx - pcx) < p["coconut_dx"]) \
                  & (fy < py + 40.0) & (fy > py - 80.0)

    cx = obs[COCONUT_X_S:COCONUT_X_E]
    cy = obs[COCONUT_Y_S:COCONUT_Y_E]
    ca = obs[COCONUT_A_S:COCONUT_A_E] > 0.5
    c_threat = jnp.any(
        ca & (jnp.abs(cx - pcx) < p["coconut_dx"] * 1.5)
           & (jnp.abs(cy - py) < 14.0)
    )

    f_side_sign = jnp.sign(fx - pcx)
    return f_threat, c_threat, f_side_sign, fx


def _horizontal_action(dx_to_target, tol):
    return jnp.where(dx_to_target > tol, RIGHT,
           jnp.where(dx_to_target < -tol, LEFT, NOOP))


def policy(obs_flat, params):
    p = _clip_params(params)

    px  = obs_flat[PLAYER_X]
    py  = obs_flat[PLAYER_Y]
    pw  = obs_flat[PLAYER_W]
    ph  = obs_flat[PLAYER_H]
    pcx = px + pw * 0.5
    feet_y = py + ph

    # ---- Reachable ladder selection (preserved 200-point logic) ----
    idx, any_valid, lcx_arr, ltop_arr, lby_arr, lx_arr, lw_arr, la_arr = \
        _select_target_ladder(obs_flat, p)

    tgt_cx   = lcx_arr[idx]
    tgt_top  = ltop_arr[idx]
    tgt_lx   = lx_arr[idx]
    tgt_lw   = lw_arr[idx]
    overlap_sel = _x_overlap_frac_arr(px, pw, tgt_lx, tgt_lw)
    on_sel_ladder = (overlap_sel >= p["overlap_frac"]) & any_valid

    dx_to_target   = tgt_cx - pcx
    traverse_action = _horizontal_action(dx_to_target, p["x_align_tol"])

    # ---- Independent on-column detection ----
    on_col_any, _ov_col, col_top, col_lx, col_lw, _col_idx = \
        _on_any_climbable_ladder(obs_flat, p)
    col_cx = col_lx + col_lw * 0.5

    # ------------------------------------------------------------------
    # Near-top detection: wider zone so dismount reliably fires.
    # Use dismount_tol AND also half of player height as minimum.
    # ------------------------------------------------------------------
    effective_dismount_tol = jnp.maximum(p["dismount_tol"], ph * 0.5)
    near_top_any = on_col_any & (jnp.abs(py - col_top) <= effective_dismount_tol)

    # Climb conditions: exclude the wider near-top zone (2x dismount_tol)
    wide_near_top = on_col_any & (jnp.abs(py - col_top) <= effective_dismount_tol * 2.0)
    can_climb_sel = on_sel_ladder & (tgt_top < py - p["min_top_above"]) & ~wide_near_top
    can_climb_col = on_col_any  & (col_top  < py - p["min_top_above"]) & ~wide_near_top
    can_climb     = can_climb_sel | can_climb_col

    # ---- Hazards ----
    mdx, mdy, m_active = _nearest_monkey(obs_flat)
    in_punch_range = m_active & (jnp.abs(mdx) < p["monkey_punch_dx"]) & \
                                (jnp.abs(mdy) < p["monkey_punch_dy"])
    monkey_right = mdx > 0
    punch_action = jnp.where(monkey_right, RIGHTFIRE, LEFTFIRE)

    f_threat, c_threat, f_side_sign, fcx = _coconut_threat(obs_flat, p)
    coc_threat = f_threat | c_threat
    sidestep   = jnp.where(f_side_sign > 0, LEFT, RIGHT)

    # ---- Post-dismount next-ladder: use actual feet_y (not col_top offset) ----
    # When near top of a column, the player's feet are near feet_y; use that.
    next_cx, has_next = _select_ladder_from_feet(obs_flat, p, feet_y)
    dx_next = next_cx - pcx

    # Also consider child direction as fallback dismount direction
    cx_child  = obs_flat[CHILD_X]
    cy_child  = obs_flat[CHILD_Y]
    c_active  = obs_flat[CHILD_ACTIVE] > 0.5
    dx_child  = cx_child - pcx

    # Dismount direction priority: next ladder > child > center > away-from-coconut
    dismount_toward_next = jnp.where(dx_next > 0, RIGHT, LEFT)
    dismount_to_child    = jnp.where(dx_child > 0, RIGHT, LEFT)
    fcx_safe_dir         = jnp.where((fcx - pcx) > 0, LEFT, RIGHT)
    dismount_center      = jnp.where(pcx < 80.0, RIGHT, LEFT)
    dismount_fallback    = jnp.where(f_threat, fcx_safe_dir, dismount_center)
    dismount_no_next     = jnp.where(c_active, dismount_to_child, dismount_fallback)
    dismount_action      = jnp.where(has_next, dismount_toward_next, dismount_no_next)

    # ---- On-ladder coconut escape: dismount rather than sidestep ----
    on_ladder_coconut_escape = on_col_any & f_threat
    ladder_escape_action     = jnp.where(has_next, dismount_toward_next, fcx_safe_dir)

    # ---- Approach child when on same tier ----
    on_child_tier = c_active & (jnp.abs(cy_child - py) < 16.0)
    child_action  = _horizontal_action(cx_child - pcx, p["x_align_tol"])

    # ---- Compose action with skill priority ----
    # Base: traverse to reachable ladder
    fallback = jnp.where(
        any_valid, traverse_action,
        jnp.where(on_child_tier, child_action,
        jnp.where(has_next, _horizontal_action(dx_next, p["x_align_tol"]),
                  RIGHT))
    )
    action = fallback

    # Layer: traverse when not yet at ladder x
    action = jnp.where(any_valid & ~on_sel_ladder, traverse_action, action)

    # Layer: climb when properly on column and not near top
    action = jnp.where(can_climb, UP, action)

    # Layer: near top -> dismount (dominates UP)
    action = jnp.where(near_top_any, dismount_action, action)

    # Layer: on-ladder coconut escape (dominates dismount direction choice)
    action = jnp.where(on_ladder_coconut_escape, ladder_escape_action, action)

    # Layer: child approach when on child tier and not climbing/dismounting
    action = jnp.where(
        on_child_tier & ~can_climb & ~near_top_any, child_action, action
    )

    # Layer: coconut sidestep when NOT on a ladder column
    action = jnp.where(coc_threat & ~on_col_any, sidestep, action)

    # PRESERVED 200-POINT PUNCH BRANCH — must remain last, no column gate
    action = jnp.where(in_punch_range, punch_action, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)


def dense_reward(obs_history, actions, rewards, total_reward, active_mask):
    mask = active_mask.astype(jnp.float32)
    n_steps = mask.shape[0]

    py  = obs_history[:, PLAYER_Y]
    px  = obs_history[:, PLAYER_X]
    pw  = obs_history[:, PLAYER_W]
    ph  = obs_history[:, PLAYER_H]
    pcx = px + pw * 0.5
    feet_y = py + ph

    lx  = obs_history[:, LADDER_X_S:LADDER_X_E]
    ly  = obs_history[:, LADDER_Y_S:LADDER_Y_E]
    lw  = obs_history[:, LADDER_W_S:LADDER_W_E]
    lh  = obs_history[:, LADDER_H_S:LADDER_H_E]
    la  = obs_history[:, LADDER_A_S:LADDER_A_E]
    lcx = lx + lw * 0.5
    lby = ly + lh
    ltop = ly

    # ---- Ladder reachability matrix ----
    feet_e  = feet_y[:, None]
    py_e    = py[:, None]
    px_e    = px[:, None]
    pw_e    = pw[:, None]

    inter   = jnp.maximum(0.0, jnp.minimum(px_e + pw_e, lx + lw) - jnp.maximum(px_e, lx))
    denom   = jnp.maximum(1.0, jnp.minimum(pw_e, lw))
    ov      = inter / denom  # [T, 20]

    reach_ok  = (jnp.abs(lby - feet_e) < 12.0) & (la > 0.5)
    above_ok  = ltop < (py_e - 4.0)
    any_reach = jnp.any(reach_ok, axis=1).astype(jnp.float32)

    # Alignment: distance to nearest reachable ladder center x
    dx_to_l = jnp.where(reach_ok, jnp.abs(lcx - pcx[:, None]), 1e4)
    min_dx  = jnp.min(dx_to_l, axis=1)
    align_term = jnp.clip(1.0 - min_dx / 40.0, -1.0, 1.0) * any_reach

    # ---- On-column overlap ----
    on_col_mat = (ov >= 0.4) & above_ok & (la > 0.5)
    on_col     = jnp.any(on_col_mat, axis=1).astype(jnp.float32)

    # ---- Upward progress ----
    dpy = jnp.concatenate([jnp.zeros((1,)), py[:-1] - py[1:]])
    climb_progress = jnp.clip(dpy, -2.0, 4.0) * on_col
    climb_term     = jnp.sum(climb_progress * mask)

    # ---- Episode height gain ----
    py_masked   = jnp.where(mask > 0.5, py, 1e6)
    min_py      = jnp.min(py_masked)
    first_py    = py[0]
    height_gain = jnp.clip(first_py - min_py, 0.0, 200.0)
    height_term = height_gain * 3.0

    # ---- Post-200 progress: bonus if player reaches well above y≈132 ----
    # y < 120 means the player got higher than the first ladder top
    post200_height_bonus = jnp.where(min_py < 120.0, 40.0, 0.0)
    # Bigger bonus for each 10px beyond the first plateau
    extra_climb = jnp.clip(132.0 - min_py, 0.0, 80.0)
    extra_climb_term = extra_climb * 1.5

    # ---- Average alignment term ----
    n_active  = jnp.maximum(1.0, jnp.sum(mask))
    align_avg = jnp.sum(align_term * mask) / n_active

    # ---- Action counts ----
    fire_acts = (
        (actions == FIRE) | (actions == RIGHTFIRE) | (actions == LEFTFIRE) |
        (actions == UPFIRE) | (actions == DOWNFIRE)
    ).astype(jnp.float32)
    fire_count = jnp.sum(fire_acts * mask)

    up_acts  = (
        (actions == UP) | (actions == UPRIGHT) | (actions == UPLEFT) |
        (actions == UPFIRE)
    ).astype(jnp.float32)
    up_count = jnp.sum(up_acts * mask)

    # ---- On-column frame count ----
    on_col_count       = jnp.sum(on_col * mask)
    column_climb_bonus = jnp.clip(on_col_count, 0.0, 60.0) * 1.0

    # ---- Dismount event proxy ----
    on_col_next    = jnp.concatenate([on_col[1:], jnp.zeros((1,))])
    dpy_next       = jnp.concatenate([py[:-1] - py[1:], jnp.zeros((1,))])
    dismount_event = (
        (on_col > 0.5) & (on_col_next < 0.5) & (dpy_next >= -2.0)
    ).astype(jnp.float32)
    dismount_term  = jnp.clip(jnp.sum(dismount_event * mask), 0.0, 8.0) * 10.0

    # ---- First reward preservation bonus ----
    first_reward_pres = jnp.where(total_reward >= 100.0, 20.0, 0.0)

    # ---- Child approach when on top tier ----
    cy_arr   = obs_history[:, CHILD_Y]
    cx_arr   = obs_history[:, CHILD_X]
    on_top_t = (jnp.abs(cy_arr - py) < 18.0).astype(jnp.float32) * mask
    child_close = on_top_t * jnp.clip(1.0 - jnp.abs(cx_arr - pcx) / 80.0, 0.0, 1.0)
    child_term  = jnp.sum(child_close) * 2.0

    # ---- Penalties ----
    # Punch farming: real reward but no height gain and no UPs
    punch_farm = jnp.where(
        (total_reward > 50.0) & (height_gain < 4.0) & (up_count < 3.0),
        -80.0, 0.0
    )

    # Stall at first plateau: scored 200 but never got above y=130
    stall_at_plateau = jnp.where(
        (total_reward >= 200.0) & (min_py >= 130.0),
        -60.0, 0.0
    )

    # No FIRE and no score: policy suppressed useful punching
    no_fire_zero = jnp.where(
        (fire_count < 1.0) & (total_reward < 1.0),
        -15.0, 0.0
    )

    # No upward progress at all
    no_progress_pen = jnp.where(height_gain < 2.0, -8.0, 0.0)

    shaped = (
        align_avg          * 5.0
        + climb_term       * 1.0
        + height_term
        + extra_climb_term
        + post200_height_bonus
        + column_climb_bonus
        + first_reward_pres
        + dismount_term
        + child_term
        + punch_farm
        + stall_at_plateau
        + no_fire_zero
        + no_progress_pen
    )

    return jnp.clip(shaped, -200.0, 600.0)
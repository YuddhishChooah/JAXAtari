"""
Auto-generated policy v3
Generated at: 2026-05-12 09:40:20
"""

"""
Kangaroo policy – conservative post-200 patch.
Preserves the stable first-reward (RIGHTFIRE) route exactly.
Adds fixes for:
  - dismount re-entry suppression after reaching ladder top
  - wider falling-coconut evasion with earlier lookahead
  - dynamic offset in next-ladder-from-upper-platform search
  - stricter on-column guard after dismount (hysteresis)
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
        "reach_y_tol":      jnp.array(9.807656288146973),
        "overlap_frac":     jnp.array(0.5350363850593567),
        "x_align_tol":      jnp.array(5.064339637756348),
        "monkey_punch_dx":  jnp.array(17.810955047607422),
        "monkey_punch_dy":  jnp.array(11.94726276397705),
        "coconut_dx":       jnp.array(8.082901954650879),
        "min_top_above":    jnp.array(6.198116302490234),
        "dismount_tol":     jnp.array(4.389926433563232),
    }


def _clip_params(p):
    return {
        "reach_y_tol":     jnp.clip(p["reach_y_tol"],     2.0,  30.0),
        "overlap_frac":    jnp.clip(p["overlap_frac"],     0.25,  0.9),
        "x_align_tol":     jnp.clip(p["x_align_tol"],     1.0,  12.0),
        "monkey_punch_dx": jnp.clip(p["monkey_punch_dx"],  6.0,  40.0),
        "monkey_punch_dy": jnp.clip(p["monkey_punch_dy"],  4.0,  24.0),
        "coconut_dx":      jnp.clip(p["coconut_dx"],       2.0,  20.0),
        "min_top_above":   jnp.clip(p["min_top_above"],    2.0,  20.0),
        "dismount_tol":    jnp.clip(p["dismount_tol"],     2.0,  12.0),
    }


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
    any_valid = jnp.any(valid)
    return idx, any_valid, lcx, ltop, lby, lx, lw, la


def _on_any_climbable_ladder(obs, p, hysteresis=4.0):
    """
    Is the player overlapping ANY active ladder column?
    Uses overlap-fraction + center-band check to avoid 1-pixel false positives.
    hysteresis: extra pixels below ladder_top before on-column clears, preventing
    re-entry immediately after dismount.
    """
    px  = obs[PLAYER_X]; py  = obs[PLAYER_Y]
    pw  = obs[PLAYER_W]; ph  = obs[PLAYER_H]
    pcx = px + pw * 0.5

    lx  = obs[LADDER_X_S:LADDER_X_E]
    ly  = obs[LADDER_Y_S:LADDER_Y_E]
    lw  = obs[LADDER_W_S:LADDER_W_E]
    lh  = obs[LADDER_H_S:LADDER_H_E]
    la  = obs[LADDER_A_S:LADDER_A_E]

    lcx = lx + lw * 0.5
    ov  = _x_overlap_frac_arr(px, pw, lx, lw)

    # center-band additional requirement
    center_ok = jnp.abs(pcx - lcx) < (lw * 0.5 + 2.0)

    # ladder column spans player vertically (with hysteresis: player must be
    # more than hysteresis pixels BELOW the ladder top to still count as on it)
    col_top_with_hyst = ly + hysteresis   # must have py > this to be "inside"
    has_height = ((ly + lh) > py) & (py > col_top_with_hyst)
    active     = la > 0.5

    valid = active & has_height & center_ok
    score = jnp.where(valid, ov, -1.0)
    idx   = jnp.argmax(score)

    best_ov  = ov[idx]
    best_top = ly[idx]
    best_lx  = lx[idx]
    best_lw  = lw[idx]
    on_col   = (best_ov >= p["overlap_frac"]) & valid[idx]
    return on_col, best_ov, best_top, best_lx, best_lw, idx


def _next_ladder_from_upper_platform(obs, p, current_top_y):
    """
    After dismounting at current_top_y, find the next active ladder whose bottom
    is near (current_top_y + player_h) — using actual player_h, not +16 constant.
    """
    px  = obs[PLAYER_X]; pw  = obs[PLAYER_W]; ph = obs[PLAYER_H]
    pcx = px + pw * 0.5

    lx  = obs[LADDER_X_S:LADDER_X_E]
    ly  = obs[LADDER_Y_S:LADDER_Y_E]
    lw  = obs[LADDER_W_S:LADDER_W_E]
    lh  = obs[LADDER_H_S:LADDER_H_E]
    la  = obs[LADDER_A_S:LADDER_A_E]

    lcx = lx + lw * 0.5
    lby = ly + lh

    # Estimated new feet y after climbing to ladder top
    est_feet_y = current_top_y + ph

    # Accept ladders whose bottom is within reach_y_tol of estimated new feet
    bottom_near = jnp.abs(lby - est_feet_y) < (p["reach_y_tol"] + 8.0)
    above_ok    = ly < (current_top_y - p["min_top_above"])
    active_ok   = la > 0.5
    # Exclude the ladder we just climbed: its top_y ≈ current_top_y
    not_same    = jnp.abs(ly - current_top_y) > 4.0
    valid       = bottom_near & above_ok & active_ok & not_same

    dx    = jnp.abs(lcx - pcx)
    score = jnp.where(valid, -dx, -1e6)
    idx   = jnp.argmax(score)
    any_valid = jnp.any(valid)
    return lcx[idx], ly[idx], any_valid


def _nearest_monkey(obs):
    px  = obs[PLAYER_X]; py  = obs[PLAYER_Y]; pw  = obs[PLAYER_W]
    mx  = obs[MONKEY_X_S:MONKEY_X_E]
    my  = obs[MONKEY_Y_S:MONKEY_Y_E]
    ma  = obs[MONKEY_A_S:MONKEY_A_E]
    pcx = px + pw * 0.5
    dx  = mx - pcx
    dy  = my - py
    active = ma > 0.5
    dist   = jnp.where(active, jnp.abs(dx) + jnp.abs(dy), 1e6)
    idx    = jnp.argmin(dist)
    return dx[idx], dy[idx], active[idx]


def _coconut_threat(obs, p):
    """
    Returns (falling_threat, thrown_threat, falling_side_sign, falling_x).
    Falling-coconut lookahead is wider: 50 px above player, not 60 below.
    Horizontal threshold for falling coconut uses 2× coconut_dx for earlier reaction.
    """
    px  = obs[PLAYER_X]; py  = obs[PLAYER_Y]; pw  = obs[PLAYER_W]
    pcx = px + pw * 0.5

    fx  = obs[FCOC_X]; fy  = obs[FCOC_Y]; fa  = obs[FCOC_A] > 0.5
    # Wider horizontal window and lookahead for falling coconut
    fall_dx_thresh = p["coconut_dx"] * 2.5
    f_threat = fa & (jnp.abs(fx - pcx) < fall_dx_thresh) & (fy < py + 20.0) & (fy > py - 70.0)

    cx  = obs[COCONUT_X_S:COCONUT_X_E]
    cy  = obs[COCONUT_Y_S:COCONUT_Y_E]
    ca  = obs[COCONUT_A_S:COCONUT_A_E] > 0.5
    c_threat = jnp.any(ca & (jnp.abs(cx - pcx) < p["coconut_dx"] * 1.5) & (jnp.abs(cy - py) < 14.0))

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

    # ---- Reachable ladder selection (preserved 200-point logic) ----
    idx, any_valid, lcx_arr, ltop_arr, lby_arr, lx_arr, lw_arr, la_arr = \
        _select_target_ladder(obs_flat, p)

    tgt_cx  = lcx_arr[idx]
    tgt_top = ltop_arr[idx]
    tgt_lx  = lx_arr[idx]
    tgt_lw  = lw_arr[idx]
    overlap_sel  = _x_overlap_frac_arr(px, pw, tgt_lx, tgt_lw)
    on_sel_ladder = (overlap_sel >= p["overlap_frac"]) & any_valid

    dx_to_target   = tgt_cx - pcx
    traverse_action = _horizontal_action(dx_to_target, p["x_align_tol"])

    # ---- On-column detection with hysteresis to prevent re-entry after dismount ----
    on_col_any, _ov_col, col_top, col_lx, col_lw, _col_idx = \
        _on_any_climbable_ladder(obs_flat, p, hysteresis=p["dismount_tol"])
    col_cx = col_lx + col_lw * 0.5

    # near-top: player y is within dismount_tol of the column's top
    near_top_any = on_col_any & (jnp.abs(py - col_top) <= p["dismount_tol"])

    # can_climb: on a column AND top is above AND NOT near top (hysteresis clears re-entry)
    can_climb_sel = on_sel_ladder & (tgt_top < py - p["min_top_above"]) & ~near_top_any
    can_climb_col = on_col_any   & (col_top  < py - p["min_top_above"]) & ~near_top_any
    can_climb     = can_climb_sel | can_climb_col

    # ---- Hazards ----
    mdx, mdy, m_active = _nearest_monkey(obs_flat)
    in_punch_range = m_active & (jnp.abs(mdx) < p["monkey_punch_dx"]) & \
                                (jnp.abs(mdy) < p["monkey_punch_dy"])
    monkey_right  = mdx > 0
    punch_action  = jnp.where(monkey_right, RIGHTFIRE, LEFTFIRE)

    f_threat, c_threat, f_side_sign, fcx = _coconut_threat(obs_flat, p)
    coc_threat = f_threat | c_threat
    sidestep   = jnp.where(f_side_sign > 0, LEFT, RIGHT)

    # ---- Next ladder from upper platform (fixed: uses player_h, not +16) ----
    next_cx, next_top_y, has_next = _next_ladder_from_upper_platform(obs_flat, p, col_top)
    dx_next = next_cx - pcx

    # ---- Dismount direction: prefer next-ladder direction ----
    dismount_toward_next = jnp.where(dx_next > 0, RIGHT, LEFT)
    fcx_safe_dir   = jnp.where((fcx - pcx) > 0, LEFT, RIGHT)
    dismount_default = jnp.where(f_threat, fcx_safe_dir,
                       jnp.where(pcx < 80.0, RIGHT, LEFT))
    dismount_action = jnp.where(has_next, dismount_toward_next, dismount_default)

    # ---- On-ladder coconut escape: move horizontally away from the coconut ----
    on_ladder_coconut_escape = on_col_any & f_threat
    ladder_escape_action = jnp.where(has_next, dismount_toward_next, fcx_safe_dir)

    # ---- Approach child fallback (top platform) ----
    cx_child   = obs_flat[CHILD_X]
    cy_child   = obs_flat[CHILD_Y]
    c_active   = obs_flat[CHILD_ACTIVE] > 0.5
    on_child_tier = c_active & (jnp.abs(cy_child - py) < 16.0)
    child_action  = _horizontal_action(cx_child - pcx, p["x_align_tol"])

    # ---- Post-dismount traversal toward next ladder ----
    post_dismount_traverse = jnp.where(has_next,
                                       _horizontal_action(dx_next, p["x_align_tol"]),
                                       jnp.where(pcx < 80.0, RIGHT, LEFT))

    # ---- Compose action with skill priority ----
    # Base: traverse toward selected reachable ladder
    fallback = jnp.where(any_valid, traverse_action,
               jnp.where(on_child_tier, child_action,
               jnp.where(has_next, post_dismount_traverse, RIGHT)))

    action = fallback

    # Approach reachable ladder horizontally if not yet on it
    action = jnp.where(any_valid & ~on_sel_ladder, traverse_action, action)

    # Climb if on column, not near top
    action = jnp.where(can_climb, UP, action)

    # Dismount at top — suppresses can_climb via near_top_any already
    action = jnp.where(near_top_any, dismount_action, action)

    # On-ladder falling-coconut escape (higher priority than plain dismount)
    action = jnp.where(on_ladder_coconut_escape, ladder_escape_action, action)

    # Child approach when on top tier
    action = jnp.where(on_child_tier & ~can_climb & ~near_top_any, child_action, action)

    # Thrown-coconut sidestep when not on ladder
    action = jnp.where(coc_threat & ~on_col_any, sidestep, action)

    # PRESERVED 200-POINT PUNCH BRANCH — highest priority, no ladder gate
    action = jnp.where(in_punch_range, punch_action, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)


def dense_reward(obs_history, actions, rewards, total_reward, active_mask):
    mask = active_mask.astype(jnp.float32)

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
    lcx  = lx + lw * 0.5
    lby  = ly + lh
    ltop = ly

    px_e    = px[:, None]
    pw_e    = pw[:, None]
    py_e    = py[:, None]
    feet_e  = feet_y[:, None]
    pcx_e   = pcx[:, None]

    inter = jnp.maximum(0.0, jnp.minimum(px_e + pw_e, lx + lw) - jnp.maximum(px_e, lx))
    denom = jnp.maximum(1.0, jnp.minimum(pw_e, lw))
    ov    = inter / denom

    # Reachable ladders: bottom near feet, top above player
    reach_ok  = (jnp.abs(lby - feet_e) < 12.0) & (la > 0.5) & (ltop < py_e - 4.0)
    any_reach = jnp.any(reach_ok, axis=1).astype(jnp.float32)

    # Alignment toward nearest reachable ladder
    dx_to_l = jnp.where(reach_ok, jnp.abs(lcx - pcx_e), 1e4)
    min_dx  = jnp.min(dx_to_l, axis=1)
    align_term = jnp.clip(1.0 - min_dx / 40.0, -1.0, 1.0) * any_reach

    # On-column: overlap ≥ 0.4, ladder top above player head, ladder bottom below head
    above    = ltop < (py_e - 4.0)
    on_col   = jnp.any((ov >= 0.4) & above & (la > 0.5), axis=1).astype(jnp.float32)

    # Upward progress only when on a real column
    dpy          = jnp.concatenate([jnp.zeros((1,)), py[:-1] - py[1:]])
    climb_prog   = jnp.clip(dpy, -2.0, 3.0) * on_col
    climb_term   = jnp.sum(climb_prog * mask)

    # Episode height gain
    py_masked = jnp.where(mask > 0.5, py, 1e6)
    min_py    = jnp.min(py_masked)
    first_py  = py[0]
    height_gain = jnp.clip(first_py - min_py, 0.0, 120.0)
    height_term = height_gain * 2.0

    # Average ladder alignment
    n_active  = jnp.maximum(1.0, jnp.sum(mask))
    align_avg = jnp.sum(align_term * mask) / n_active

    # Action-class counts
    fire_acts = ((actions == FIRE) | (actions == RIGHTFIRE) | (actions == LEFTFIRE) |
                 (actions == UPFIRE) | (actions == DOWNFIRE)).astype(jnp.float32)
    fire_count = jnp.sum(fire_acts * mask)
    up_acts    = ((actions == UP) | (actions == UPRIGHT) | (actions == UPLEFT) |
                  (actions == UPFIRE)).astype(jnp.float32)
    up_count   = jnp.sum(up_acts * mask)

    # Column climb event count
    on_col_count       = jnp.sum(on_col * mask)
    column_climb_bonus = jnp.clip(on_col_count, 0.0, 60.0) * 0.8

    # Dismount event: on_col → off_col while not descending
    on_col_next    = jnp.concatenate([on_col[1:], jnp.zeros((1,))])
    dpy_next       = jnp.concatenate([py[:-1] - py[1:], jnp.zeros((1,))])
    dismount_event = ((on_col > 0.5) & (on_col_next < 0.5) & (dpy_next >= -1.0)).astype(jnp.float32)
    dismount_term  = jnp.clip(jnp.sum(dismount_event * mask), 0.0, 8.0) * 10.0

    # Post-200 transition bonus: scored ≥ 100 AND made meaningful upward progress
    post200_bonus = jnp.where(
        (total_reward >= 100.0) & (height_gain > 15.0),
        40.0, 0.0
    )

    # First-reward preservation bonus
    first_reward_pres = jnp.where(total_reward >= 100.0, 20.0, 0.0)

    # Punch-farming penalty: real reward but NO upward progress and almost no UP actions
    punch_farm = jnp.where(
        (total_reward > 50.0) & (height_gain < 4.0) & (up_count < 3.0),
        -60.0, 0.0
    )

    # No-FIRE scoreless penalty
    no_fire_zero = jnp.where(
        (fire_count < 1.0) & (total_reward < 1.0),
        -15.0, 0.0
    )

    # Idle / no progress penalty
    no_progress_pen = jnp.where(height_gain < 2.0, -5.0, 0.0)

    # Child approach when on top tier
    cy_arr    = obs_history[:, CHILD_Y]
    cx_arr    = obs_history[:, CHILD_X]
    on_top    = (jnp.abs(cy_arr - py) < 18.0).astype(jnp.float32) * mask
    child_close = on_top * jnp.clip(1.0 - jnp.abs(cx_arr - pcx) / 80.0, 0.0, 1.0)
    child_term  = jnp.sum(child_close) * 2.0

    # Falling-coconut survival: reward frames where agent was near a falling coconut
    # but survived (proxy: active_mask still True)
    fc_x    = obs_history[:, FCOC_X]
    fc_y    = obs_history[:, FCOC_Y]
    fc_a    = (obs_history[:, FCOC_A] > 0.5).astype(jnp.float32)
    fc_near = fc_a * (jnp.abs(fc_x - pcx) < 20.0).astype(jnp.float32) * mask
    survival_term = jnp.clip(jnp.sum(fc_near), 0.0, 30.0) * 0.5

    shaped = (
        align_avg          *  6.0
        + climb_term       *  1.0
        + height_term
        + column_climb_bonus
        + dismount_term
        + post200_bonus
        + first_reward_pres
        + child_term
        + survival_term
        + punch_farm
        + no_fire_zero
        + no_progress_pen
    )

    return jnp.clip(shaped, -200.0, 600.0)
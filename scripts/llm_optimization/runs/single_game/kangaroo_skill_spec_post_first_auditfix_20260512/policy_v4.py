"""
Auto-generated policy v4
Generated at: 2026-05-12 10:08:38
"""

"""
Auto-generated policy v3 - conservative post-200 patch
Preserves first-reward route, fixes dismount/next-ladder/coconut-escape.
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
        "x_align_tol":      jnp.array(4.0),
        "monkey_punch_dx":  jnp.array(18.0),
        "monkey_punch_dy":  jnp.array(12.0),
        "coconut_dx":       jnp.array(8.0),
        "min_top_above":    jnp.array(6.0),
        "dismount_tol":     jnp.array(5.0),
    }


def _clip_params(p):
    return {
        "reach_y_tol":     jnp.clip(p["reach_y_tol"],     2.0, 30.0),
        "overlap_frac":    jnp.clip(p["overlap_frac"],     0.25, 0.9),
        "x_align_tol":     jnp.clip(p["x_align_tol"],     1.0, 12.0),
        "monkey_punch_dx": jnp.clip(p["monkey_punch_dx"],  6.0, 40.0),
        "monkey_punch_dy": jnp.clip(p["monkey_punch_dy"],  4.0, 24.0),
        "coconut_dx":      jnp.clip(p["coconut_dx"],       2.0, 20.0),
        "min_top_above":   jnp.clip(p["min_top_above"],    2.0, 20.0),
        "dismount_tol":    jnp.clip(p["dismount_tol"],     2.0, 12.0),
    }


def _x_overlap_frac_arr(px, pw, lx, lw):
    inter = jnp.maximum(0.0, jnp.minimum(px + pw, lx + lw) - jnp.maximum(px, lx))
    denom = jnp.maximum(1.0, jnp.minimum(pw, lw))
    return inter / denom


# ---------------------------------------------------------------------------
# PRESERVED: exact first-reward ladder selector
# ---------------------------------------------------------------------------
def _select_target_ladder(obs, p):
    """Select reachable ladder from current platform: bottom near feet, top above."""
    px = obs[PLAYER_X]; py = obs[PLAYER_Y]; pw = obs[PLAYER_W]; ph = obs[PLAYER_H]
    feet_y = py + ph
    pcx = px + pw * 0.5

    lx = obs[LADDER_X_S:LADDER_X_E]
    ly = obs[LADDER_Y_S:LADDER_Y_E]
    lw = obs[LADDER_W_S:LADDER_W_E]
    lh = obs[LADDER_H_S:LADDER_H_E]
    la = obs[LADDER_A_S:LADDER_A_E]

    lcx = lx + lw * 0.5
    lby = ly + lh
    ltop = ly

    reach_ok = jnp.abs(lby - feet_y) < p["reach_y_tol"]
    above_ok = ltop < (py - p["min_top_above"])
    active_ok = la > 0.5
    valid = reach_ok & above_ok & active_ok

    dx = jnp.abs(lcx - pcx)
    score = jnp.where(valid, -dx, -1e6)
    idx = jnp.argmax(score)
    any_valid = jnp.any(valid)
    return idx, any_valid, lcx, ltop, lby, lx, lw, la


# ---------------------------------------------------------------------------
# PRESERVED: on-any-climbable-ladder detection
# ---------------------------------------------------------------------------
def _on_any_climbable_ladder(obs, p):
    px = obs[PLAYER_X]; py = obs[PLAYER_Y]; pw = obs[PLAYER_W]
    lx = obs[LADDER_X_S:LADDER_X_E]
    ly = obs[LADDER_Y_S:LADDER_Y_E]
    lw = obs[LADDER_W_S:LADDER_W_E]
    lh = obs[LADDER_H_S:LADDER_H_E]
    la = obs[LADDER_A_S:LADDER_A_E]

    ov = _x_overlap_frac_arr(px, pw, lx, lw)
    active = la > 0.5
    has_height = (ly + lh) > py
    valid = active & has_height
    score = jnp.where(valid, ov, -1.0)
    idx = jnp.argmax(score)
    best_ov = ov[idx]
    best_top = ly[idx]
    best_lx = lx[idx]
    best_lw = lw[idx]
    on_col = (best_ov >= p["overlap_frac"]) & valid[idx]
    return on_col, best_ov, best_top, best_lx, best_lw, idx


# ---------------------------------------------------------------------------
# FIXED: next-ladder search uses player feet, not ladder-top offset
# ---------------------------------------------------------------------------
def _next_ladder_from_feet(obs, p):
    """After dismount, find next upward ladder using player's own feet_y
    (mirrors _select_target_ladder logic exactly)."""
    px = obs[PLAYER_X]; py = obs[PLAYER_Y]; pw = obs[PLAYER_W]; ph = obs[PLAYER_H]
    feet_y = py + ph
    pcx = px + pw * 0.5

    lx = obs[LADDER_X_S:LADDER_X_E]
    ly = obs[LADDER_Y_S:LADDER_Y_E]
    lw = obs[LADDER_W_S:LADDER_W_E]
    lh = obs[LADDER_H_S:LADDER_H_E]
    la = obs[LADDER_A_S:LADDER_A_E]

    lcx = lx + lw * 0.5
    lby = ly + lh
    ltop = ly

    reach_ok = jnp.abs(lby - feet_y) < p["reach_y_tol"]
    above_ok = ltop < (py - p["min_top_above"])
    active_ok = la > 0.5
    valid = reach_ok & above_ok & active_ok

    dx = jnp.abs(lcx - pcx)
    score = jnp.where(valid, -dx, -1e6)
    idx = jnp.argmax(score)
    any_valid = jnp.any(valid)
    return lcx[idx], any_valid, lcx, lx[idx], lw[idx]


# ---------------------------------------------------------------------------
# PRESERVED: monkey threat
# ---------------------------------------------------------------------------
def _nearest_monkey(obs):
    px = obs[PLAYER_X]; py = obs[PLAYER_Y]; pw = obs[PLAYER_W]
    mx = obs[MONKEY_X_S:MONKEY_X_E]
    my = obs[MONKEY_Y_S:MONKEY_Y_E]
    ma = obs[MONKEY_A_S:MONKEY_A_E]
    pcx = px + pw * 0.5
    dx = mx - pcx
    dy = my - py
    active = ma > 0.5
    dist = jnp.where(active, jnp.abs(dx) + jnp.abs(dy), 1e6)
    idx = jnp.argmin(dist)
    return dx[idx], dy[idx], active[idx]


# ---------------------------------------------------------------------------
# PRESERVED: coconut threat detection
# ---------------------------------------------------------------------------
def _coconut_threat(obs, p):
    px = obs[PLAYER_X]; py = obs[PLAYER_Y]; pw = obs[PLAYER_W]
    pcx = px + pw * 0.5

    fx = obs[FCOC_X]; fy = obs[FCOC_Y]; fa = obs[FCOC_A] > 0.5
    f_threat = fa & (jnp.abs(fx - pcx) < p["coconut_dx"]) & (fy < py + 30.0) & (fy > py - 60.0)

    cx = obs[COCONUT_X_S:COCONUT_X_E]
    cy = obs[COCONUT_Y_S:COCONUT_Y_E]
    ca = obs[COCONUT_A_S:COCONUT_A_E] > 0.5
    c_threat = jnp.any(ca & (jnp.abs(cx - pcx) < p["coconut_dx"] * 1.5) & (jnp.abs(cy - py) < 14.0))

    f_side_sign = jnp.sign(fx - pcx)
    return f_threat, c_threat, f_side_sign, fx


def _horizontal_action(dx_to_target, tol):
    return jnp.where(dx_to_target > tol, RIGHT,
                     jnp.where(dx_to_target < -tol, LEFT, NOOP))


# ---------------------------------------------------------------------------
# MAIN POLICY
# ---------------------------------------------------------------------------
def policy(obs_flat, params):
    p = _clip_params(params)

    px = obs_flat[PLAYER_X]
    py = obs_flat[PLAYER_Y]
    pw = obs_flat[PLAYER_W]
    ph = obs_flat[PLAYER_H]
    pcx = px + pw * 0.5

    # ---- Reachable ladder selection (preserved 200-point logic) ----
    idx, any_valid, lcx_arr, ltop_arr, lby_arr, lx_arr, lw_arr, la_arr = \
        _select_target_ladder(obs_flat, p)

    tgt_cx = lcx_arr[idx]
    tgt_top = ltop_arr[idx]
    tgt_lx = lx_arr[idx]
    tgt_lw = lw_arr[idx]
    overlap_sel = _x_overlap_frac_arr(px, pw, tgt_lx, tgt_lw)
    on_sel_ladder = (overlap_sel >= p["overlap_frac"]) & any_valid

    dx_to_target = tgt_cx - pcx
    traverse_action = _horizontal_action(dx_to_target, p["x_align_tol"])

    # ---- Independent on-column / near-top detection ----
    on_col_any, _ov_col, col_top, col_lx, col_lw, _col_idx = \
        _on_any_climbable_ladder(obs_flat, p)

    # FIXED near_top: require player to be well above col_top, not just within dismount_tol.
    # Use dismount_tol as the window but require py is genuinely near or above the top.
    near_top_any = on_col_any & (py <= col_top + p["dismount_tol"])

    # Climbing conditions
    can_climb_sel = on_sel_ladder & (tgt_top < py - p["min_top_above"]) & ~near_top_any
    can_climb_col = on_col_any & (col_top < py - p["min_top_above"]) & ~near_top_any
    can_climb = can_climb_sel | can_climb_col

    # ---- Hazards ----
    mdx, mdy, m_active = _nearest_monkey(obs_flat)
    in_punch_range = m_active & (jnp.abs(mdx) < p["monkey_punch_dx"]) & \
                     (jnp.abs(mdy) < p["monkey_punch_dy"])
    monkey_right = mdx > 0
    punch_action = jnp.where(monkey_right, RIGHTFIRE, LEFTFIRE)

    f_threat, c_threat, f_side_sign, fcx = _coconut_threat(obs_flat, p)
    coc_threat = f_threat | c_threat
    # Sidestep away from coconut — move away from its x
    sidestep = jnp.where(f_side_sign > 0, LEFT, RIGHT)

    # ---- FIXED: post-dismount next-ladder search uses player feet ----
    next_cx, has_next, next_lcx_arr, _, _ = _next_ladder_from_feet(obs_flat, p)
    dx_next = next_cx - pcx

    # ---- Dismount direction ----
    cx_child = obs_flat[CHILD_X]
    cy_child = obs_flat[CHILD_Y]
    c_active = obs_flat[CHILD_ACTIVE] > 0.5
    # Prefer toward next ladder; else toward child; else screen center
    dx_to_child = cx_child - pcx
    toward_child = jnp.where(dx_to_child > 0, RIGHT, LEFT)
    toward_center = jnp.where(pcx < 80.0, RIGHT, LEFT)
    dismount_base = jnp.where(has_next,
                               jnp.where(dx_next > p["x_align_tol"], RIGHT,
                                         jnp.where(dx_next < -p["x_align_tol"], LEFT, toward_center)),
                               jnp.where(c_active, toward_child, toward_center))
    # If falling coconut threatens, escape away from it during dismount too
    fcx_escape = jnp.where(f_side_sign > 0, LEFT, RIGHT)
    dismount_action = jnp.where(f_threat, fcx_escape, dismount_base)

    # ---- FIXED: coconut escape applies on-column too (not gated by ~on_col_any) ----
    # When on ladder and falling coconut threatens, escape horizontally
    ladder_coc_escape = on_col_any & f_threat
    ladder_escape_action = fcx_escape

    # ---- Approach child when on same tier ----
    on_child_tier = c_active & (jnp.abs(cy_child - py) < 16.0)
    child_action = _horizontal_action(cx_child - pcx, p["x_align_tol"])

    # ---- Fallback when no reachable ladder found: move toward next or child ----
    no_ladder_fallback = jnp.where(
        has_next,
        _horizontal_action(dx_next, p["x_align_tol"]),
        jnp.where(c_active, toward_child, toward_center)
    )

    # ---- Compose action with skill priority ----
    # Start with base traversal or fallback
    action = jnp.where(any_valid, traverse_action, no_ladder_fallback)

    # On-approach: move toward selected ladder if not yet overlapping
    action = jnp.where(any_valid & ~on_sel_ladder, traverse_action, action)

    # Climb when on a column and not near top
    action = jnp.where(can_climb, UP, action)

    # Near top: dismount
    action = jnp.where(near_top_any, dismount_action, action)

    # On-column coconut escape (FIXED: no ~on_col_any gate)
    action = jnp.where(ladder_coc_escape, ladder_escape_action, action)

    # Child approach when on same tier and not climbing
    action = jnp.where(on_child_tier & ~can_climb & ~near_top_any, child_action, action)

    # Off-column coconut sidestep
    action = jnp.where(coc_threat & ~on_col_any, sidestep, action)

    # PRESERVED 200-POINT PUNCH BRANCH — always last, never gated by ladder state
    action = jnp.where(in_punch_range, punch_action, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)


def dense_reward(obs_history, actions, rewards, total_reward, active_mask):
    mask = active_mask.astype(jnp.float32)
    T = obs_history.shape[0]

    py = obs_history[:, PLAYER_Y]
    px = obs_history[:, PLAYER_X]
    pw = obs_history[:, PLAYER_W]
    ph = obs_history[:, PLAYER_H]
    pcx = px + pw * 0.5
    feet_y = py + ph

    lx = obs_history[:, LADDER_X_S:LADDER_X_E]
    ly = obs_history[:, LADDER_Y_S:LADDER_Y_E]
    lw = obs_history[:, LADDER_W_S:LADDER_W_E]
    lh = obs_history[:, LADDER_H_S:LADDER_H_E]
    la = obs_history[:, LADDER_A_S:LADDER_A_E]
    lcx = lx + lw * 0.5
    lby = ly + lh
    ltop = ly

    px_e = px[:, None]; pw_e = pw[:, None]
    py_e = py[:, None]; feet_e = feet_y[:, None]
    inter = jnp.maximum(0.0, jnp.minimum(px_e + pw_e, lx + lw) - jnp.maximum(px_e, lx))
    denom = jnp.maximum(1.0, jnp.minimum(pw_e, lw))
    ov = inter / denom

    # Reachability: ladder bottom near player feet
    reach_ok = (jnp.abs(lby - feet_e) < 12.0) & (la > 0.5)
    any_reach = jnp.any(reach_ok, axis=1).astype(jnp.float32)

    # Alignment toward nearest reachable ladder
    dx_to_l = jnp.where(reach_ok, jnp.abs(lcx - pcx[:, None]), 1e4)
    min_dx = jnp.min(dx_to_l, axis=1)
    align_term = jnp.clip(1.0 - min_dx / 40.0, -1.0, 1.0) * any_reach

    # On-column: genuine overlap with active ladder whose top is above player head
    above = ltop < (py_e - 4.0)
    on_col = jnp.any((ov >= 0.4) & above & (la > 0.5), axis=1).astype(jnp.float32)

    # Upward progress only when on a real column
    dpy = jnp.concatenate([jnp.zeros((1,)), py[:-1] - py[1:]])
    climb_progress = jnp.clip(dpy, -2.0, 3.0) * on_col
    climb_term = jnp.sum(climb_progress * mask)

    # Episode height gain (bounded): how much lower (higher on screen) did we get?
    py_masked = jnp.where(mask > 0.5, py, 1e6)
    min_py = jnp.min(py_masked)
    first_py = py[0]
    height_gain = jnp.clip(first_py - min_py, 0.0, 120.0)
    height_term = height_gain * 2.0

    # Average alignment reward
    n_active = jnp.maximum(1.0, jnp.sum(mask))
    align_avg = jnp.sum(align_term * mask) / n_active

    # Action-class counts
    fire_acts = (
        (actions == FIRE) | (actions == RIGHTFIRE) | (actions == LEFTFIRE) |
        (actions == UPFIRE) | (actions == DOWNFIRE)
    ).astype(jnp.float32)
    fire_count = jnp.sum(fire_acts * mask)

    up_acts = (
        (actions == UP) | (actions == UPRIGHT) | (actions == UPLEFT) | (actions == UPFIRE)
    ).astype(jnp.float32)
    up_count = jnp.sum(up_acts * mask)

    # On-column step count (clipped)
    on_col_count = jnp.sum(on_col * mask)
    column_climb_bonus = jnp.clip(on_col_count, 0.0, 50.0) * 1.0

    # Punch farming penalty: real reward > 0 but NO upward progress AND almost no UP actions
    punch_farm = jnp.where(
        (total_reward > 50.0) & (height_gain < 6.0) & (up_count < 5.0),
        -80.0, 0.0
    )

    # No-FIRE scoreless penalty
    no_fire_zero = jnp.where(
        (fire_count < 1.0) & (total_reward < 1.0),
        -15.0, 0.0
    )

    # First-reward preservation bonus
    first_reward_pres = jnp.where(total_reward >= 100.0, 20.0, 0.0)

    # Post-200 transition: reward climbing significantly beyond the first reward position.
    # Require both first reward AND meaningful height gain (>20 px).
    post200_bonus = jnp.where(
        (total_reward >= 200.0) & (height_gain > 20.0),
        50.0, 0.0
    )

    # Additional post-200 graded height bonus: reward each pixel beyond y≈132.
    # first_py ≈ 148 at start, so height_gain > 16 means player went above y≈132.
    beyond_first_platform = jnp.clip(height_gain - 16.0, 0.0, 80.0)
    beyond_term = beyond_first_platform * 1.5

    # Dismount event proxy: on_col transitions to off_col while upward progress happened.
    on_col_next = jnp.concatenate([on_col[1:], jnp.zeros((1,))])
    dpy_next = jnp.concatenate([py[:-1] - py[1:], jnp.zeros((1,))])
    dismount_event = (
        (on_col > 0.5) & (on_col_next < 0.5) & (dpy_next >= -2.0)
    ).astype(jnp.float32)
    dismount_term = jnp.clip(jnp.sum(dismount_event * mask), 0.0, 6.0) * 10.0

    # Stall penalty: if player y has not improved beyond the first-platform level
    # and total_reward is exactly 200 (one event, no further progress).
    stall_pen = jnp.where(
        (total_reward >= 200.0) & (height_gain < 6.0),
        -40.0, 0.0
    )

    # Idle (no-progress) penalty
    no_progress_pen = jnp.where(height_gain < 2.0, -5.0, 0.0)

    # Child approach reward when on top tier
    cy_arr = obs_history[:, CHILD_Y]
    cx_arr = obs_history[:, CHILD_X]
    on_top_tier = (jnp.abs(cy_arr - py) < 18.0).astype(jnp.float32) * mask
    child_close = on_top_tier * jnp.clip(1.0 - jnp.abs(cx_arr - pcx) / 80.0, 0.0, 1.0)
    child_term = jnp.sum(child_close) * 2.0

    # UP-action bonus ONLY when genuinely on column (avoids rewarding spurious UP)
    productive_up = jnp.sum(up_acts * on_col * mask)
    productive_up_bonus = jnp.clip(productive_up, 0.0, 30.0) * 0.5

    shaped = (
        align_avg * 5.0
        + climb_term * 1.5
        + height_term
        + column_climb_bonus
        + first_reward_pres
        + post200_bonus
        + beyond_term
        + dismount_term
        + child_term
        + productive_up_bonus
        + punch_farm
        + no_fire_zero
        + no_progress_pen
        + stall_pen
    )

    return jnp.clip(shaped, -200.0, 600.0)
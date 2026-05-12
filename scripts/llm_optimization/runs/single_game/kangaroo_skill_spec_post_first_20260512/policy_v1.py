"""
Auto-generated policy v1
Generated at: 2026-05-10 02:22:49
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
        "reach_y_tol": jnp.array(10.0),
        "overlap_frac": jnp.array(0.45),
        "x_align_tol": jnp.array(4.0),
        "monkey_punch_dx": jnp.array(18.0),
        "monkey_punch_dy": jnp.array(12.0),
        "coconut_dx": jnp.array(8.0),
        "min_top_above": jnp.array(6.0),
        "dismount_tol": jnp.array(5.0),
    }


def _clip_params(p):
    return {
        "reach_y_tol": jnp.clip(p["reach_y_tol"], 2.0, 30.0),
        "overlap_frac": jnp.clip(p["overlap_frac"], 0.25, 0.9),
        "x_align_tol": jnp.clip(p["x_align_tol"], 1.0, 12.0),
        "monkey_punch_dx": jnp.clip(p["monkey_punch_dx"], 6.0, 40.0),
        "monkey_punch_dy": jnp.clip(p["monkey_punch_dy"], 4.0, 24.0),
        "coconut_dx": jnp.clip(p["coconut_dx"], 2.0, 20.0),
        "min_top_above": jnp.clip(p["min_top_above"], 2.0, 20.0),
        "dismount_tol": jnp.clip(p["dismount_tol"], 2.0, 12.0),
    }


def _x_overlap_frac(px, pw, lx, lw):
    inter = jnp.maximum(0.0, jnp.minimum(px + pw, lx + lw) - jnp.maximum(px, lx))
    denom = jnp.maximum(1.0, jnp.minimum(pw, lw))
    return inter / denom


def _select_target_ladder(obs, p):
    px = obs[PLAYER_X]
    py = obs[PLAYER_Y]
    pw = obs[PLAYER_W]
    ph = obs[PLAYER_H]
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


def _nearest_monkey(obs):
    px = obs[PLAYER_X]
    py = obs[PLAYER_Y]
    pw = obs[PLAYER_W]
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


def _coconut_threat(obs, p):
    px = obs[PLAYER_X]
    py = obs[PLAYER_Y]
    pw = obs[PLAYER_W]
    pcx = px + pw * 0.5

    fx = obs[FCOC_X]
    fy = obs[FCOC_Y]
    fa = obs[FCOC_A] > 0.5
    f_threat = fa & (jnp.abs(fx - pcx) < p["coconut_dx"]) & (fy < py + 30.0) & (fy > py - 50.0)

    cx = obs[COCONUT_X_S:COCONUT_X_E]
    cy = obs[COCONUT_Y_S:COCONUT_Y_E]
    ca = obs[COCONUT_A_S:COCONUT_A_E] > 0.5
    c_threat = jnp.any(ca & (jnp.abs(cx - pcx) < p["coconut_dx"] * 1.5) & (jnp.abs(cy - py) < 14.0))

    # sidestep direction: + means coconut is to the right, so move left
    side_sign = jnp.sign(fx - pcx)
    return f_threat, c_threat, side_sign


def _horizontal_action(dx_to_target, tol):
    return jnp.where(dx_to_target > tol, RIGHT,
                     jnp.where(dx_to_target < -tol, LEFT, NOOP))


def policy(obs_flat, params):
    p = _clip_params(params)

    px = obs_flat[PLAYER_X]
    py = obs_flat[PLAYER_Y]
    pw = obs_flat[PLAYER_W]
    ph = obs_flat[PLAYER_H]
    pcx = px + pw * 0.5
    orient = obs_flat[PLAYER_ORIENT]
    facing_right = orient > 180.0  # 270=left, 90=right => >180 means left actually
    # orientation: 90=right, 270=left
    facing_right = jnp.abs(orient - 90.0) < 45.0

    idx, any_valid, lcx_arr, ltop_arr, lby_arr, lx_arr, lw_arr, la_arr = _select_target_ladder(obs_flat, p)

    tgt_cx = lcx_arr[idx]
    tgt_top = ltop_arr[idx]
    tgt_lx = lx_arr[idx]
    tgt_lw = lw_arr[idx]
    overlap = _x_overlap_frac(px, pw, tgt_lx, tgt_lw)
    on_ladder = (overlap >= p["overlap_frac"]) & any_valid
    can_climb = on_ladder & (tgt_top < py - p["min_top_above"])
    near_top = on_ladder & (jnp.abs(py - tgt_top) <= p["dismount_tol"])

    dx_to_target = tgt_cx - pcx

    # Monkey threat
    mdx, mdy, m_active = _nearest_monkey(obs_flat)
    in_punch_range = m_active & (jnp.abs(mdx) < p["monkey_punch_dx"]) & (jnp.abs(mdy) < p["monkey_punch_dy"])
    monkey_right = mdx > 0
    # punch toward monkey
    punch_action = jnp.where(monkey_right, RIGHTFIRE, LEFTFIRE)

    # Coconut threat
    f_threat, c_threat, side_sign = _coconut_threat(obs_flat, p)
    coc_threat = f_threat | c_threat
    # sidestep away: if coconut to the right (side_sign>0), move left
    sidestep = jnp.where(side_sign > 0, LEFT, RIGHT)

    # Approach child fallback
    cx = obs_flat[CHILD_X]
    cy = obs_flat[CHILD_Y]
    c_active = obs_flat[CHILD_ACTIVE] > 0.5
    on_child_tier = c_active & (jnp.abs(cy - py) < 16.0)
    child_dx = cx - pcx
    child_action = _horizontal_action(child_dx, p["x_align_tol"])

    # Traversal action toward ladder
    traverse_action = _horizontal_action(dx_to_target, p["x_align_tol"])

    # Climb action
    climb_action = UP

    # Dismount: step horizontally off ladder. Prefer direction with more room (toward center 80).
    dismount_action = jnp.where(pcx < 80.0, RIGHT, LEFT)

    # Skill priority:
    # 1. coconut hazard -> sidestep
    # 2. monkey in punch range -> punch
    # 3. near top of ladder -> dismount
    # 4. on ladder and can climb -> climb
    # 5. target ladder valid -> traverse
    # 6. on child tier -> approach child
    # 7. else -> fallback search (move right toward likely ladder)

    fallback = jnp.where(any_valid, traverse_action,
                         jnp.where(on_child_tier, child_action, RIGHT))

    action = fallback
    action = jnp.where(any_valid & ~on_ladder, traverse_action, action)
    action = jnp.where(can_climb, climb_action, action)
    action = jnp.where(near_top, dismount_action, action)
    action = jnp.where(in_punch_range, punch_action, action)
    action = jnp.where(coc_threat, sidestep, action)
    action = jnp.where(on_child_tier & ~can_climb, child_action, action)

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

    # Ladder fields
    lx = obs_history[:, LADDER_X_S:LADDER_X_E]
    ly = obs_history[:, LADDER_Y_S:LADDER_Y_E]
    lw = obs_history[:, LADDER_W_S:LADDER_W_E]
    lh = obs_history[:, LADDER_H_S:LADDER_H_E]
    la = obs_history[:, LADDER_A_S:LADDER_A_E]
    lcx = lx + lw * 0.5
    lby = ly + lh
    ltop = ly

    # Per-step: best ladder overlap fraction with any reachable ladder
    px_e = px[:, None]
    pw_e = pw[:, None]
    py_e = py[:, None]
    feet_e = feet_y[:, None]
    inter = jnp.maximum(0.0, jnp.minimum(px_e + pw_e, lx + lw) - jnp.maximum(px_e, lx))
    denom = jnp.maximum(1.0, jnp.minimum(pw_e, lw))
    ov = inter / denom

    reach_ok = (jnp.abs(lby - feet_e) < 12.0) & (la > 0.5)
    any_reach = jnp.any(reach_ok, axis=1).astype(jnp.float32)

    # alignment: min |pcx - lcx| over reachable ladders
    dx_to_l = jnp.where(reach_ok, jnp.abs(lcx - pcx[:, None]), 1e4)
    min_dx = jnp.min(dx_to_l, axis=1)
    align_term = jnp.clip(1.0 - min_dx / 40.0, -1.0, 1.0) * any_reach

    # column overlap with any active ladder where ladder is above
    above = ltop < (py_e - 4.0)
    on_col = jnp.any((ov >= 0.4) & above & (la > 0.5), axis=1).astype(jnp.float32)

    # Upward progress: decrease in py while on column
    dpy = jnp.concatenate([jnp.zeros((1,)), py[:-1] - py[1:]])  # positive when moving up
    climb_progress = jnp.clip(dpy, -2.0, 3.0) * on_col
    climb_term = jnp.sum(climb_progress * mask)  # bounded by ~3 per step but gated by on_col

    # Total upward delta from start (bounded)
    valid_idx = jnp.argmax(mask[::-1])  # not used; instead use first/last
    first_py = py[0]
    # last active py
    last_py = jnp.sum(py * mask) / jnp.maximum(1.0, jnp.sum(mask))  # average proxy
    # better: use min py during episode (highest point reached)
    py_masked = jnp.where(mask > 0.5, py, 1e6)
    min_py = jnp.min(py_masked)
    height_gain = jnp.clip(first_py - min_py, 0.0, 80.0)  # bounded gain in pixels

    # Alignment shaping (bounded average)
    n_active = jnp.maximum(1.0, jnp.sum(mask))
    align_avg = jnp.sum(align_term * mask) / n_active  # in [-1,1]

    # Punch-farming penalty: many FIRE-direction actions but no upward progress
    fire_acts = ((actions == FIRE) | (actions == RIGHTFIRE) | (actions == LEFTFIRE) |
                 (actions == UPFIRE) | (actions == DOWNFIRE)).astype(jnp.float32)
    fire_count = jnp.sum(fire_acts * mask)
    up_acts = ((actions == UP) | (actions == UPRIGHT) | (actions == UPLEFT) |
               (actions == UPFIRE)).astype(jnp.float32)
    up_count = jnp.sum(up_acts * mask)

    # If reward earned but no upward progress and no UP actions -> punch farming
    punch_farm = jnp.where(
        (total_reward > 50.0) & (height_gain < 4.0) & (up_count < 3.0),
        -50.0, 0.0
    )

    # No-FIRE scoreless penalty: if no FIRE used and no reward, slight nudge
    no_fire_zero = jnp.where(
        (fire_count < 1.0) & (total_reward < 1.0),
        -5.0, 0.0
    )

    # First-reward preservation bonus: if total_reward >= 100 and some upward effort, reward
    first_reward_pres = jnp.where(total_reward >= 100.0, 10.0, 0.0)

    # Real upward-progress bonus (large enough to compete)
    height_term = height_gain * 2.0  # up to 160

    # On-column climb event bonus (real ladder use)
    on_col_count = jnp.sum(on_col * mask)
    column_climb_bonus = jnp.clip(on_col_count, 0.0, 30.0) * 1.0  # up to 30

    # Reach the child x while on top tier
    cy_arr = obs_history[:, CHILD_Y]
    cx_arr = obs_history[:, CHILD_X]
    on_top_tier = (jnp.abs(cy_arr - py) < 18.0).astype(jnp.float32) * mask
    child_dx = jnp.abs(cx_arr - pcx)
    child_close = on_top_tier * jnp.clip(1.0 - child_dx / 80.0, 0.0, 1.0)
    child_term = jnp.sum(child_close) * 2.0  # bounded by tier presence

    # Idle-near-same-state penalty: if min_py never improves below start
    no_progress_pen = jnp.where(height_gain < 2.0, -3.0, 0.0)

    shaped = (
        align_avg * 5.0
        + climb_term * 1.0
        + height_term
        + column_climb_bonus
        + first_reward_pres
        + child_term
        + punch_farm
        + no_fire_zero
        + no_progress_pen
    )

    return jnp.clip(shaped, -200.0, 400.0)
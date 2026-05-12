"""
Auto-generated policy v1
Generated at: 2026-05-08 15:56:09
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
RIGHTFIRE, LEFTFIRE = 11, 12


def init_params():
    return {
        "reach_tol": jnp.float32(8.0),     # |ladder_bottom - player_bottom| tolerance
        "align_tol": jnp.float32(4.0),     # x-center alignment tolerance to start climbing
        "climb_top_tol": jnp.float32(6.0), # how close to ladder top before dismount
        "monkey_punch_dx": jnp.float32(18.0),
        "monkey_danger_r": jnp.float32(22.0),
        "coco_danger_r": jnp.float32(16.0),
        "overlap_frac": jnp.float32(0.45), # fraction of player width that must overlap ladder
    }


# ---- skills ----

def _select_reachable_ladder(obs, p_cx, p_by, reach_tol):
    lx = jax.lax.dynamic_slice(obs, (LADDER_X0,), (20,))
    ly = jax.lax.dynamic_slice(obs, (LADDER_Y0,), (20,))
    lw = jax.lax.dynamic_slice(obs, (LADDER_W0,), (20,))
    lh = jax.lax.dynamic_slice(obs, (LADDER_H0,), (20,))
    la = jax.lax.dynamic_slice(obs, (LADDER_ACT0,), (20,))

    lcx = lx + lw * 0.5
    lby = ly + lh
    lty = ly

    # reachable from current platform: ladder bottom near player feet
    reach_ok = (jnp.abs(lby - p_by) < reach_tol) & (la > 0.5) & (lty < p_by - 8.0)
    # cost: prefer reachable, then nearest in x
    big = jnp.float32(1e6)
    cost = jnp.where(reach_ok, jnp.abs(lcx - p_cx), big)
    idx = jnp.argmin(cost)

    has_ladder = jnp.any(reach_ok)
    return idx, has_ladder, lcx, lty, lby, lw


def _nearest_monkey_threat(obs, p_x, p_y):
    mx = jax.lax.dynamic_slice(obs, (MONK_X0,), (4,))
    my = jax.lax.dynamic_slice(obs, (MONK_Y0,), (4,))
    ma = jax.lax.dynamic_slice(obs, (MONK_ACT0,), (4,))
    dx = mx - p_x
    dy = my - p_y
    same_row = jnp.abs(dy) < 16.0
    active = ma > 0.5
    valid = active & same_row
    big = jnp.float32(1e6)
    abs_dx = jnp.where(valid, jnp.abs(dx), big)
    idx = jnp.argmin(abs_dx)
    near_dx = abs_dx[idx]
    near_sx = jnp.sign(dx[idx])
    has = jnp.any(valid)
    return has, near_dx, near_sx


def _coconut_danger(obs, p_x, p_y, r):
    cx = jax.lax.dynamic_slice(obs, (COCO_X0,), (4,))
    cy = jax.lax.dynamic_slice(obs, (COCO_Y0,), (4,))
    ca = jax.lax.dynamic_slice(obs, (COCO_ACT0,), (4,))
    d = jnp.sqrt((cx - p_x) ** 2 + (cy - p_y) ** 2)
    danger = jnp.any((ca > 0.5) & (d < r))

    fx = obs[FCOCO_X]
    fy = obs[FCOCO_Y]
    fa = obs[FCOCO_ACT]
    fd = jnp.sqrt((fx - p_x) ** 2 + (fy - p_y) ** 2)
    fdanger = (fa > 0.5) & (fd < r) & (fy < p_y + 8.0)
    # also flag falling coconut directly above
    above = (fa > 0.5) & (jnp.abs(fx - p_x) < r * 0.7) & (fy < p_y)
    return danger | fdanger | above, jnp.sign(fx - p_x)


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
    danger_r = params["monkey_danger_r"]
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

    # column overlap fraction: how much of player x-extent is inside ladder x-extent
    overlap_left = jnp.maximum(p_x, tgt_cx - tgt_w * 0.5)
    overlap_right = jnp.minimum(p_x + p_w, tgt_cx + tgt_w * 0.5)
    overlap = jnp.maximum(0.0, overlap_right - overlap_left)
    on_column = (overlap / jnp.maximum(p_w, 1.0)) > overlap_frac

    aligned = jnp.abs(dx_to_lad) < align_tol
    on_ladder_band = on_column & (p_by > tgt_top + climb_top_tol) & (p_by <= tgt_bot + 4.0)
    near_top = on_column & (p_by <= tgt_top + climb_top_tol + 6.0)

    # monkey threat / punch
    m_has, m_dx, m_sx = _nearest_monkey_threat(obs, p_x, p_y)
    punch_in_range = m_has & (m_dx < punch_dx)
    monkey_danger = m_has & (m_dx < danger_r)

    # coconut danger
    coco_danger, fc_sx = _coconut_danger(obs, p_x, p_y, coco_r)

    # --- decide ---
    # Default: navigate horizontally toward selected ladder
    move_right = dx_to_lad > 0
    horiz_action = jnp.where(move_right, RIGHT, LEFT)

    # If aligned (or on column) and ladder still useful, climb up
    climb_action = UP

    # If at top of ladder, dismount sideways (toward child if known else right)
    child_x = obs[CHILD_X]
    child_act = obs[CHILD_ACT]
    dismount_right = jnp.where(child_act > 0.5, child_x > p_cx, jnp.bool_(True))
    dismount_action = jnp.where(dismount_right, RIGHT, LEFT)

    # Punch action: face monkey and fire
    punch_action = jnp.where(m_sx >= 0, RIGHTFIRE, LEFTFIRE)

    # Coconut dodge: step away from incoming
    dodge_action = jnp.where(fc_sx >= 0, LEFT, RIGHT)

    # Compose with priority
    # 1) punch monkey if in range (preserve first-route punch)
    # 2) dodge coconut when very close
    # 3) climb if on ladder column and ladder still has height above
    # 4) dismount if near top
    # 5) otherwise navigate horizontally toward chosen ladder
    # 6) if no ladder reachable, move toward center / child x
    fallback_dx = jnp.where(obs[CHILD_ACT] > 0.5, obs[CHILD_X] - p_cx, jnp.float32(1.0))
    fallback_action = jnp.where(fallback_dx > 0, RIGHT, LEFT)

    nav_action = jnp.where(has_ladder, horiz_action, fallback_action)

    action = nav_action
    action = jnp.where((on_ladder_band | aligned) & has_ladder & ~near_top,
                       climb_action, action)
    action = jnp.where(near_top & has_ladder, dismount_action, action)
    action = jnp.where(coco_danger, dodge_action, action)
    action = jnp.where(punch_in_range & ~on_ladder_band, punch_action, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)


def dense_reward(obs_history, actions, rewards, total_reward, active_mask):
    obs = obs_history.astype(jnp.float32)
    mask = active_mask.astype(jnp.float32)
    T = obs.shape[0]

    p_x = obs[:, PLAYER_X]
    p_y = obs[:, PLAYER_Y]
    p_h = obs[:, PLAYER_H]
    p_w = obs[:, PLAYER_W]
    p_by = p_y + p_h
    p_cx = p_x + p_w * 0.5

    # 1) Upward progress: reward decrease in player_y vs initial
    y0 = p_y[0]
    upward = jnp.clip((y0 - p_y) / 100.0, 0.0, 1.5) * mask
    upward_term = jnp.sum(upward) / jnp.maximum(jnp.sum(mask), 1.0)

    # Best (lowest) y reached -> strong bonus
    masked_y = jnp.where(mask > 0.5, p_y, 1e6)
    min_y = jnp.min(masked_y)
    best_up = jnp.clip((y0 - min_y) / 100.0, 0.0, 2.0)

    # 2) Ladder alignment frequency: fraction of steps with some active ladder
    #    whose center_x is within 6 px of player center and ladder is reachable-ish
    lcx = obs[:, LADDER_X0:LADDER_X1] + obs[:, LADDER_W0:LADDER_W1] * 0.5
    lby = obs[:, LADDER_Y0:LADDER_Y1] + obs[:, LADDER_H0:LADDER_H1]
    la = obs[:, LADDER_ACT0:LADDER_ACT1]
    align = (jnp.abs(lcx - p_cx[:, None]) < 6.0) & (la > 0.5)
    reach = (jnp.abs(lby - p_by[:, None]) < 12.0) & align
    aligned_step = jnp.any(reach, axis=1).astype(jnp.float32) * mask
    align_term = jnp.sum(aligned_step) / jnp.maximum(jnp.sum(mask), 1.0)

    # 3) UP-action usage: encourage at least some climbing actions
    is_up = ((actions == UP) | (actions == UPRIGHT) | (actions == UPLEFT)).astype(jnp.float32) * mask
    up_frac = jnp.sum(is_up) / jnp.maximum(jnp.sum(mask), 1.0)
    # Only reward UP when also somewhat aligned, to avoid blind UP
    up_term = jnp.minimum(up_frac, 0.2) * 2.0  # bounded ~0.4

    # 4) Punch-farming penalty: many FIRE actions but no upward progress
    is_fire = ((actions == FIRE) | (actions == RIGHTFIRE) | (actions == LEFTFIRE)).astype(jnp.float32) * mask
    fire_frac = jnp.sum(is_fire) / jnp.maximum(jnp.sum(mask), 1.0)
    no_climb = (best_up < 0.2).astype(jnp.float32)
    punch_farm_pen = fire_frac * no_climb * 1.0  # bounded ~1.0

    # 5) Goal approach: distance from player to child y when child active
    child_y = obs[:, CHILD_Y]
    child_act = obs[:, CHILD_ACT]
    dy_child = jnp.abs(p_y - child_y)
    have_child = (child_act > 0.5).astype(jnp.float32) * mask
    init_dy = jnp.where(have_child[0] > 0.5, dy_child[0], 150.0)
    closeness = jnp.clip((init_dy - dy_child) / 150.0, 0.0, 1.5) * have_child
    goal_term = jnp.sum(closeness) / jnp.maximum(jnp.sum(have_child), 1.0)

    # 6) Idle penalty: many NOOP
    is_noop = (actions == NOOP).astype(jnp.float32) * mask
    noop_frac = jnp.sum(is_noop) / jnp.maximum(jnp.sum(mask), 1.0)
    idle_pen = jnp.minimum(noop_frac, 0.5)

    # 7) Survival
    surv = jnp.sum(mask) / jnp.maximum(jnp.float32(T), 1.0)

    shaped = (
        1.0 * upward_term
        + 1.5 * best_up
        + 0.8 * align_term
        + 1.0 * up_term
        + 1.2 * goal_term
        + 0.3 * surv
        - 1.5 * punch_farm_pen
        - 0.4 * idle_pen
    )
    return shaped
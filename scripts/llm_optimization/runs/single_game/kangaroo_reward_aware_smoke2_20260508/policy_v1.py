"""
Auto-generated policy v1
Generated at: 2026-05-08 16:01:32
"""

import jax
import jax.numpy as jnp


# Observation index aliases
PLAYER_X, PLAYER_Y, PLAYER_W, PLAYER_H = 0, 1, 2, 3
PLAYER_ORIENT = 7

LADDER_X_S, LADDER_X_E = 168, 188
LADDER_Y_S, LADDER_Y_E = 188, 208
LADDER_W_S, LADDER_W_E = 208, 228
LADDER_H_S, LADDER_H_E = 228, 248
LADDER_ACT_S, LADDER_ACT_E = 248, 268

CHILD_X, CHILD_Y, CHILD_ACT = 360, 361, 364
FCOCO_X, FCOCO_Y, FCOCO_ACT = 368, 369, 372

MONKEY_X_S, MONKEY_X_E = 376, 380
MONKEY_Y_S, MONKEY_Y_E = 380, 384
MONKEY_ACT_S, MONKEY_ACT_E = 392, 396

COCO_X_S, COCO_X_E = 408, 412
COCO_Y_S, COCO_Y_E = 412, 416
COCO_ACT_S, COCO_ACT_E = 424, 428

# Actions
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = 6, 7, 8, 9
UPFIRE, RIGHTFIRE, LEFTFIRE = 10, 11, 12


def init_params():
    return {
        "reach_tol": jnp.float32(14.0),     # ladder-bottom reach tolerance
        "align_tol": jnp.float32(4.0),      # ladder x alignment tolerance for climbing
        "top_tol": jnp.float32(8.0),        # how close to ladder top to dismount
        "punch_dx": jnp.float32(18.0),      # punch range
        "danger_r": jnp.float32(22.0),      # generic danger radius
        "coco_dx": jnp.float32(12.0),       # falling coconut dodge x range
    }


def _select_ladder(obs, px_center, p_bot):
    lx = jax.lax.dynamic_slice(obs, (LADDER_X_S,), (20,))
    ly = jax.lax.dynamic_slice(obs, (LADDER_Y_S,), (20,))
    lw = jax.lax.dynamic_slice(obs, (LADDER_W_S,), (20,))
    lh = jax.lax.dynamic_slice(obs, (LADDER_H_S,), (20,))
    la = jax.lax.dynamic_slice(obs, (LADDER_ACT_S,), (20,))

    lcx = lx + lw * 0.5
    lbot = ly + lh
    ltop = ly

    # reachable: bottom of ladder near player feet, and ladder goes up
    bot_dy = jnp.abs(lbot - p_bot)
    upward = ltop < (p_bot - 4.0)
    active = la > 0.5
    reachable = active & upward & (bot_dy < 18.0)

    # score: prefer reachable, then small horizontal distance
    dx = jnp.abs(lcx - px_center)
    big = jnp.float32(1e6)
    score = jnp.where(reachable, dx, big)
    idx = jnp.argmin(score)

    has_target = jnp.min(score) < big * 0.5
    return lcx[idx], ltop[idx], lbot[idx], lw[idx], has_target


def _on_ladder_column(px_center, lcx, lw, align_tol):
    # require the player center to be within ladder's central band
    half = jnp.maximum(lw * 0.5 - 1.0, align_tol)
    return jnp.abs(px_center - lcx) <= half


def _hazard_action(obs, px, py, p_bot, params):
    # Falling coconut directly above
    fcx = obs[FCOCO_X]
    fcy = obs[FCOCO_Y]
    fca = obs[FCOCO_ACT]
    fc_threat = (fca > 0.5) & (jnp.abs(fcx - (px + 8.0)) < params["coco_dx"]) & (fcy < p_bot) & (fcy > py - 40.0)

    # Thrown coconut
    cx = jax.lax.dynamic_slice(obs, (COCO_X_S,), (4,))
    cy = jax.lax.dynamic_slice(obs, (COCO_Y_S,), (4,))
    ca = jax.lax.dynamic_slice(obs, (COCO_ACT_S,), (4,))
    cdx = jnp.abs(cx - (px + 8.0))
    cdy = jnp.abs(cy - (py + 8.0))
    cthreat = jnp.any((ca > 0.5) & (cdx < params["danger_r"]) & (cdy < params["danger_r"]))

    return fc_threat, cthreat


def _nearest_monkey(obs, px, py):
    mx = jax.lax.dynamic_slice(obs, (MONKEY_X_S,), (4,))
    my = jax.lax.dynamic_slice(obs, (MONKEY_Y_S,), (4,))
    ma = jax.lax.dynamic_slice(obs, (MONKEY_ACT_S,), (4,))
    dx = mx - (px + 8.0)
    dy = jnp.abs(my - (py + 8.0))
    big = jnp.float32(1e6)
    dist = jnp.where(ma > 0.5, jnp.abs(dx) + dy, big)
    idx = jnp.argmin(dist)
    return mx[idx], my[idx], ma[idx] > 0.5, dist[idx] < big * 0.5


def policy(obs_flat, params):
    obs = obs_flat
    px = obs[PLAYER_X]
    py = obs[PLAYER_Y]
    pw = obs[PLAYER_W]
    ph = obs[PLAYER_H]
    px_c = px + pw * 0.5
    p_bot = py + ph

    # target ladder
    lcx, ltop, lbot, lw, has_ladder = _select_ladder(obs, px_c, p_bot)

    align_tol = params["align_tol"]
    on_col = _on_ladder_column(px_c, lcx, lw, align_tol) & has_ladder
    near_top = (py - ltop) < params["top_tol"]
    above_bot = py < (lbot - 4.0)
    climbing = on_col & above_bot & ~near_top

    # horizontal direction toward ladder
    dx_lad = lcx - px_c
    go_right_lad = dx_lad > align_tol
    go_left_lad = dx_lad < -align_tol

    # monkey punch
    mx, my, m_active, m_any = _nearest_monkey(obs, px, py)
    mdx = mx - (px + 8.0)
    mdy = jnp.abs(my - (py + 8.0))
    same_row = mdy < 12.0
    in_punch_range = (jnp.abs(mdx) < params["punch_dx"]) & same_row & m_active
    monkey_right = mdx > 0

    # hazards
    fc_threat, cthreat = _hazard_action(obs, px, py, p_bot, params)

    # ---- Action priority ----
    # 1. If climbing column: keep going up (or dismount near top by going horizontal)
    # 2. Hazard dodge (falling coconut overhead)
    # 3. Punch nearby monkey
    # 4. Move toward ladder
    # 5. Default small motion

    # Default: move toward ladder horizontally
    horiz = jnp.where(go_right_lad, RIGHT, jnp.where(go_left_lad, LEFT, NOOP))

    # If aligned but not yet on column due to small offset, nudge
    base_action = horiz

    # Punch branch
    punch_action = jnp.where(monkey_right, RIGHTFIRE, LEFTFIRE)
    base_action = jnp.where(in_punch_range & ~climbing, punch_action, base_action)

    # Climb branch
    climb_action = UP
    base_action = jnp.where(climbing, climb_action, base_action)

    # Near top of ladder: dismount horizontally toward next ladder direction
    # After topping out, lcx will refer to current ladder; encourage stepping off.
    dismount_dir = jnp.where(px_c < 80.0, RIGHT, jnp.where(px_c > 120.0, RIGHT, RIGHT))
    base_action = jnp.where(on_col & near_top, dismount_dir, base_action)

    # Hazard override: falling coconut overhead -> step aside
    dodge = jnp.where((px + 8.0) < obs[FCOCO_X], LEFT, RIGHT)
    base_action = jnp.where(fc_threat & ~climbing, dodge, base_action)

    # Thrown coconut threat: jump (FIRE)
    base_action = jnp.where(cthreat & ~climbing, FIRE, base_action)

    return base_action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)


def dense_reward(obs_history, actions, rewards, total_reward, active_mask):
    mask = active_mask.astype(jnp.float32)
    T = obs_history.shape[0]

    py = obs_history[:, PLAYER_Y]
    px = obs_history[:, PLAYER_X]
    pw = obs_history[:, PLAYER_W]
    ph = obs_history[:, PLAYER_H]
    px_c = px + pw * 0.5
    p_bot = py + ph

    # Upward progress: lower py is better. Reward decreases in py vs initial.
    # Use per-step reduction weighted by mask.
    py0 = py[0]
    upward = jnp.maximum(py0 - py, 0.0) * mask  # positive when above start
    upward_term = jnp.sum(upward) / jnp.maximum(jnp.sum(mask), 1.0)
    # Scale: typical climb ~30-100 px
    upward_reward = jnp.clip(upward_term, 0.0, 120.0) * 1.5  # up to ~180

    # Best (lowest) py reached
    py_masked = jnp.where(mask > 0.5, py, 1e6)
    py_min = jnp.min(py_masked)
    best_climb = jnp.clip(py0 - py_min, 0.0, 120.0)
    best_climb_reward = best_climb * 1.0  # up to 120

    # Ladder alignment progress: count steps where player center near any active ladder column AND moving up
    lcx_all = obs_history[:, LADDER_X_S:LADDER_X_E] + 0.5 * obs_history[:, LADDER_W_S:LADDER_W_E]
    la_all = obs_history[:, LADDER_ACT_S:LADDER_ACT_E]
    ltop_all = obs_history[:, LADDER_Y_S:LADDER_Y_E]
    lbot_all = ltop_all + obs_history[:, LADDER_H_S:LADDER_H_E]

    px_c_b = px_c[:, None]
    p_bot_b = p_bot[:, None]
    py_b = py[:, None]
    aligned = (jnp.abs(lcx_all - px_c_b) < 6.0) & (la_all > 0.5) & (py_b < lbot_all) & (py_b > ltop_all - 4.0)
    aligned_any = jnp.any(aligned, axis=1).astype(jnp.float32) * mask
    align_reward = jnp.sum(aligned_any) * 0.3  # bounded by episode length

    # UP-action effectiveness: UP-action only counts when also aligned to a ladder column
    up_actions = (actions == UP) | (actions == UPFIRE) | (actions == UPRIGHT) | (actions == UPLEFT)
    valid_up = up_actions.astype(jnp.float32) * aligned_any
    up_reward = jnp.sum(valid_up) * 0.5

    # Penalty: punch farming -- many FIRE actions but no upward progress
    fire_actions = (actions == FIRE) | (actions == RIGHTFIRE) | (actions == LEFTFIRE) | (actions == UPFIRE)
    fire_count = jnp.sum(fire_actions.astype(jnp.float32) * mask)
    no_climb = (best_climb < 8.0).astype(jnp.float32)
    punch_farm_penalty = fire_count * no_climb * 0.4

    # Penalty: UP when not aligned to any ladder
    up_not_aligned = up_actions.astype(jnp.float32) * (1.0 - aligned_any) * mask
    bad_up_penalty = jnp.sum(up_not_aligned) * 0.05

    # Survival bonus
    survival = jnp.sum(mask) * 0.02

    # Approach to child (only as small bonus)
    cy = obs_history[:, CHILD_Y]
    cx = obs_history[:, CHILD_X]
    child_dy = jnp.maximum(py - cy, 0.0)
    init_dy = jnp.maximum(py0 - cy[0], 1.0)
    approach = jnp.clip(1.0 - child_dy / init_dy, 0.0, 1.0) * mask
    approach_reward = jnp.sum(approach) * 0.1

    aux = (upward_reward + best_climb_reward + align_reward + up_reward
           + survival + approach_reward
           - punch_farm_penalty - bad_up_penalty)

    return aux
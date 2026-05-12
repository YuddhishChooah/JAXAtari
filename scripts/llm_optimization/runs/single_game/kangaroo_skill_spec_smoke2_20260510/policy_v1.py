"""
Auto-generated policy v1
Generated at: 2026-05-10 02:14:14
"""

import jax
import jax.numpy as jnp


# ---------- Observation indices (440-feature single-frame layout) ----------
PLAYER_X, PLAYER_Y, PLAYER_W, PLAYER_H = 0, 1, 2, 3
PLAYER_ACTIVE = 4
PLAYER_ORIENT = 7

LADDER_X_S, LADDER_X_E = 168, 188
LADDER_Y_S, LADDER_Y_E = 188, 208
LADDER_W_S, LADDER_W_E = 208, 228
LADDER_H_S, LADDER_H_E = 228, 248
LADDER_ACT_S, LADDER_ACT_E = 248, 268

CHILD_X, CHILD_Y, CHILD_ACT = 360, 361, 364
FALLCOCO_X, FALLCOCO_Y, FALLCOCO_ACT = 368, 369, 372

MONKEY_X_S, MONKEY_X_E = 376, 380
MONKEY_Y_S, MONKEY_Y_E = 380, 384
MONKEY_ACT_S, MONKEY_ACT_E = 392, 396

COCO_X_S, COCO_X_E = 408, 412
COCO_Y_S, COCO_Y_E = 412, 416
COCO_ACT_S, COCO_ACT_E = 424, 428

# Actions
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT = 6, 7
RIGHTFIRE, LEFTFIRE = 11, 12
UPFIRE = 10


def init_params():
    return {
        "reach_y_tol": jnp.float32(40.0),
        "climb_x_band": jnp.float32(6.0),
        "min_overlap_frac": jnp.float32(0.5),
        "dismount_margin": jnp.float32(8.0),
        "hazard_radius": jnp.float32(18.0),
        "punch_x_range": jnp.float32(22.0),
        "row_tol": jnp.float32(10.0),
    }


# ---------- Helpers ----------
def _clip_params(params):
    p = {}
    p["reach_y_tol"] = jnp.clip(params["reach_y_tol"], 8.0, 80.0)
    p["climb_x_band"] = jnp.clip(params["climb_x_band"], 2.0, 16.0)
    p["min_overlap_frac"] = jnp.clip(params["min_overlap_frac"], 0.25, 0.9)
    p["dismount_margin"] = jnp.clip(params["dismount_margin"], 2.0, 20.0)
    p["hazard_radius"] = jnp.maximum(params["hazard_radius"], 6.0)
    p["punch_x_range"] = jnp.maximum(params["punch_x_range"], 8.0)
    p["row_tol"] = jnp.maximum(params["row_tol"], 4.0)
    return p


def _ladder_overlap_frac(px, pw, lx, lw):
    inter = jnp.maximum(0.0, jnp.minimum(px + pw, lx + lw) - jnp.maximum(px, lx))
    denom = jnp.maximum(1.0, jnp.minimum(pw, lw))
    return inter / denom


def _select_target_ladder(obs, p):
    px = obs[PLAYER_X]
    py = obs[PLAYER_Y]
    pw = obs[PLAYER_W]
    ph = obs[PLAYER_H]
    pcx = px + 0.5 * pw
    pby = py + ph

    lx = jax.lax.dynamic_slice(obs, (LADDER_X_S,), (20,))
    ly = jax.lax.dynamic_slice(obs, (LADDER_Y_S,), (20,))
    lw = jax.lax.dynamic_slice(obs, (LADDER_W_S,), (20,))
    lh = jax.lax.dynamic_slice(obs, (LADDER_H_S,), (20,))
    la = jax.lax.dynamic_slice(obs, (LADDER_ACT_S,), (20,))

    lcx = lx + 0.5 * lw
    lby = ly + lh
    ltop = ly

    reach_dy = jnp.abs(lby - pby)
    reachable = (la > 0.5) & (reach_dy < p["reach_y_tol"]) & (ltop < py - 1.0)

    dx = jnp.abs(lcx - pcx)
    # cost: prefer reachable, then small horizontal distance, then more upward gain
    big = jnp.float32(1e6)
    cost = jnp.where(reachable, dx - 0.1 * (py - ltop), big)
    idx = jnp.argmin(cost)
    any_reach = jnp.any(reachable)
    return idx, any_reach, lcx, ltop, lby, lx, lw, la


def _nearest_monkey(obs):
    px = obs[PLAYER_X]
    py = obs[PLAYER_Y]
    pw = obs[PLAYER_W]
    ph = obs[PLAYER_H]
    pcx = px + 0.5 * pw
    pcy = py + 0.5 * ph

    mx = jax.lax.dynamic_slice(obs, (MONKEY_X_S,), (4,))
    my = jax.lax.dynamic_slice(obs, (MONKEY_Y_S,), (4,))
    ma = jax.lax.dynamic_slice(obs, (MONKEY_ACT_S,), (4,))

    dx = mx - pcx
    dy = my - pcy
    dist = jnp.sqrt(dx * dx + dy * dy)
    big = jnp.float32(1e6)
    eff = jnp.where(ma > 0.5, dist, big)
    idx = jnp.argmin(eff)
    return mx[idx], my[idx], ma[idx], dx[idx], dy[idx], eff[idx]


def _falling_coco_threat(obs):
    fa = obs[FALLCOCO_ACT]
    fx = obs[FALLCOCO_X]
    fy = obs[FALLCOCO_Y]
    px = obs[PLAYER_X]
    pw = obs[PLAYER_W]
    pcx = px + 0.5 * pw
    py = obs[PLAYER_Y]
    dx = jnp.abs(fx - pcx)
    dy = fy - py
    threat = (fa > 0.5) & (dx < 14.0) & (dy < 50.0) & (dy > -10.0)
    return threat, fx - pcx


def _thrown_coco_threat(obs):
    px = obs[PLAYER_X]
    py = obs[PLAYER_Y]
    pw = obs[PLAYER_W]
    ph = obs[PLAYER_H]
    pcx = px + 0.5 * pw
    pcy = py + 0.5 * ph
    cx = jax.lax.dynamic_slice(obs, (COCO_X_S,), (4,))
    cy = jax.lax.dynamic_slice(obs, (COCO_Y_S,), (4,))
    ca = jax.lax.dynamic_slice(obs, (COCO_ACT_S,), (4,))
    dx = jnp.abs(cx - pcx)
    dy = jnp.abs(cy - pcy)
    near = (ca > 0.5) & (dx < 16.0) & (dy < 12.0)
    return jnp.any(near)


# ---------- Policy ----------
def policy(obs_flat, params):
    p = _clip_params(params)

    px = obs_flat[PLAYER_X]
    py = obs_flat[PLAYER_Y]
    pw = obs_flat[PLAYER_W]
    ph = obs_flat[PLAYER_H]
    pcx = px + 0.5 * pw

    idx, any_reach, lcx, ltop, lby, lx_arr, lw_arr, la_arr = _select_target_ladder(
        obs_flat, p
    )
    tgt_cx = lcx[idx]
    tgt_top = ltop[idx]
    tgt_lx = lx_arr[idx]
    tgt_lw = lw_arr[idx]

    overlap = _ladder_overlap_frac(px, pw, tgt_lx, tgt_lw)
    in_column = (overlap >= p["min_overlap_frac"]) & any_reach
    near_top = py <= (tgt_top + p["dismount_margin"])

    dx_to_ladder = tgt_cx - pcx

    # Hazards
    mx, my, ma, mdx, mdy, mdist = _nearest_monkey(obs_flat)
    monkey_close = (ma > 0.5) & (mdist < p["hazard_radius"] * 2.0)
    monkey_punchable = (
        (ma > 0.5)
        & (jnp.abs(mdx) < p["punch_x_range"])
        & (jnp.abs(mdy) < p["row_tol"])
    )
    monkey_right = mdx > 0.0

    fall_threat, fall_dx = _falling_coco_threat(obs_flat)
    thrown_threat = _thrown_coco_threat(obs_flat)

    # ---- Action selection ----
    # Default: select & traverse to ladder
    move_right = dx_to_ladder > p["climb_x_band"]
    move_left = dx_to_ladder < -p["climb_x_band"]
    horiz_action = jnp.where(
        move_right, RIGHT, jnp.where(move_left, LEFT, NOOP)
    )

    # If not reachable target, fallback move right toward middle
    fallback_action = jnp.where(pcx < 80.0, RIGHT, jnp.where(pcx > 80.0, LEFT, NOOP))
    horiz_action = jnp.where(any_reach, horiz_action, fallback_action)

    # Climb when in column and not at top
    climb_action = jnp.where(near_top, NOOP, UP)
    # Dismount: at top, move horizontally off the ladder
    dismount_action = jnp.where(pcx < 80.0, RIGHT, LEFT)
    in_column_action = jnp.where(near_top, dismount_action, climb_action)

    # Hazard handling
    # Punch nearest monkey if punchable (preserves first-route stable-200 RIGHTFIRE)
    punch_action = jnp.where(monkey_right, RIGHTFIRE, LEFTFIRE)

    # Falling coconut: sidestep
    dodge_action = jnp.where(fall_dx > 0.0, LEFT, RIGHT)

    # Compose: priority order
    # base = select/traverse or climb
    base_action = jnp.where(in_column, in_column_action, horiz_action)

    # post: if thrown coconut very close, try jump (FIRE)
    base_action = jnp.where(thrown_threat, FIRE, base_action)
    # falling coconut: sidestep
    base_action = jnp.where(fall_threat, dodge_action, base_action)
    # punch monkey if aligned and in punch range (highest priority for first-route preservation)
    action = jnp.where(monkey_punchable, punch_action, base_action)

    return action.astype(jnp.int32)


# ---------- Measurement ----------
def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)


# ---------- Dense reward ----------
def dense_reward(obs_history, actions, rewards, total_reward, active_mask):
    mask = active_mask.astype(jnp.float32)
    T = obs_history.shape[0]

    py = obs_history[:, PLAYER_Y]
    ph = obs_history[:, PLAYER_H]
    px = obs_history[:, PLAYER_X]
    pw = obs_history[:, PLAYER_W]
    pact = obs_history[:, PLAYER_ACTIVE]

    # --- Upward progress: reward decrease in player_y vs episode start ---
    # Use first active y as baseline.
    first_y = py[0]
    y_drop = jnp.clip(first_y - py, 0.0, 120.0)
    upward_progress = jnp.max(y_drop * mask)  # event-style, bounded ~120

    # --- Ladder-column climb signal: UP action while overlap >= 0.5 with any reachable ladder ---
    lx = obs_history[:, LADDER_X_S:LADDER_X_E]
    ly = obs_history[:, LADDER_Y_S:LADDER_Y_E]
    lw = obs_history[:, LADDER_W_S:LADDER_W_E]
    lh = obs_history[:, LADDER_H_S:LADDER_H_E]
    la = obs_history[:, LADDER_ACT_S:LADDER_ACT_E]

    pby = py + ph
    lby = ly + lh
    # broadcasting: (T,1) vs (T,20)
    px_e = px[:, None]
    pw_e = pw[:, None]
    inter = jnp.maximum(
        0.0, jnp.minimum(px_e + pw_e, lx + lw) - jnp.maximum(px_e, lx)
    )
    denom = jnp.maximum(1.0, jnp.minimum(pw_e, lw))
    overlap = inter / denom

    reach_dy = jnp.abs(lby - pby[:, None])
    upward = ly < (py[:, None] - 1.0)
    reachable = (la > 0.5) & (reach_dy < 40.0) & upward
    in_any_real_column = jnp.any((overlap >= 0.5) & reachable, axis=1)  # (T,)

    is_up = (
        (actions == UP) | (actions == UPRIGHT) | (actions == UPLEFT) | (actions == UPFIRE)
    )
    valid_climb = (in_any_real_column & is_up).astype(jnp.float32) * mask
    climb_term = jnp.clip(jnp.sum(valid_climb), 0.0, 60.0)

    # Penalize UP when NOT in any real ladder column
    bad_up = (is_up & (~in_any_real_column)).astype(jnp.float32) * mask
    bad_up_pen = jnp.clip(jnp.sum(bad_up), 0.0, 40.0)

    # --- First-route preservation: reward presence of FIRE in early episode ---
    is_fire_act = (
        (actions == FIRE)
        | (actions == RIGHTFIRE)
        | (actions == LEFTFIRE)
        | (actions == UPFIRE)
    )
    early = jnp.arange(T) < 200
    early_fire = (is_fire_act & early).astype(jnp.float32) * mask
    fire_used_early = jnp.clip(jnp.sum(early_fire), 0.0, 1.0) * 20.0  # one-shot bonus

    # Real reward event preservation: any positive reward in the episode
    got_real = (jnp.sum(rewards * mask) > 0.0).astype(jnp.float32) * 40.0

    # --- Anti-punch-farming: if there is real reward but no UP action and no upward progress, penalize ---
    any_up = (jnp.sum(valid_climb) > 0.0).astype(jnp.float32)
    no_climb_no_progress = (
        ((jnp.sum(rewards * mask) > 0.0) & (upward_progress < 4.0) & (any_up < 0.5))
        .astype(jnp.float32)
        * 30.0
    )

    # --- Survival term: fraction of steps with player_active==1, capped ---
    survival = jnp.sum((pact > 0.5).astype(jnp.float32) * mask) / jnp.maximum(
        1.0, jnp.sum(mask)
    )
    survival_term = jnp.clip(survival, 0.0, 1.0) * 10.0

    # --- Approach term: decrease in horizontal distance to nearest reachable ladder center ---
    lcx = lx + 0.5 * lw
    pcx = (px + 0.5 * pw)[:, None]
    dx_abs = jnp.abs(lcx - pcx)
    big = jnp.float32(1e6)
    dx_eff = jnp.where(reachable, dx_abs, big)
    min_dx = jnp.min(dx_eff, axis=1)
    # event: best (min) horizontal distance achieved across episode
    valid_min = jnp.where(mask > 0.5, min_dx, big)
    best_dx = jnp.min(valid_min)
    approach_term = jnp.clip(80.0 - best_dx, 0.0, 80.0) * 0.25  # up to 20

    shaped = (
        upward_progress * 1.5      # up to ~180
        + climb_term               # up to 60
        + fire_used_early          # up to 20
        + got_real                 # up to 40
        + survival_term            # up to 10
        + approach_term            # up to 20
        - bad_up_pen               # up to -40
        - no_climb_no_progress     # up to -30
    )
    return shaped
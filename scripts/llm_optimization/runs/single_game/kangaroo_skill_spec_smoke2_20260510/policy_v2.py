"""
Auto-generated policy v2
Generated at: 2026-05-10 02:16:58
"""

import jax
import jax.numpy as jnp


# ---------- Observation indices ----------
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
DOWNRIGHT, DOWNLEFT = 8, 9
UPFIRE, RIGHTFIRE, LEFTFIRE = 10, 11, 12


def init_params():
    return {
        "reach_y_tol": jnp.float32(40.0),
        "climb_x_band": jnp.float32(6.0),
        "enter_overlap": jnp.float32(0.4),
        "dismount_margin": jnp.float32(8.0),
        "hazard_radius": jnp.float32(18.0),
        "punch_x_range": jnp.float32(22.0),
        "row_tol": jnp.float32(10.0),
    }


def _clip_params(params):
    p = {}
    p["reach_y_tol"] = jnp.clip(params["reach_y_tol"], 8.0, 80.0)
    p["climb_x_band"] = jnp.clip(params["climb_x_band"], 2.0, 16.0)
    # Cap entry overlap below 0.6 so CMA-ES cannot make climbing impossible.
    p["enter_overlap"] = jnp.clip(params["enter_overlap"], 0.2, 0.6)
    p["dismount_margin"] = jnp.clip(params["dismount_margin"], 2.0, 20.0)
    p["hazard_radius"] = jnp.maximum(params["hazard_radius"], 6.0)
    p["punch_x_range"] = jnp.maximum(params["punch_x_range"], 8.0)
    p["row_tol"] = jnp.maximum(params["row_tol"], 4.0)
    return p


# ---------- Perception helpers ----------
def _ladder_overlap_frac(px, pw, lx, lw):
    inter = jnp.maximum(0.0, jnp.minimum(px + pw, lx + lw) - jnp.maximum(px, lx))
    denom = jnp.maximum(1.0, jnp.minimum(pw, lw))
    return inter / denom


def _select_reachable_ladder(obs, p):
    """Pick reachable ladder: bottom near player_bottom AND top above player."""
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
    big = jnp.float32(1e6)
    cost = jnp.where(reachable, dx, big)
    idx = jnp.argmin(cost)
    any_reach = jnp.any(reachable)
    return idx, any_reach, lcx, ltop, lx, lw, la


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
    any_monkey = jnp.any(ma > 0.5)
    return dx[idx], dy[idx], eff[idx], any_monkey


def _falling_coco_threat(obs, p):
    fa = obs[FALLCOCO_ACT]
    fx = obs[FALLCOCO_X]
    fy = obs[FALLCOCO_Y]
    px = obs[PLAYER_X]
    pw = obs[PLAYER_W]
    py = obs[PLAYER_Y]
    pcx = px + 0.5 * pw
    dx = jnp.abs(fx - pcx)
    dy = fy - py
    threat = (fa > 0.5) & (dx < p["hazard_radius"]) & (dy < 60.0) & (dy > -20.0)
    sign = fx - pcx
    return threat, sign


def _thrown_coco_threat(obs, p):
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
    near = (ca > 0.5) & (dx < p["hazard_radius"]) & (dy < 14.0)
    return jnp.any(near)


# ---------- Policy ----------
def policy(obs_flat, params):
    p = _clip_params(params)

    px = obs_flat[PLAYER_X]
    py = obs_flat[PLAYER_Y]
    pw = obs_flat[PLAYER_W]
    ph = obs_flat[PLAYER_H]
    pcx = px + 0.5 * pw

    idx, any_reach, lcx_arr, ltop_arr, lx_arr, lw_arr, la_arr = _select_reachable_ladder(
        obs_flat, p
    )
    tgt_cx = lcx_arr[idx]
    tgt_top = ltop_arr[idx]
    tgt_lx = lx_arr[idx]
    tgt_lw = lw_arr[idx]

    overlap = _ladder_overlap_frac(px, pw, tgt_lx, tgt_lw)
    in_column = (overlap >= p["enter_overlap"]) & any_reach
    near_top = py <= (tgt_top + p["dismount_margin"])

    dx_to_ladder = tgt_cx - pcx

    # Hazards
    mdx, mdy, mdist, any_monkey = _nearest_monkey(obs_flat)
    monkey_punchable = (
        any_monkey
        & (jnp.abs(mdx) < p["punch_x_range"])
        & (jnp.abs(mdy) < p["row_tol"])
    )
    monkey_right = mdx > 0.0
    punch_action = jnp.where(monkey_right, RIGHTFIRE, LEFTFIRE)

    fall_threat, fall_sign = _falling_coco_threat(obs_flat, p)
    thrown_threat = _thrown_coco_threat(obs_flat, p)

    # ---- Horizontal traverse ----
    move_right = dx_to_ladder > p["climb_x_band"]
    move_left = dx_to_ladder < -p["climb_x_band"]
    # If inside band but not enough overlap, nudge toward exact center (no NOOP freeze).
    nudge = jnp.where(dx_to_ladder >= 0.0, RIGHT, LEFT)
    horiz_action = jnp.where(
        move_right, RIGHT, jnp.where(move_left, LEFT, nudge)
    )
    # If no reachable ladder selected, drift toward x≈132 (known first ladder x).
    fallback_action = jnp.where(pcx < 132.0, RIGHT, LEFT)
    horiz_action = jnp.where(any_reach, horiz_action, fallback_action)

    # ---- Climb / dismount ----
    # Dismount direction: pick side based on whether x≈20 or x≈132 ladder is to either side
    # Heuristic: continue in player_x direction toward screen middle/right; if pcx>100 go left to find next.
    dismount_action = jnp.where(pcx < 76.0, RIGHT, LEFT)
    climb_or_dismount = jnp.where(near_top, dismount_action, UP)

    # ---- Compose base ----
    base_action = jnp.where(in_column, climb_or_dismount, horiz_action)

    # Falling coconut: if not in column, sidestep; if in column, climb harder (UPFIRE) to escape upward.
    coco_dodge = jnp.where(fall_sign > 0.0, LEFT, RIGHT)
    coco_response = jnp.where(in_column & ~near_top, UPFIRE, coco_dodge)
    base_action = jnp.where(fall_threat, coco_response, base_action)

    # Thrown coconut close: jump
    base_action = jnp.where(thrown_threat, FIRE, base_action)

    # ---- Punch branch (preserved exactly): highest priority ----
    punch_in_range = any_monkey & (jnp.abs(mdx) < p["punch_x_range"]) & (
        jnp.abs(mdy) < p["row_tol"]
    )
    action = jnp.where(punch_in_range, punch_action, base_action)

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

    # Upward progress: reward decrease in player_y vs episode start (event-style).
    first_y = py[0]
    y_drop = jnp.clip(first_y - py, 0.0, 120.0) * mask
    upward_progress = jnp.max(y_drop)

    # Ladder column overlap with reachable ladders
    lx = obs_history[:, LADDER_X_S:LADDER_X_E]
    ly = obs_history[:, LADDER_Y_S:LADDER_Y_E]
    lw = obs_history[:, LADDER_W_S:LADDER_W_E]
    lh = obs_history[:, LADDER_H_S:LADDER_H_E]
    la = obs_history[:, LADDER_ACT_S:LADDER_ACT_E]

    pby = py + ph
    lby = ly + lh
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
    in_real_column = jnp.any((overlap >= 0.3) & reachable, axis=1)
    in_any_overlap = jnp.any(overlap >= 0.2, axis=1)

    # Climb term: UP-family while in real column
    is_up = (
        (actions == UP) | (actions == UPRIGHT) | (actions == UPLEFT) | (actions == UPFIRE)
    )
    valid_climb = (in_real_column & is_up).astype(jnp.float32) * mask
    climb_term = jnp.clip(jnp.sum(valid_climb), 0.0, 80.0)

    # Penalize UP with no overlap with any ladder at all
    bad_up = (is_up & (~in_any_overlap)).astype(jnp.float32) * mask
    bad_up_pen = jnp.clip(jnp.sum(bad_up), 0.0, 30.0)

    # Real reward event
    got_real = (jnp.sum(rewards * mask) > 0.0).astype(jnp.float32)

    # Approach term: best minimum horizontal distance to any reachable ladder
    lcx = lx + 0.5 * lw
    pcx = (px + 0.5 * pw)[:, None]
    dx_abs = jnp.abs(lcx - pcx)
    big = jnp.float32(1e6)
    dx_eff = jnp.where(reachable, dx_abs, big)
    min_dx = jnp.min(dx_eff, axis=1)
    valid_min = jnp.where(mask > 0.5, min_dx, big)
    best_dx = jnp.min(valid_min)
    approach_term = jnp.clip(60.0 - best_dx, 0.0, 60.0) * 0.15  # up to ~9

    # Ladder-top one-shot bonus: did min_player_y ever drop below any reachable ladder top + margin?
    masked_py = jnp.where(mask > 0.5, py, jnp.float32(1e6))
    min_py = jnp.min(masked_py)
    # Use per-step reachable ladder tops; one-shot if min_py beats any active ladder's top.
    ltop_active = jnp.where(la > 0.5, ly, jnp.float32(1e6))
    best_ltop = jnp.min(ltop_active)  # smallest (highest) ladder top across episode
    crossed_a_top = (min_py < (best_ltop + 12.0)).astype(jnp.float32) * 50.0

    # Survival ratio
    survival = jnp.sum((pact > 0.5).astype(jnp.float32) * mask) / jnp.maximum(
        1.0, jnp.sum(mask)
    )
    survival_term = jnp.clip(survival, 0.0, 1.0) * 6.0

    # Gate "got_real" reward: only valuable if there is also some upward progress OR climb attempt.
    progress_gate = jnp.clip(upward_progress / 8.0, 0.0, 1.0)
    climb_gate = jnp.clip(jnp.sum(valid_climb) / 4.0, 0.0, 1.0)
    gate = jnp.maximum(progress_gate, climb_gate)
    got_real_term = got_real * (10.0 + 30.0 * gate)  # 10 baseline (preserve), up to 40 with climb

    # Post-200 stagnation penalty: if got real reward but y_range==0
    masked_py2 = jnp.where(mask > 0.5, py, py[0])
    y_range = jnp.max(masked_py2) - jnp.min(masked_py2)
    stagnation_pen = (
        ((got_real > 0.5) & (y_range < 4.0)).astype(jnp.float32) * 25.0
    )

    shaped = (
        upward_progress * 2.0     # up to ~240, dominant
        + climb_term              # up to 80
        + crossed_a_top           # up to 50
        + got_real_term           # up to 40 (preserves first-reward incentive)
        + approach_term           # up to ~9
        + survival_term           # up to 6
        - bad_up_pen              # up to -30
        - stagnation_pen          # up to -25
    )
    return shaped
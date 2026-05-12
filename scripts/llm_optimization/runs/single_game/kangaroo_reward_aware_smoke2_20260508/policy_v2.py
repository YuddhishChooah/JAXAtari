"""
Auto-generated policy v2
Generated at: 2026-05-08 16:04:09
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
        "reach_tol": jnp.float32(20.0),     # ladder bottom-dy reach window
        "align_tol": jnp.float32(5.0),      # ladder x-alignment band
        "top_tol": jnp.float32(8.0),        # near-top dismount window
        "punch_dx": jnp.float32(20.0),      # punch x range
        "punch_dy": jnp.float32(18.0),      # punch y tolerance
        "danger_r": jnp.float32(22.0),      # thrown-coconut danger radius
        "coco_dx": jnp.float32(12.0),       # falling coconut x dodge range
    }


def _select_reachable_ladder(obs, px_c, p_bot, reach_tol):
    lx = jax.lax.dynamic_slice(obs, (LADDER_X_S,), (20,))
    ly = jax.lax.dynamic_slice(obs, (LADDER_Y_S,), (20,))
    lw = jax.lax.dynamic_slice(obs, (LADDER_W_S,), (20,))
    lh = jax.lax.dynamic_slice(obs, (LADDER_H_S,), (20,))
    la = jax.lax.dynamic_slice(obs, (LADDER_ACT_S,), (20,))

    lcx = lx + lw * 0.5
    lbot = ly + lh
    ltop = ly

    bot_dy = jnp.abs(lbot - p_bot)
    upward = ltop < (p_bot - 4.0)
    active = la > 0.5
    reachable = active & upward & (bot_dy < reach_tol)

    dx = jnp.abs(lcx - px_c)
    big = jnp.float32(1e6)
    score = jnp.where(reachable, dx, big)
    idx = jnp.argmin(score)
    has = jnp.min(score) < big * 0.5

    # fallback: nearest active upward ladder by dx (so navigation never blanks)
    score_fb = jnp.where(active & upward, dx, big)
    idx_fb = jnp.argmin(score_fb)
    has_fb = jnp.min(score_fb) < big * 0.5

    use_idx = jnp.where(has, idx, idx_fb)
    use_has = has | has_fb

    return lcx[use_idx], ltop[use_idx], lbot[use_idx], lw[use_idx], has, use_has, lcx[idx_fb]


def _player_inside_ladder_span(py, p_bot, ltop, lbot):
    # player vertically inside the ladder rectangle (allows climbing without
    # depending solely on the starting-platform reach test)
    return (p_bot > ltop + 2.0) & (py < lbot - 2.0)


def _on_ladder_column(px_c, lcx, lw, align_tol):
    half = jnp.maximum(lw * 0.5, align_tol)
    return jnp.abs(px_c - lcx) <= half


def _nearest_monkey(obs, px_c, py_c):
    mx = jax.lax.dynamic_slice(obs, (MONKEY_X_S,), (4,))
    my = jax.lax.dynamic_slice(obs, (MONKEY_Y_S,), (4,))
    ma = jax.lax.dynamic_slice(obs, (MONKEY_ACT_S,), (4,))
    dx = mx - px_c
    dy = my - py_c
    big = jnp.float32(1e6)
    dist = jnp.where(ma > 0.5, jnp.abs(dx) + jnp.abs(dy), big)
    idx = jnp.argmin(dist)
    return mx[idx], my[idx], ma[idx] > 0.5


def _hazards(obs, px_c, py, p_bot, params):
    fcx = obs[FCOCO_X]
    fcy = obs[FCOCO_Y]
    fca = obs[FCOCO_ACT]
    fc_threat = (fca > 0.5) & (jnp.abs(fcx - px_c) < params["coco_dx"]) & (fcy < p_bot) & (fcy > py - 50.0)

    cx = jax.lax.dynamic_slice(obs, (COCO_X_S,), (4,))
    cy = jax.lax.dynamic_slice(obs, (COCO_Y_S,), (4,))
    ca = jax.lax.dynamic_slice(obs, (COCO_ACT_S,), (4,))
    cdx = jnp.abs(cx - px_c)
    cdy = jnp.abs(cy - (py + 8.0))
    cthreat = jnp.any((ca > 0.5) & (cdx < params["danger_r"]) & (cdy < params["danger_r"]))
    return fc_threat, cthreat, fcx


def _move_toward_x(dx, align_tol):
    return jnp.where(dx > align_tol, RIGHT, jnp.where(dx < -align_tol, LEFT, NOOP))


def policy(obs_flat, params):
    obs = obs_flat
    px = obs[PLAYER_X]
    py = obs[PLAYER_Y]
    pw = obs[PLAYER_W]
    ph = obs[PLAYER_H]
    px_c = px + pw * 0.5
    py_c = py + ph * 0.5
    p_bot = py + ph

    align_tol = params["align_tol"]
    reach_tol = params["reach_tol"]

    # Primary ladder = reachable from current platform; fallback = best active.
    lcx, ltop, lbot, lw, has_reach, has_any, lcx_fb = _select_reachable_ladder(
        obs, px_c, p_bot, reach_tol
    )

    inside_span = _player_inside_ladder_span(py, p_bot, ltop, lbot)
    on_col = _on_ladder_column(px_c, lcx, lw, align_tol)
    near_top = (py - ltop) < params["top_tol"]

    # Climb when on the ladder column AND vertically inside the ladder span.
    climbing = on_col & inside_span & ~near_top & has_any

    # Approach direction: toward the reachable ladder if any, else fallback.
    target_x = jnp.where(has_reach, lcx, lcx_fb)
    dx_lad = target_x - px_c
    horiz = _move_toward_x(dx_lad, align_tol)

    # Monkey punch (kept independent from climb/column gates per rules 13-14).
    mx, my, m_active = _nearest_monkey(obs, px_c, py_c)
    mdx = mx - px_c
    mdy = jnp.abs(my - py_c)
    in_punch_range = m_active & (jnp.abs(mdx) < params["punch_dx"]) & (mdy < params["punch_dy"])
    punch_action = jnp.where(mdx >= 0.0, RIGHTFIRE, LEFTFIRE)

    # Hazards
    fc_threat, cthreat, fcx = _hazards(obs, px_c, py, p_bot, params)
    dodge_dir = jnp.where(px_c < fcx, LEFT, RIGHT)

    # ----- Action priority -----
    # default: horizontal toward target ladder
    action = horiz

    # punch overrides horizontal traversal (preserves first-route RIGHTFIRE)
    action = jnp.where(in_punch_range & ~climbing, punch_action, action)

    # climb overrides horizontal/punch when truly on column
    action = jnp.where(climbing, UP, action)

    # near top of current ladder: dismount toward next reachable ladder.
    # Re-evaluate ladder selection from a hypothetical position just above top.
    # We approximate the post-dismount platform by using ltop as the new p_bot.
    next_lcx, _, _, _, has_next_reach, has_next_any, next_lcx_fb = _select_reachable_ladder(
        obs, px_c, ltop + 2.0, reach_tol
    )
    next_target = jnp.where(has_next_reach, next_lcx, next_lcx_fb)
    dismount_dir = jnp.where(next_target > px_c, RIGHT, LEFT)
    action = jnp.where(on_col & near_top & has_any, dismount_dir, action)

    # hazard dodges (don't break a committed climb)
    action = jnp.where(fc_threat & ~climbing, dodge_dir, action)
    action = jnp.where(cthreat & ~climbing, FIRE, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)


def dense_reward(obs_history, actions, rewards, total_reward, active_mask):
    mask = active_mask.astype(jnp.float32)
    msum = jnp.maximum(jnp.sum(mask), 1.0)

    py = obs_history[:, PLAYER_Y]
    px = obs_history[:, PLAYER_X]
    pw = obs_history[:, PLAYER_W]
    ph = obs_history[:, PLAYER_H]
    px_c = px + pw * 0.5
    p_bot = py + ph

    py0 = py[0]

    # ---- ladder alignment (vector of [T,20]) ----
    lcx_all = obs_history[:, LADDER_X_S:LADDER_X_E] + 0.5 * obs_history[:, LADDER_W_S:LADDER_W_E]
    lw_all = obs_history[:, LADDER_W_S:LADDER_W_E]
    la_all = obs_history[:, LADDER_ACT_S:LADDER_ACT_E]
    ltop_all = obs_history[:, LADDER_Y_S:LADDER_Y_E]
    lbot_all = ltop_all + obs_history[:, LADDER_H_S:LADDER_H_E]

    px_c_b = px_c[:, None]
    py_b = py[:, None]
    p_bot_b = p_bot[:, None]

    half = jnp.maximum(lw_all * 0.5, 4.0)
    on_col = (jnp.abs(lcx_all - px_c_b) <= half) & (la_all > 0.5)
    inside_span = (p_bot_b > ltop_all + 2.0) & (py_b < lbot_all - 2.0)
    truly_on_ladder = on_col & inside_span
    aligned_any = jnp.any(truly_on_ladder, axis=1).astype(jnp.float32) * mask

    # ---- upward / best-y progress (real geometric progress) ----
    py_masked = jnp.where(mask > 0.5, py, 1e6)
    py_min = jnp.min(py_masked)
    best_climb = jnp.clip(py0 - py_min, 0.0, 140.0)  # bounded
    best_climb_reward = best_climb * 1.0  # up to 140

    # delta progress per step (only count when player goes higher than ever before)
    cummin_py = jax.lax.cummin(jnp.where(mask > 0.5, py, py0))
    new_best = jnp.maximum(0.0, jnp.concatenate([jnp.array([0.0]), cummin_py[:-1] - cummin_py[1:]]))
    progress_reward = jnp.sum(new_best) * 1.0  # up to ~140 total

    # ---- effective UP: UP-action AND on real ladder column AND upward step ----
    up_actions = (actions == UP) | (actions == UPFIRE) | (actions == UPRIGHT) | (actions == UPLEFT)
    valid_up = up_actions.astype(jnp.float32) * aligned_any
    up_reward = jnp.minimum(jnp.sum(valid_up) * 0.5, 60.0)

    # ---- punch farming penalty: many FIRE actions but no real upward progress ----
    fire_actions = (actions == FIRE) | (actions == RIGHTFIRE) | (actions == LEFTFIRE) | (actions == UPFIRE)
    fire_count = jnp.sum(fire_actions.astype(jnp.float32) * mask)
    no_climb = (best_climb < 8.0).astype(jnp.float32)
    punch_farm_penalty = jnp.minimum(fire_count * no_climb * 0.5, 80.0)

    # ---- bad UP: UP action when not actually on a ladder ----
    bad_up = up_actions.astype(jnp.float32) * (1.0 - aligned_any) * mask
    bad_up_penalty = jnp.minimum(jnp.sum(bad_up) * 0.05, 30.0)

    # ---- first-reward preservation: small bonus once score events appear ----
    score_events = jnp.sum((rewards > 0.5).astype(jnp.float32))
    first_reward_bonus = jnp.minimum(score_events * 20.0, 100.0)

    # ---- idle penalty: many steps with same player_y near start ----
    stuck = ((jnp.abs(py - py0) < 4.0).astype(jnp.float32) * mask)
    stuck_frac = jnp.sum(stuck) / msum
    idle_penalty = jnp.clip(stuck_frac - 0.5, 0.0, 1.0) * 40.0

    aux = (best_climb_reward
           + progress_reward
           + up_reward
           + first_reward_bonus
           - punch_farm_penalty
           - bad_up_penalty
           - idle_penalty)
    return aux
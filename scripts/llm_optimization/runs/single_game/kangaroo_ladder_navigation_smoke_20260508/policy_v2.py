"""
Auto-generated policy v2
Generated at: 2026-05-08 16:19:33
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Observation index aliases (440-feature single-frame layout)
# ---------------------------------------------------------------------------
P_X, P_Y, P_W, P_H = 0, 1, 2, 3
P_ORIENT = 7

LAD_X_S = 168
LAD_Y_S = 188
LAD_W_S = 208
LAD_H_S = 228
LAD_A_S = 248

CHILD_X, CHILD_Y = 360, 361

FCOCO_X, FCOCO_Y, FCOCO_A = 368, 369, 372

MON_X_S = 376
MON_Y_S = 380
MON_A_S = 392

COCO_X_S = 408
COCO_Y_S = 412
COCO_A_S = 424

# Action constants
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = 6, 7, 8, 9
UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE = 10, 11, 12, 13


def init_params():
    return {
        "ladder_reach_v": jnp.float32(18.0),    # |ladder_bottom - player_bottom|
        "ladder_align_x": jnp.float32(5.0),     # column-alignment tolerance
        "climb_top_margin": jnp.float32(6.0),   # near-top dismount band
        "punch_dx": jnp.float32(20.0),          # forward punch range
        "punch_dy_band": jnp.float32(5.0),      # tight same-row monkey gate
        "danger_r": jnp.float32(14.0),          # hazard avoidance radius
        "x_move_eps": jnp.float32(2.0),         # horizontal deadzone
    }


# ---------------------------------------------------------------------------
# Skill: select a reachable active ladder useful for upward progress
# Prefer minimum vertical reach distance, then x-distance as tie-break.
# ---------------------------------------------------------------------------
def _select_ladder(obs, params):
    lx = jax.lax.dynamic_slice(obs, (LAD_X_S,), (20,))
    ly = jax.lax.dynamic_slice(obs, (LAD_Y_S,), (20,))
    lw = jax.lax.dynamic_slice(obs, (LAD_W_S,), (20,))
    lh = jax.lax.dynamic_slice(obs, (LAD_H_S,), (20,))
    la = jax.lax.dynamic_slice(obs, (LAD_A_S,), (20,))

    p_y = obs[P_Y]
    p_h = obs[P_H]
    p_bot = p_y + p_h
    p_cx = obs[P_X] + obs[P_W] * 0.5

    lcx = lx + lw * 0.5
    lbot = ly + lh
    ltop = ly

    reach_dy = jnp.abs(lbot - p_bot)
    useful = ltop < (p_y - 2.0)
    active = la > 0.5
    reachable = reach_dy < params["ladder_reach_v"]

    valid = active & useful & reachable
    # Score: vertical reach dominates, x distance is tiebreaker
    cost = reach_dy * 4.0 + jnp.abs(lcx - p_cx) + jnp.where(valid, 0.0, 1e6)
    idx = jnp.argmin(cost)
    any_valid = jnp.any(valid)

    return lcx[idx], ltop[idx], lbot[idx], any_valid


# ---------------------------------------------------------------------------
# Skill: column-overlap check (parametric, not 1-pixel)
# ---------------------------------------------------------------------------
def _is_on_ladder_column(obs, lcx, params):
    p_cx = obs[P_X] + obs[P_W] * 0.5
    return jnp.abs(p_cx - lcx) < params["ladder_align_x"]


# ---------------------------------------------------------------------------
# Skill: detect a punchable monkey on the same row, in facing direction
# ---------------------------------------------------------------------------
def _can_punch_monkey(obs, params):
    mx = jax.lax.dynamic_slice(obs, (MON_X_S,), (4,))
    my = jax.lax.dynamic_slice(obs, (MON_Y_S,), (4,))
    ma = jax.lax.dynamic_slice(obs, (MON_A_S,), (4,))

    p_x = obs[P_X]
    p_y = obs[P_Y]
    orient = obs[P_ORIENT]  # 90=right, 270=left
    facing_right = orient < 180.0

    dx = mx - p_x
    same_row = jnp.abs(my - p_y) < params["punch_dy_band"]
    in_range = jnp.abs(dx) < params["punch_dx"]

    # require monkey on the side the player is facing
    side_ok = jnp.where(facing_right, dx > 0.0, dx < 0.0)

    near = same_row & in_range & side_ok & (ma > 0.5)
    return jnp.any(near), facing_right


# ---------------------------------------------------------------------------
# Skill: hazard detection - returns (hazard_present, hazard_x)
# ---------------------------------------------------------------------------
def _danger_near_player(obs, params):
    p_x = obs[P_X]
    p_y = obs[P_Y]
    r = params["danger_r"]

    fcx = obs[FCOCO_X]
    fcy = obs[FCOCO_Y]
    fca = obs[FCOCO_A]
    fc_close = (fca > 0.5) & (jnp.abs(fcx - p_x) < r) & \
               (jnp.abs(fcy - p_y) < r * 2.0)

    cx = jax.lax.dynamic_slice(obs, (COCO_X_S,), (4,))
    cy = jax.lax.dynamic_slice(obs, (COCO_Y_S,), (4,))
    ca = jax.lax.dynamic_slice(obs, (COCO_A_S,), (4,))
    co_mask = (ca > 0.5) & (jnp.abs(cx - p_x) < r) & (jnp.abs(cy - p_y) < r)
    co_close = jnp.any(co_mask)

    hazard = fc_close | co_close
    # representative hazard x: prefer falling coconut if active
    haz_x = jnp.where(fc_close, fcx,
                      jnp.where(co_close, jnp.sum(jnp.where(co_mask, cx, 0.0)) /
                                jnp.maximum(jnp.sum(co_mask.astype(jnp.float32)), 1.0),
                                p_x))
    return hazard, haz_x


# ---------------------------------------------------------------------------
# Skill: convert horizontal target into LEFT/RIGHT/NOOP
# ---------------------------------------------------------------------------
def _move_toward_x(p_cx, target_x, eps):
    dx = target_x - p_cx
    return jnp.where(dx > eps, RIGHT, jnp.where(dx < -eps, LEFT, NOOP))


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------
def policy(obs_flat, params):
    obs = obs_flat
    p_x = obs[P_X]
    p_w = obs[P_W]
    p_y = obs[P_Y]
    p_cx = p_x + p_w * 0.5

    lcx, ltop, lbot, any_valid = _select_ladder(obs, params)
    on_col = _is_on_ladder_column(obs, lcx, params)
    near_top = p_y < (ltop + params["climb_top_margin"])

    has_monkey, face_right = _can_punch_monkey(obs, params)
    punch_action = jnp.where(face_right, RIGHTFIRE, LEFTFIRE)

    hazard, haz_x = _danger_near_player(obs, params)

    # Approach action: move toward selected ladder x
    approach = _move_toward_x(p_cx, lcx, params["x_move_eps"])

    # Dismount action: step off ladder top horizontally toward ladder column's
    # nearest direction away from current ladder; if no clear direction, go right.
    child_x = obs[CHILD_X]
    dismount = jnp.where(child_x > p_cx, RIGHT, LEFT)

    # Hazard dodge: move horizontally away from hazard x (directional, not in-place FIRE)
    dodge_dir = jnp.where(haz_x > p_cx, LEFT, RIGHT)

    # Decision tree (ordered):
    # 1. Climbing wins when truly on ladder column and not at the top.
    # 2. Punch when safely on a platform (not on column) and a monkey is in facing range.
    # 3. Hazard dodge directionally.
    # 4. Dismount when near top of ladder.
    # 5. Approach selected ladder horizontally.
    # 6. Fallback: move toward child x.

    climbing = any_valid & on_col & ~near_top
    dismounting = any_valid & on_col & near_top

    fallback = _move_toward_x(p_cx, child_x, params["x_move_eps"])

    action = jnp.where(
        climbing, UP,
        jnp.where(
            has_monkey & ~on_col, punch_action,
            jnp.where(
                hazard & ~on_col, dodge_dir,
                jnp.where(
                    dismounting, dismount,
                    jnp.where(any_valid, approach, fallback)
                )
            )
        )
    )
    return action.astype(jnp.int32)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)


# ---------------------------------------------------------------------------
# Dense reward (auxiliary shaping for hybrid LeGPS)
# ---------------------------------------------------------------------------
def dense_reward(obs_history, actions, rewards, total_reward, active_mask):
    mask = active_mask.astype(jnp.float32)
    n_active = jnp.maximum(jnp.sum(mask), 1.0)

    p_x = obs_history[:, P_X]
    p_y = obs_history[:, P_Y]
    p_w = obs_history[:, P_W]
    p_h = obs_history[:, P_H]
    p_cx = p_x + p_w * 0.5
    p_bot = p_y + p_h

    lx = jax.lax.dynamic_slice_in_dim(obs_history, LAD_X_S, 20, axis=1)
    ly = jax.lax.dynamic_slice_in_dim(obs_history, LAD_Y_S, 20, axis=1)
    lw = jax.lax.dynamic_slice_in_dim(obs_history, LAD_W_S, 20, axis=1)
    lh = jax.lax.dynamic_slice_in_dim(obs_history, LAD_H_S, 20, axis=1)
    la = jax.lax.dynamic_slice_in_dim(obs_history, LAD_A_S, 20, axis=1)
    lcx = lx + lw * 0.5
    lbot = ly + lh
    ltop = ly

    align_dx = jnp.abs(lcx - p_cx[:, None])
    useful = (ltop < (p_y[:, None] - 2.0)) & (la > 0.5)
    reach = jnp.abs(lbot - p_bot[:, None]) < 18.0
    aligned_any = jnp.any((align_dx < 5.0) & useful, axis=1)

    # 1) Upward progress: bounded by 120
    start_y = p_y[0]
    min_y = jnp.min(jnp.where(mask > 0.5, p_y, 1e6))
    upward_gain = jnp.clip(start_y - min_y, 0.0, 120.0)
    upward_term = upward_gain * 1.5  # up to 180

    # 2) Real climb events: UP-like aligned and y decreased
    is_up_like = (actions == UP) | (actions == UPFIRE) | \
                 (actions == UPRIGHT) | (actions == UPLEFT)
    next_y = jnp.concatenate([p_y[1:], p_y[-1:]], axis=0)
    dy_up = p_y - next_y
    real_climb = is_up_like & aligned_any & (dy_up > 0.0) & (mask > 0.5)
    climb_count = jnp.sum(real_climb.astype(jnp.float32))
    climb_term = jnp.clip(climb_count, 0.0, 60.0)  # up to 60

    # 3) Approach: reduce avg distance to nearest reachable ladder
    reachable_dx = jnp.where(reach & (la > 0.5), align_dx, 1e4)
    nearest_dx = jnp.min(reachable_dx, axis=1)
    nearest_dx = jnp.where(mask > 0.5, nearest_dx, 0.0)
    avg_dx = jnp.sum(nearest_dx) / n_active
    approach_term = jnp.clip(40.0 - jnp.minimum(avg_dx, 40.0), 0.0, 40.0)

    # 4) Real reward preservation (the 200 first-RIGHTFIRE branch)
    got_reward = jnp.sum(rewards * mask)
    first_reward_term = jnp.clip(got_reward, 0.0, 400.0) * 0.3  # up to 120

    # 5) Platform bonus: reached significantly higher
    platform_bonus = jnp.where(upward_gain > 30.0, 50.0, 0.0)

    # 6) Bad-UP penalty: UP issued without ladder alignment
    bad_up = is_up_like & (~aligned_any) & (mask > 0.5)
    bad_up_pen = jnp.clip(jnp.sum(bad_up.astype(jnp.float32)) - 15.0, 0.0, 80.0) * 0.4

    # 7) Idle penalty: tiny y range AND tiny x range AND no reward
    y_range = jnp.max(jnp.where(mask > 0.5, p_y, -1e6)) - \
              jnp.min(jnp.where(mask > 0.5, p_y, 1e6))
    x_range = jnp.max(jnp.where(mask > 0.5, p_x, -1e6)) - \
              jnp.min(jnp.where(mask > 0.5, p_x, 1e6))
    idle_pen = jnp.where((y_range < 4.0) & (x_range < 8.0) & (got_reward < 1.0),
                         30.0, 0.0)

    # 8) Punch-farming penalty: many FIRE-like actions but no upward gain
    is_fire_like = (actions == FIRE) | (actions == RIGHTFIRE) | \
                   (actions == LEFTFIRE) | (actions == UPFIRE)
    fire_count = jnp.sum((is_fire_like & (mask > 0.5)).astype(jnp.float32))
    no_up_progress = upward_gain < 8.0
    farm_penalty = jnp.where(no_up_progress,
                             jnp.clip(fire_count - 10.0, 0.0, 60.0) * 0.5,
                             0.0)

    aux = (upward_term + climb_term + approach_term + first_reward_term
           + platform_bonus
           - bad_up_pen - idle_pen - farm_penalty)
    return aux
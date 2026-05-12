"""
Auto-generated policy v1
Generated at: 2026-05-08 16:16:54
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Observation index aliases (440-feature single-frame layout)
# ---------------------------------------------------------------------------
P_X, P_Y, P_W, P_H = 0, 1, 2, 3
P_ORIENT = 7

LAD_X_S, LAD_X_E = 168, 188
LAD_Y_S, LAD_Y_E = 188, 208
LAD_W_S, LAD_W_E = 208, 228
LAD_H_S, LAD_H_E = 228, 248
LAD_A_S, LAD_A_E = 248, 268

CHILD_X, CHILD_Y = 360, 361

FCOCO_X, FCOCO_Y, FCOCO_A = 368, 369, 372

MON_X_S, MON_X_E = 376, 380
MON_Y_S, MON_Y_E = 380, 384
MON_A_S, MON_A_E = 392, 396

COCO_X_S, COCO_X_E = 408, 412
COCO_Y_S, COCO_Y_E = 412, 416
COCO_A_S, COCO_A_E = 424, 428

# Action constants
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = 6, 7, 8, 9
UPFIRE, RIGHTFIRE, LEFTFIRE = 10, 11, 12


def init_params():
    return {
        "ladder_reach_v": jnp.float32(14.0),   # |ladder_bottom - player_bottom| tolerance
        "ladder_align_x": jnp.float32(4.0),    # |ladder_center - player_center| tight band
        "climb_top_margin": jnp.float32(8.0),  # how close to ladder top before dismounting
        "punch_dx": jnp.float32(18.0),         # punch range in x for nearby monkey
        "danger_r": jnp.float32(14.0),         # coconut/falling-coconut avoidance radius
        "x_move_eps": jnp.float32(2.0),        # deadzone for horizontal motion
    }


# ---------------------------------------------------------------------------
# Skill: select a reachable active ladder useful for upward progress
# ---------------------------------------------------------------------------
def select_ladder(obs, params):
    lx = jax.lax.dynamic_slice(obs, (LAD_X_S,), (20,))
    ly = jax.lax.dynamic_slice(obs, (LAD_Y_S,), (20,))
    lw = jax.lax.dynamic_slice(obs, (LAD_W_S,), (20,))
    lh = jax.lax.dynamic_slice(obs, (LAD_H_S,), (20,))
    la = jax.lax.dynamic_slice(obs, (LAD_A_S,), (20,))

    p_y = obs[P_Y]
    p_h = obs[P_H]
    p_bot = p_y + p_h

    lcx = lx + lw * 0.5
    lbot = ly + lh
    ltop = ly

    reach_dy = jnp.abs(lbot - p_bot)
    useful = (ltop < p_y - 2.0)
    active = la > 0.5
    reachable = reach_dy < params["ladder_reach_v"]

    valid = active & useful & reachable
    # Score: prefer reachable, then closest in x
    p_cx = obs[P_X] + obs[P_W] * 0.5
    cost = jnp.abs(lcx - p_cx) + jnp.where(valid, 0.0, 1e6)
    idx = jnp.argmin(cost)
    any_valid = jnp.any(valid)

    return lcx[idx], ltop[idx], lbot[idx], lw[idx], any_valid


# ---------------------------------------------------------------------------
# Skill: detect column overlap with selected ladder (real overlap, not 1px)
# ---------------------------------------------------------------------------
def on_ladder_column(obs, lcx, lw, params):
    p_cx = obs[P_X] + obs[P_W] * 0.5
    return jnp.abs(p_cx - lcx) < params["ladder_align_x"]


# ---------------------------------------------------------------------------
# Skill: detect a punchable monkey in front of player
# ---------------------------------------------------------------------------
def punch_in_range(obs, params):
    mx = jax.lax.dynamic_slice(obs, (MON_X_S,), (4,))
    my = jax.lax.dynamic_slice(obs, (MON_Y_S,), (4,))
    ma = jax.lax.dynamic_slice(obs, (MON_A_S,), (4,))

    p_x = obs[P_X]
    p_y = obs[P_Y]
    p_h = obs[P_H]

    dx = mx - p_x
    dy_ok = jnp.abs(my - p_y) < (p_h + 6.0)
    near = (jnp.abs(dx) < params["punch_dx"]) & dy_ok & (ma > 0.5)
    any_near = jnp.any(near)
    # direction: positive dx -> right
    avg_dx = jnp.sum(jnp.where(near, dx, 0.0))
    face_right = avg_dx >= 0.0
    return any_near, face_right


# ---------------------------------------------------------------------------
# Skill: hazard avoidance signal (falling coconut + thrown coconuts close)
# ---------------------------------------------------------------------------
def hazard_near(obs, params):
    p_x = obs[P_X]
    p_y = obs[P_Y]

    fcx = obs[FCOCO_X]
    fcy = obs[FCOCO_Y]
    fca = obs[FCOCO_A]
    fc_close = (fca > 0.5) & (jnp.abs(fcx - p_x) < params["danger_r"]) & \
               (jnp.abs(fcy - p_y) < params["danger_r"] * 2.0)

    cx = jax.lax.dynamic_slice(obs, (COCO_X_S,), (4,))
    cy = jax.lax.dynamic_slice(obs, (COCO_Y_S,), (4,))
    ca = jax.lax.dynamic_slice(obs, (COCO_A_S,), (4,))
    co_close = jnp.any((ca > 0.5) &
                       (jnp.abs(cx - p_x) < params["danger_r"]) &
                       (jnp.abs(cy - p_y) < params["danger_r"]))
    return fc_close | co_close


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------
def policy(obs_flat, params):
    obs = obs_flat
    p_x = obs[P_X]
    p_w = obs[P_W]
    p_cx = p_x + p_w * 0.5
    p_y = obs[P_Y]

    lcx, ltop, lbot, lw, any_valid = select_ladder(obs, params)
    on_col = on_ladder_column(obs, lcx, lw, params)
    near_top = p_y < (ltop + params["climb_top_margin"])

    # Horizontal direction toward ladder center
    dx = lcx - p_cx
    move_right = dx > params["x_move_eps"]
    move_left = dx < -params["x_move_eps"]

    # Skill: punch nearby monkey (preserve first-reward route)
    has_monkey, face_right = punch_in_range(obs, params)
    punch_action = jnp.where(face_right, RIGHTFIRE, LEFTFIRE)

    # Skill: hazard dodge - jump (FIRE) is safer than standing
    hazard = hazard_near(obs, params)

    # Decision tree (ordered, simple):
    # 1. If a monkey is in punch range -> punch (preserves stable 200 reward).
    # 2. Else if hazard near -> FIRE (jump) to dodge.
    # 3. Else if on a ladder column and ladder useful and not near top -> UP.
    # 4. Else if on column and near top -> dismount horizontally toward child.
    # 5. Else move horizontally toward selected ladder.
    # 6. Fallback: move toward child x.

    child_dx = obs[CHILD_X] - p_cx
    child_right = child_dx > 0.0
    dismount = jnp.where(child_right, RIGHT, LEFT)

    horiz = jnp.where(move_right, RIGHT,
                      jnp.where(move_left, LEFT, NOOP))

    climb_action = UP
    no_ladder_action = jnp.where(child_right, RIGHT, LEFT)

    action = jnp.where(
        has_monkey, punch_action,
        jnp.where(
            hazard, FIRE,
            jnp.where(
                any_valid & on_col & ~near_top, climb_action,
                jnp.where(
                    any_valid & on_col & near_top, dismount,
                    jnp.where(any_valid, horiz, no_ladder_action)
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

    p_y = obs_history[:, P_Y]
    p_x = obs_history[:, P_X]
    p_w = obs_history[:, P_W]
    p_h = obs_history[:, P_H]
    p_cx = p_x + p_w * 0.5
    p_bot = p_y + p_h

    # Ladder fields per timestep
    lx = jax.lax.dynamic_slice_in_dim(obs_history, LAD_X_S, 20, axis=1)
    ly = jax.lax.dynamic_slice_in_dim(obs_history, LAD_Y_S, 20, axis=1)
    lw = jax.lax.dynamic_slice_in_dim(obs_history, LAD_W_S, 20, axis=1)
    lh = jax.lax.dynamic_slice_in_dim(obs_history, LAD_H_S, 20, axis=1)
    la = jax.lax.dynamic_slice_in_dim(obs_history, LAD_A_S, 20, axis=1)
    lcx = lx + lw * 0.5
    lbot = ly + lh
    ltop = ly

    # Per-step: is the player column-aligned with any active useful ladder?
    align_dx = jnp.abs(lcx - p_cx[:, None])
    useful = (ltop < (p_y[:, None] - 2.0)) & (la > 0.5)
    reach = jnp.abs(lbot - p_bot[:, None]) < 16.0
    aligned_any = jnp.any((align_dx < 4.0) & useful, axis=1)
    reach_any = jnp.any((align_dx < 8.0) & reach & (la > 0.5), axis=1)

    # Upward progress: how much lower y the player reached vs starting y
    start_y = p_y[0]
    min_y = jnp.min(jnp.where(mask > 0.5, p_y, 1e6))
    upward_gain = jnp.clip(start_y - min_y, 0.0, 120.0)  # bounded
    upward_term = upward_gain * 1.5  # up to ~180

    # Real ladder-column climb events: UP-like action AND aligned AND y decreased next step
    is_up_like = (actions == UP) | (actions == UPFIRE) | \
                 (actions == UPRIGHT) | (actions == UPLEFT)
    # next y change
    next_y = jnp.concatenate([p_y[1:], p_y[-1:]], axis=0)
    dy = p_y - next_y  # positive -> moved up
    real_climb = is_up_like & aligned_any & (dy > 0.0) & (mask > 0.5)
    climb_count = jnp.sum(real_climb.astype(jnp.float32))
    climb_term = jnp.clip(climb_count, 0.0, 60.0) * 1.0  # up to 60

    # Approach term: average distance to nearest reachable ladder center, lower is better
    reachable_dx = jnp.where(reach & (la > 0.5), align_dx, 1e4)
    nearest_dx = jnp.min(reachable_dx, axis=1)
    nearest_dx = jnp.where(mask > 0.5, nearest_dx, 0.0)
    avg_dx = jnp.sum(nearest_dx) / n_active
    approach_term = jnp.clip(40.0 - jnp.minimum(avg_dx, 40.0), 0.0, 40.0)  # up to 40

    # First-reward preservation: any positive reward
    got_reward = jnp.sum(rewards * mask)
    first_reward_term = jnp.clip(got_reward, 0.0, 200.0) * 0.25  # up to 50

    # Penalty: punch farming - many FIRE-like actions but no upward gain
    is_fire_like = (actions == FIRE) | (actions == RIGHTFIRE) | \
                   (actions == LEFTFIRE) | (actions == UPFIRE)
    fire_count = jnp.sum((is_fire_like & (mask > 0.5)).astype(jnp.float32))
    no_up_progress = upward_gain < 8.0
    farm_penalty = jnp.where(no_up_progress,
                             jnp.clip(fire_count - 5.0, 0.0, 60.0) * 1.0,
                             0.0)

    # Penalty: UP action when not actually column-aligned (premature climb)
    bad_up = is_up_like & (~aligned_any) & (mask > 0.5)
    bad_up_pen = jnp.clip(jnp.sum(bad_up.astype(jnp.float32)) - 10.0, 0.0, 80.0) * 0.5

    # Penalty: zero FIRE and zero reward (scoreless passive)
    zero_fire = fire_count < 1.0
    zero_reward = got_reward < 1.0
    passive_pen = jnp.where(zero_fire & zero_reward, 30.0, 0.0)

    # Penalty: idling - very small total y range
    y_range = jnp.max(jnp.where(mask > 0.5, p_y, -1e6)) - \
              jnp.min(jnp.where(mask > 0.5, p_y, 1e6))
    idle_pen = jnp.where(y_range < 4.0, 20.0, 0.0)

    # Bonus: reach a higher platform (significant upward gain)
    platform_bonus = jnp.where(upward_gain > 30.0, 40.0, 0.0)

    aux = (upward_term + climb_term + approach_term + first_reward_term
           + platform_bonus
           - farm_penalty - bad_up_pen - passive_pen - idle_pen)
    return aux
"""
Auto-generated policy v3
Generated at: 2026-04-26 19:54:24
"""

import jax
import jax.numpy as jnp

NOOP = 0
FIRE = 1
MOVE_UP = 2
MOVE_DOWN = 3
FIRE_UP = 4
FIRE_DOWN = 5

FRAME_SIZE = 26
PLAYER_X = 140.0
PADDLE_CENTER_OFFSET = 8.0
PADDLE_HEIGHT = 16.0
PLAYFIELD_TOP = 24.0
PLAYFIELD_BOTTOM = 194.0
SCREEN_MID = 105.0


def init_params():
    return {
        "dead_zone": jnp.array(3.5),
        "aim_gain": jnp.array(0.7),
        "idle_track_gain": jnp.array(0.7),  # blend ball_y vs screen mid when receding
        "fire_dist": jnp.array(25.0),
        "aim_close_dist": jnp.array(30.0),  # only apply aim offset when ball is close
    }


def _fold(y):
    span = PLAYFIELD_BOTTOM - PLAYFIELD_TOP
    rel = y - PLAYFIELD_TOP
    mod = jnp.mod(rel, 2.0 * span)
    folded = jnp.where(mod > span, 2.0 * span - mod, mod)
    return PLAYFIELD_TOP + folded


def _predict_intercept(bx, by, dx, dy):
    # Predict only when ball is meaningfully approaching
    approaching = dx > 0.5
    t = jnp.where(approaching, (PLAYER_X - bx) / jnp.maximum(dx, 0.5), 0.0)
    t = jnp.clip(t, 0.0, 60.0)
    raw_y = by + dy * t
    pred = _fold(raw_y)
    # If not approaching, fall back to current ball y
    return jnp.where(approaching, pred, by)


def _hard_sign(x, thresh):
    pos = x > thresh
    neg = x < -thresh
    return jnp.where(pos, 1.0, jnp.where(neg, -1.0, 0.0))


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    player_y = curr[1]
    paddle_center = player_y + PADDLE_CENTER_OFFSET

    prev_bx, prev_by = prev[16], prev[17]
    bx, by = curr[16], curr[17]
    ball_active = curr[20]

    dx = bx - prev_bx
    dy = by - prev_by

    approaching = dx > 0.5
    dist_to_player = PLAYER_X - bx
    ball_close = dist_to_player < params["aim_close_dist"]

    predicted = _predict_intercept(bx, by, dx, dy)

    # Stable aim offset: only when ball is close and approaching, hard-sign on dy
    dy_dir = _hard_sign(dy, 0.5)
    apply_aim = approaching & ball_close
    aim_offset = jnp.where(apply_aim,
                           -params["aim_gain"] * dy_dir * (PADDLE_HEIGHT * 0.5),
                           0.0)
    aim = predicted + aim_offset

    aim = jnp.clip(aim,
                   PLAYFIELD_TOP + PADDLE_CENTER_OFFSET,
                   PLAYFIELD_BOTTOM - PADDLE_CENTER_OFFSET)

    # Receding fallback: track ball Y blended with center
    g = jnp.clip(params["idle_track_gain"], 0.0, 1.0)
    idle_target = g * by + (1.0 - g) * SCREEN_MID
    idle_target = jnp.clip(idle_target,
                           PLAYFIELD_TOP + PADDLE_CENTER_OFFSET,
                           PLAYFIELD_BOTTOM - PADDLE_CENTER_OFFSET)

    target_active = jnp.where(approaching, aim, idle_target)
    target = jnp.where(ball_active > 0.5, target_active, SCREEN_MID)

    error = target - paddle_center
    dz = params["dead_zone"]

    move_up = error < -dz
    move_down = error > dz

    use_fire = ball_close & approaching & (ball_active > 0.5) & (dist_to_player < params["fire_dist"])

    action_up = jnp.where(use_fire, FIRE_UP, MOVE_UP)
    action_down = jnp.where(use_fire, FIRE_DOWN, MOVE_DOWN)
    action_idle = jnp.where(use_fire, FIRE, NOOP)

    action = jnp.where(move_up, action_up,
                       jnp.where(move_down, action_down, action_idle))
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
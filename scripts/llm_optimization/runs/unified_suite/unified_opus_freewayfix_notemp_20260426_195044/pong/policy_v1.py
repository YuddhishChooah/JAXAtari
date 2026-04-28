"""
Auto-generated policy v1
Generated at: 2026-04-26 19:51:00
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
PLAYFIELD_TOP = 24.0
PLAYFIELD_BOTTOM = 194.0


def init_params():
    return {
        "dead_zone": jnp.array(3.0),
        "predict_gain": jnp.array(1.0),
        "aim_offset": jnp.array(0.0),
        "fallback_y": jnp.array(105.0),
        "fire_bias": jnp.array(0.5),
    }


def _predict_intercept(bx, by, dx, dy, gain):
    # time to reach player paddle
    dist = PLAYER_X - bx
    # avoid div by zero; if dx <= 0, prediction not useful
    safe_dx = jnp.where(jnp.abs(dx) < 1e-3, 1e-3, dx)
    t = dist / safe_dx
    raw_y = by + dy * t * gain
    # reflect off top/bottom walls
    span = PLAYFIELD_BOTTOM - PLAYFIELD_TOP
    rel = raw_y - PLAYFIELD_TOP
    mod = jnp.mod(rel, 2.0 * span)
    folded = jnp.where(mod > span, 2.0 * span - mod, mod)
    return PLAYFIELD_TOP + folded


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

    predicted = _predict_intercept(bx, by, dx, dy, params["predict_gain"])
    # if ball is moving away (dx <= 0), aim toward current ball y, lightly
    target_when_away = by
    target = jnp.where(dx > 0.0, predicted + params["aim_offset"], target_when_away)
    # if ball not active, recenter
    target = jnp.where(ball_active > 0.5, target, params["fallback_y"])

    error = target - paddle_center
    dz = params["dead_zone"]

    move_up = error < -dz
    move_down = error > dz

    # fire bias: occasionally combine FIRE based on signed bias param
    use_fire = params["fire_bias"] > 0.0

    action_up = jnp.where(use_fire, FIRE_UP, MOVE_UP)
    action_down = jnp.where(use_fire, FIRE_DOWN, MOVE_DOWN)
    action_idle = jnp.where(use_fire, FIRE, NOOP)

    action = jnp.where(move_up, action_up,
                       jnp.where(move_down, action_down, action_idle))
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
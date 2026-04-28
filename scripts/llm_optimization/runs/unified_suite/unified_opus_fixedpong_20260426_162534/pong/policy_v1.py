"""
Auto-generated policy v1
Generated at: 2026-04-26 16:25:49
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
        "fire_bias": jnp.array(0.0),
        "fallback_y": jnp.array(105.0),
    }


def _predict_intercept(bx, by, dx, dy, gain):
    # time to reach player x (only meaningful if dx > 0)
    safe_dx = jnp.where(jnp.abs(dx) < 1e-3, 1e-3, dx)
    t = (PLAYER_X - bx) / safe_dx
    t = jnp.clip(t, 0.0, 200.0)
    raw_y = by + dy * t * gain
    # reflect off top/bottom walls
    span = PLAYFIELD_BOTTOM - PLAYFIELD_TOP
    rel = raw_y - PLAYFIELD_TOP
    period = 2.0 * span
    m = jnp.mod(rel, period)
    folded = jnp.where(m > span, period - m, m)
    return PLAYFIELD_TOP + folded


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    player_y = curr[1]
    paddle_center = player_y + PADDLE_CENTER_OFFSET

    prev_bx = prev[16]
    prev_by = prev[17]
    curr_bx = curr[16]
    curr_by = curr[17]
    ball_active = curr[20]

    dx = curr_bx - prev_bx
    dy = curr_by - prev_by

    moving_toward = dx > 0.0

    intercept = _predict_intercept(curr_bx, curr_by, dx, dy, params["predict_gain"])
    target = jnp.where(moving_toward, intercept + params["aim_offset"], params["fallback_y"])
    target = jnp.where(ball_active > 0.5, target, params["fallback_y"])

    error = target - paddle_center
    dz = params["dead_zone"]

    move_up = error < -dz
    move_down = error > dz

    fire_on = params["fire_bias"] > 0.0

    action_up = jnp.where(fire_on, FIRE_UP, MOVE_UP)
    action_down = jnp.where(fire_on, FIRE_DOWN, MOVE_DOWN)
    action_idle = jnp.where(fire_on, FIRE, NOOP)

    action = jnp.where(move_up, action_up, jnp.where(move_down, action_down, action_idle))
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
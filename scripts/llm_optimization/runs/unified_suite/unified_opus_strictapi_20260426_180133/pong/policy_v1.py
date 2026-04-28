"""
Auto-generated policy v1
Generated at: 2026-04-26 18:01:51
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
        "dead_zone": jnp.array(2.5),
        "predict_gain": jnp.array(1.0),
        "aim_offset": jnp.array(0.0),
        "fire_bias": jnp.array(0.5),
        "fallback_y": jnp.array(109.0),
    }


def _predict_intercept(curr_ball_x, curr_ball_y, ball_dx, ball_dy, gain):
    # Estimate time to reach player x; clamp to avoid div by zero
    dx_safe = jnp.where(jnp.abs(ball_dx) < 1e-3, 1e-3, ball_dx)
    t = (PLAYER_X - curr_ball_x) / dx_safe
    t = jnp.clip(t, 0.0, 80.0)
    raw_y = curr_ball_y + ball_dy * t * gain

    # Reflect off top/bottom walls
    span = PLAYFIELD_BOTTOM - PLAYFIELD_TOP
    rel = jnp.mod(raw_y - PLAYFIELD_TOP, 2.0 * span)
    rel = jnp.where(rel > span, 2.0 * span - rel, rel)
    return PLAYFIELD_TOP + rel


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    player_y = curr[1]
    ball_x = curr[16]
    ball_y = curr[17]
    ball_active = curr[20]
    prev_ball_x = prev[16]
    prev_ball_y = prev[17]

    ball_dx = ball_x - prev_ball_x
    ball_dy = ball_y - prev_ball_y

    moving_toward = ball_dx > 0.0

    target_y = jnp.where(
        moving_toward,
        _predict_intercept(ball_x, ball_y, ball_dx, ball_dy, params["predict_gain"]),
        params["fallback_y"],
    )
    target_y = target_y + params["aim_offset"]

    paddle_center = player_y + PADDLE_CENTER_OFFSET
    error = target_y - paddle_center

    dz = params["dead_zone"]
    move_up = error < -dz
    move_down = error > dz

    # Fire when ball is close and moving toward us
    close = (PLAYER_X - ball_x) < 40.0
    fire_active = moving_toward & close & (params["fire_bias"] > 0.0)

    action = jnp.where(
        move_up,
        jnp.where(fire_active, FIRE_UP, MOVE_UP),
        jnp.where(
            move_down,
            jnp.where(fire_active, FIRE_DOWN, MOVE_DOWN),
            jnp.where(ball_active < 0.5, FIRE, NOOP),
        ),
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
"""
Auto-generated policy v1
Generated at: 2026-04-28 13:45:14
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
        "vy_damp": jnp.array(0.8),
    }


def _predict_intercept(bx, by, dx, dy, gain):
    # Time to reach the player paddle plane
    safe_dx = jnp.where(jnp.abs(dx) < 1e-3, 1e-3, dx)
    t = (PLAYER_X - bx) / safe_dx
    t = jnp.maximum(t, 0.0)
    raw_y = by + dy * t * gain
    # Reflect off top/bottom walls
    span = PLAYFIELD_BOTTOM - PLAYFIELD_TOP
    rel = raw_y - PLAYFIELD_TOP
    mod = jnp.mod(rel, 2.0 * span)
    folded = jnp.where(mod > span, 2.0 * span - mod, mod)
    return PLAYFIELD_TOP + folded


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    player_y = curr[1]
    ball_x_prev = prev[16]
    ball_y_prev = prev[17]
    ball_x = curr[16]
    ball_y = curr[17]

    dx = ball_x - ball_x_prev
    dy = ball_y - ball_y_prev

    paddle_center = player_y + PADDLE_CENTER_OFFSET

    # If the ball is moving toward the player, predict intercept; else track ball y
    moving_toward = dx > 0.0
    predicted = _predict_intercept(ball_x, ball_y, dx, dy * params["vy_damp"], params["predict_gain"])
    target_y = jnp.where(moving_toward, predicted, ball_y)
    target_y = target_y + params["aim_offset"]

    error = target_y - paddle_center
    dz = params["dead_zone"]

    # Decide movement
    move_up = error < -dz
    move_down = error > dz

    # Fire when ball is close to paddle and aligned
    close = jnp.abs(PLAYER_X - ball_x) < 20.0
    aligned = jnp.abs(error) < (dz + 6.0)
    use_fire = close & aligned & (params["fire_bias"] > 0.0)

    action = jnp.where(
        move_up,
        jnp.where(use_fire, FIRE_UP, MOVE_UP),
        jnp.where(
            move_down,
            jnp.where(use_fire, FIRE_DOWN, MOVE_DOWN),
            jnp.where(use_fire, FIRE, NOOP),
        ),
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
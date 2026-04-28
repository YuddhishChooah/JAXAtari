"""
Auto-generated policy v1
Generated at: 2026-04-27 17:19:52
"""

import jax
import jax.numpy as jnp

NOOP = 0
FIRE = 1
MOVE_UP = 2
MOVE_DOWN = 3
FIRE_UP = 4
FIRE_DOWN = 5

PLAYER_X = 140.0
PLAYFIELD_TOP = 24.0
PLAYFIELD_BOTTOM = 194.0
PADDLE_HEIGHT = 16.0
PADDLE_CENTER_OFFSET = 8.0


def init_params():
    return {
        "dead_zone": 3.0,
        "lead": 1.5,
        "aim_offset": 4.0,
        "center_y": 110.0,
        "fire_threshold": 6.0,
    }


def predict_intercept(ball_x, ball_y, ball_dx, ball_dy, target_x):
    # Time until ball reaches target_x (guard against zero dx)
    safe_dx = jnp.where(jnp.abs(ball_dx) < 1e-3, 1e-3, ball_dx)
    t = (target_x - ball_x) / safe_dx
    t = jnp.maximum(t, 0.0)
    raw_y = ball_y + ball_dy * t
    # Reflect off top/bottom walls
    span = PLAYFIELD_BOTTOM - PLAYFIELD_TOP
    y_shift = raw_y - PLAYFIELD_TOP
    period = 2.0 * span
    y_mod = jnp.mod(y_shift, period)
    y_reflected = jnp.where(y_mod > span, period - y_mod, y_mod)
    return PLAYFIELD_TOP + y_reflected


def policy(obs_flat, params):
    prev_ball_x = obs_flat[16]
    prev_ball_y = obs_flat[17]
    curr_player_y = obs_flat[26 + 1]
    curr_ball_x = obs_flat[26 + 16]
    curr_ball_y = obs_flat[26 + 17]
    curr_ball_active = obs_flat[26 + 20]

    ball_dx = curr_ball_x - prev_ball_x
    ball_dy = curr_ball_y - prev_ball_y

    paddle_center = curr_player_y + PADDLE_CENTER_OFFSET

    # When ball moving toward player (dx > 0), predict intercept; else track ball/center
    moving_toward = ball_dx > 0.1

    intercept = predict_intercept(
        curr_ball_x, curr_ball_y, ball_dx, ball_dy, PLAYER_X
    )

    # Aim slightly off-center: bias toward hitting with paddle edge depending on ball_dy sign
    aim_bias = jnp.where(ball_dy >= 0, params["aim_offset"], -params["aim_offset"])
    target_attack = intercept + aim_bias

    # When ball not coming, drift toward center / track ball loosely with lead
    target_idle = params["center_y"] + params["lead"] * ball_dy

    # If ball inactive, go to center
    target = jnp.where(moving_toward, target_attack, target_idle)
    target = jnp.where(curr_ball_active > 0.5, target, params["center_y"])

    # Clamp target to playfield
    target = jnp.clip(
        target,
        PLAYFIELD_TOP + PADDLE_CENTER_OFFSET,
        PLAYFIELD_BOTTOM - PADDLE_CENTER_OFFSET,
    )

    error = target - paddle_center
    abs_err = jnp.abs(error)

    # Decide movement
    in_dead = abs_err < params["dead_zone"]
    move_up = error < 0
    # Use FIRE-combo when error large (more aggressive) and FIRE alone when ball idle
    big_err = abs_err > params["fire_threshold"]

    action = jnp.where(
        in_dead,
        jnp.where(curr_ball_active > 0.5, NOOP, FIRE),
        jnp.where(
            move_up,
            jnp.where(big_err, FIRE_UP, MOVE_UP),
            jnp.where(big_err, FIRE_DOWN, MOVE_DOWN),
        ),
    )

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
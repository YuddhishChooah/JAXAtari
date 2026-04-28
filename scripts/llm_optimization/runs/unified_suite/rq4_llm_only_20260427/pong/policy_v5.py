"""
Auto-generated policy v5
Generated at: 2026-04-27 17:24:16
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
ENEMY_X = 16.0
PLAYFIELD_TOP = 24.0
PLAYFIELD_BOTTOM = 194.0
PADDLE_HEIGHT = 16.0
PADDLE_CENTER_OFFSET = 8.0
BALL_WIDTH = 2.0
INTERCEPT_X = PLAYER_X - BALL_WIDTH  # 138.0
CENTER_Y = 0.5 * (PLAYFIELD_TOP + PLAYFIELD_BOTTOM)  # 109.0


def init_params():
    return {
        "dead_zone": 4.0,
        "aim_offset": 6.0,
        "approach_x": 110.0,
        "attack_x": 128.0,
        "idle_lead": 0.6,
    }


def reflect_y(raw_y):
    span = PLAYFIELD_BOTTOM - PLAYFIELD_TOP
    period = 2.0 * span
    y_shift = jnp.mod(raw_y - PLAYFIELD_TOP, period)
    y_reflected = jnp.where(y_shift > span, period - y_shift, y_shift)
    return PLAYFIELD_TOP + y_reflected


def predict_intercept(ball_x, ball_y, ball_dx, ball_dy, target_x):
    safe_dx = jnp.where(jnp.abs(ball_dx) < 1e-3,
                        1e-3 * jnp.sign(ball_dx + 1e-6),
                        ball_dx)
    t = (target_x - ball_x) / safe_dx
    t = jnp.maximum(t, 0.0)
    raw_y = ball_y + ball_dy * t
    return reflect_y(raw_y)


def select_target(curr_ball_x, curr_ball_y, ball_dx, ball_dy,
                  curr_ball_active, params):
    # Forward intercept when approaching
    forward_intercept = predict_intercept(
        curr_ball_x, curr_ball_y, ball_dx, ball_dy, INTERCEPT_X
    )

    moving_toward = ball_dx > 0.0
    close = curr_ball_x > params["approach_x"]
    very_close = curr_ball_x > params["attack_x"]

    # Stable aim sign: based on intercept position relative to playfield center.
    # If intercept is upper half, aim to hit with bottom edge (target slightly
    # below intercept) to send ball downward; vice versa. This keeps sign
    # constant across wall bounces during approach.
    aim_sign = jnp.where(forward_intercept < CENTER_Y, 1.0, -1.0)
    aim_bias = jnp.where(very_close, aim_sign * params["aim_offset"], 0.0)

    attack_target = forward_intercept + aim_bias
    approach_target = forward_intercept

    # When ball moves away: park near center, with small lead toward where
    # the ball currently is so the paddle is roughly pre-positioned.
    away_target = CENTER_Y + params["idle_lead"] * (curr_ball_y - CENTER_Y)

    target = jnp.where(
        moving_toward,
        jnp.where(close, attack_target, approach_target),
        away_target,
    )

    target = jnp.where(curr_ball_active > 0.5, target, CENTER_Y)
    target = jnp.clip(
        target,
        PLAYFIELD_TOP + PADDLE_CENTER_OFFSET,
        PLAYFIELD_BOTTOM - PADDLE_CENTER_OFFSET,
    )
    return target, moving_toward & close


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

    target, contact_phase = select_target(
        curr_ball_x, curr_ball_y, ball_dx, ball_dy, curr_ball_active, params
    )

    error = target - paddle_center
    abs_err = jnp.abs(error)
    in_dead = abs_err < params["dead_zone"]
    move_up = error < 0.0

    # Use FIRE-combo movement near contact (helps end rallies),
    # plain movement otherwise (cleaner control, less state churn).
    move_action = jnp.where(
        contact_phase,
        jnp.where(move_up, FIRE_UP, FIRE_DOWN),
        jnp.where(move_up, MOVE_UP, MOVE_DOWN),
    )
    idle_action = jnp.where(curr_ball_active > 0.5, NOOP, FIRE)

    action = jnp.where(in_dead, idle_action, move_action)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
"""
Auto-generated policy v2
Generated at: 2026-04-27 17:21:00
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


def init_params():
    return {
        "dead_zone": 2.0,
        "aim_offset": 5.0,
        "approach_x": 70.0,
        "track_lead": 3.0,
    }


def reflect_y(raw_y):
    span = PLAYFIELD_BOTTOM - PLAYFIELD_TOP
    period = 2.0 * span
    y_shift = jnp.mod(raw_y - PLAYFIELD_TOP, period)
    y_reflected = jnp.where(y_shift > span, period - y_shift, y_shift)
    return PLAYFIELD_TOP + y_reflected


def predict_intercept(ball_x, ball_y, ball_dx, ball_dy, target_x):
    safe_dx = jnp.where(jnp.abs(ball_dx) < 1e-3, 1e-3 * jnp.sign(ball_dx + 1e-6), ball_dx)
    t = (target_x - ball_x) / safe_dx
    t = jnp.maximum(t, 0.0)
    raw_y = ball_y + ball_dy * t
    return reflect_y(raw_y)


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

    # Forward intercept: ball heading toward player
    forward_intercept = predict_intercept(
        curr_ball_x, curr_ball_y, ball_dx, ball_dy, INTERCEPT_X
    )

    # Reverse case: ball moving away. Predict where enemy will intercept,
    # then bounce back to player as if enemy reflects at ENEMY_X.
    enemy_y = predict_intercept(
        curr_ball_x, curr_ball_y, ball_dx, ball_dy, ENEMY_X
    )
    # Assume enemy returns ball; reflect dy and predict re-intercept at player x.
    return_intercept = predict_intercept(
        ENEMY_X, enemy_y, jnp.abs(ball_dx), ball_dy, INTERCEPT_X
    )

    moving_toward = ball_dx > 0.0

    # Use forward intercept when ball approaches; else use predicted return
    intercept = jnp.where(moving_toward, forward_intercept, return_intercept)

    # Aim offset: hit with edge opposite to incoming dy to create spiky returns.
    # When ball is descending (dy>0), hit with upper edge -> target slightly above intercept.
    close = curr_ball_x > params["approach_x"]
    aim_sign = jnp.where(ball_dy >= 0, -1.0, 1.0)
    aim_bias = jnp.where(moving_toward & close, aim_sign * params["aim_offset"], 0.0)

    target_attack = intercept + aim_bias

    # When ball is far or moving away, also blend in direct ball tracking with lead
    target_track = curr_ball_y + params["track_lead"] * ball_dy

    use_attack = moving_toward & close
    target = jnp.where(use_attack, target_attack, jnp.where(moving_toward, intercept, target_track))

    # Inactive ball: hold near vertical center of playfield
    center = 0.5 * (PLAYFIELD_TOP + PLAYFIELD_BOTTOM)
    target = jnp.where(curr_ball_active > 0.5, target, center)

    # Clamp to legal paddle center range
    target = jnp.clip(
        target,
        PLAYFIELD_TOP + PADDLE_CENTER_OFFSET,
        PLAYFIELD_BOTTOM - PADDLE_CENTER_OFFSET,
    )

    error = target - paddle_center
    abs_err = jnp.abs(error)

    in_dead = abs_err < params["dead_zone"]
    move_up = error < 0.0

    # Default to FIRE-combo movement (reported to outperform). Idle FIREs when ball inactive.
    move_action = jnp.where(move_up, FIRE_UP, FIRE_DOWN)
    idle_action = jnp.where(curr_ball_active > 0.5, NOOP, FIRE)

    action = jnp.where(in_dead, idle_action, move_action)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
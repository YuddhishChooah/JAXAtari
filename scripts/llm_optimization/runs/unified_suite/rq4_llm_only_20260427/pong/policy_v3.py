"""
Auto-generated policy v3
Generated at: 2026-04-27 17:22:10
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
CENTER_Y = 0.5 * (PLAYFIELD_TOP + PLAYFIELD_BOTTOM)


def init_params():
    return {
        "dead_zone": 5.0,
        "aim_offset": 7.0,
        "aim_align_thresh": 6.0,
        "idle_center_blend": 0.35,
        "track_lead": 4.0,
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

    moving_toward = ball_dx > 0.0

    # Forward intercept with wall reflection (the strong part)
    forward_intercept = predict_intercept(
        curr_ball_x, curr_ball_y, ball_dx, ball_dy, INTERCEPT_X
    )

    # Away-phase: pre-position defensively. Blend ball-y (with small lead)
    # toward field center so we recover quickly to hit either return.
    track_y = curr_ball_y + params["track_lead"] * ball_dy
    blend = params["idle_center_blend"]
    defensive_y = (1.0 - blend) * track_y + blend * CENTER_Y

    base_target = jnp.where(moving_toward, forward_intercept, defensive_y)

    # Aim offset: only when already aligned and ball is approaching.
    # Hit with edge opposite to incoming dy for spiky returns.
    error_pre = base_target - paddle_center
    aligned = jnp.abs(error_pre) < params["aim_align_thresh"]
    aim_sign = jnp.where(ball_dy >= 0.0, -1.0, 1.0)
    aim_bias = jnp.where(moving_toward & aligned,
                         aim_sign * params["aim_offset"],
                         0.0)

    target = base_target + aim_bias

    # Inactive ball: hold center
    target = jnp.where(curr_ball_active > 0.5, target, CENTER_Y)

    # Clamp target
    target = jnp.clip(
        target,
        PLAYFIELD_TOP + PADDLE_CENTER_OFFSET,
        PLAYFIELD_BOTTOM - PADDLE_CENTER_OFFSET,
    )

    error = target - paddle_center
    abs_err = jnp.abs(error)
    in_dead = abs_err < params["dead_zone"]
    move_up = error < 0.0

    # FIRE only when ball inactive (to serve) or when actively contesting
    # an approaching ball; otherwise plain movement.
    contact_zone = moving_toward & (curr_ball_x > 110.0)
    move_action = jnp.where(
        contact_zone,
        jnp.where(move_up, FIRE_UP, FIRE_DOWN),
        jnp.where(move_up, MOVE_UP, MOVE_DOWN),
    )
    idle_action = jnp.where(curr_ball_active > 0.5, NOOP, FIRE)

    action = jnp.where(in_dead, idle_action, move_action)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
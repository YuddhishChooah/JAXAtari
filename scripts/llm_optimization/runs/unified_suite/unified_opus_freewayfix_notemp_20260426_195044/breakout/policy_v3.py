"""
Auto-generated policy v3
Generated at: 2026-04-26 20:52:48
"""

import jax
import jax.numpy as jnp

# Action constants
NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3

# Observation indices
PLAYER_X = 0
BALL_X = 8
BALL_Y = 9
BALL_ACTIVE = 12

# Geometry
PADDLE_HALF = 8.0
SCREEN_MID = 80.0  # Atari Breakout playfield is ~160 wide


def init_params():
    return {
        "dead_zone_high": jnp.array(5.0),
        "dead_zone_low": jnp.array(1.5),
        "y_threshold": jnp.array(120.0),
        "aim_offset": jnp.array(3.0),
        "home_x": jnp.array(80.0),
        "fire_bias": jnp.array(0.5),
    }


def _select_move(err, dead_zone):
    go_right = err > dead_zone
    go_left = err < -dead_zone
    return jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))


def _descent_factor(ball_y, y_threshold):
    # 0 when ball is high, 1 when ball is at/below threshold (descending into paddle zone)
    return jnp.clip(ball_y / jnp.maximum(y_threshold, 1.0), 0.0, 1.0)


def _target_x(ball_x, descent, aim_offset, home_x):
    # When ball is descending, aim at ball_x with a small steering offset
    # that pushes the paddle slightly toward the side opposite the ball,
    # so the ball gets struck off-center and reaches the wall side.
    # Sign: if ball is left of mid, aim a bit further left; if right, further right.
    side = jnp.sign(ball_x - SCREEN_MID)
    aim_target = ball_x + side * aim_offset

    # When ball is high, drift toward home_x to prepare.
    return descent * aim_target + (1.0 - descent) * home_x


def policy(obs_flat, params):
    paddle_x = obs_flat[PLAYER_X]
    ball_x = obs_flat[BALL_X]
    ball_y = obs_flat[BALL_Y]
    ball_active = obs_flat[BALL_ACTIVE]

    paddle_center = paddle_x + PADDLE_HALF

    descent = _descent_factor(ball_y, params["y_threshold"])

    target_x = _target_x(
        ball_x, descent, params["aim_offset"], params["home_x"]
    )

    err = target_x - paddle_center

    # Dead zone shrinks as ball descends (more decisive when urgent).
    dz = (1.0 - descent) * params["dead_zone_high"] + descent * params["dead_zone_low"]
    dz = jnp.maximum(dz, 0.5)

    move_action = _select_move(err, dz)

    need_fire = ball_active < params["fire_bias"]
    action = jnp.where(need_fire, FIRE, move_action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
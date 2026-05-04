"""
Auto-generated policy v5
Generated at: 2026-04-28 18:13:05
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
LIVES = 124

# Screen geometry (Atari Breakout playfield)
SCREEN_CENTER_X = 76.0


def init_params():
    return {
        "dead_zone": jnp.array(2.5),
        "paddle_offset": jnp.array(8.0),
        "lead_gain": jnp.array(0.25),
        "urgency_y": jnp.array(120.0),
        "tight_factor": jnp.array(0.4),
    }


def _urgency(ball_y, urgency_y):
    # 0 when ball is high, 1 when ball is at/below urgency_y level.
    return jnp.clip(ball_y / urgency_y, 0.0, 1.0)


def _target_x(ball_x, paddle_center, urgency, lead_gain):
    # Predictive bias: push target further in the direction the ball
    # is displaced from the paddle, scaled by descent urgency.
    displacement = ball_x - paddle_center
    return ball_x + lead_gain * urgency * displacement


def _move_action(paddle_center, target, dead_zone):
    err = target - paddle_center
    go_right = err > dead_zone
    go_left = err < -dead_zone
    a = jnp.where(go_right, RIGHT, NOOP)
    a = jnp.where(go_left, LEFT, a)
    return a


def policy(obs_flat, params):
    ball_active = obs_flat[BALL_ACTIVE]
    ball_x = obs_flat[BALL_X]
    ball_y = obs_flat[BALL_Y]
    paddle_center = obs_flat[PLAYER_X] + params["paddle_offset"]

    urgency = _urgency(ball_y, params["urgency_y"])

    # Adaptive dead zone: tighter when ball is descending (urgency high),
    # looser when ball is high up (prevents jitter).
    tight = params["tight_factor"]
    dz_scale = 1.0 - (1.0 - tight) * urgency
    dead_zone = params["dead_zone"] * dz_scale

    target = _target_x(ball_x, paddle_center, urgency, params["lead_gain"])

    move = _move_action(paddle_center, target, dead_zone)

    # When ball inactive: center the paddle and FIRE to launch.
    center_move = _move_action(paddle_center, SCREEN_CENTER_X, params["dead_zone"])
    centered = jnp.abs(paddle_center - SCREEN_CENTER_X) <= params["dead_zone"]
    pre_launch = jnp.where(centered, FIRE, center_move)

    action = jnp.where(ball_active > 0.5, move, pre_launch)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
"""
Auto-generated policy v2
Generated at: 2026-04-28 17:43:39
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
BALL_ORIENT = 15
LIVES = 124

SCREEN_CENTER = 80.0


def init_params():
    return {
        "paddle_offset": jnp.array(8.0),
        "dead_zone_base": jnp.array(3.0),
        "dead_zone_urgency": jnp.array(2.0),
        "lead_gain": jnp.array(6.0),
        "urgency_y": jnp.array(140.0),
        "center_bias": jnp.array(0.3),
    }


def _urgency(ball_y, urgency_y):
    return jnp.clip(ball_y / urgency_y, 0.0, 1.0)


def _direction_proxy(obs):
    # Use orientation if it carries sign info; fall back to off-center sign.
    orient = obs[BALL_ORIENT]
    off_center = obs[BALL_X] - SCREEN_CENTER
    # Combine: orientation term plus a small off-center contribution.
    return jnp.tanh(orient) + 0.01 * off_center


def _target_x(obs, params):
    ball_x = obs[BALL_X]
    ball_y = obs[BALL_Y]
    u = _urgency(ball_y, params["urgency_y"])
    direction = _direction_proxy(obs)
    lead = params["lead_gain"] * u * direction
    # When ball is high, bias target toward center to reduce wasted travel.
    high_factor = 1.0 - u
    center_pull = params["center_bias"] * high_factor * (SCREEN_CENTER - ball_x)
    return ball_x + lead + center_pull


def _move_action(paddle_center, target, dead_zone):
    err = target - paddle_center
    go_right = err > dead_zone
    go_left = err < -dead_zone
    a = jnp.where(go_right, RIGHT, NOOP)
    a = jnp.where(go_left, LEFT, a)
    return a


def _adaptive_dead_zone(ball_y, params):
    u = _urgency(ball_y, params["urgency_y"])
    # Larger dead zone when ball is high (less jitter), smaller when descending.
    dz = params["dead_zone_base"] - params["dead_zone_urgency"] * u
    return jnp.maximum(dz, 0.5)


def policy(obs_flat, params):
    ball_active = obs_flat[BALL_ACTIVE]
    paddle_center = obs_flat[PLAYER_X] + params["paddle_offset"]
    target = _target_x(obs_flat, params)
    dz = _adaptive_dead_zone(obs_flat[BALL_Y], params)

    move = _move_action(paddle_center, target, dz)
    action = jnp.where(ball_active > 0.5, move, FIRE)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
"""
Auto-generated policy v4
Generated at: 2026-04-28 18:02:33
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
BALL_STATE = 14
BALL_ORIENT = 15
LIVES = 124


def init_params():
    return {
        "dead_zone": jnp.array(2.0),
        "paddle_offset": jnp.array(8.0),
        "lead_gain": jnp.array(4.0),
        "aim_bias": jnp.array(0.0),
        "urgency_y": jnp.array(120.0),
        "far_dz_mult": jnp.array(2.0),
    }


def _direction_sign(obs):
    # Use ball orientation as a direction proxy in [-1, 1].
    # Many JAX Atari encodings keep orientation small; tanh squashes safely.
    o = obs[BALL_ORIENT]
    s = obs[BALL_STATE]
    return jnp.tanh(o) + 0.0 * s


def _target_x(obs, params):
    ball_x = obs[BALL_X]
    ball_y = obs[BALL_Y]
    urgency = jnp.clip(ball_y / params["urgency_y"], 0.0, 1.0)
    # Lead in the ball's travel direction, stronger when descending.
    direction = _direction_sign(obs)
    lead = params["lead_gain"] * urgency * direction
    # Aim bias is applied more when ball is high (to angle returns),
    # and faded out as the ball descends (precise interception).
    aim = params["aim_bias"] * (1.0 - urgency)
    return ball_x + lead + aim


def _adaptive_dead_zone(obs, params):
    ball_y = obs[BALL_Y]
    urgency = jnp.clip(ball_y / params["urgency_y"], 0.0, 1.0)
    # Wider dead zone when ball is far (low urgency), tighter when close.
    dz = params["dead_zone"] * (1.0 + (params["far_dz_mult"] - 1.0) * (1.0 - urgency))
    return dz


def _move_action(paddle_center, target, dead_zone):
    err = target - paddle_center
    go_right = err > dead_zone
    go_left = err < -dead_zone
    a = jnp.where(go_right, RIGHT, NOOP)
    a = jnp.where(go_left, LEFT, a)
    return a


def policy(obs_flat, params):
    ball_active = obs_flat[BALL_ACTIVE]
    paddle_center = obs_flat[PLAYER_X] + params["paddle_offset"]
    target = _target_x(obs_flat, params)
    dz = _adaptive_dead_zone(obs_flat, params)

    move = _move_action(paddle_center, target, dz)
    action = jnp.where(ball_active > 0.5, move, FIRE)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
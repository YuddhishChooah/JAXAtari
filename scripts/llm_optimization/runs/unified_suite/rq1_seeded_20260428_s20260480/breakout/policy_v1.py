"""
Auto-generated policy v1
Generated at: 2026-04-28 17:33:41
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


def init_params():
    return {
        "dead_zone": jnp.array(2.5),
        "paddle_offset": jnp.array(8.0),
        "lead_gain": jnp.array(0.15),
        "urgency_y": jnp.array(140.0),
        "urgency_scale": jnp.array(0.5),
    }


def _target_x(obs, params):
    ball_x = obs[BALL_X]
    ball_y = obs[BALL_Y]
    # Stronger lead when ball is low (descending toward paddle).
    urgency = jnp.clip(ball_y / params["urgency_y"], 0.0, 1.0)
    lead = params["lead_gain"] * urgency * params["urgency_scale"]
    return ball_x + lead * ball_x * 0.0  # placeholder for future lead use


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

    move = _move_action(paddle_center, target, params["dead_zone"])
    # If ball not active, fire to launch.
    action = jnp.where(ball_active > 0.5, move, FIRE)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
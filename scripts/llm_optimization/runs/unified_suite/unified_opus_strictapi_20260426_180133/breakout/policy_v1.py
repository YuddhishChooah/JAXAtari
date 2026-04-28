"""
Auto-generated policy v1
Generated at: 2026-04-26 19:03:33
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

# Paddle half-width (paddle width ~16)
PADDLE_HALF = 8.0


def init_params():
    return {
        "dead_zone": jnp.array(3.0),
        "lead": jnp.array(0.15),
        "urgency_y": jnp.array(140.0),
        "urgency_gain": jnp.array(0.5),
        "fire_bias": jnp.array(1.0),
    }


def _select_move(error, dead_zone):
    # error > 0 means target is to the right
    go_right = error > dead_zone
    go_left = error < -dead_zone
    action = jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))
    return action


def policy(obs_flat, params):
    paddle_center = obs_flat[PLAYER_X] + PADDLE_HALF
    ball_x = obs_flat[BALL_X]
    ball_y = obs_flat[BALL_Y]
    ball_active = obs_flat[BALL_ACTIVE]

    # Lead term: shift target slightly based on ball y (descending => more urgent)
    # Normalize y roughly into [0,1]
    y_norm = ball_y / params["urgency_y"]
    urgency = 1.0 + params["urgency_gain"] * y_norm

    # Target is ball x (with simple lead based on offset from paddle scaled by urgency)
    raw_error = (ball_x - paddle_center) * urgency
    # Slight lead toward ball x direction
    target_error = raw_error + params["lead"] * (ball_x - paddle_center)

    move_action = _select_move(target_error, params["dead_zone"])

    # If ball not active, FIRE to launch (with a small bias parameter to keep it tunable)
    fire_when_inactive = params["fire_bias"] > 0.0
    inactive = ball_active < 0.5
    action = jnp.where(
        inactive & fire_when_inactive,
        FIRE,
        move_action,
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
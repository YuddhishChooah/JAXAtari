"""
Auto-generated policy v1
Generated at: 2026-04-26 20:47:15
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
        "fire_bias": jnp.array(0.5),
    }


def _select_move(err, dead_zone):
    # err > 0 means target is to the right of paddle
    go_right = err > dead_zone
    go_left = err < -dead_zone
    action = jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))
    return action


def policy(obs_flat, params):
    paddle_x = obs_flat[PLAYER_X]
    ball_x = obs_flat[BALL_X]
    ball_y = obs_flat[BALL_Y]
    ball_active = obs_flat[BALL_ACTIVE]

    paddle_center = paddle_x + PADDLE_HALF

    # Lead target: a small bias toward where ball will be (using y as urgency proxy).
    # When ball_y is large (near paddle), urgency increases.
    urgency = jnp.clip(ball_y / params["urgency_y"], 0.0, 1.5)
    target_x = ball_x + params["lead"] * (ball_x - paddle_center) * urgency

    err = target_x - paddle_center

    move_action = _select_move(err, params["dead_zone"])

    # If ball is not active, fire to launch.
    need_fire = ball_active < params["fire_bias"]
    action = jnp.where(need_fire, FIRE, move_action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
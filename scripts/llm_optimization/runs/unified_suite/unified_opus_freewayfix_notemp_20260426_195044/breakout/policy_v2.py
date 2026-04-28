"""
Auto-generated policy v2
Generated at: 2026-04-26 20:50:00
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
PLAYER_Y = 1
BALL_X = 8
BALL_Y = 9
BALL_ACTIVE = 12
LIVES = 124

# Geometry
PADDLE_HALF = 8.0
SCREEN_CENTER_X = 80.0


def init_params():
    return {
        "dead_zone": jnp.array(2.5),
        "descend_dead_zone": jnp.array(1.2),
        "aim_offset": jnp.array(3.0),
        "offset_side_gain": jnp.array(1.0),
        "urgency_y": jnp.array(140.0),
        "fire_bias": jnp.array(0.85),
    }


def _select_move(err, dead_zone):
    go_right = err > dead_zone
    go_left = err < -dead_zone
    return jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))


def _aim_target(ball_x, ball_y, params):
    # Sign: push ball toward the side with more screen room (away from center).
    # If ball is on the right half, hit it on the right side of paddle to send it left, and vice versa.
    side = jnp.sign(ball_x - SCREEN_CENTER_X)
    # Urgency rises as ball descends toward paddle.
    urgency = jnp.clip(ball_y / params["urgency_y"], 0.0, 1.0)
    offset = params["aim_offset"] * params["offset_side_gain"] * side * urgency
    # target_x is where we want the paddle CENTER to be.
    # To strike the ball with an off-center contact, paddle_center = ball_x - offset.
    return ball_x - offset, urgency


def policy(obs_flat, params):
    paddle_x = obs_flat[PLAYER_X]
    ball_x = obs_flat[BALL_X]
    ball_y = obs_flat[BALL_Y]
    ball_active = obs_flat[BALL_ACTIVE]

    paddle_center = paddle_x + PADDLE_HALF

    target_x, urgency = _aim_target(ball_x, ball_y, params)
    err = target_x - paddle_center

    # Tighten dead zone when ball is descending fast / near paddle.
    dz = params["dead_zone"] * (1.0 - urgency) + params["descend_dead_zone"] * urgency
    dz = jnp.maximum(dz, 0.3)

    move_action = _select_move(err, dz)

    need_fire = ball_active < params["fire_bias"]
    action = jnp.where(need_fire, FIRE, move_action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
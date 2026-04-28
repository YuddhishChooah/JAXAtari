"""
Auto-generated policy v2
Generated at: 2026-04-26 19:06:04
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

PADDLE_HALF = 8.0
SCREEN_BOTTOM = 195.0
SCREEN_CENTER = 80.0


def init_params():
    return {
        "dead_zone": jnp.array(5.0),
        "lead": jnp.array(0.35),
        "descend_y": jnp.array(100.0),
        "descend_gain": jnp.array(1.5),
        "fire_bias": jnp.array(1.0),
        "center_pull": jnp.array(0.2),
    }


def _vx_from_orient(orient):
    # orientation field: try to extract a horizontal sign in [-1, 1].
    # Many JAXAtari encodings use small integer codes; sign() gives a robust proxy,
    # and a tanh keeps magnitude bounded if the field is continuous.
    return jnp.tanh(orient)


def _select_move(error, dead_zone):
    go_right = error > dead_zone
    go_left = error < -dead_zone
    return jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))


def policy(obs_flat, params):
    paddle_center = obs_flat[PLAYER_X] + PADDLE_HALF
    ball_x = obs_flat[BALL_X]
    ball_y = obs_flat[BALL_Y]
    ball_active = obs_flat[BALL_ACTIVE]
    ball_orient = obs_flat[BALL_ORIENT]

    # Horizontal velocity proxy from orientation field.
    vx_proxy = _vx_from_orient(ball_orient)

    # Time-to-impact proxy: how far ball still has to fall.
    time_to_impact = jnp.maximum(SCREEN_BOTTOM - ball_y, 0.0)

    # Predictive interception target.
    target_x = ball_x + params["lead"] * vx_proxy * time_to_impact

    # One-sided urgency: only amplify error when ball is in lower (dangerous) zone.
    descending = ball_y > params["descend_y"]
    urgency = jnp.where(descending, 1.0 + params["descend_gain"], 1.0)

    error = (target_x - paddle_center) * urgency
    move_action = _select_move(error, params["dead_zone"])

    # When ball not active, drift paddle toward screen center for clean launch,
    # and FIRE to launch.
    center_error = (SCREEN_CENTER - paddle_center) * params["center_pull"]
    center_move = _select_move(center_error, params["dead_zone"])

    inactive = ball_active < 0.5
    fire_on = params["fire_bias"] > 0.0

    # When inactive: alternate centering and firing — prefer FIRE when paddle near center.
    near_center = jnp.abs(SCREEN_CENTER - paddle_center) < (params["dead_zone"] + 2.0)
    inactive_action = jnp.where(near_center & fire_on, FIRE, center_move)

    action = jnp.where(inactive, inactive_action, move_action)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
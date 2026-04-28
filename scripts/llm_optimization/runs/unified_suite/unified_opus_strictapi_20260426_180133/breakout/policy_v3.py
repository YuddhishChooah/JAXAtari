"""
Auto-generated policy v3
Generated at: 2026-04-26 19:08:37
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

PADDLE_HALF = 8.0
SCREEN_CENTER = 80.0


def init_params():
    return {
        "dead_zone_air": jnp.array(5.0),
        "dead_zone_descend": jnp.array(2.5),
        "descend_y": jnp.array(100.0),
        "lead_offset": jnp.array(3.0),
        "center_pull": jnp.array(0.2),
        "launch_tol": jnp.array(6.0),
    }


def _select_move(error, dead_zone):
    go_right = error > dead_zone
    go_left = error < -dead_zone
    return jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))


def policy(obs_flat, params):
    paddle_center = obs_flat[PLAYER_X] + PADDLE_HALF
    ball_x = obs_flat[BALL_X]
    ball_y = obs_flat[BALL_Y]
    ball_active = obs_flat[BALL_ACTIVE]

    # Sign of horizontal offset between ball and paddle: a robust proxy
    # for which side of the paddle the ball is currently on.
    side_sign = jnp.sign(ball_x - paddle_center)

    descending = ball_y > params["descend_y"]

    # When descending, bias the target slightly in the direction the ball is
    # offset from the paddle to encourage steeper return angles.
    # When ascending, just track ball_x directly.
    target_x = jnp.where(
        descending,
        ball_x + params["lead_offset"] * side_sign,
        ball_x,
    )

    # Tighter dead zone near impact for precision; wider during ascent to limit jitter.
    dz = jnp.where(descending, params["dead_zone_descend"], params["dead_zone_air"])

    error = target_x - paddle_center
    move_action = _select_move(error, dz)

    # Inactive ball: drift toward center, FIRE when close enough.
    center_error = (SCREEN_CENTER - paddle_center) * params["center_pull"]
    center_move = _select_move(center_error, params["dead_zone_air"])
    near_center = jnp.abs(SCREEN_CENTER - paddle_center) < params["launch_tol"]
    inactive_action = jnp.where(near_center, FIRE, center_move)

    inactive = ball_active < 0.5
    action = jnp.where(inactive, inactive_action, move_action)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
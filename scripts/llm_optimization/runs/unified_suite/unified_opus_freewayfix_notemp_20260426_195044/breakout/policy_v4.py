"""
Auto-generated policy v4
Generated at: 2026-04-26 20:55:42
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
BALL_ORIENT = 15
LIVES = 124

PADDLE_HALF = 8.0
PADDLE_Y_APPROX = 189.0  # approximate paddle row in Breakout


def init_params():
    return {
        "lead": jnp.array(0.25),
        "dead_zone_near": jnp.array(2.0),
        "dead_zone_far": jnp.array(5.0),
        "y_split": jnp.array(120.0),
        "aim_offset": jnp.array(2.0),
    }


def _select_move(err, dead_zone):
    go_right = err > dead_zone
    go_left = err < -dead_zone
    return jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))


def _orient_sign(orient):
    # Map ball orientation to a horizontal direction sign in {-1, 0, +1}.
    # Many Atari encodings store direction as a small int; sign() is robust.
    s = jnp.sign(orient - 0.5)  # treat 0 as left-ish, >0 as right-ish baseline
    return jnp.where(orient == 0, -1.0, jnp.where(orient > 0, 1.0, -1.0))


def policy(obs_flat, params):
    paddle_x = obs_flat[PLAYER_X]
    ball_x = obs_flat[BALL_X]
    ball_y = obs_flat[BALL_Y]
    ball_active = obs_flat[BALL_ACTIVE]
    ball_orient = obs_flat[BALL_ORIENT]

    paddle_center = paddle_x + PADDLE_HALF

    # Linear projection: predict ball x at paddle y using orientation as direction sign.
    dy = jnp.maximum(PADDLE_Y_APPROX - ball_y, 0.0)
    dir_sign = _orient_sign(ball_orient)
    predicted_x = ball_x + dir_sign * params["lead"] * dy

    # Aim offset: bias contact point off-center to vary bounce angle.
    # Use direction sign so we aim into the ball's travel direction.
    target_x = predicted_x + dir_sign * params["aim_offset"]

    err = target_x - paddle_center

    # Stratified dead zone: tighter when ball is near paddle (descending phase).
    near = ball_y > params["y_split"]
    dead_zone = jnp.where(near, params["dead_zone_near"], params["dead_zone_far"])

    move_action = _select_move(err, dead_zone)

    # FIRE only when ball is clearly inactive.
    need_fire = ball_active < 0.5
    action = jnp.where(need_fire, FIRE, move_action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
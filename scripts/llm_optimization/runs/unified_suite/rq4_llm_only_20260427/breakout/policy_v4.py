"""
Auto-generated policy v4
Generated at: 2026-04-27 17:46:45
"""

import jax
import jax.numpy as jnp


# Action ids
NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3

# Observation indices
PLAYER_X = 0
PLAYER_Y = 1
PLAYER_W = 2
BALL_X = 8
BALL_Y = 9
BALL_ACTIVE = 12
BALL_STATE = 14
BALL_ORIENT = 15
LIVES = 124


def init_params():
    return {
        "dead_zone": jnp.float32(1.5),
        "lead_gain": jnp.float32(0.42),
        "anchor_x": jnp.float32(76.0),
        "anchor_pull": jnp.float32(0.35),
        "descend_y": jnp.float32(110.0),
        "aim_offset": jnp.float32(3.0),
    }


def _paddle_center(obs):
    return obs[PLAYER_X] + obs[PLAYER_W] * 0.5


def _move_action(err, dead_zone):
    go_right = err > dead_zone
    go_left = err < -dead_zone
    return jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))


def _dir_sign(orient, ball_x, anchor_x):
    # orientation field may encode horizontal direction; fall back to
    # sign of (ball_x - anchor) which correlates with where bounce sends it.
    o = jnp.where(jnp.abs(orient) > 1e-3, jnp.sign(orient), 0.0)
    fallback = jnp.sign(ball_x - anchor_x)
    return jnp.where(jnp.abs(o) > 0.0, o, fallback)


def policy(obs_flat, params):
    obs = obs_flat.astype(jnp.float32)

    paddle_cx = _paddle_center(obs)
    paddle_y = obs[PLAYER_Y]
    ball_x = obs[BALL_X]
    ball_y = obs[BALL_Y]
    ball_active = obs[BALL_ACTIVE] > 0.5
    ball_orient = obs[BALL_ORIENT]

    anchor_x = params["anchor_x"]
    lead_gain = params["lead_gain"]
    descend_y = params["descend_y"]
    anchor_pull = params["anchor_pull"]
    aim_offset = params["aim_offset"]
    dead_zone = params["dead_zone"]

    # Vertical distance from ball to paddle (positive when ball above paddle).
    dy = jnp.maximum(paddle_y - ball_y, 0.0)

    # Direction proxy for horizontal motion.
    dir_sign = _dir_sign(ball_orient, ball_x, anchor_x)

    # Predicted intercept x = ball_x + slope * dy.
    predicted_x = ball_x + lead_gain * dir_sign * dy

    # Aim slightly off-center on the paddle to impart angle: bias target
    # opposite to direction so ball is hit on trailing side.
    aim_bias = -aim_offset * dir_sign
    intercept_target = predicted_x + aim_bias

    # Descending regime: ball below descend_y threshold OR moving down.
    # We approximate "descending" by ball being in lower half of screen.
    descending = ball_y > descend_y

    # When ascending / high: blend toward anchor for pre-positioning.
    # When descending: commit fully to intercept.
    anchored = (1.0 - anchor_pull) * intercept_target + anchor_pull * anchor_x
    target = jnp.where(descending, intercept_target, anchored)

    # Asymmetric dead zone: tighter when descending, looser when high.
    dz = jnp.where(descending, dead_zone, dead_zone * 2.5)

    err = target - paddle_cx
    move = _move_action(err, dz)

    action = jnp.where(ball_active, move, FIRE)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
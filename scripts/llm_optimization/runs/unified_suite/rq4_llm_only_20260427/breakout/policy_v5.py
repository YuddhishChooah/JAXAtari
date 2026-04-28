"""
Auto-generated policy v5
Generated at: 2026-04-27 17:47:36
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
PLAYER_W = 2
BALL_X = 8
BALL_Y = 9
BALL_ACTIVE = 12
LIVES = 124


def init_params():
    return {
        "dead_zone": jnp.float32(3.0),
        "lead_gain": jnp.float32(0.35),
        "descent_y": jnp.float32(110.0),
        "urgency_sharp": jnp.float32(0.08),
        "anchor_x": jnp.float32(80.0),
        "edge_bias": jnp.float32(4.0),
    }


def _paddle_center(obs):
    return obs[PLAYER_X] + obs[PLAYER_W] * 0.5


def _move_action(err, dead_zone):
    go_right = err > dead_zone
    go_left = err < -dead_zone
    return jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))


def _urgency(ball_y, descent_y, sharp):
    # Sigmoid-like ramp: ~0 above descent_y line, ~1 well below it.
    z = (ball_y - descent_y) * sharp
    return jax.nn.sigmoid(z)


def policy(obs_flat, params):
    obs = obs_flat.astype(jnp.float32)

    paddle_cx = _paddle_center(obs)
    ball_x = obs[BALL_X]
    ball_y = obs[BALL_Y]
    ball_active = obs[BALL_ACTIVE] > 0.5

    anchor_x = params["anchor_x"]

    # Directional bias: push paddle slightly toward the side the ball is on,
    # giving stateless anticipation of where the ball will land.
    side = jnp.sign(ball_x - anchor_x)
    edge = params["edge_bias"] * side

    # Position-proportional lead amplifies tracking when ball is far.
    lead = params["lead_gain"] * (ball_x - paddle_cx)

    # Tracking target with anticipatory bias.
    track_target = ball_x + lead + edge

    # Urgency: sharp, asymmetric — committed to anchor when ball is high,
    # committed to track when ball is descending toward paddle.
    u = _urgency(ball_y, params["descent_y"], params["urgency_sharp"])

    target = u * track_target + (1.0 - u) * anchor_x

    err = target - paddle_cx
    move = _move_action(err, params["dead_zone"])

    action = jnp.where(ball_active, move, FIRE)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
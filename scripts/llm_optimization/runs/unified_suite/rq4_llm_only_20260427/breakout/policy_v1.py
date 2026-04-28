"""
Auto-generated policy v1
Generated at: 2026-04-27 17:43:25
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
        "dead_zone": jnp.float32(2.0),
        "lead": jnp.float32(0.18),
        "anchor_x": jnp.float32(80.0),
        "anchor_pull": jnp.float32(0.05),
    }


def _paddle_center(obs):
    return obs[PLAYER_X] + obs[PLAYER_W] * 0.5


def _move_action(err, dead_zone):
    # err = target_x - paddle_center; positive => move RIGHT
    go_right = err > dead_zone
    go_left = err < -dead_zone
    a = jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))
    return a


def policy(obs_flat, params):
    obs = obs_flat.astype(jnp.float32)

    paddle_cx = _paddle_center(obs)
    ball_x = obs[BALL_X]
    ball_y = obs[BALL_Y]
    ball_active = obs[BALL_ACTIVE] > 0.5

    # Approximate ball horizontal lead using ball_x relative to paddle.
    # When ball is far from paddle horizontally, lead a bit toward it.
    lead = params["lead"] * (ball_x - paddle_cx)

    # Anchor toward center of play area when ball is high (less urgent),
    # so the paddle returns to a good default position.
    # Urgency rises as ball descends. ball_y ~ small at top, larger near paddle.
    # We blend target between (ball_x + lead) and anchor_x.
    # Use a simple normalized urgency in [0,1] without extra params.
    urgency = jnp.clip(ball_y / 180.0, 0.0, 1.0)

    target_when_active = ball_x + lead
    anchored = (1.0 - params["anchor_pull"]) * target_when_active \
               + params["anchor_pull"] * params["anchor_x"]
    # Mix: when urgent (ball low), trust ball_x more directly.
    target = urgency * target_when_active + (1.0 - urgency) * anchored

    err = target - paddle_cx
    move = _move_action(err, params["dead_zone"])

    # If ball is not active (waiting to launch), FIRE.
    action = jnp.where(ball_active, move, FIRE)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
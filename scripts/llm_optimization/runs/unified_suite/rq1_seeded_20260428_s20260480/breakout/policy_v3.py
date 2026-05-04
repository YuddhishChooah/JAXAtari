"""
Auto-generated policy v3
Generated at: 2026-04-28 17:53:08
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
BALL_STATE = 14
BALL_ORIENT = 15
LIVES = 124


def init_params():
    return {
        "dead_zone": jnp.array(2.5),
        "paddle_offset": jnp.array(7.9),
        "lead_gain": jnp.array(6.0),
        "urgency_y": jnp.array(120.0),
        "fire_y": jnp.array(40.0),
    }


def _orient_sign(orient):
    # Map orientation value to a sign in [-1, 1]; treat 0 as neutral.
    return jnp.sign(orient)


def _urgency(ball_y, urgency_y):
    # 0 when ball is high (safe), 1 when ball is at/below urgency_y (danger).
    return jnp.clip(ball_y / jnp.maximum(urgency_y, 1.0), 0.0, 1.0)


def _target_x(obs, params):
    ball_x = obs[BALL_X]
    ball_y = obs[BALL_Y]
    orient = obs[BALL_ORIENT]
    u = _urgency(ball_y, params["urgency_y"])
    # Lead in the direction the ball is moving, more aggressively when descending.
    lead = params["lead_gain"] * _orient_sign(orient) * u
    return ball_x + lead


def _move_action(paddle_center, target, eff_dead_zone):
    err = target - paddle_center
    go_right = err > eff_dead_zone
    go_left = err < -eff_dead_zone
    a = jnp.where(go_right, RIGHT, NOOP)
    a = jnp.where(go_left, LEFT, a)
    return a


def _should_fire(obs, fire_y):
    ball_active = obs[BALL_ACTIVE]
    ball_y = obs[BALL_Y]
    not_active = ball_active < 0.5
    # Also fire if the ball appears stuck near the top (relaunch guard).
    stuck_top = ball_y < fire_y
    return jnp.logical_or(not_active, stuck_top)


def policy(obs_flat, params):
    paddle_center = obs_flat[PLAYER_X] + params["paddle_offset"]
    target = _target_x(obs_flat, params)

    # Effective dead zone shrinks when urgency is high (ball low/descending).
    u = _urgency(obs_flat[BALL_Y], params["urgency_y"])
    eff_dead_zone = params["dead_zone"] * (1.0 - 0.75 * u)

    move = _move_action(paddle_center, target, eff_dead_zone)

    fire_now = _should_fire(obs_flat, params["fire_y"])
    action = jnp.where(fire_now, FIRE, move)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
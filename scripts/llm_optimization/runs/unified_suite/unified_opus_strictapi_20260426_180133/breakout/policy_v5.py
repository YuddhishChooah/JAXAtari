"""
Auto-generated policy v5
Generated at: 2026-04-26 19:13:49
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
BALL_Y_TOP = 30.0
BALL_Y_BOTTOM = 190.0


def init_params():
    return {
        "dead_zone": jnp.array(3.0),
        "descend_weight": jnp.array(2.0),
        "ascend_relax": jnp.array(0.4),
        "center_offset": jnp.array(0.0),
        "launch_offset": jnp.array(6.0),
        "track_gain": jnp.array(1.0),
    }


def _select_move(error, dead_zone):
    go_right = error > dead_zone
    go_left = error < -dead_zone
    return jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))


def _descend_weight(ball_y, descend_w, ascend_w):
    # Smooth ramp in [0, 1] over the vertical play area.
    frac = (ball_y - BALL_Y_TOP) / (BALL_Y_BOTTOM - BALL_Y_TOP)
    frac = jnp.clip(frac, 0.0, 1.0)
    # Blend: ascending (top) uses ascend_w, descending (bottom) uses descend_w.
    return ascend_w + (descend_w - ascend_w) * frac


def policy(obs_flat, params):
    paddle_center = obs_flat[PLAYER_X] + PADDLE_HALF
    ball_x = obs_flat[BALL_X]
    ball_y = obs_flat[BALL_Y]
    ball_active = obs_flat[BALL_ACTIVE]

    # Smooth urgency weight: small when ball is high, large when descending.
    weight = _descend_weight(ball_y, params["descend_weight"], params["ascend_relax"])

    # Tracking error scaled by urgency and a global gain.
    raw_error = ball_x - paddle_center
    error = raw_error * weight * params["track_gain"]

    # Effective dead zone shrinks when ball is dangerous (weight large).
    eff_dead_zone = params["dead_zone"] / jnp.maximum(weight, 0.25)

    move_action = _select_move(error, eff_dead_zone)

    # Inactive ball: aim slightly off-center to vary launch angle, then FIRE.
    launch_target = SCREEN_CENTER + params["launch_offset"]
    launch_error = launch_target - paddle_center
    near_launch = jnp.abs(launch_error) < (params["dead_zone"] + 2.0)
    centering_move = _select_move(-launch_error, params["dead_zone"])
    # When near launch position, FIRE; otherwise move toward it.
    inactive_action = jnp.where(near_launch, FIRE, jnp.where(launch_error > 0, RIGHT, LEFT))
    # Suppress spurious centering_move warning; use it as a fallback.
    inactive_action = jnp.where(near_launch, FIRE, jnp.where(launch_error > params["dead_zone"], RIGHT,
                                  jnp.where(launch_error < -params["dead_zone"], LEFT, FIRE)))

    inactive = ball_active < 0.5
    action = jnp.where(inactive, inactive_action, move_action)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
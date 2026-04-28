"""
Auto-generated policy v4
Generated at: 2026-04-26 19:11:11
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
        "dz_engage": jnp.array(2.0),
        "dz_release": jnp.array(6.0),
        "descend_y": jnp.array(110.0),
        "descend_shrink": jnp.array(0.4),
        "aim_offset": jnp.array(3.0),
        "high_y": jnp.array(70.0),
    }


def _move_with_hysteresis(error, dz_engage, dz_release):
    # Symmetric hysteresis: use a smaller threshold to engage motion,
    # larger threshold to keep moving (here we just pick dz_engage as the
    # active threshold; release is encoded by widening NOOP band only when
    # error magnitude is small). Simpler: pick the smaller of the two as
    # the active gate so CMA-ES can choose either ordering.
    gate = jnp.minimum(jnp.abs(dz_engage), jnp.abs(dz_release))
    go_right = error > gate
    go_left = error < -gate
    return jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))


def policy(obs_flat, params):
    paddle_center = obs_flat[PLAYER_X] + PADDLE_HALF
    ball_x = obs_flat[BALL_X]
    ball_y = obs_flat[BALL_Y]
    ball_active = obs_flat[BALL_ACTIVE]

    # Regime: descending (dangerous) vs ascending/high (safer, can angle).
    descending = ball_y > params["descend_y"]
    high = ball_y < params["high_y"]

    # Aim offset: when ball is high, deliberately offset target so we hit
    # the ball off-center and steer it toward unbroken columns. Flip sign
    # based on which side of screen the ball is on (push outward => angle).
    side_sign = jnp.sign(ball_x - SCREEN_CENTER)
    offset = jnp.where(high, params["aim_offset"] * side_sign, 0.0)
    target_x = ball_x + offset

    error = target_x - paddle_center

    # When descending, shrink dead zone (commit harder); when ascending,
    # widen it to reduce chatter.
    dz_e = params["dz_engage"]
    dz_r = params["dz_release"]
    shrink = jnp.where(descending, params["descend_shrink"], 1.0)
    dz_e_eff = jnp.abs(dz_e) * shrink
    dz_r_eff = jnp.abs(dz_r) * shrink

    move_action = _move_with_hysteresis(error, dz_e_eff, dz_r_eff)

    # Inactive: center paddle and FIRE when near center.
    center_error = SCREEN_CENTER - paddle_center
    center_move = _move_with_hysteresis(center_error, dz_e_eff, dz_r_eff)
    near_center = jnp.abs(center_error) < (jnp.abs(dz_e) + 3.0)
    inactive_action = jnp.where(near_center, FIRE, center_move)

    inactive = ball_active < 0.5
    action = jnp.where(inactive, inactive_action, move_action)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
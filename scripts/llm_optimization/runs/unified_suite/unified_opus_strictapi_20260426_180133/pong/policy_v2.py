"""
Auto-generated policy v2
Generated at: 2026-04-26 18:04:09
"""

import jax
import jax.numpy as jnp

NOOP = 0
FIRE = 1
MOVE_UP = 2
MOVE_DOWN = 3
FIRE_UP = 4
FIRE_DOWN = 5

FRAME_SIZE = 26
PLAYER_X = 140.0
ENEMY_X = 16.0
PLAYFIELD_TOP = 24.0
PLAYFIELD_BOTTOM = 194.0


def init_params():
    return {
        "dead_zone": jnp.array(3.0),
        "predict_gain": jnp.array(1.0),
        "aim_spike": jnp.array(4.0),
        "fallback_gain": jnp.array(0.6),
        "fire_window": jnp.array(12.0),
    }


def _reflect_y(raw_y):
    span = PLAYFIELD_BOTTOM - PLAYFIELD_TOP
    rel = jnp.mod(raw_y - PLAYFIELD_TOP, 2.0 * span)
    rel = jnp.where(rel > span, 2.0 * span - rel, rel)
    return PLAYFIELD_TOP + rel


def _predict_y(ball_x, ball_y, ball_dx, ball_dy, target_x, gain):
    dx_safe = jnp.where(jnp.abs(ball_dx) < 1e-2, 1e-2, ball_dx)
    t = (target_x - ball_x) / dx_safe
    t = jnp.clip(t, 0.0, 120.0)
    raw_y = ball_y + ball_dy * t * gain
    return _reflect_y(raw_y)


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    player_y = curr[1]
    player_h = curr[3]
    ball_x = curr[16]
    ball_y = curr[17]
    ball_active = curr[20]
    prev_ball_x = prev[16]
    prev_ball_y = prev[17]

    ball_dx = ball_x - prev_ball_x
    ball_dy = ball_y - prev_ball_y

    eps_dx = 0.25
    moving_toward = ball_dx > eps_dx
    moving_away = ball_dx < -eps_dx

    # Predicted intercept on player side
    pred_player = _predict_y(ball_x, ball_y, ball_dx, ball_dy,
                             PLAYER_X, params["predict_gain"])

    # When ball moves away, predict where the enemy will return it,
    # then mirror back toward player as a soft fallback target.
    pred_enemy = _predict_y(ball_x, ball_y, ball_dx, ball_dy,
                            ENEMY_X, params["predict_gain"])
    fallback_target = ball_y + (pred_enemy - ball_y) * params["fallback_gain"]

    # When ball is essentially stationary in x, drift gently to ball_y
    target_y = jnp.where(moving_toward, pred_player,
                         jnp.where(moving_away, fallback_target, ball_y))

    # Spike aim: bias hit point off-center based on ball vertical direction
    spike = params["aim_spike"] * jnp.sign(ball_dy)
    target_y = target_y + spike * jnp.where(moving_toward, 1.0, 0.0)

    paddle_center = player_y + player_h * 0.5
    error = target_y - paddle_center

    dz = params["dead_zone"]
    move_up = error < -dz
    move_down = error > dz

    # FIRE only near contact and roughly aligned (spike attempt)
    close = (PLAYER_X - ball_x) < params["fire_window"]
    aligned = jnp.abs(error) < (player_h * 0.5 + 2.0)
    fire_active = moving_toward & close & aligned

    serve = ball_active < 0.5

    action = jnp.where(
        serve,
        FIRE,
        jnp.where(
            move_up,
            jnp.where(fire_active, FIRE_UP, MOVE_UP),
            jnp.where(
                move_down,
                jnp.where(fire_active, FIRE_DOWN, MOVE_DOWN),
                jnp.where(fire_active, FIRE, NOOP),
            ),
        ),
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
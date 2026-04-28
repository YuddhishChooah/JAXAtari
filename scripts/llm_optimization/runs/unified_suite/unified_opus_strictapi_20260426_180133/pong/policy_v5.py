"""
Auto-generated policy v5
Generated at: 2026-04-26 18:10:04
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
PLAYFIELD_MID = 0.5 * (PLAYFIELD_TOP + PLAYFIELD_BOTTOM)


def init_params():
    return {
        "dead_zone": jnp.array(2.5),
        "aim_spike": jnp.array(5.0),
        "fire_window": jnp.array(14.0),
        "spike_engage": jnp.array(8.0),
        "home_bias": jnp.array(0.3),
    }


def _reflect_y(raw_y):
    span = PLAYFIELD_BOTTOM - PLAYFIELD_TOP
    rel = jnp.mod(raw_y - PLAYFIELD_TOP, 2.0 * span)
    rel = jnp.where(rel > span, 2.0 * span - rel, rel)
    return PLAYFIELD_TOP + rel


def _predict_intercept(ball_x, ball_y, ball_dx, ball_dy, target_x):
    dx_safe = jnp.where(jnp.abs(ball_dx) < 1e-2,
                        jnp.sign(ball_dx) * 1e-2 + 1e-2, ball_dx)
    t = (target_x - ball_x) / dx_safe
    t = jnp.clip(t, 0.0, 200.0)
    raw_y = ball_y + ball_dy * t
    return _reflect_y(raw_y)


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    player_y = curr[1]
    player_h = curr[3]
    enemy_y = curr[9]
    enemy_h = curr[11]
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

    paddle_center = player_y + player_h * 0.5
    enemy_center = enemy_y + enemy_h * 0.5

    # True intercept on player side (no gain distortion).
    pred_player = _predict_intercept(ball_x, ball_y, ball_dx, ball_dy, PLAYER_X)

    # Defensive home: bias toward mid-court when ball is moving away.
    home_y = PLAYFIELD_MID + (paddle_center - PLAYFIELD_MID) * params["home_bias"]

    # Base target: track intercept when incoming, hold home when outgoing.
    base_target = jnp.where(moving_toward, pred_player,
                            jnp.where(moving_away, home_y, ball_y))

    # Distance to ball along x; smaller -> closer to contact.
    dx_to_player = PLAYER_X - ball_x

    # Spike only when paddle is already aligned with intercept and ball is close.
    err_pre = pred_player - paddle_center
    aligned_pre = jnp.abs(err_pre) < params["spike_engage"]
    close_x = (dx_to_player > 0.0) & (dx_to_player < 40.0)
    # Spike direction: away from enemy paddle (force them to travel).
    spike_sign = jnp.sign(pred_player - enemy_center)
    spike_sign = jnp.where(jnp.abs(spike_sign) < 0.5, 1.0, spike_sign)
    spike_offset = params["aim_spike"] * spike_sign
    apply_spike = moving_toward & aligned_pre & close_x
    target_y = base_target + jnp.where(apply_spike, spike_offset, 0.0)

    error = target_y - paddle_center

    # Asymmetric dead zone: tighter when ball is close & incoming.
    dz_base = params["dead_zone"]
    dz = jnp.where(moving_toward & (dx_to_player < 60.0),
                   dz_base, dz_base + 2.0)

    move_up = error < -dz
    move_down = error > dz

    # FIRE gating: incoming, within window, well aligned.
    in_window = (dx_to_player > 0.0) & (dx_to_player < params["fire_window"])
    aligned_hit = jnp.abs(error) < (player_h * 0.5)
    fire_active = moving_toward & in_window & aligned_hit

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
"""
Auto-generated policy v3
Generated at: 2026-04-26 18:06:19
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
PLAYFIELD_CENTER = 0.5 * (PLAYFIELD_TOP + PLAYFIELD_BOTTOM)
PADDLE_HEIGHT = 16.0


def init_params():
    return {
        "dead_zone": jnp.array(4.0),
        "aim_spike": jnp.array(5.0),
        "recenter_bias": jnp.array(0.6),
        "fire_window": jnp.array(8.0),
        "fire_tol": jnp.array(4.0),
    }


def _reflect_y(raw_y):
    span = PLAYFIELD_BOTTOM - PLAYFIELD_TOP
    rel = jnp.mod(raw_y - PLAYFIELD_TOP, 2.0 * span)
    rel = jnp.where(rel > span, 2.0 * span - rel, rel)
    return PLAYFIELD_TOP + rel


def _intercept(ball_x, ball_y, ball_dx, ball_dy, target_x):
    dx_safe = jnp.where(jnp.abs(ball_dx) < 1e-2,
                        jnp.sign(ball_dx) * 1e-2 + 1e-3, ball_dx)
    t = (target_x - ball_x) / dx_safe
    t = jnp.clip(t, 0.0, 240.0)
    return _reflect_y(ball_y + ball_dy * t)


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

    # Pure geometric intercept on player side (no gain).
    pred_player = _intercept(ball_x, ball_y, ball_dx, ball_dy, PLAYER_X)

    # When ball moves away: predict where enemy hits, then reflect that
    # trajectory back toward the player side, blended with screen center
    # to keep paddle near a defensive resting spot.
    pred_enemy_y = _intercept(ball_x, ball_y, ball_dx, ball_dy, ENEMY_X)
    # Mirror back: assume enemy returns ball roughly straight; aim for
    # the reflected y-line, which is just pred_enemy_y itself as a proxy.
    away_target = (params["recenter_bias"] * PLAYFIELD_CENTER
                   + (1.0 - params["recenter_bias"]) * pred_enemy_y)

    target_y = jnp.where(moving_toward, pred_player,
                         jnp.where(moving_away, away_target, PLAYFIELD_CENTER))

    paddle_center = player_y + player_h * 0.5
    raw_error = target_y - paddle_center

    # Spike only when already nearly aligned and very close to contact:
    # avoids target snaps from sign(ball_dy) flips during travel.
    dx_to_player = PLAYER_X - ball_x
    near_contact = moving_toward & (dx_to_player < params["fire_window"])
    nearly_aligned = jnp.abs(raw_error) < params["fire_tol"]
    spike_on = near_contact & nearly_aligned
    spike = params["aim_spike"] * jnp.sign(ball_dy)
    error = raw_error + jnp.where(spike_on, spike, 0.0)

    dz = params["dead_zone"]
    move_up = error < -dz
    move_down = error > dz

    # FIRE: tight gate — close and aligned within paddle quarter.
    fire_active = near_contact & nearly_aligned

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
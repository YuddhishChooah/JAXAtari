"""
Auto-generated policy v4
Generated at: 2026-04-26 18:08:08
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
MID_Y = 0.5 * (PLAYFIELD_TOP + PLAYFIELD_BOTTOM)


def init_params():
    return {
        "dead_zone": jnp.array(1.8),
        "aim_spike": jnp.array(3.0),
        "fallback_gain": jnp.array(0.5),
        "fire_window": jnp.array(14.0),
        "commit_thresh": jnp.array(1.0),
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
    t = jnp.clip(t, 0.0, 200.0)
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

    # True physics intercept on the player side.
    intercept_player = _intercept(ball_x, ball_y, ball_dx, ball_dy, PLAYER_X)

    # Idle/ready target when ball is going away: blend toward mid screen.
    fg = jnp.clip(params["fallback_gain"], 0.0, 1.0)
    idle_target = MID_Y + (intercept_player - MID_Y) * (1.0 - fg)

    # Default target by ball direction.
    target_y = jnp.where(moving_toward, intercept_player,
                         jnp.where(moving_away, idle_target, intercept_player))

    paddle_center = player_y + player_h * 0.5
    base_error = target_y - paddle_center

    # Contact-gated spike: only when close AND already aligned.
    dx_to_player = PLAYER_X - ball_x
    close = (dx_to_player < params["fire_window"]) & (dx_to_player > -2.0)
    half_h = player_h * 0.5
    aligned = jnp.abs(base_error) < (half_h - 1.0)

    # Spike bias only at contact, capped so it never flips error sign badly.
    spike_raw = params["aim_spike"] * jnp.sign(ball_dy)
    spike_cap = jnp.maximum(half_h - jnp.abs(base_error) - 1.0, 0.0)
    spike = jnp.clip(spike_raw, -spike_cap, spike_cap)
    spike_active = moving_toward & close & aligned
    target_y = target_y + jnp.where(spike_active, spike, 0.0)

    error = target_y - paddle_center

    dz = params["dead_zone"]
    commit = params["commit_thresh"]

    # Commit early when ball is approaching and close in x.
    approaching_close = moving_toward & (dx_to_player < 60.0)
    effective_dz = jnp.where(approaching_close, commit, dz)

    move_up = error < -effective_dz
    move_down = error > effective_dz

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
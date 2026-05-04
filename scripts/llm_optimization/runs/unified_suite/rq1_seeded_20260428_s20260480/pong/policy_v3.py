"""
Auto-generated policy v3
Generated at: 2026-04-28 13:53:20
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
PADDLE_CENTER_OFFSET = 8.0
PLAYFIELD_TOP = 24.0
PLAYFIELD_BOTTOM = 194.0
PLAYFIELD_MID = 0.5 * (PLAYFIELD_TOP + PLAYFIELD_BOTTOM)
BALL_HEIGHT = 4.0


def init_params():
    return {
        "dead_zone": jnp.array(2.5),
        "time_bias": jnp.array(0.0),
        "aim_offset": jnp.array(-0.8),
        "spike_gain": jnp.array(2.0),
        "fire_range": jnp.array(18.0),
        "idle_pull": jnp.array(0.5),
    }


def _reflect(y, lo, hi):
    span = hi - lo
    rel = y - lo
    mod = jnp.mod(rel, 2.0 * span)
    folded = jnp.where(mod > span, 2.0 * span - mod, mod)
    return lo + folded


def _predict_intercept(bx, by, dx, dy, time_bias):
    safe_dx = jnp.where(jnp.abs(dx) < 1e-3, 1e-3, dx)
    t = (PLAYER_X - bx) / safe_dx
    t = jnp.maximum(t, 0.0) + time_bias
    raw_y = by + dy * t
    lo = PLAYFIELD_TOP + 0.5 * BALL_HEIGHT
    hi = PLAYFIELD_BOTTOM - 0.5 * BALL_HEIGHT
    return _reflect(raw_y, lo, hi)


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    player_y = curr[1]
    ball_x_prev = prev[16]
    ball_y_prev = prev[17]
    ball_x = curr[16]
    ball_y = curr[17]

    dx = ball_x - ball_x_prev
    dy = ball_y - ball_y_prev

    paddle_center = player_y + PADDLE_CENTER_OFFSET

    moving_toward = dx > 0.0
    predicted = _predict_intercept(ball_x, ball_y, dx, dy, params["time_bias"])

    # Idle target: blend between mid-court and current ball y when ball recedes
    idle_target = PLAYFIELD_MID + params["idle_pull"] * (ball_y - PLAYFIELD_MID)

    target_y = jnp.where(moving_toward, predicted, idle_target)

    # Spiky aim: bias hit point in the direction of dy so we add angle
    spike = params["aim_offset"] + params["spike_gain"] * jnp.sign(dy)
    target_y = target_y + jnp.where(moving_toward, spike, 0.0)

    error = target_y - paddle_center

    # Two-tier dead zone: tighter when ball is approaching and close
    close_x = jnp.abs(PLAYER_X - ball_x)
    dz_base = params["dead_zone"]
    dz = jnp.where(moving_toward & (close_x < 40.0), dz_base, dz_base * 2.0)

    move_up = error < -dz
    move_down = error > dz

    # FIRE only on imminent contact and reasonable alignment
    in_fire_range = close_x < params["fire_range"]
    aligned = jnp.abs(error) < (dz_base + 8.0)
    use_fire = moving_toward & in_fire_range & aligned

    up_act = jnp.where(use_fire, FIRE_UP, MOVE_UP)
    down_act = jnp.where(use_fire, FIRE_DOWN, MOVE_DOWN)
    still_act = jnp.where(use_fire, FIRE, NOOP)

    action = jnp.where(move_up, up_act, jnp.where(move_down, down_act, still_act))
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
"""
Auto-generated policy v5
Generated at: 2026-04-28 14:00:08
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
        "dead_zone": jnp.array(2.8),
        "aim_offset": jnp.array(-1.1),
        "spike_gain": jnp.array(2.4),
        "spike_scale": jnp.array(1.5),
        "fire_range": jnp.array(18.0),
        "shadow_gain": jnp.array(0.85),
    }


def _reflect(y, lo, hi):
    span = hi - lo
    rel = y - lo
    mod = jnp.mod(rel, 2.0 * span)
    folded = jnp.where(mod > span, 2.0 * span - mod, mod)
    return lo + folded


def _predict_intercept(bx, by, dx, dy):
    safe_dx = jnp.where(jnp.abs(dx) < 1e-3, 1e-3, dx)
    t = (PLAYER_X - bx) / safe_dx
    t = jnp.clip(t, 0.0, 60.0)
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
    predicted = _predict_intercept(ball_x, ball_y, dx, dy)

    # Idle target: shadow the ball's Y when it recedes (replaces mid-court pull).
    idle_target = paddle_center + params["shadow_gain"] * (ball_y - paddle_center)

    target_y = jnp.where(moving_toward, predicted, idle_target)

    # Smooth spike term using tanh, only when approaching.
    smooth_dy = jnp.tanh(dy / params["spike_scale"])
    spike = params["aim_offset"] + params["spike_gain"] * smooth_dy
    target_y = target_y + jnp.where(moving_toward, spike, 0.0)

    error = target_y - paddle_center

    dz = params["dead_zone"]
    move_up = error < -dz
    move_down = error > dz

    # FIRE on imminent contact with alignment that accounts for spike magnitude.
    close_x = jnp.abs(PLAYER_X - ball_x)
    in_fire_range = close_x < params["fire_range"]
    align_margin = dz + jnp.abs(params["aim_offset"]) + 0.5 * params["spike_gain"]
    aligned = jnp.abs(error) < align_margin
    use_fire = moving_toward & in_fire_range & aligned

    up_act = jnp.where(use_fire, FIRE_UP, MOVE_UP)
    down_act = jnp.where(use_fire, FIRE_DOWN, MOVE_DOWN)
    still_act = jnp.where(use_fire, FIRE, NOOP)

    action = jnp.where(move_up, up_act, jnp.where(move_down, down_act, still_act))
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
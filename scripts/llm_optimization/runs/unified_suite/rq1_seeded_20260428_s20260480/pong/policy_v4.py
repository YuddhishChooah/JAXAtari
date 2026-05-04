"""
Auto-generated policy v4
Generated at: 2026-04-28 13:56:39
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
        "dead_zone": jnp.array(2.0),
        "aim_offset": jnp.array(-0.5),
        "spike_gain": jnp.array(1.5),
        "fire_range": jnp.array(18.0),
        "close_dz_scale": jnp.array(0.4),
        "hysteresis": jnp.array(1.2),
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
    t = jnp.maximum(t, 0.0)
    raw_y = by + dy * t
    lo = PLAYFIELD_TOP + 0.5 * BALL_HEIGHT
    hi = PLAYFIELD_BOTTOM - 0.5 * BALL_HEIGHT
    return _reflect(raw_y, lo, hi)


def _adaptive_dz(dz_base, close_x, close_dz_scale):
    # Tighter near contact, wider when far. Smooth ramp on close_x in [0, 80].
    far_frac = jnp.clip(close_x / 80.0, 0.0, 1.0)
    scale = close_dz_scale + (1.0 - close_dz_scale) * far_frac
    return dz_base * scale


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
    close_x = jnp.abs(PLAYER_X - ball_x)

    predicted = _predict_intercept(ball_x, ball_y, dx, dy)

    # When ball recedes, pre-position at mid-court (no chasing ball Y).
    target_y = jnp.where(moving_toward, predicted, PLAYFIELD_MID)

    # Spike only near contact, when already roughly aligned. Stable mid-flight.
    near_contact = close_x < params["fire_range"]
    rough_align = jnp.abs(predicted - paddle_center) < 6.0
    apply_spike = moving_toward & near_contact & rough_align
    dy_sign = jnp.sign(dy)
    spike = params["aim_offset"] + params["spike_gain"] * dy_sign
    target_y = target_y + jnp.where(apply_spike, spike, 0.0)

    error = target_y - paddle_center

    # Distance-adaptive dead zone: smaller near contact for precision.
    dz = _adaptive_dz(params["dead_zone"], close_x, params["close_dz_scale"])

    # Hysteresis: require margin beyond dz to actually move.
    h = params["hysteresis"]
    move_up = error < -(dz + h)
    move_down = error > (dz + h)

    # FIRE only on imminent contact and good alignment.
    aligned = jnp.abs(error) < (params["dead_zone"] + 6.0)
    use_fire = moving_toward & near_contact & aligned

    up_act = jnp.where(use_fire, FIRE_UP, MOVE_UP)
    down_act = jnp.where(use_fire, FIRE_DOWN, MOVE_DOWN)
    still_act = jnp.where(use_fire, FIRE, NOOP)

    action = jnp.where(move_up, up_act, jnp.where(move_down, down_act, still_act))
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
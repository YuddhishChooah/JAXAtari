"""
Auto-generated policy v2
Generated at: 2026-04-26 16:27:47
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
PADDLE_CENTER_OFFSET = 8.0
PADDLE_HALF = 8.0
PLAYFIELD_TOP = 24.0
PLAYFIELD_BOTTOM = 194.0


def init_params():
    return {
        "dead_zone": jnp.array(2.5),
        "time_gain": jnp.array(1.0),
        "aim_spike": jnp.array(3.0),
        "idle_track": jnp.array(0.6),
        "fire_dist": jnp.array(40.0),
    }


def _fold(y):
    span = PLAYFIELD_BOTTOM - PLAYFIELD_TOP
    rel = y - PLAYFIELD_TOP
    period = 2.0 * span
    m = jnp.mod(rel, period)
    folded = jnp.where(m > span, period - m, m)
    return PLAYFIELD_TOP + folded


def _intercept(bx, by, dx, dy, target_x, time_gain):
    safe_dx = jnp.where(jnp.abs(dx) < 1e-3, 1e-3 * jnp.sign(dx) + 1e-3, dx)
    t = (target_x - bx) / safe_dx
    t = jnp.clip(t * time_gain, 0.0, 400.0)
    return _fold(by + dy * t)


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    player_y = curr[1]
    paddle_center = player_y + PADDLE_CENTER_OFFSET

    prev_bx, prev_by = prev[16], prev[17]
    curr_bx, curr_by = curr[16], curr[17]
    ball_active = curr[20]

    dx = curr_bx - prev_bx
    dy = curr_by - prev_by

    moving_toward = dx > 0.0

    # Direct intercept when ball approaches.
    intercept_player = _intercept(curr_bx, curr_by, dx, dy, PLAYER_X, params["time_gain"])

    # When ball moves away, predict where it will hit enemy paddle, then mirror back.
    intercept_enemy = _intercept(curr_bx, curr_by, dx, dy, ENEMY_X, params["time_gain"])
    # Pre-position partially toward the enemy intercept so we can react fast.
    idle_target = curr_by + (intercept_enemy - curr_by) * params["idle_track"]

    # Aim bias: hit ball off-center to create spiky returns.
    # When ball moving down, aim slightly low (hit with top of paddle) -> steeper up.
    aim_bias = -jnp.sign(dy) * params["aim_spike"]

    target = jnp.where(moving_toward, intercept_player + aim_bias, idle_target)
    target = jnp.where(ball_active > 0.5, target, 105.0)

    # Clamp target to keep paddle inside playfield.
    lo = PLAYFIELD_TOP + PADDLE_HALF
    hi = PLAYFIELD_BOTTOM - PADDLE_HALF
    target = jnp.clip(target, lo, hi)

    error = target - paddle_center
    dz = params["dead_zone"]

    move_up = error < -dz
    move_down = error > dz

    # FIRE only when ball is close and approaching (contact moment).
    dist_to_player = PLAYER_X - curr_bx
    fire_on = (dist_to_player > 0.0) & (dist_to_player < params["fire_dist"]) & moving_toward

    action_up = jnp.where(fire_on, FIRE_UP, MOVE_UP)
    action_down = jnp.where(fire_on, FIRE_DOWN, MOVE_DOWN)
    action_idle = jnp.where(fire_on, FIRE, NOOP)

    action = jnp.where(move_up, action_up,
                       jnp.where(move_down, action_down, action_idle))
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
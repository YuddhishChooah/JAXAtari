"""
Auto-generated policy v3
Generated at: 2026-04-26 16:29:52
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
PLAYFIELD_MID = 0.5 * (PLAYFIELD_TOP + PLAYFIELD_BOTTOM)


def init_params():
    return {
        "dead_zone": jnp.array(4.0),
        "aim_spike": jnp.array(6.0),
        "aim_margin": jnp.array(20.0),
        "idle_track": jnp.array(0.9),
        "fire_dist": jnp.array(35.0),
        "lead_bias": jnp.array(0.0),
    }


def _fold(y):
    span = PLAYFIELD_BOTTOM - PLAYFIELD_TOP
    rel = y - PLAYFIELD_TOP
    period = 2.0 * span
    m = jnp.mod(rel, period)
    folded = jnp.where(m > span, period - m, m)
    return PLAYFIELD_TOP + folded


def _intercept_y(bx, by, dx, dy, target_x):
    safe_dx = jnp.where(jnp.abs(dx) < 1e-3, 1e-3, jnp.abs(dx)) * jnp.sign(dx + 1e-6)
    t = (target_x - bx) / safe_dx
    t = jnp.clip(t, 0.0, 400.0)
    return _fold(by + dy * t)


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    player_y = curr[1]
    paddle_center = player_y + PADDLE_CENTER_OFFSET

    enemy_y = curr[9]
    enemy_center = enemy_y + PADDLE_CENTER_OFFSET

    prev_bx, prev_by = prev[16], prev[17]
    curr_bx, curr_by = curr[16], curr[17]
    ball_active = curr[20]

    dx = curr_bx - prev_bx
    dy = curr_by - prev_by

    moving_toward = dx > 0.0

    # True geometric intercept (no time_gain corruption)
    intercept_player = _intercept_y(curr_bx, curr_by, dx, dy, PLAYER_X)
    intercept_enemy = _intercept_y(curr_bx, curr_by, dx, dy, ENEMY_X)

    # Idle: pre-position at where ball will return after enemy hits.
    # Mirror around enemy paddle: assume return path tracks toward our side.
    idle_target = curr_by + (intercept_enemy - curr_by) * params["idle_track"]

    dist_to_player = PLAYER_X - curr_bx
    far_from_paddle = dist_to_player > params["aim_margin"]

    # Aim away from enemy: hit ball with the paddle edge opposite enemy_center.
    # If enemy is above mid, we want the return to go down -> hit ball with top of paddle
    # so paddle center sits BELOW intercept by aim_spike.
    aim_dir = jnp.sign(enemy_center - PLAYFIELD_MID)
    # If enemy above mid (aim_dir<0), bias paddle below intercept (positive offset).
    aim_offset = -aim_dir * params["aim_spike"]
    # Only apply when ball is far enough that we have margin to recover.
    aim_offset = jnp.where(far_from_paddle, aim_offset, 0.0)

    # Lead bias: small additive correction in direction of dy.
    lead = jnp.sign(dy) * params["lead_bias"]

    target_attack = intercept_player + aim_offset + lead
    target = jnp.where(moving_toward, target_attack, idle_target)
    target = jnp.where(ball_active > 0.5, target, PLAYFIELD_MID)

    lo = PLAYFIELD_TOP + PADDLE_HALF
    hi = PLAYFIELD_BOTTOM - PADDLE_HALF
    target = jnp.clip(target, lo, hi)

    error = target - paddle_center
    dz = params["dead_zone"]

    move_up = error < -dz
    move_down = error > dz

    fire_on = (dist_to_player > 0.0) & (dist_to_player < params["fire_dist"]) & moving_toward

    action_up = jnp.where(fire_on, FIRE_UP, MOVE_UP)
    action_down = jnp.where(fire_on, FIRE_DOWN, MOVE_DOWN)
    action_idle = jnp.where(fire_on, FIRE, NOOP)

    action = jnp.where(move_up, action_up,
                       jnp.where(move_down, action_down, action_idle))
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
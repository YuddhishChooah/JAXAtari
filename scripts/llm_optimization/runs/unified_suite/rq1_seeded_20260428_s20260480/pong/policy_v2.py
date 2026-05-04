"""
Auto-generated policy v2
Generated at: 2026-04-28 13:49:36
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
        "dead_zone": jnp.array(2.5),
        "aim_offset": jnp.array(0.0),
        "fire_align": jnp.array(6.0),
        "vy_damp": jnp.array(0.95),
        "idle_pull": jnp.array(0.6),
        "jitter_margin": jnp.array(1.5),
    }


def _fold(y):
    span = PLAYFIELD_BOTTOM - PLAYFIELD_TOP
    rel = y - PLAYFIELD_TOP
    mod = jnp.mod(rel, 2.0 * span)
    folded = jnp.where(mod > span, 2.0 * span - mod, mod)
    return PLAYFIELD_TOP + folded


def _intercept_at(bx, by, dx, dy, target_x):
    safe_dx = jnp.where(jnp.abs(dx) < 1e-3, jnp.sign(dx) * 1e-3 + 1e-3, dx)
    t = (target_x - bx) / safe_dx
    t = jnp.maximum(t, 0.0)
    raw_y = by + dy * t
    return _fold(raw_y)


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    player_y = curr[1]
    ball_x_prev = prev[16]
    ball_y_prev = prev[17]
    ball_x = curr[16]
    ball_y = curr[17]

    dx = ball_x - ball_x_prev
    dy_raw = ball_y - ball_y_prev
    dy = dy_raw * params["vy_damp"]

    paddle_center = player_y + PADDLE_CENTER_OFFSET

    moving_toward = dx > 0.0

    # Direct intercept on player plane
    direct_intercept = _intercept_at(ball_x, ball_y, dx, dy, PLAYER_X)

    # Anticipated intercept after enemy bounce: predict where ball reaches enemy,
    # then mirror dx and project to player plane.
    enemy_intercept_y = _intercept_at(ball_x, ball_y, dx, dy, ENEMY_X)
    # Distance from enemy back to player plane
    bounce_target = _fold(enemy_intercept_y + dy * ((PLAYER_X - ENEMY_X) / jnp.where(jnp.abs(dx) < 1e-3, 1e-3, jnp.abs(dx))))

    # Idle target: blend midline and bounce prediction
    idle_target = PLAYFIELD_MID + params["idle_pull"] * (bounce_target - PLAYFIELD_MID)

    target_y = jnp.where(moving_toward, direct_intercept, idle_target)

    # Aim offset sign-dependent on ball vertical direction (creates spiky returns)
    offset = params["aim_offset"] * jnp.sign(dy_raw + 1e-6)
    target_y = target_y + offset

    # Clamp to reachable range
    target_y = jnp.clip(target_y,
                       PLAYFIELD_TOP + PADDLE_HALF,
                       PLAYFIELD_BOTTOM - PADDLE_HALF)

    error = target_y - paddle_center
    abs_err = jnp.abs(error)

    # Hysteretic dead zone: grows with vertical ball speed to suppress jitter
    dz = params["dead_zone"] + params["jitter_margin"] * jnp.abs(dy_raw)

    move_up = error < -dz
    move_down = error > dz

    # FIRE gating: only when ball is close, approaching, and well aligned
    close = (PLAYER_X - ball_x) < 25.0
    close = close & (ball_x < PLAYER_X)
    aligned = abs_err < params["fire_align"]
    use_fire = close & aligned & moving_toward

    action = jnp.where(
        move_up,
        jnp.where(use_fire, FIRE_UP, MOVE_UP),
        jnp.where(
            move_down,
            jnp.where(use_fire, FIRE_DOWN, MOVE_DOWN),
            jnp.where(use_fire, FIRE, NOOP),
        ),
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
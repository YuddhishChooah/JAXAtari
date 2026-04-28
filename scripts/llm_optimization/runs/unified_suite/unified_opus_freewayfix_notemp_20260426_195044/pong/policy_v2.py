"""
Auto-generated policy v2
Generated at: 2026-04-26 19:52:45
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
PADDLE_HEIGHT = 16.0
PLAYFIELD_TOP = 24.0
PLAYFIELD_BOTTOM = 194.0
SCREEN_MID = 105.0


def init_params():
    return {
        "dead_zone": jnp.array(2.5),
        "aim_gain": jnp.array(0.6),       # off-center aim strength based on dy sign
        "idle_y": jnp.array(105.0),       # ready position when ball recedes
        "fire_dist": jnp.array(20.0),     # x-distance to ball at which FIRE engages
        "horizon_clip": jnp.array(60.0),  # clamp predicted travel time*|dx|
    }


def _fold(y):
    span = PLAYFIELD_BOTTOM - PLAYFIELD_TOP
    rel = y - PLAYFIELD_TOP
    mod = jnp.mod(rel, 2.0 * span)
    folded = jnp.where(mod > span, 2.0 * span - mod, mod)
    return PLAYFIELD_TOP + folded


def _predict_intercept(bx, by, dx, dy, horizon_clip):
    # Only meaningful when dx > 0 (ball moving toward player).
    safe_dx = jnp.maximum(dx, 0.5)
    t = (PLAYER_X - bx) / safe_dx
    # Clamp horizon to keep prediction stable.
    t = jnp.clip(t, 0.0, horizon_clip)
    raw_y = by + dy * t
    return _fold(raw_y)


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    player_y = curr[1]
    paddle_center = player_y + PADDLE_CENTER_OFFSET

    prev_bx, prev_by = prev[16], prev[17]
    bx, by = curr[16], curr[17]
    ball_active = curr[20]

    dx = bx - prev_bx
    dy = by - prev_by

    predicted = _predict_intercept(bx, by, dx, dy, params["horizon_clip"])

    # Strategic off-center aim: hit ball with paddle offset opposite to dy
    # so returns get steeper. sign(dy) ~ direction ball is moving vertically.
    dy_sign = jnp.tanh(dy)  # smooth sign in [-1,1]
    aim = predicted - params["aim_gain"] * dy_sign * (PADDLE_HEIGHT * 0.5)

    # Clamp aim so paddle never wedges into walls.
    aim = jnp.clip(aim,
                   PLAYFIELD_TOP + PADDLE_CENTER_OFFSET,
                   PLAYFIELD_BOTTOM - PADDLE_CENTER_OFFSET)

    # Target selection:
    # - ball inactive: idle_y
    # - ball receding (dx <= 0): idle_y (ready position)
    # - ball approaching: aim
    approaching = dx > 0.0
    target_active = jnp.where(approaching, aim, params["idle_y"])
    target = jnp.where(ball_active > 0.5, target_active, params["idle_y"])

    error = target - paddle_center
    dz = params["dead_zone"]

    move_up = error < -dz
    move_down = error > dz

    # FIRE only when the ball is close to the player paddle and approaching.
    ball_close = (PLAYER_X - bx) < params["fire_dist"]
    use_fire = ball_close & approaching & (ball_active > 0.5)

    action_up = jnp.where(use_fire, FIRE_UP, MOVE_UP)
    action_down = jnp.where(use_fire, FIRE_DOWN, MOVE_DOWN)
    action_idle = jnp.where(use_fire, FIRE, NOOP)

    action = jnp.where(move_up, action_up,
                       jnp.where(move_down, action_down, action_idle))
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
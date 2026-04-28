"""
Auto-generated policy v5
Generated at: 2026-04-26 19:58:03
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
PADDLE_HEIGHT = 16.0
PLAYFIELD_TOP = 24.0
PLAYFIELD_BOTTOM = 194.0
SCREEN_MID = 105.0


def init_params():
    return {
        "dead_zone": jnp.array(4.0),
        "aim_gain": jnp.array(0.6),
        "aim_softness": jnp.array(8.0),
        "aim_range": jnp.array(60.0),
        "fire_dist": jnp.array(22.0),
        "return_damp": jnp.array(0.85),
    }


def _fold(y):
    span = PLAYFIELD_BOTTOM - PLAYFIELD_TOP
    rel = y - PLAYFIELD_TOP
    mod = jnp.mod(rel, 2.0 * span)
    folded = jnp.where(mod > span, 2.0 * span - mod, mod)
    return PLAYFIELD_TOP + folded


def _predict_forward(bx, by, dx, dy):
    # Ball moving toward player (dx > 0): time to reach PLAYER_X
    safe_dx = jnp.maximum(jnp.abs(dx), 0.5)
    t = (PLAYER_X - bx) / safe_dx
    t = jnp.clip(t, 0.0, 80.0)
    return _fold(by + dy * t)


def _predict_return(bx, by, dx, dy):
    # Ball moving toward enemy (dx < 0): bounce off enemy paddle area, then come back.
    safe_dx = jnp.maximum(jnp.abs(dx), 0.5)
    t_to_enemy = (bx - ENEMY_X) / safe_dx
    t_to_enemy = jnp.clip(t_to_enemy, 0.0, 80.0)
    y_at_enemy = _fold(by + dy * t_to_enemy)
    t_back = (PLAYER_X - ENEMY_X) / safe_dx
    t_back = jnp.clip(t_back, 0.0, 120.0)
    # After bounce, dy roughly persists (walls already folded). Predict return Y.
    return _fold(y_at_enemy + dy * t_back)


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    player_y = curr[1]
    paddle_center = player_y + PADDLE_CENTER_OFFSET

    enemy_y = curr[9]
    enemy_center = enemy_y + PADDLE_CENTER_OFFSET

    prev_bx, prev_by = prev[16], prev[17]
    bx, by = curr[16], curr[17]
    ball_active = curr[20]

    dx = bx - prev_bx
    dy = by - prev_by

    approaching = dx > 0.3
    dist_to_player = PLAYER_X - bx

    forward_pred = _predict_forward(bx, by, dx, dy)
    return_pred = _predict_return(bx, by, dx, dy)

    # Damp the return prediction toward screen mid (uncertainty grows over time).
    damp = jnp.clip(params["return_damp"], 0.0, 1.0)
    return_target = damp * return_pred + (1.0 - damp) * SCREEN_MID

    base_target = jnp.where(approaching, forward_pred, return_target)

    # Smooth aim offset: bias contact away from enemy paddle to score.
    # If enemy is above center, aim low; if below, aim high.
    enemy_rel = enemy_center - SCREEN_MID
    aim_dir = jnp.tanh(enemy_rel / params["aim_softness"])
    # Smooth distance gating: full effect when dist <= aim_range, fades beyond.
    rng = jnp.maximum(params["aim_range"], 5.0)
    dist_w = jnp.clip(1.0 - dist_to_player / rng, 0.0, 1.0)
    aim_offset = approaching * dist_w * params["aim_gain"] * aim_dir * (PADDLE_HEIGHT * 0.5)

    target = base_target + aim_offset
    target = jnp.clip(target,
                      PLAYFIELD_TOP + PADDLE_CENTER_OFFSET,
                      PLAYFIELD_BOTTOM - PADDLE_CENTER_OFFSET)

    # When ball inactive, hold center.
    target = jnp.where(ball_active > 0.5, target, SCREEN_MID)

    error = target - paddle_center
    dz = params["dead_zone"]

    move_up = error < -dz
    move_down = error > dz

    use_fire = approaching & (ball_active > 0.5) & (dist_to_player < params["fire_dist"]) & (dist_to_player > 0.0)

    action_up = jnp.where(use_fire, FIRE_UP, MOVE_UP)
    action_down = jnp.where(use_fire, FIRE_DOWN, MOVE_DOWN)
    action_idle = jnp.where(use_fire, FIRE, NOOP)

    action = jnp.where(move_up, action_up,
                       jnp.where(move_down, action_down, action_idle))
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
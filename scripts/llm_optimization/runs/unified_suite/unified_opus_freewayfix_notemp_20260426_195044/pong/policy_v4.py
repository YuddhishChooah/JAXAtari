"""
Auto-generated policy v4
Generated at: 2026-04-26 19:56:12
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
        "dead_zone": jnp.array(2.0),
        "aim_gain": jnp.array(0.6),
        "anticipation": jnp.array(0.5),     # blend predicted return vs center for receding
        "fire_dist": jnp.array(8.0),        # tight contact window for FIRE
        "aim_close_dist": jnp.array(35.0),
        "vel_floor": jnp.array(1.0),        # min |dx| used in time-to-intercept
    }


def _fold(y):
    span = PLAYFIELD_BOTTOM - PLAYFIELD_TOP
    rel = y - PLAYFIELD_TOP
    mod = jnp.mod(rel, 2.0 * span)
    folded = jnp.where(mod > span, 2.0 * span - mod, mod)
    return PLAYFIELD_TOP + folded


def _predict_y_at(bx, by, dx, dy, target_x, vel_floor):
    speed = jnp.maximum(jnp.abs(dx), vel_floor)
    t = (target_x - bx) / jnp.where(dx >= 0, speed, -speed)
    t = jnp.clip(t, 0.0, 120.0)
    raw_y = by + dy * t
    return _fold(raw_y)


def _clip_target(y):
    return jnp.clip(y,
                    PLAYFIELD_TOP + PADDLE_CENTER_OFFSET,
                    PLAYFIELD_BOTTOM - PADDLE_CENTER_OFFSET)


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
    dx = jnp.clip(dx, -6.0, 6.0)
    dy = jnp.clip(dy, -6.0, 6.0)

    vel_floor = jnp.maximum(params["vel_floor"], 0.3)
    dist_to_player = PLAYER_X - bx

    # Soft approach weight: 1 when dx>>0, 0 when dx<<0
    approach_w = jax.nn.sigmoid(dx * 1.5)

    # Predict intercept at player x (forward direction)
    pred_intercept = _predict_y_at(bx, by, dx, dy, PLAYER_X, vel_floor)

    # Predict where ball will reach enemy then bounce back to player x
    y_at_enemy = _predict_y_at(bx, by, dx, dy, ENEMY_X, vel_floor)
    # assume enemy returns it; predict from enemy back to player using mirrored dx
    return_y = _fold(y_at_enemy + dy * ((PLAYER_X - ENEMY_X) / vel_floor) * 0.0 + 0.0)
    # Simpler anticipation: blend ball-y mirror with center
    anticip = params["anticipation"]
    receding_target = anticip * y_at_enemy + (1.0 - anticip) * SCREEN_MID

    # Aim offset: deliberate off-center, only when ball close and approaching
    ball_close = dist_to_player < params["aim_close_dist"]
    error_pre = pred_intercept - paddle_center
    # offset toward the side that creates a sharper return: opposite of current error sign
    aim_dir = jnp.tanh(error_pre * 0.2)
    apply_aim = ball_close & (dx > 0.3)
    aim_offset = jnp.where(apply_aim,
                           params["aim_gain"] * aim_dir * (PADDLE_HEIGHT * 0.4),
                           0.0)
    forward_target = _clip_target(pred_intercept + aim_offset)
    backward_target = _clip_target(receding_target)

    target = approach_w * forward_target + (1.0 - approach_w) * backward_target
    target = jnp.where(ball_active > 0.5, target, SCREEN_MID)

    error = target - paddle_center

    # Adaptive dead zone: tight when ball close+approaching, looser when far
    closeness = jnp.clip(1.0 - dist_to_player / 120.0, 0.0, 1.0)
    dz = params["dead_zone"] * (1.5 - closeness)

    move_up = error < -dz
    move_down = error > dz

    # FIRE only in tight contact window
    use_fire = (dist_to_player < params["fire_dist"]) & (dx > 0.0) & (ball_active > 0.5)

    action_up = jnp.where(use_fire, FIRE_UP, MOVE_UP)
    action_down = jnp.where(use_fire, FIRE_DOWN, MOVE_DOWN)
    action_idle = jnp.where(use_fire, FIRE, NOOP)

    action = jnp.where(move_up, action_up,
                       jnp.where(move_down, action_down, action_idle))
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
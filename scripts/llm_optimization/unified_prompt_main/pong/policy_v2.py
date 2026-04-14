"""
Auto-generated policy v2
Generated at: 2026-04-07 21:26:35
"""

"""
Parametric Pong policy for CMA-ES optimization.
Improved ball intercept prediction with direct tracking and off-center targeting.
"""

import jax.numpy as jnp

# --- Action constants ---
NOOP      = 0
FIRE      = 1
MOVE_DOWN = 3
MOVE_UP   = 4
FIRE_DOWN = 11
FIRE_UP   = 12

# --- Screen / game constants ---
SCREEN_WIDTH         = 160
SCREEN_HEIGHT        = 210
PLAYFIELD_TOP        = 24
PLAYFIELD_BOTTOM     = 194
PLAYER_X             = 140
ENEMY_X              = 16
PADDLE_HEIGHT        = 16
PADDLE_CENTER_OFFSET = 8


def init_params() -> dict:
    return {
        "intercept_scale": jnp.float32(15.0),
        "dead_zone":       jnp.float32(2.0),
        "offset_bias":     jnp.float32(0.0),
        "fire_threshold":  jnp.float32(40.0),
        "approach_bias":   jnp.float32(0.5),
    }


def _clamp(v, lo, hi):
    return jnp.clip(v, lo, hi)


def _bounce(projected, top, bottom):
    """Single-bounce wall reflection into [top, bottom]."""
    play_range = jnp.float32(bottom - top)
    shifted = projected - top
    # triangular wrap: fold into [0, play_range]
    folded = jnp.abs(
        (shifted % (2.0 * play_range)) - play_range
    )
    return folded + top


def _predict_intercept(ball_x, ball_y, ball_dx, ball_dy, target_x):
    """
    Predict ball Y when it reaches target_x using linear extrapolation
    with a single wall-bounce correction.
    """
    # Steps until ball reaches target_x; avoid division by zero
    safe_dx = jnp.where(jnp.abs(ball_dx) < 0.5, jnp.float32(0.5), ball_dx)
    steps = (target_x - ball_x) / safe_dx
    steps = jnp.clip(steps, 0.0, 120.0)

    projected_y = ball_y + ball_dy * steps
    return _bounce(projected_y, jnp.float32(PLAYFIELD_TOP), jnp.float32(PLAYFIELD_BOTTOM))


def _choose_action(paddle_y, target_y, dead_zone, use_fire):
    """Move paddle toward target_y; use FIRE variants when requested."""
    paddle_center = paddle_y + jnp.float32(PADDLE_CENTER_OFFSET)
    error = target_y - paddle_center

    action_up   = jnp.where(use_fire > 0.5, jnp.int32(FIRE_UP),   jnp.int32(MOVE_UP))
    action_down = jnp.where(use_fire > 0.5, jnp.int32(FIRE_DOWN), jnp.int32(MOVE_DOWN))

    action = jnp.where(
        error < -dead_zone,
        action_up,
        jnp.where(
            error > dead_zone,
            action_down,
            jnp.int32(NOOP)
        )
    )
    return action


def policy(obs_flat: jnp.ndarray, params: dict) -> int:
    # Unpack params
    intercept_scale = params["intercept_scale"]
    dead_zone       = params["dead_zone"]
    offset_bias     = params["offset_bias"]
    fire_threshold  = params["fire_threshold"]
    approach_bias   = params["approach_bias"]

    # Unpack observations
    curr_player_y = obs_flat[15]   # curr[1]
    curr_ball_x   = obs_flat[22]   # curr[8]
    curr_ball_y   = obs_flat[23]   # curr[9]
    prev_ball_x   = obs_flat[8]    # prev[8]
    prev_ball_y   = obs_flat[9]    # prev[9]

    ball_dx = curr_ball_x - prev_ball_x
    ball_dy = curr_ball_y - prev_ball_y

    # --- Target computation ---
    screen_center_y = jnp.float32((PLAYFIELD_TOP + PLAYFIELD_BOTTOM) / 2.0)

    # Intercept at player paddle X
    player_x_f = jnp.float32(PLAYER_X)
    intercept_y = _predict_intercept(
        curr_ball_x, curr_ball_y, ball_dx, ball_dy, player_x_f
    )

    # When ball moves away (ball_dx < 0), blend intercept toward screen center
    # so paddle returns to a neutral position rather than chasing a bad projection
    away_target = approach_bias * screen_center_y + (1.0 - approach_bias) * intercept_y
    target_y_base = jnp.where(ball_dx > 0.0, intercept_y, away_target)

    # Off-center bias: aim so the ball hits slightly above/below paddle center
    target_y = _clamp(
        target_y_base - jnp.float32(PADDLE_CENTER_OFFSET) + offset_bias,
        jnp.float32(PLAYFIELD_TOP),
        jnp.float32(PLAYFIELD_BOTTOM - PADDLE_HEIGHT)
    )

    # FIRE when ball is approaching and close to player paddle
    dist_to_player = jnp.abs(curr_ball_x - player_x_f)
    use_fire = jnp.where(
        (ball_dx > 0.0) & (dist_to_player < fire_threshold),
        1.0,
        0.0
    )

    action = _choose_action(curr_player_y, target_y, dead_zone, use_fire)
    return action


def measure_main(episode_rewards: jnp.ndarray, episode_scores: dict) -> float:
    """Fitness: total reward across all episodes."""
    return jnp.sum(episode_rewards)
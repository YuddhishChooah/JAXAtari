"""
Auto-generated policy v3
Generated at: 2026-04-07 21:29:15
"""

"""
Parametric Pong policy for CMA-ES optimization.
Improved intercept prediction with better wall-bounce handling and
direct ball tracking when prediction is uncertain.
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
PLAYFIELD_MID        = (PLAYFIELD_TOP + PLAYFIELD_BOTTOM) / 2.0


def init_params() -> dict:
    """
    Tunable float parameters for CMA-ES.
      intercept_scale : steps ahead to project ball (0..40)
      dead_zone       : pixel dead zone to suppress jitter (0..8)
      offset_bias     : preferred hit offset from paddle center (-8..8)
      fire_threshold  : ball x distance to player to use FIRE (0..60)
      center_bias     : blend toward center when ball moves away (0..1)
    """
    return {
        "intercept_scale": jnp.float32(15.0),
        "dead_zone":       jnp.float32(2.0),
        "offset_bias":     jnp.float32(0.0),
        "fire_threshold":  jnp.float32(40.0),
        "center_bias":     jnp.float32(0.2),
    }


def _bounce_y(projected, top, bottom):
    """Fold projected Y into [top, bottom] with triangle-wave bounce."""
    span = jnp.float32(bottom - top)
    rel  = projected - top
    # triangle wave period = 2*span
    period = 2.0 * span
    phase  = rel % period
    folded = jnp.where(phase < span, phase, period - phase)
    return folded + top


def _predict_intercept(ball_x, ball_y, ball_dx, ball_dy, intercept_scale):
    """
    Project ball to player paddle x using velocity ratio,
    then compute Y with wall bounces.
    """
    # Steps until ball reaches player paddle
    dist_to_player = jnp.float32(PLAYER_X) - ball_x
    # Use intercept_scale as a fixed horizon when dx is near zero
    steps_by_vel = jnp.where(
        jnp.abs(ball_dx) > 0.5,
        dist_to_player / (ball_dx + 1e-6),
        intercept_scale
    )
    # Clamp steps to reasonable range
    steps = jnp.clip(steps_by_vel, 0.0, 80.0)
    projected_y = ball_y + ball_dy * steps
    return _bounce_y(projected_y, PLAYFIELD_TOP, PLAYFIELD_BOTTOM)


def _choose_action(paddle_y, target_y, dead_zone, use_fire):
    """Return movement action based on signed error."""
    paddle_center = paddle_y + PADDLE_CENTER_OFFSET
    error = target_y - paddle_center

    action_up   = jnp.where(use_fire > 0.5, jnp.int32(FIRE_UP),   jnp.int32(MOVE_UP))
    action_down = jnp.where(use_fire > 0.5, jnp.int32(FIRE_DOWN), jnp.int32(MOVE_DOWN))

    action = jnp.where(
        error < -dead_zone,
        action_up,
        jnp.where(error > dead_zone, action_down, jnp.int32(NOOP))
    )
    return action


def policy(obs_flat: jnp.ndarray, params: dict) -> int:
    """
    Main policy: predict intercept Y, aim slightly off-center, return action.
    """
    intercept_scale = params["intercept_scale"]
    dead_zone       = params["dead_zone"]
    offset_bias     = params["offset_bias"]
    fire_threshold  = params["fire_threshold"]
    center_bias     = params["center_bias"]

    # Unpack observations
    curr_player_y = obs_flat[15]   # curr player y
    curr_ball_x   = obs_flat[22]   # curr ball x
    curr_ball_y   = obs_flat[23]   # curr ball y
    prev_ball_x   = obs_flat[8]    # prev ball x
    prev_ball_y   = obs_flat[9]    # prev ball y

    ball_dx = curr_ball_x - prev_ball_x
    ball_dy = curr_ball_y - prev_ball_y

    # Predict where ball will reach player paddle
    intercept_y = _predict_intercept(
        curr_ball_x, curr_ball_y, ball_dx, ball_dy, intercept_scale
    )

    # When ball moves away, blend intercept toward screen center
    center_y = jnp.float32(PLAYFIELD_MID)
    blended_y = center_bias * center_y + (1.0 - center_bias) * intercept_y
    target_y_base = jnp.where(ball_dx > 0.0, intercept_y, blended_y)

    # Apply offset bias to aim off-center for spiky returns
    target_y = jnp.clip(
        target_y_base + offset_bias,
        jnp.float32(PLAYFIELD_TOP),
        jnp.float32(PLAYFIELD_BOTTOM - PADDLE_HEIGHT)
    )

    # Use FIRE when ball is close to player paddle
    dist_to_player = jnp.abs(curr_ball_x - jnp.float32(PLAYER_X))
    use_fire = jnp.where(dist_to_player < fire_threshold, 1.0, 0.0)

    action = _choose_action(curr_player_y, target_y, dead_zone, use_fire)
    return action


def measure_main(episode_rewards: jnp.ndarray, episode_scores: dict) -> float:
    """Fitness: total reward across all episodes."""
    return jnp.sum(episode_rewards)
"""
Auto-generated policy v1
Generated at: 2026-04-07 21:23:59
"""

"""
Parametric Pong policy for CMA-ES optimization.
Uses ball intercept prediction with dead-zone control and off-center targeting.
"""

import jax.numpy as jnp
import jax

# --- Action constants ---
NOOP      = 0
FIRE      = 1
MOVE_DOWN = 3
MOVE_UP   = 4
FIRE_DOWN = 11
FIRE_UP   = 12

# --- Screen / game constants ---
SCREEN_WIDTH        = 160
SCREEN_HEIGHT       = 210
PLAYFIELD_TOP       = 24
PLAYFIELD_BOTTOM    = 194
PLAYER_X            = 140
ENEMY_X             = 16
PADDLE_HEIGHT       = 16
PADDLE_CENTER_OFFSET = 8


def init_params() -> dict:
    """
    3-8 tunable float parameters for CMA-ES.
      intercept_scale : how many steps ahead to project the ball (0..~30)
      dead_zone       : pixel dead zone around target to suppress jitter (0..10)
      offset_bias     : preferred hit offset from paddle center (-8..8)
      fire_threshold  : closeness to player x to trigger FIRE actions (0..50)
      approach_bias   : fractional blend toward center when ball moving away (0..1)
    """
    return {
        "intercept_scale": jnp.float32(20.0),
        "dead_zone":       jnp.float32(3.0),
        "offset_bias":     jnp.float32(4.0),
        "fire_threshold":  jnp.float32(30.0),
        "approach_bias":   jnp.float32(0.3),
    }


def _clamp(v, lo, hi):
    return jnp.clip(v, lo, hi)


def _predict_intercept(ball_y, ball_dy, intercept_scale):
    """Project ball Y position forward by intercept_scale steps."""
    projected = ball_y + ball_dy * intercept_scale
    # Bounce off top/bottom walls (single bounce approximation)
    playfield_range = jnp.float32(PLAYFIELD_BOTTOM - PLAYFIELD_TOP)
    projected_clamped = projected - PLAYFIELD_TOP
    # Fold into [0, range] with a triangular wrap
    folded = jnp.abs(
        (projected_clamped % (2.0 * playfield_range)) - playfield_range
    )
    return folded + PLAYFIELD_TOP


def _choose_action(paddle_y, target_y, dead_zone, use_fire):
    """
    Choose move action based on error relative to dead zone.
    use_fire: 0.0 or 1.0 scalar, whether to use FIRE variants.
    """
    paddle_center = paddle_y + PADDLE_CENTER_OFFSET
    error = target_y - paddle_center

    # actions: up, down, noop
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
    """
    Main policy: predict intercept, aim for off-center hit, return action.
    """
    # Unpack params
    intercept_scale = params["intercept_scale"]
    dead_zone       = params["dead_zone"]
    offset_bias     = params["offset_bias"]
    fire_threshold  = params["fire_threshold"]
    approach_bias   = params["approach_bias"]

    # Unpack observations
    # prev frame: indices 0-13
    # curr frame: indices 14-27
    curr_player_y = obs_flat[15]   # curr[1]
    curr_ball_x   = obs_flat[22]   # curr[8]
    curr_ball_y   = obs_flat[23]   # curr[9]
    prev_ball_x   = obs_flat[8]    # prev[8]
    prev_ball_y   = obs_flat[9]    # prev[9]

    ball_dx = curr_ball_x - prev_ball_x
    ball_dy = curr_ball_y - prev_ball_y

    # Predict intercept when ball is heading toward player (ball_dx > 0)
    intercept_y = _predict_intercept(curr_ball_y, ball_dy, intercept_scale)

    # When ball moves away, blend toward screen center
    screen_center_y = jnp.float32((PLAYFIELD_TOP + PLAYFIELD_BOTTOM) / 2.0)
    target_when_away = (
        approach_bias * screen_center_y + (1.0 - approach_bias) * intercept_y
    )

    target_y_base = jnp.where(ball_dx > 0.0, intercept_y, target_when_away)

    # Apply offset bias: aim slightly off-center of paddle
    target_y = _clamp(
        target_y_base - PADDLE_CENTER_OFFSET + offset_bias,
        jnp.float32(PLAYFIELD_TOP),
        jnp.float32(PLAYFIELD_BOTTOM - PADDLE_HEIGHT)
    )

    # Decide whether to use FIRE: ball is close to player paddle
    dist_to_player = jnp.abs(curr_ball_x - jnp.float32(PLAYER_X))
    use_fire = jnp.where(dist_to_player < fire_threshold, 1.0, 0.0)

    action = _choose_action(curr_player_y, target_y, dead_zone, use_fire)
    return action


def measure_main(episode_rewards: jnp.ndarray, episode_scores: dict) -> float:
    """Fitness: total reward across all episodes."""
    return jnp.sum(episode_rewards)
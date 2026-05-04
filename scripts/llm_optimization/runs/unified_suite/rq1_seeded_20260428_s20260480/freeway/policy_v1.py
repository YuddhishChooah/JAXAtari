"""
Auto-generated policy v1
Generated at: 2026-04-28 14:03:40
"""

import jax
import jax.numpy as jnp

FRAME_SIZE = 88
NOOP = 0
UP = 1
DOWN = 2
CHICKEN_X = 44.0
CHICKEN_WIDTH = 6.0
CHICKEN_HEIGHT = 8.0
NUM_CARS = 10


def init_params():
    return {
        "safe_gap": jnp.array(12.0),       # x-margin around chicken to call lane safe
        "lookahead": jnp.array(4.0),       # frames to project car motion
        "next_lane_dy": jnp.array(10.0),   # y-distance threshold to evaluate "next lane"
        "danger_thresh": jnp.array(8.0),   # current-lane danger triggers escape
        "down_safe_gap": jnp.array(16.0),  # required margin below to allow DOWN
        "up_bias": jnp.array(0.5),         # bias making UP the default
    }


def _lane_threat(curr, prev, chicken_y, params, lane_y_offset):
    """Return minimum signed safety margin across cars whose lane is near chicken_y+offset."""
    cx_curr = curr[8:18]
    cy = curr[18:28]
    cw = curr[28:38]
    active = curr[48:58]
    cx_prev = prev[8:18]
    dx = cx_curr - cx_prev

    target_y = chicken_y + lane_y_offset
    # weight: 1 if car lane y is within ~8 px of target_y
    lane_match = (jnp.abs(cy - target_y) < 8.0).astype(jnp.float32)
    lane_match = lane_match * active

    # projected car x at lookahead frames
    proj_x = cx_curr + dx * params["lookahead"]
    car_center = proj_x + cw * 0.5
    chicken_center = CHICKEN_X + CHICKEN_WIDTH * 0.5
    # horizontal distance between chicken and car (after projection)
    gap = jnp.abs(car_center - chicken_center) - (cw * 0.5 + CHICKEN_WIDTH * 0.5)

    # also consider current overlap distance (no projection) to catch immediate threats
    cur_gap = jnp.abs(car_center - dx * params["lookahead"] - chicken_center) - (
        cw * 0.5 + CHICKEN_WIDTH * 0.5
    )
    min_gap = jnp.minimum(gap, cur_gap)

    # for non-matching lanes set gap large
    big = jnp.full_like(min_gap, 1e3)
    effective = jnp.where(lane_match > 0.5, min_gap, big)
    return jnp.min(effective)


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    chicken_y = curr[1]

    # safety margins (positive = safe, negative = overlap)
    cur_margin = _lane_threat(curr, prev, chicken_y, params, lane_y_offset=0.0)
    next_margin = _lane_threat(curr, prev, chicken_y, params, lane_y_offset=-16.0)
    down_margin = _lane_threat(curr, prev, chicken_y, params, lane_y_offset=+16.0)

    safe_gap = params["safe_gap"]
    danger = params["danger_thresh"]
    down_gap = params["down_safe_gap"]

    # Default: UP if both current and next lane are safe enough
    up_ok = (cur_margin > 0.0) & (next_margin > -safe_gap)

    # Emergency: current lane very dangerous, lower lane very safe -> DOWN
    in_danger = cur_margin < -danger
    down_ok = down_margin > down_gap

    # Always finish at top
    near_top = chicken_y < 25.0

    action = jnp.where(
        near_top,
        UP,
        jnp.where(
            in_danger & down_ok,
            DOWN,
            jnp.where(up_ok, UP, NOOP),
        ),
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
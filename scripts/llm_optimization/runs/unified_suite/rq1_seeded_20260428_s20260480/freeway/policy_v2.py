"""
Auto-generated policy v2
Generated at: 2026-04-28 14:08:08
"""

import jax
import jax.numpy as jnp

FRAME_SIZE = 88
NOOP = 0
UP = 1
DOWN = 2
CHICKEN_X = 44.0
CHICKEN_WIDTH = 6.0
NUM_CARS = 10

LANE_Y = jnp.array([27.0, 43.0, 59.0, 75.0, 91.0, 107.0, 123.0, 139.0, 155.0, 171.0])


def init_params():
    return {
        "ttc_curr": jnp.array(8.0),     # min time-to-collision needed in current lane
        "ttc_next": jnp.array(6.0),     # min time-to-collision needed in next lane
        "gap_min": jnp.array(7.0),      # min spatial gap regardless of TTC
        "horizon": jnp.array(12.0),     # max frames to look ahead
        "commit_band": jnp.array(6.0),  # if mid-transit, force UP
    }


def _lane_safe(curr, prev, lane_idx, params):
    """Return True if the lane indexed by lane_idx is safe to occupy soon."""
    cx_curr = curr[8:18]
    cy = curr[18:28]
    cw = curr[28:38]
    active = curr[48:58]
    cx_prev = prev[8:18]
    dx = cx_curr - cx_prev

    target_y = LANE_Y[lane_idx]
    lane_match = (jnp.abs(cy - target_y) < 8.0).astype(jnp.float32) * active

    car_center = cx_curr + cw * 0.5
    chicken_center = CHICKEN_X + CHICKEN_WIDTH * 0.5
    half_widths = cw * 0.5 + CHICKEN_WIDTH * 0.5

    # Signed horizontal offset: positive means car is to the right of chicken
    offset = car_center - chicken_center
    # Closing speed: positive means car approaching chicken
    closing = -jnp.sign(offset) * dx
    closing = jnp.maximum(closing, 1e-3)

    # Current spatial gap (negative if overlapping)
    gap_now = jnp.abs(offset) - half_widths
    # Time-to-collision: how many frames until car edge reaches chicken edge
    ttc = gap_now / closing

    # If car moving away (dx*sign opposite), ttc large; if approaching, finite
    moving_away = (jnp.sign(dx) * jnp.sign(offset)) > 0
    ttc_eff = jnp.where(moving_away, 1e3, ttc)

    # A car is dangerous if it's projected to hit within horizon AND currently close
    horizon = params["horizon"]
    gap_min = params["gap_min"]

    # Per-car danger score: low ttc OR small current gap
    danger_ttc = ttc_eff
    # for non-matching/inactive lanes, push ttc to large
    danger_ttc = jnp.where(lane_match > 0.5, danger_ttc, 1e3)
    danger_gap = jnp.where(lane_match > 0.5, gap_now, 1e3)

    min_ttc = jnp.min(danger_ttc)
    min_gap = jnp.min(danger_gap)

    return min_ttc, min_gap


def _current_lane_idx(chicken_y):
    diffs = jnp.abs(LANE_Y - chicken_y)
    return jnp.argmin(diffs)


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    chicken_y = curr[1]

    cur_idx = _current_lane_idx(chicken_y)
    next_idx = jnp.maximum(cur_idx - 1, 0)

    cur_ttc, cur_gap = _lane_safe(curr, prev, cur_idx, params)
    nxt_ttc, nxt_gap = _lane_safe(curr, prev, next_idx, params)

    # Safe = enough TTC AND enough current spatial gap
    cur_safe = (cur_ttc > params["ttc_curr"]) & (cur_gap > params["gap_min"])
    nxt_safe = (nxt_ttc > params["ttc_next"]) & (nxt_gap > params["gap_min"])

    # Mid-transit commit: if chicken is between lanes, keep moving UP
    nearest_lane_y = LANE_Y[cur_idx]
    mid_transit = jnp.abs(chicken_y - nearest_lane_y) > params["commit_band"]

    near_top = chicken_y < 25.0

    up_ok = cur_safe & nxt_safe

    action = jnp.where(
        near_top,
        UP,
        jnp.where(
            mid_transit,
            UP,
            jnp.where(up_ok, UP, NOOP),
        ),
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
"""
Auto-generated policy v3
Generated at: 2026-04-28 14:13:12
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


def init_params():
    return {
        "safe_gap": jnp.array(8.0),       # required positive x-margin to step UP
        "lookahead": jnp.array(4.0),      # frames to project car motion
        "danger_gap": jnp.array(4.0),     # current-lane margin below this -> escape
        "down_gap": jnp.array(4.0),       # required positive margin in lower lane for DOWN
        "lane_window": jnp.array(6.0),    # half-height of lane-match window
    }


def _min_margin(curr, prev, target_y, params):
    """Signed margin to nearest threatening car in lane near target_y.
    Uses signed approach: cars moving away or already past chicken are ignored.
    Returns min over current and projected gap.
    """
    cx_curr = curr[8:18]
    cy = curr[18:28]
    cw = curr[28:38]
    active = curr[48:58]
    cx_prev = prev[8:18]
    dx = cx_curr - cx_prev

    lane_window = params["lane_window"]
    lookahead = params["lookahead"]

    in_lane = (jnp.abs(cy - target_y) < lane_window).astype(jnp.float32) * active

    car_half = cw * 0.5
    chick_half = CHICKEN_WIDTH * 0.5
    chick_c = CHICKEN_X + chick_half

    car_c_now = cx_curr + car_half
    car_c_fut = cx_curr + car_half + dx * lookahead

    # Signed offset: positive = car is to the right of chicken
    off_now = car_c_now - chick_c
    off_fut = car_c_fut - chick_c

    # A car is threatening if it is currently overlapping OR crosses chicken
    # between now and the future projection (sign change of off, or stays near zero).
    crosses = (off_now * off_fut) <= 0.0
    # Also threatening if it's near and approaching
    approaching = (jnp.abs(off_fut) < jnp.abs(off_now))

    threat = crosses | approaching

    # Gap = min absolute distance during the horizon, minus combined half-widths
    min_abs_off = jnp.minimum(jnp.abs(off_now), jnp.abs(off_fut))
    # If crosses, true min during interval is ~0
    min_abs_off = jnp.where(crosses, jnp.zeros_like(min_abs_off), min_abs_off)

    gap = min_abs_off - (car_half + chick_half)

    # Cars not in lane, inactive, or not threatening -> safe (large gap)
    big = jnp.full_like(gap, 1e3)
    effective = jnp.where((in_lane > 0.5) & threat, gap, big)
    # Also: even non-threatening but currently overlapping cars should count
    cur_overlap = jnp.abs(off_now) - (car_half + chick_half)
    cur_eff = jnp.where(in_lane > 0.5, cur_overlap, big)

    return jnp.minimum(jnp.min(effective), jnp.min(cur_eff))


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    chicken_y = curr[1]

    cur_margin = _min_margin(curr, prev, chicken_y, params)
    next_margin = _min_margin(curr, prev, chicken_y - 16.0, params)
    down_margin = _min_margin(curr, prev, chicken_y + 16.0, params)

    safe_gap = params["safe_gap"]
    danger_gap = params["danger_gap"]
    down_gap = params["down_gap"]

    near_top = chicken_y < 25.0

    # UP allowed when current lane is safe and next lane has positive margin
    up_ok = (cur_margin > 0.0) & (next_margin > safe_gap)

    # DOWN escape: current lane is dangerously overlapping AND below is clear
    in_danger = cur_margin < -danger_gap
    down_ok = down_margin > down_gap

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
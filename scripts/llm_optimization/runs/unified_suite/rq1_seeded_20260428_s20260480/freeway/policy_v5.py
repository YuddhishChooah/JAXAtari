"""
Auto-generated policy v5
Generated at: 2026-04-28 14:22:29
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
        "cur_horizon": jnp.array(5.0),       # frames to look ahead on current lane
        "next_horizon": jnp.array(5.0),      # frames to look ahead on entry lane
        "entry_delay": jnp.array(4.0),       # frames until chicken enters next lane
        "x_slack": jnp.array(2.5),           # extra x slack on collision check
        "danger_thresh": jnp.array(1.0),     # overlap depth that triggers DOWN escape
    }


def _lane_min_clearance(curr, prev, lane_idx, t_start, t_end, x_slack):
    """Return the minimum x-clearance (negative = collision) over frames
    [t_start, t_end] for cars in given lane.
    """
    cx_curr = curr[8:18]
    cy = curr[18:28]
    cw = curr[28:38]
    active = curr[48:58]
    cx_prev = prev[8:18]
    dx = cx_curr - cx_prev

    target_y = LANE_Y[lane_idx]
    in_lane = (jnp.abs(cy - target_y) < 5.0).astype(jnp.float32) * active

    chicken_cx = CHICKEN_X + CHICKEN_WIDTH * 0.5
    half_sum = cw * 0.5 + CHICKEN_WIDTH * 0.5 + x_slack

    # Sample several timesteps and take the minimum clearance per car.
    ts = jnp.linspace(t_start, t_end, 5)

    def clearance_at(t):
        car_cx_t = cx_curr + dx * t
        gap = jnp.abs(chicken_cx - car_cx_t) - half_sum
        return gap

    gaps = jax.vmap(clearance_at)(ts)  # shape (5, 10)
    min_per_car = jnp.min(gaps, axis=0)

    # Inactive / out-of-lane cars: report large clearance.
    min_per_car = jnp.where(in_lane > 0.5, min_per_car, jnp.full_like(min_per_car, 1e3))
    return jnp.min(min_per_car)


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    chicken_y = curr[1]

    dists = jnp.abs(LANE_Y - chicken_y)
    cur_lane = jnp.argmin(dists)
    next_lane = jnp.maximum(cur_lane - 1, 0)
    down_lane = jnp.minimum(cur_lane + 1, 9)

    cur_h = params["cur_horizon"]
    nxt_h = params["next_horizon"]
    delay = params["entry_delay"]
    slack = params["x_slack"]

    # Current lane: frames 0..cur_h
    cur_clear = _lane_min_clearance(curr, prev, cur_lane, 0.0, cur_h, slack)
    # Next lane: evaluated around the time chicken arrives there
    next_clear = _lane_min_clearance(curr, prev, next_lane,
                                     delay - 1.0, delay + nxt_h, slack)
    # Down lane: immediate safety
    down_clear = _lane_min_clearance(curr, prev, down_lane, 0.0, cur_h, slack)

    nearest_y = LANE_Y[cur_lane]
    between_lanes = jnp.abs(chicken_y - nearest_y) > 3.0
    near_top = chicken_y < 25.0

    cur_safe = cur_clear > 0.0
    next_safe = next_clear > 0.0
    up_ok = cur_safe & next_safe

    # DOWN escape: only if current lane is clearly dangerous AND down lane safe.
    in_danger = cur_clear < (-params["danger_thresh"])
    down_safe = down_clear > 1.0
    escape = in_danger & down_safe & (~between_lanes) & (~near_top)

    action = jnp.where(
        near_top, UP,
        jnp.where(between_lanes, UP,
                  jnp.where(escape, DOWN,
                            jnp.where(up_ok, UP, NOOP)))
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
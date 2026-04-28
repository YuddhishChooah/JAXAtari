"""
Auto-generated policy v1
Generated at: 2026-04-27 17:24:59
"""

import jax
import jax.numpy as jnp

FRAME_SIZE = 88
NOOP = 0
UP = 1
DOWN = 2
CHICKEN_X = 44
CHICKEN_WIDTH = 6
CHICKEN_HEIGHT = 8
TOP_BORDER = 15
BOTTOM_BORDER = 180
N_CARS = 10


def init_params():
    return {
        "lookahead": 6.0,        # frames to project car positions
        "x_margin": 10.0,        # extra horizontal safety margin around chicken
        "lane_band": 10.0,       # vertical band defining "in lane"
        "next_lane_dy": 14.0,    # how far ahead to look for next lane
        "danger_thresh": 0.5,    # threshold for danger score
        "emergency_thresh": 0.9, # threshold for DOWN escape
    }


def _lane_danger(frame_curr, frame_prev, cy, params):
    # cy: chicken y to evaluate the lane around
    cars_x = frame_curr[8:18]
    cars_y = frame_curr[18:28]
    cars_w = frame_curr[28:38]
    cars_active = frame_curr[48:58]
    prev_x = frame_prev[8:18]
    dx = cars_x - prev_x

    # Is this car in the lane near cy?
    in_lane = jnp.abs(cars_y - cy) < params["lane_band"]

    # Project car x positions a few frames ahead
    future_x = cars_x + dx * params["lookahead"]

    # Check overlap between [future_x, future_x + w] and chicken bounds
    chick_lo = CHICKEN_X - params["x_margin"]
    chick_hi = CHICKEN_X + CHICKEN_WIDTH + params["x_margin"]

    # overlap if any frame between now and lookahead has overlap
    # Use both current and projected
    cur_overlap = (cars_x < chick_hi) & (cars_x + cars_w > chick_lo)
    fut_overlap = (future_x < chick_hi) & (future_x + cars_w > chick_lo)

    # Also check approach: a car heading toward chicken
    # If car is to the left of chicken and moving right, or right and moving left
    car_center = cars_x + cars_w * 0.5
    approaching = ((car_center < CHICKEN_X) & (dx > 0)) | (
        (car_center > CHICKEN_X) & (dx < 0)
    )

    danger_per_car = in_lane & cars_active.astype(bool) & (
        cur_overlap | fut_overlap | approaching & (jnp.abs(car_center - CHICKEN_X) < 30.0)
    )
    return jnp.any(danger_per_car).astype(jnp.float32)


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    cy = curr[1]

    # Danger of staying in current lane
    cur_danger = _lane_danger(curr, prev, cy, params)
    # Danger of next (upper) lane
    next_y = cy - params["next_lane_dy"]
    next_danger = _lane_danger(curr, prev, next_y, params)
    # Danger of lane below (for emergency DOWN)
    down_y = cy + params["next_lane_dy"]
    down_danger = _lane_danger(curr, prev, down_y, params)

    near_top = cy < (TOP_BORDER + 12.0)

    # Default: go UP
    action = jnp.int32(UP)

    # If next lane is dangerous, wait (NOOP) unless near top
    safe_up = (next_danger < params["danger_thresh"]) | near_top
    action = jnp.where(safe_up, jnp.int32(UP), jnp.int32(NOOP))

    # Emergency: current lane very dangerous and lane below safe -> DOWN
    emergency = (cur_danger > params["emergency_thresh"]) & (
        down_danger < params["danger_thresh"]
    ) & (~near_top) & (cy < BOTTOM_BORDER - 8.0) & (next_danger > params["danger_thresh"])
    action = jnp.where(emergency, jnp.int32(DOWN), action)

    return action


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
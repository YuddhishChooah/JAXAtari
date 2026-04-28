"""
Auto-generated policy v1
Generated at: 2026-04-26 18:11:42
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
NUM_CARS = 10


def init_params():
    return {
        "safe_gap": jnp.float32(12.0),
        "lookahead": jnp.float32(6.0),
        "lane_band": jnp.float32(10.0),
        "down_thresh": jnp.float32(4.0),
        "up_bias": jnp.float32(0.5),
        "finish_y": jnp.float32(30.0),
    }


def _split(obs_flat):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]
    return prev, curr


def _danger(prev, curr, params):
    # For each car, compute future overlap risk with chicken column.
    cx = curr[0]
    cy = curr[1]
    car_x = curr[8:8 + NUM_CARS]
    car_y = curr[18:18 + NUM_CARS]
    car_w = curr[28:28 + NUM_CARS]
    car_h = curr[38:38 + NUM_CARS]
    car_act = curr[48:48 + NUM_CARS]
    prev_x = prev[8:8 + NUM_CARS]
    dx = car_x - prev_x

    look = params["lookahead"]
    band = params["lane_band"]
    gap = params["safe_gap"]

    # project car center forward
    fx = car_x + dx * look
    # horizontal distance from chicken column edges
    chicken_left = cx
    chicken_right = cx + CHICKEN_WIDTH
    car_left = jnp.minimum(car_x, fx)
    car_right = jnp.maximum(car_x + car_w, fx + car_w)

    horiz_clear = (car_right + gap < chicken_left) | (car_left - gap > chicken_right)
    # vertical proximity to chicken (per lane)
    dy = jnp.abs(car_y - cy)
    vert_close = dy < band

    risk = vert_close & (~horiz_clear) & (car_act > 0.5)
    return risk


def _next_lane_danger(prev, curr, params, dy_offset):
    # Simulate chicken moved by dy_offset (negative = up) and check danger near new y.
    cx = curr[0]
    cy = curr[1] + dy_offset
    car_x = curr[8:8 + NUM_CARS]
    car_y = curr[18:18 + NUM_CARS]
    car_w = curr[28:28 + NUM_CARS]
    car_act = curr[48:48 + NUM_CARS]
    prev_x = prev[8:8 + NUM_CARS]
    dx = car_x - prev_x

    look = params["lookahead"]
    band = params["lane_band"]
    gap = params["safe_gap"]

    fx = car_x + dx * look
    chicken_left = cx
    chicken_right = cx + CHICKEN_WIDTH
    car_left = jnp.minimum(car_x, fx)
    car_right = jnp.maximum(car_x + car_w, fx + car_w)

    horiz_clear = (car_right + gap < chicken_left) | (car_left - gap > chicken_right)
    dy = jnp.abs(car_y - cy)
    vert_close = dy < band

    risk = vert_close & (~horiz_clear) & (car_act > 0.5)
    return jnp.sum(risk.astype(jnp.float32))


def policy(obs_flat, params):
    prev, curr = _split(obs_flat)
    cy = curr[1]

    # Risk if we step UP into next lane (~ -16 in y) and current lane.
    risk_up = _next_lane_danger(prev, curr, params, -16.0)
    risk_stay = _next_lane_danger(prev, curr, params, 0.0)
    risk_down = _next_lane_danger(prev, curr, params, 16.0)

    # Finish: if near top, force UP.
    near_top = cy < params["finish_y"]

    # Decision logic.
    up_safe = risk_up < params["up_bias"] + 0.5
    stay_dangerous = risk_stay > params["down_thresh"]

    action = jnp.where(
        near_top,
        UP,
        jnp.where(
            up_safe,
            UP,
            jnp.where(
                stay_dangerous & (risk_down < risk_stay),
                DOWN,
                NOOP,
            ),
        ),
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
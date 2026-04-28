"""
Auto-generated policy v1
Generated at: 2026-04-26 19:59:37
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
TOP_BORDER = 15.0
NUM_CARS = 10


def init_params():
    return {
        "safe_gap": jnp.array(8.0),
        "lookahead": jnp.array(4.0),
        "lane_radius": jnp.array(12.0),
        "down_margin": jnp.array(3.0),
        "up_bias": jnp.array(0.5),
    }


def _lane_danger(frame_curr, frame_prev, y_ref, lookahead, params):
    # For each car, project x position `lookahead` frames ahead and check overlap
    car_x = frame_curr[8:18]
    car_y = frame_curr[18:28]
    car_w = frame_curr[28:38]
    car_h = frame_curr[38:48]
    car_active = frame_curr[48:58]
    car_dx = car_x - frame_prev[8:18]

    # only cars in the lane near y_ref
    in_lane = jnp.abs(car_y - y_ref) < params["lane_radius"]

    # overlap check at current and projected times
    chicken_left = CHICKEN_X
    chicken_right = CHICKEN_X + CHICKEN_WIDTH
    safe_gap = params["safe_gap"]

    def overlap_at(t):
        cx = car_x + car_dx * t
        car_left = cx - safe_gap
        car_right = cx + car_w + safe_gap
        return (car_right > chicken_left) & (car_left < chicken_right)

    danger_now = overlap_at(0.0)
    danger_soon = overlap_at(lookahead)
    danger = (danger_now | danger_soon) & in_lane & (car_active > 0.5)
    return jnp.any(danger)


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    chicken_y = curr[1]
    lookahead = params["lookahead"]

    # current lane, next lane (up = smaller y), prev lane (down = larger y)
    y_curr = chicken_y
    y_up = chicken_y - 16.0
    y_down = chicken_y + 16.0

    danger_curr = _lane_danger(curr, prev, y_curr, lookahead, params)
    danger_up = _lane_danger(curr, prev, y_up, lookahead, params)
    danger_down = _lane_danger(curr, prev, y_down, lookahead, params * 0 + params["safe_gap"]) if False else _lane_danger(curr, prev, y_down, params["down_margin"], params)

    near_top = chicken_y < (TOP_BORDER + 18.0)

    # default action: UP
    # if next lane up is dangerous and current lane is safe, NOOP
    # if current lane is dangerous and down lane is clearly safe, DOWN (emergency)
    safe_up = jnp.logical_not(danger_up)
    safe_curr = jnp.logical_not(danger_curr)
    safe_down = jnp.logical_not(danger_down)

    # bias toward UP when up_bias is high
    go_up = safe_up | near_top | (params["up_bias"] > 0.8)
    emergency = danger_curr & safe_down & jnp.logical_not(near_top)

    action = jnp.where(go_up, UP, NOOP)
    action = jnp.where(emergency & jnp.logical_not(safe_up), DOWN, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
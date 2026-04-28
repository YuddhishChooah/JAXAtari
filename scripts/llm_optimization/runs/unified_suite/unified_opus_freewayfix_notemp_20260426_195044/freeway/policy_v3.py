"""
Auto-generated policy v3
Generated at: 2026-04-26 20:03:01
"""

import jax
import jax.numpy as jnp

FRAME_SIZE = 88
NOOP = 0
UP = 1
DOWN = 2
CHICKEN_X = 44.0
CHICKEN_WIDTH = 6.0
TOP_BORDER = 15.0

LANE_Y = jnp.array([27.0, 43.0, 59.0, 75.0, 91.0, 107.0, 123.0, 139.0, 155.0, 171.0])


def init_params():
    return {
        "gap_base": jnp.array(3.0),
        "gap_approach": jnp.array(7.0),
        "lookahead_curr": jnp.array(3.5),
        "lookahead_next": jnp.array(9.0),
        "lane_radius": jnp.array(5.0),
    }


def _lane_danger(curr, prev, y_ref, params, horizon):
    car_x = curr[8:18]
    car_y = curr[18:28]
    car_w = curr[28:38]
    car_active = curr[48:58]
    car_dx = car_x - prev[8:18]

    in_lane = jnp.abs(car_y - y_ref) < params["lane_radius"]
    active = car_active > 0.5
    valid = in_lane & active

    chicken_left = CHICKEN_X
    chicken_right = CHICKEN_X + CHICKEN_WIDTH

    gap_base = params["gap_base"]
    gap_app = params["gap_approach"]

    # Both sides get base gap; approaching side gets extra inflation
    moving_left = car_dx < 0.0
    moving_right = car_dx > 0.0
    gap_left = gap_base + jnp.where(moving_left, gap_app, 0.0)
    gap_right = gap_base + jnp.where(moving_right, gap_app, 0.0)

    def overlap_at(t):
        cx = car_x + car_dx * t
        car_left = cx - gap_left
        car_right = cx + car_w + gap_right
        return (car_right > chicken_left) & (car_left < chicken_right)

    d0 = overlap_at(0.0)
    d1 = overlap_at(horizon * 0.5)
    d2 = overlap_at(horizon)

    danger = (d0 | d1 | d2) & valid
    return jnp.any(danger)


def _nearest_lane_idx(chicken_y):
    diffs = jnp.abs(LANE_Y - chicken_y)
    return jnp.argmin(diffs)


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    chicken_y = curr[1]

    idx = _nearest_lane_idx(chicken_y)
    idx_up = jnp.maximum(idx - 1, 0)
    idx_down = jnp.minimum(idx + 1, 9)

    y_curr = LANE_Y[idx]
    y_up = LANE_Y[idx_up]
    y_down = LANE_Y[idx_down]

    L_curr = params["lookahead_curr"]
    L_next = params["lookahead_next"]

    danger_curr = _lane_danger(curr, prev, y_curr, params, L_curr)
    danger_up = _lane_danger(curr, prev, y_up, params, L_next)
    danger_down = _lane_danger(curr, prev, y_down, params, L_curr)

    safe_up = jnp.logical_not(danger_up)
    safe_down = jnp.logical_not(danger_down)

    # Default: UP if next lane is safe, else NOOP
    action = jnp.where(safe_up, UP, NOOP)

    # Emergency DOWN: current lane dangerous, down is safe
    emergency = danger_curr & safe_down & jnp.logical_not(safe_up)
    action = jnp.where(emergency, DOWN, action)

    # If at top already, just push UP to score
    at_top = chicken_y < TOP_BORDER + 8.0
    action = jnp.where(at_top, UP, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
"""
Auto-generated policy v2
Generated at: 2026-04-26 20:01:19
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


def init_params():
    return {
        "safe_gap_up": jnp.array(6.0),
        "safe_gap_down": jnp.array(10.0),
        "lookahead": jnp.array(5.0),
        "lane_radius": jnp.array(8.0),
        "vel_scale": jnp.array(1.5),
    }


def _lane_danger(curr, prev, y_ref, params, gap):
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

    # direction-aware gap: inflate only on the approaching side
    approach_from_right = car_dx < 0.0  # car moving left, threat on its left side
    # left edge gap: if car moves left (approaching from right), extend left
    gap_left = jnp.where(approach_from_right, gap, 1.0)
    # right edge gap: if car moves right, extend right
    gap_right = jnp.where(car_dx > 0.0, gap, 1.0)

    L = params["lookahead"]

    def overlap_at(t):
        cx = car_x + car_dx * t
        car_left = cx - gap_left
        car_right = cx + car_w + gap_right
        return (car_right > chicken_left) & (car_left < chicken_right)

    d0 = overlap_at(0.0)
    d1 = overlap_at(L * 0.5)
    d2 = overlap_at(L)
    d3 = overlap_at(L * params["vel_scale"])

    danger = (d0 | d1 | d2 | d3) & valid
    return jnp.any(danger)


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    chicken_y = curr[1]

    y_curr = chicken_y
    y_up = chicken_y - 16.0
    y_down = chicken_y + 16.0

    gap_up = params["safe_gap_up"]
    gap_down = params["safe_gap_down"]

    danger_curr = _lane_danger(curr, prev, y_curr, params, gap_up)
    danger_up = _lane_danger(curr, prev, y_up, params, gap_up)
    danger_down = _lane_danger(curr, prev, y_down, params, gap_down)

    safe_up = jnp.logical_not(danger_up)
    safe_down = jnp.logical_not(danger_down)

    near_top = chicken_y < (TOP_BORDER + 24.0)

    # Default: UP if next lane safe, else NOOP
    action = jnp.where(safe_up, UP, NOOP)

    # Force UP when near top to grab high-value last lanes
    action = jnp.where(near_top, UP, action)

    # Emergency DOWN: only when both current and up are dangerous AND down is strongly safe
    emergency = danger_curr & danger_up & safe_down & jnp.logical_not(near_top)
    action = jnp.where(emergency, DOWN, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
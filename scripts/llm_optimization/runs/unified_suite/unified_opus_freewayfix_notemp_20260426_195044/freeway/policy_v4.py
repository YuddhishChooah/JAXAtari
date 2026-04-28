"""
Auto-generated policy v4
Generated at: 2026-04-26 20:05:05
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
        "gap_curr": jnp.array(9.0),
        "gap_up": jnp.array(5.0),
        "lookahead": jnp.array(6.0),
        "lane_radius": jnp.array(8.0),
        "horizon_mult": jnp.array(2.0),
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

    gap_left = jnp.where(car_dx < 0.0, gap, 1.0)
    gap_right = jnp.where(car_dx > 0.0, gap, 1.0)

    L = params["lookahead"]
    H = L * params["horizon_mult"]

    def overlap_at(t):
        cx = car_x + car_dx * t
        car_left = cx - gap_left
        car_right = cx + car_w + gap_right
        return (car_right > chicken_left) & (car_left < chicken_right)

    d0 = overlap_at(0.0)
    d1 = overlap_at(L * 0.5)
    d2 = overlap_at(L)
    d3 = overlap_at(H)

    danger = (d0 | d1 | d2 | d3) & valid
    return jnp.any(danger)


def _nearest_lane_above(y):
    # pick the closest LANE_Y strictly above y; fallback to top lane
    diffs = y - LANE_Y  # positive when lane is above chicken
    valid = diffs > 1.0
    big = jnp.where(valid, diffs, 1e6)
    idx = jnp.argmin(big)
    return LANE_Y[idx]


def _nearest_lane_below(y):
    diffs = LANE_Y - y
    valid = diffs > 1.0
    big = jnp.where(valid, diffs, 1e6)
    idx = jnp.argmin(big)
    return LANE_Y[idx]


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    chicken_y = curr[1]

    y_up = _nearest_lane_above(chicken_y)
    y_down = _nearest_lane_below(chicken_y)

    # nearest lane (chicken's current band)
    dists = jnp.abs(LANE_Y - chicken_y)
    y_curr = LANE_Y[jnp.argmin(dists)]

    gap_curr = params["gap_curr"]
    gap_up = params["gap_up"]

    danger_curr = _lane_danger(curr, prev, y_curr, params, gap_curr)
    danger_up = _lane_danger(curr, prev, y_up, params, gap_up)
    danger_down = _lane_danger(curr, prev, y_down, params, gap_up)

    # straddle: if chicken not close to a lane center, also include the other adjacent lane
    straddle = jnp.min(dists) > 4.0
    danger_curr = jnp.where(straddle, danger_curr | danger_up, danger_curr)

    safe_up = jnp.logical_not(danger_up)
    safe_down = jnp.logical_not(danger_down)

    # default: UP when next lane safe, else NOOP
    action = jnp.where(safe_up, UP, NOOP)

    # emergency DOWN: current lane dangerous, up dangerous, down clearly safe
    emergency = danger_curr & danger_up & safe_down
    action = jnp.where(emergency, DOWN, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
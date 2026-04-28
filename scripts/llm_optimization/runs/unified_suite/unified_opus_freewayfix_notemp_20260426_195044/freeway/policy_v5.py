"""
Auto-generated policy v5
Generated at: 2026-04-26 20:07:19
"""

import jax
import jax.numpy as jnp

FRAME_SIZE = 88
NOOP = 0
UP = 1
DOWN = 2
CHICKEN_X = 44.0
CHICKEN_WIDTH = 6.0

LANE_Y = jnp.array([27.0, 43.0, 59.0, 75.0, 91.0, 107.0, 123.0, 139.0, 155.0, 171.0])


def init_params():
    return {
        "gap_up": jnp.array(7.0),
        "gap_curr": jnp.array(4.0),
        "gap_down": jnp.array(9.0),
        "lookahead": jnp.array(6.0),
        "imminent": jnp.array(3.0),
        "lane_radius": jnp.array(7.0),
    }


def _lane_overlap(curr, prev, y_ref, params, gap, t_samples):
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

    def overlap_at(t):
        cx = car_x + car_dx * t
        car_left = cx - gap_left
        car_right = cx + car_w + gap_right
        hit = (car_right > chicken_left) & (car_left < chicken_right)
        return jnp.any(hit & valid)

    flags = jnp.array([overlap_at(t) for t in t_samples])
    return jnp.any(flags)


def _nearest_lane_above(y):
    diffs = y - LANE_Y
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
    dists = jnp.abs(LANE_Y - chicken_y)
    y_curr = LANE_Y[jnp.argmin(dists)]

    L = params["lookahead"]
    imm = params["imminent"]

    # next-lane gate: 2 samples at projected entry y
    y_entry = chicken_y - 3.0
    danger_up = _lane_overlap(
        curr, prev, y_up, params, params["gap_up"],
        (0.0, L)
    )
    # also check at entry y in case mid-transition
    danger_entry = _lane_overlap(
        curr, prev, y_entry, params, params["gap_up"],
        (0.0, L * 0.5)
    )
    danger_up = danger_up | danger_entry

    # current-lane imminent danger (short horizon, more samples)
    danger_curr = _lane_overlap(
        curr, prev, y_curr, params, params["gap_curr"],
        (0.0, imm * 0.5, imm)
    )

    # down lane must be very safe
    danger_down = _lane_overlap(
        curr, prev, y_down, params, params["gap_down"],
        (0.0, L * 0.5, L)
    )

    safe_up = jnp.logical_not(danger_up)
    safe_down = jnp.logical_not(danger_down)

    # mid-transition: chicken not snapped to lane center
    mid_transition = jnp.min(dists) > 4.0

    # default: UP whenever next lane safe
    action = jnp.where(safe_up, UP, NOOP)

    # emergency DOWN: only when current lane imminently dangerous,
    # next lane dangerous, lower lane very safe, and not mid-transition
    emergency = danger_curr & danger_up & safe_down & jnp.logical_not(mid_transition)
    action = jnp.where(emergency, DOWN, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
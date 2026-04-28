"""
Auto-generated policy v3
Generated at: 2026-04-26 18:16:04
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

LANE_Y = jnp.array([27., 43., 59., 75., 91., 107., 123., 139., 155., 171.],
                   dtype=jnp.float32)


def init_params():
    return {
        "entry_delay": jnp.float32(2.5),     # frames until chicken arrives in next lane
        "dwell": jnp.float32(4.0),           # frames to remain safe after arrival
        "margin_px": jnp.float32(3.0),       # extra horizontal body margin
        "finish_y": jnp.float32(30.0),       # force UP when y < finish_y
        "danger_horizon": jnp.float32(2.5),  # frames-ahead to trigger DOWN from current lane
        "down_safe_time": jnp.float32(5.0),  # required clearance in down lane to retreat
    }


def _split(obs_flat):
    return obs_flat[0:FRAME_SIZE], obs_flat[FRAME_SIZE:2 * FRAME_SIZE]


def _lane_index(cy):
    d = jnp.abs(LANE_Y - cy)
    return jnp.argmin(d)


def _overlap_at_time(prev, curr, lane_idx, t, margin):
    """Will the car in lane_idx overlap the chicken column at time t (frames ahead)?"""
    car_x = curr[8 + lane_idx]
    car_w = curr[28 + lane_idx]
    car_act = curr[48 + lane_idx]
    prev_x = prev[8 + lane_idx]
    dx = car_x - prev_x

    future_x = car_x + dx * t
    car_l = future_x - margin
    car_r = future_x + car_w + margin
    chicken_l = CHICKEN_X
    chicken_r = CHICKEN_X + CHICKEN_WIDTH

    overlap = (car_r >= chicken_l) & (car_l <= chicken_r)
    overlap = overlap & (car_act > 0.5)
    return overlap


def _lane_safe_window(prev, curr, lane_idx, t_start, t_end, margin):
    """Safe if no overlap at any of several sample times in [t_start, t_end]."""
    # Sample 5 points across the window.
    ts = jnp.linspace(t_start, t_end, 5)
    overlaps = jax.vmap(
        lambda t: _overlap_at_time(prev, curr, lane_idx, t, margin)
    )(ts)
    return ~jnp.any(overlaps)


def _imminent_in_lane(prev, curr, lane_idx, horizon, margin):
    """Will lane_idx have car overlap within `horizon` frames (including now)?"""
    ts = jnp.linspace(0.0, horizon, 4)
    overlaps = jax.vmap(
        lambda t: _overlap_at_time(prev, curr, lane_idx, t, margin)
    )(ts)
    return jnp.any(overlaps)


def policy(obs_flat, params):
    prev, curr = _split(obs_flat)
    cy = curr[1]

    lane = _lane_index(cy)
    next_lane = jnp.maximum(lane - 1, 0)
    nn_lane = jnp.maximum(lane - 2, 0)
    down_lane = jnp.minimum(lane + 1, 9)

    margin = params["margin_px"]
    t_in = params["entry_delay"]
    t_out = params["entry_delay"] + params["dwell"]

    # UP is safe iff next lane stays clear over arrival window
    up_safe = _lane_safe_window(prev, curr, next_lane, t_in, t_out, margin)

    # Two-lane lookahead: lane after next must be enterable a bit later.
    nn_safe = _lane_safe_window(prev, curr, nn_lane,
                                t_in + params["dwell"],
                                t_out + params["dwell"] + 2.0,
                                margin)

    near_top = cy < params["finish_y"]

    # DOWN trigger: imminent overlap in current lane and down lane is safe.
    cur_danger = _imminent_in_lane(prev, curr, lane,
                                   params["danger_horizon"], margin)
    down_safe = _lane_safe_window(prev, curr, down_lane,
                                  0.5, params["down_safe_time"], margin)

    # Decision:
    # 1. near top -> UP (sprint to finish)
    # 2. up_safe and (nn_safe or near top half) -> UP
    # 3. current lane imminent collision and down safe -> DOWN
    # 4. else NOOP
    go_up = up_safe & nn_safe
    retreat = cur_danger & down_safe & (~up_safe)

    action = jnp.where(
        near_top, UP,
        jnp.where(go_up, UP,
                  jnp.where(retreat, DOWN, NOOP))
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
"""
Auto-generated policy v5
Generated at: 2026-04-26 18:24:14
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
        "entry_delay": jnp.float32(2.0),     # frames until chicken enters next lane
        "occupancy": jnp.float32(8.0),       # frames chicken occupies a lane
        "safety_pad": jnp.float32(3.0),      # extra px around chicken for safety
        "down_window": jnp.float32(6.0),     # frames lookahead for current-lane danger
        "plan_horizon": jnp.float32(14.0),   # cap on lookahead frames
    }


def _split(obs_flat):
    return obs_flat[0:FRAME_SIZE], obs_flat[FRAME_SIZE:2 * FRAME_SIZE]


def _lane_index(cy):
    d = jnp.abs(LANE_Y - cy)
    return jnp.argmin(d)


def _will_overlap(prev, curr, lane_idx, t_start, t_end, pad):
    """Predict whether car in lane_idx overlaps chicken column during [t_start, t_end].
    Linear extrapolation using dx = curr - prev."""
    car_x = curr[8 + lane_idx]
    car_w = curr[28 + lane_idx]
    car_act = curr[48 + lane_idx]
    prev_x = prev[8 + lane_idx]
    dx = car_x - prev_x

    chicken_l = CHICKEN_X - pad
    chicken_r = CHICKEN_X + CHICKEN_WIDTH + pad

    # Car edges at t_start and t_end
    car_l_a = car_x + dx * t_start
    car_r_a = car_x + car_w + dx * t_start
    car_l_b = car_x + dx * t_end
    car_r_b = car_x + car_w + dx * t_end

    # Span of car left/right over the window
    car_l_min = jnp.minimum(car_l_a, car_l_b)
    car_r_max = jnp.maximum(car_r_a, car_r_b)

    overlap = (car_r_max >= chicken_l) & (car_l_min <= chicken_r)
    overlap = overlap & (car_act > 0.5)
    return overlap


def _lane_safe_entry(prev, curr, lane_idx, params):
    """Is it safe to step into lane_idx now?"""
    t0 = params["entry_delay"]
    t1 = jnp.minimum(t0 + params["occupancy"], params["plan_horizon"])
    pad = params["safety_pad"]
    hit = _will_overlap(prev, curr, lane_idx, t0, t1, pad)
    return ~hit


def _lane_danger_soon(prev, curr, lane_idx, params):
    """Is current lane about to be hit within down_window?"""
    pad = params["safety_pad"]
    hit = _will_overlap(prev, curr, lane_idx, 0.0, params["down_window"], pad)
    return hit


def policy(obs_flat, params):
    prev, curr = _split(obs_flat)
    cy = curr[1]

    lane = _lane_index(cy)
    next_lane = jnp.maximum(lane - 1, 0)
    nn_lane = jnp.maximum(lane - 2, 0)
    down_lane = jnp.minimum(lane + 1, 9)

    up_safe = _lane_safe_entry(prev, curr, next_lane, params)
    up_safe2 = _lane_safe_entry(prev, curr, nn_lane, params)
    cur_danger = _lane_danger_soon(prev, curr, lane, params)
    down_safe = _lane_safe_entry(prev, curr, down_lane, params)

    # Near top: just go up; final lane handled by predictor anyway
    near_top = cy < 20.0

    # Prefer UP when the next lane is safe.
    # If next lane unsafe but current lane is about to be hit, try DOWN if safe.
    # Otherwise NOOP and wait.
    go_up = near_top | up_safe
    go_down = (~up_safe) & cur_danger & down_safe

    action = jnp.where(
        go_up, UP,
        jnp.where(go_down, DOWN, NOOP)
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
"""
Auto-generated policy v2
Generated at: 2026-04-26 18:13:38
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
        "safe_time": jnp.float32(8.0),     # frames of TTC margin required
        "clear_px": jnp.float32(10.0),     # horizontal clearance considered safe now
        "finish_y": jnp.float32(30.0),     # force UP when y < finish_y
        "down_ttc": jnp.float32(3.0),      # if current lane TTC < this, consider DOWN
        "wait_horizon": jnp.float32(20.0), # frames willing to wait
    }


def _split(obs_flat):
    return obs_flat[0:FRAME_SIZE], obs_flat[FRAME_SIZE:2 * FRAME_SIZE]


def _lane_index(cy):
    # nearest lane index by y
    d = jnp.abs(LANE_Y - cy)
    return jnp.argmin(d)


def _ttc_for_lane(prev, curr, lane_idx):
    """Time-to-collision (frames) for the car in `lane_idx`.
    Returns a large value if safe / passed / inactive."""
    car_x = curr[8 + lane_idx]
    car_w = curr[28 + lane_idx]
    car_act = curr[48 + lane_idx]
    prev_x = prev[8 + lane_idx]
    dx = car_x - prev_x

    chicken_l = CHICKEN_X
    chicken_r = CHICKEN_X + CHICKEN_WIDTH
    car_l = car_x
    car_r = car_x + car_w

    # Currently overlapping horizontally?
    overlap_now = (car_r >= chicken_l) & (car_l <= chicken_r)

    # Distance to the chicken column along travel direction.
    # If car moving right (dx>0): gap = chicken_l - car_r (positive => car is behind/left)
    # If car moving left  (dx<0): gap = car_l - chicken_r
    moving_right = dx > 0.05
    moving_left = dx < -0.05

    gap_right = chicken_l - car_r  # frames until front reaches chicken_l
    gap_left = car_l - chicken_r

    # Compute TTC; only positive gaps count as approaching
    speed = jnp.abs(dx) + 1e-3
    ttc_right = jnp.where(gap_right > 0, gap_right / speed, 1e6)
    ttc_left = jnp.where(gap_left > 0, gap_left / speed, 1e6)

    ttc = jnp.where(moving_right, ttc_right,
                    jnp.where(moving_left, ttc_left, 1e6))

    # If car is currently overlapping the chicken column, treat as imminent.
    ttc = jnp.where(overlap_now, 0.0, ttc)

    # Inactive cars: no threat.
    ttc = jnp.where(car_act > 0.5, ttc, 1e6)
    return ttc


def _min_horizontal_clear(curr, lane_idx):
    car_x = curr[8 + lane_idx]
    car_w = curr[28 + lane_idx]
    car_act = curr[48 + lane_idx]
    chicken_l = CHICKEN_X
    chicken_r = CHICKEN_X + CHICKEN_WIDTH
    car_l = car_x
    car_r = car_x + car_w
    # signed clearance: distance from chicken column to car edges
    clear = jnp.minimum(jnp.abs(car_r - chicken_l), jnp.abs(car_l - chicken_r))
    overlap = (car_r >= chicken_l) & (car_l <= chicken_r)
    clear = jnp.where(overlap, 0.0, clear)
    clear = jnp.where(car_act > 0.5, clear, 1e6)
    return clear


def _lane_safe(prev, curr, lane_idx, params):
    ttc = _ttc_for_lane(prev, curr, lane_idx)
    clear = _min_horizontal_clear(curr, lane_idx)
    # safe if TTC large OR currently far clear
    return (ttc > params["safe_time"]) | (clear > params["clear_px"])


def policy(obs_flat, params):
    prev, curr = _split(obs_flat)
    cy = curr[1]

    lane = _lane_index(cy)
    next_lane = jnp.maximum(lane - 1, 0)
    nn_lane = jnp.maximum(lane - 2, 0)
    down_lane = jnp.minimum(lane + 1, 9)

    up_safe = _lane_safe(prev, curr, next_lane, params)
    up_safe2 = _lane_safe(prev, curr, nn_lane, params)
    cur_ttc = _ttc_for_lane(prev, curr, lane)

    near_top = cy < params["finish_y"]

    # Wait window: if next lane unsafe but TTC there is short and will clear soon,
    # NOOP rather than backing up.
    next_ttc = _ttc_for_lane(prev, curr, next_lane)
    will_clear = next_ttc < params["wait_horizon"]

    # DOWN only if we're about to be hit in current lane and down is safe.
    cur_danger = cur_ttc < params["down_ttc"]
    down_safe = _lane_safe(prev, curr, down_lane, params)

    # Decision tree, flat:
    # 1. near top -> UP
    # 2. up_safe (and prefer up_safe2 too, but not required) -> UP
    # 3. current lane imminent collision and down safe -> DOWN
    # 4. otherwise NOOP (wait for window)
    action = jnp.where(
        near_top, UP,
        jnp.where(
            up_safe, UP,
            jnp.where(cur_danger & down_safe, DOWN, NOOP)
        )
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
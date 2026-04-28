"""
Auto-generated policy v4
Generated at: 2026-04-26 18:20:43
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
        "arrival_frames": jnp.float32(6.0),   # frames to settle in new lane
        "post_pass_margin": jnp.float32(3.0), # px past trailing edge to be safe
        "pad": jnp.float32(2.0),              # extra px around chicken column
        "finish_y": jnp.float32(32.0),        # force UP when y < finish_y
        "fast_band_start": jnp.float32(4.5),  # lanes >= this need 2-lane lookahead
        "danger_horizon": jnp.float32(4.0),   # frames horizon for current-lane danger
    }


def _split(obs_flat):
    return obs_flat[0:FRAME_SIZE], obs_flat[FRAME_SIZE:2 * FRAME_SIZE]


def _lane_index(cy):
    return jnp.argmin(jnp.abs(LANE_Y - cy))


def _lane_safe(prev, curr, lane_idx, params):
    """A lane is safe if the car will not overlap the chicken column at t=0
    and at t=arrival_frames, with a post-pass margin on the trailing side."""
    car_x = curr[8 + lane_idx]
    car_w = curr[28 + lane_idx]
    car_act = curr[48 + lane_idx]
    prev_x = prev[8 + lane_idx]
    dx = car_x - prev_x

    pad = params["pad"]
    chicken_l = CHICKEN_X - pad
    chicken_r = CHICKEN_X + CHICKEN_WIDTH + pad

    # Overlap test at time t
    def overlap_at(t):
        x = car_x + dx * t
        cl = x
        cr = x + car_w
        return (cr >= chicken_l) & (cl <= chicken_r)

    overlap_now = overlap_at(0.0)
    overlap_arrival = overlap_at(params["arrival_frames"])
    overlap_mid = overlap_at(params["arrival_frames"] * 0.5)

    # Post-pass margin: if car has just passed (trailing edge close), wait
    # Trailing edge depends on direction
    moving_right = dx > 0.05
    # If moving right, trailing edge is car_l; we want chicken_l - car_r > margin
    # or car_l > chicken_r (car ahead, safe).
    # Simpler: require trailing-side clearance.
    trailing_clear_right = (chicken_l - (car_x + car_w)) > params["post_pass_margin"]
    trailing_clear_left = (car_x - chicken_r) > params["post_pass_margin"]
    # Just-passed unsafe: car overlapped recently (within margin)
    just_passed_right = (~moving_right) | trailing_clear_right | (car_x > chicken_r)
    # if moving right and car is to the left of chicken (car_r < chicken_l) that's "behind", fine
    # if moving right and car just passed (car_l just past chicken_r) -> car_x > chicken_r, also fine
    # The risky case is moving_right and car_r is between chicken_l-margin and chicken_l (just-passed wrap? no, that's behind)
    # Actually with moving_right, "just passed" means car has gone through; car_l > chicken_r.
    # Re-think: car moving right starts left, ends right of chicken. "Just passed" = car_l slightly > chicken_r.
    # That's safe to step in. The dangerous trailing case is when car is right at chicken (overlapping or about to leave)
    # which the overlap tests already cover.

    unsafe = overlap_now | overlap_arrival | overlap_mid
    safe = ~unsafe
    # Inactive cars are always safe
    safe = safe | (car_act <= 0.5)
    return safe


def _current_lane_danger(prev, curr, lane_idx, params):
    """Is current lane about to be hit within danger_horizon frames?"""
    car_x = curr[8 + lane_idx]
    car_w = curr[28 + lane_idx]
    car_act = curr[48 + lane_idx]
    prev_x = prev[8 + lane_idx]
    dx = car_x - prev_x

    pad = params["pad"]
    chicken_l = CHICKEN_X - pad
    chicken_r = CHICKEN_X + CHICKEN_WIDTH + pad

    t = params["danger_horizon"]
    x_future = car_x + dx * t
    overlap_soon = ((x_future + car_w) >= chicken_l) & (x_future <= chicken_r)
    overlap_now = ((car_x + car_w) >= chicken_l) & (car_x <= chicken_r)
    danger = (overlap_now | overlap_soon) & (car_act > 0.5)
    return danger


def policy(obs_flat, params):
    prev, curr = _split(obs_flat)
    cy = curr[1]

    lane = _lane_index(cy)
    next_lane = jnp.maximum(lane - 1, 0)
    nn_lane = jnp.maximum(lane - 2, 0)
    down_lane = jnp.minimum(lane + 1, 9)

    up_safe = _lane_safe(prev, curr, next_lane, params)
    up_safe2 = _lane_safe(prev, curr, nn_lane, params)
    down_safe = _lane_safe(prev, curr, down_lane, params)

    near_top = cy < params["finish_y"]
    in_fast_band = lane.astype(jnp.float32) >= params["fast_band_start"]

    # In fast band, require both next and next-next safe (when not at top)
    up_gate = jnp.where(in_fast_band, up_safe & up_safe2, up_safe)

    cur_danger = _current_lane_danger(prev, curr, lane, params)

    # Hysteresis: if chicken is already mid-transition (cy below lane center),
    # be more permissive about continuing UP.
    lane_center = LANE_Y[lane]
    moved_up = cy < (lane_center - 2.0)
    up_gate = up_gate | (moved_up & up_safe)

    # Decision:
    # 1. near top -> UP
    # 2. up_gate -> UP
    # 3. current lane dangerous and down lane safe -> DOWN
    # 4. else NOOP
    action = jnp.where(
        near_top, UP,
        jnp.where(up_gate, UP,
                  jnp.where(cur_danger & down_safe, DOWN, NOOP))
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
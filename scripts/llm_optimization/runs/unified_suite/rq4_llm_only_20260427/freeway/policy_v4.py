"""
Auto-generated policy v4
Generated at: 2026-04-27 17:29:43
"""

import jax
import jax.numpy as jnp

FRAME_SIZE = 88
NOOP = 0
UP = 1
DOWN = 2
CHICKEN_X = 44
CHICKEN_WIDTH = 6
TOP_BORDER = 15
BOTTOM_BORDER = 180


def init_params():
    return {
        "horizon_cur": 4.0,       # window for current lane danger
        "horizon_next": 6.0,      # window for near probe
        "arrive_w": 4.0,          # half-width of arrival window
        "lane_band": 5.0,         # lane membership half-band
        "near_dy": 8.0,           # half-lane probe offset
        "next_dy": 16.0,          # full-lane probe offset
        "margin_base": 2.0,       # base x-margin
        "margin_k": 1.6,          # margin scale per |dx|
    }


def _lane_overlap(frame_curr, frame_prev, cy, lane_band,
                  t_lo, t_hi, margin_base, margin_k):
    """Check if any active car in the lane overlaps chicken column
    within time window [t_lo, t_hi]. Margin scales with |dx|."""
    cars_x = frame_curr[8:18]
    cars_y = frame_curr[18:28]
    cars_w = frame_curr[28:38]
    cars_active = frame_curr[48:58]
    prev_x = frame_prev[8:18]
    dx = cars_x - prev_x
    abs_dx = jnp.abs(dx)

    in_lane = (jnp.abs(cars_y - cy) < lane_band) & (cars_active > 0.5)

    # Speed-aware margin per car
    xm = margin_base + margin_k * abs_dx
    chick_lo = CHICKEN_X - xm
    chick_hi = CHICKEN_X + CHICKEN_WIDTH + xm

    a = chick_hi - cars_x
    b = chick_lo - cars_x - cars_w

    eps = 1e-3
    safe_dx = jnp.where(abs_dx < eps, eps, dx)
    t1 = a / safe_dx
    t2 = b / safe_dx
    t_enter = jnp.minimum(t1, t2)
    t_exit = jnp.maximum(t1, t2)

    static_overlap = (a > 0) & (b < 0)
    near_zero = abs_dx < eps
    t_enter = jnp.where(near_zero, jnp.where(static_overlap, -1e3, 1e6), t_enter)
    t_exit = jnp.where(near_zero, jnp.where(static_overlap, 1e6, -1e3), t_exit)

    # "Just passed" suppression: if exit is barely behind us, treat as clear.
    just_passed = t_exit < 0.5

    lo = jnp.maximum(t_enter, t_lo)
    hi = jnp.minimum(t_exit, t_hi)
    overlaps = in_lane & (lo <= hi) & (~just_passed)
    return jnp.any(overlaps)


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    cy = curr[1]

    lane_band = params["lane_band"]
    mb = params["margin_base"]
    mk = params["margin_k"]
    h_cur = params["horizon_cur"]
    h_next = params["horizon_next"]

    # Self-calibrated arrival time: how long to traverse near_dy at ~1.5 px/frame
    arrive_t = params["near_dy"] / 1.5
    aw = params["arrive_w"]

    # Current lane: imminent danger window
    cur_hit = _lane_overlap(curr, prev, cy, lane_band,
                            0.0, h_cur, mb, mk)

    # Near probe (half lane up): danger during/just after step
    near_hit = _lane_overlap(curr, prev, cy - params["near_dy"], lane_band,
                             0.0, h_next, mb, mk)

    # Next lane arrival window: blocks UP if a car will be there on arrival
    t_lo = arrive_t - aw
    t_hi = arrive_t + aw
    next_hit = _lane_overlap(curr, prev, cy - params["next_dy"], lane_band,
                             t_lo, t_hi, mb, mk)

    # Emergency: very imminent current-lane hit
    cur_emerg = _lane_overlap(curr, prev, cy, lane_band,
                              0.0, 1.5, mb, mk)

    # Lower-lane safety for DOWN escape
    down_hit = _lane_overlap(curr, prev, cy + params["near_dy"], lane_band,
                             0.0, h_next, mb, mk)

    near_top = cy < (TOP_BORDER + 12.0)
    can_down = cy < (BOTTOM_BORDER - 8.0)

    # OR semantics: if anything is unsafe, wait. Single high-leverage fix.
    unsafe = cur_hit | near_hit | next_hit
    go_up = (~unsafe) | near_top
    action = jnp.where(go_up, jnp.int32(UP), jnp.int32(NOOP))

    # Emergency DOWN only when current lane will hit very soon and below is clear
    emergency = cur_emerg & (~down_hit) & can_down & (~near_top)
    action = jnp.where(emergency, jnp.int32(DOWN), action)

    return action


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
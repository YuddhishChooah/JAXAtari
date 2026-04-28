"""
Auto-generated policy v5
Generated at: 2026-04-27 17:31:19
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
        "horizon_cur": 4.0,       # frames to look ahead in current lane
        "arrive_t": 5.0,          # expected frames to reach next lane
        "arrive_w": 3.0,          # half-width of arrival window
        "lane_band": 6.0,         # lane membership band
        "near_dy": 8.0,           # half-lane probe
        "next_dy": 16.0,          # full-lane probe
        "x_margin_base": 3.0,     # base x guard band
        "x_margin_k": 1.6,        # margin scale per |dx|
    }


def _lane_overlap(frame_curr, frame_prev, cy, lane_band,
                  t_lo, t_hi, x_margin_base, x_margin_k):
    """True if any active car in the given lane projects to overlap the
    chicken column at any time within [t_lo, t_hi]."""
    cars_x = frame_curr[8:18]
    cars_y = frame_curr[18:28]
    cars_w = frame_curr[28:38]
    cars_active = frame_curr[48:58]
    prev_x = frame_prev[8:18]
    dx = cars_x - prev_x

    in_lane = (jnp.abs(cars_y - cy) < lane_band) & (cars_active > 0.5)

    # Margin scales with car speed: fast lanes get larger guard band.
    abs_dx = jnp.abs(dx)
    x_margin = x_margin_base + x_margin_k * abs_dx

    chick_lo = CHICKEN_X - x_margin
    chick_hi = CHICKEN_X + CHICKEN_WIDTH + x_margin

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

    lo = jnp.maximum(t_enter, t_lo)
    hi = jnp.minimum(t_exit, t_hi)
    return jnp.any(in_lane & (lo <= hi))


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    cy = curr[1]

    band = params["lane_band"]
    mb = params["x_margin_base"]
    mk = params["x_margin_k"]
    hcur = params["horizon_cur"]

    # Current lane: any imminent overlap in [0, horizon_cur]?
    cur_hit = _lane_overlap(curr, prev, cy, band,
                            0.0, hcur, mb, mk)

    # Half-lane probe: collisions during the step itself.
    near_hit = _lane_overlap(curr, prev, cy - params["near_dy"], band,
                             0.0, hcur + 1.0, mb, mk)

    # Destination lane: arrival-window check.
    t_lo = params["arrive_t"] - params["arrive_w"]
    t_hi = params["arrive_t"] + params["arrive_w"]
    next_hit = _lane_overlap(curr, prev, cy - params["next_dy"], band,
                             t_lo, t_hi, mb, mk)

    # Emergency: very imminent current-lane collision, looser margin.
    cur_emerg = _lane_overlap(curr, prev, cy, band,
                              0.0, 2.0, mb + 2.0, mk)

    # Lower-lane check for DOWN escape, looser margin.
    down_safe = ~_lane_overlap(curr, prev, cy + params["near_dy"], band,
                               0.0, hcur + 1.0, mb, mk)

    near_top = cy < (TOP_BORDER + 6.0)
    can_down = cy < (BOTTOM_BORDER - 8.0)

    # OR-gated safety: any of the three checks failing blocks UP.
    blocked = cur_hit | near_hit | next_hit
    go_up = (~blocked) | (near_top & (~cur_hit) & (~near_hit))

    action = jnp.where(go_up, jnp.int32(UP), jnp.int32(NOOP))

    # Emergency DOWN only when current lane is hot AND below is safe.
    emergency = cur_emerg & down_safe & can_down & (~near_top)
    action = jnp.where(emergency, jnp.int32(DOWN), action)

    return action


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
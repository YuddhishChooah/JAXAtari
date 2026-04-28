"""
Auto-generated policy v3
Generated at: 2026-04-27 17:27:57
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
        "horizon_cur": 3.0,       # short window for current lane
        "arrive_t": 8.0,          # expected frames to reach next lane
        "arrive_w": 4.0,          # half-width of arrival window
        "lane_band_cur": 5.0,     # tight band for current lane
        "lane_band_next": 6.0,    # band for probe lanes
        "near_dy": 8.0,           # half-lane probe
        "next_dy": 16.0,          # full-lane probe
        "ttc_emergency": 2.5,     # imminent collision threshold
    }


def _overlap_in_window(frame_curr, frame_prev, cy, lane_band,
                       t_lo_win, t_hi_win, x_margin):
    """Return True if any active car in the lane overlaps the chicken column
    at any time within [t_lo_win, t_hi_win]."""
    cars_x = frame_curr[8:18]
    cars_y = frame_curr[18:28]
    cars_w = frame_curr[28:38]
    cars_active = frame_curr[48:58]
    prev_x = frame_prev[8:18]
    dx = cars_x - prev_x

    in_lane = (jnp.abs(cars_y - cy) < lane_band) & (cars_active > 0.5)

    chick_lo = CHICKEN_X - x_margin
    chick_hi = CHICKEN_X + CHICKEN_WIDTH + x_margin

    a = chick_hi - cars_x
    b = chick_lo - cars_x - cars_w

    eps = 1e-3
    safe_dx = jnp.where(jnp.abs(dx) < eps, eps, dx)
    t1 = a / safe_dx
    t2 = b / safe_dx
    t_enter = jnp.minimum(t1, t2)
    t_exit = jnp.maximum(t1, t2)

    static_overlap = (a > 0) & (b < 0)
    near_zero = jnp.abs(dx) < eps
    t_enter = jnp.where(near_zero, jnp.where(static_overlap, -1e3, 1e6), t_enter)
    t_exit = jnp.where(near_zero, jnp.where(static_overlap, 1e6, -1e3), t_exit)

    # Intersect [t_enter, t_exit] with [t_lo_win, t_hi_win]
    lo = jnp.maximum(t_enter, t_lo_win)
    hi = jnp.minimum(t_exit, t_hi_win)
    overlaps = in_lane & (lo <= hi)
    return jnp.any(overlaps)


def _direction_margin(frame_curr, frame_prev, cy, lane_band):
    """Use a moderate x_margin; approaching traffic needs more room."""
    cars_y = frame_curr[18:28]
    cars_x = frame_curr[8:18]
    prev_x = frame_prev[8:18]
    dx = cars_x - prev_x
    in_lane = jnp.abs(cars_y - cy) < lane_band
    # Approaching: car is right of chicken moving left (dx<0), or left of chicken moving right (dx>0)
    rel = cars_x - CHICKEN_X
    approaching = in_lane & (jnp.sign(rel) != jnp.sign(dx)) & (jnp.abs(dx) > 0.1)
    any_approach = jnp.any(approaching)
    return jnp.where(any_approach, 4.0, 2.0)


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    cy = curr[1]

    # Direction-aware margins (small constant choice)
    xm_cur = _direction_margin(curr, prev, cy, params["lane_band_cur"])
    xm_next = _direction_margin(curr, prev, cy - params["near_dy"],
                                params["lane_band_next"])

    # Current lane: any collision in short window [0, horizon_cur]?
    cur_hit = _overlap_in_window(curr, prev, cy,
                                 params["lane_band_cur"],
                                 0.0, params["horizon_cur"], xm_cur)

    # Near probe (half lane up): collision soon as we move?
    near_hit = _overlap_in_window(curr, prev, cy - params["near_dy"],
                                  params["lane_band_next"],
                                  0.0, params["horizon_cur"] + 2.0, xm_next)

    # Next lane: arrival-window check, not full horizon
    t_lo = params["arrive_t"] - params["arrive_w"]
    t_hi = params["arrive_t"] + params["arrive_w"]
    next_hit = _overlap_in_window(curr, prev, cy - params["next_dy"],
                                  params["lane_band_next"],
                                  t_lo, t_hi, xm_next)

    # Emergency: very imminent current-lane hit
    cur_emerg = _overlap_in_window(curr, prev, cy,
                                   params["lane_band_cur"],
                                   0.0, params["ttc_emergency"], xm_cur)

    # Lower-lane safety for DOWN escape
    down_hit = _overlap_in_window(curr, prev, cy + params["near_dy"],
                                  params["lane_band_next"],
                                  0.0, params["horizon_cur"] + 1.0, xm_next)

    near_top = cy < (TOP_BORDER + 12.0)
    can_down = cy < (BOTTOM_BORDER - 8.0)

    # UP if current and near probes are clear; next-lane check only blocks
    # if both near and far are unsafe (avoid overconservative full-lane gating)
    safe_to_step = (~cur_hit) & (~near_hit)
    next_blocks = next_hit & near_hit  # only block UP when both fail

    go_up = (safe_to_step & (~next_blocks)) | near_top
    action = jnp.where(go_up, jnp.int32(UP), jnp.int32(NOOP))

    # Emergency DOWN: independent of next-lane state
    emergency = cur_emerg & (~down_hit) & can_down & (~near_top)
    action = jnp.where(emergency, jnp.int32(DOWN), action)

    return action


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
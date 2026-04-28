"""
Auto-generated policy v2
Generated at: 2026-04-27 17:26:29
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
N_CARS = 10


def init_params():
    return {
        "horizon_cur": 4.0,      # short horizon for current lane (chicken already there)
        "horizon_next": 10.0,    # longer horizon for next lane (we'll arrive later)
        "x_margin": 4.0,         # tight horizontal margin
        "lane_band_cur": 7.0,    # vertical band for current lane
        "lane_band_next": 7.0,   # vertical band for next lane
        "next_lane_dy": 16.0,    # offset to next lane probe
        "ttc_emergency": 3.0,    # very low TTC -> emergency
    }


def _min_ttc(frame_curr, frame_prev, cy, lane_band, horizon, x_margin):
    """Return minimum time (in frames, within [0, horizon]) at which any active
    car in this lane overlaps the chicken column. Returns horizon+1 if no overlap."""
    cars_x = frame_curr[8:18]
    cars_y = frame_curr[18:28]
    cars_w = frame_curr[28:38]
    cars_active = frame_curr[48:58]
    prev_x = frame_prev[8:18]
    dx = cars_x - prev_x

    in_lane = (jnp.abs(cars_y - cy) < lane_band) & (cars_active > 0.5)

    chick_lo = CHICKEN_X - x_margin
    chick_hi = CHICKEN_X + CHICKEN_WIDTH + x_margin

    # Car interval at time t: [cars_x + dx*t, cars_x + cars_w + dx*t]
    # Overlap condition with chicken column:
    #   cars_x + dx*t < chick_hi   AND   cars_x + cars_w + dx*t > chick_lo
    # => dx*t < chick_hi - cars_x          (A)
    # => dx*t > chick_lo - cars_x - cars_w (B)
    a = chick_hi - cars_x          # upper bound on dx*t
    b = chick_lo - cars_x - cars_w # lower bound on dx*t

    eps = 1e-3
    safe_dx = jnp.where(jnp.abs(dx) < eps, eps, dx)

    # Solve dx*t = a and dx*t = b to find entry/exit times
    t1 = a / safe_dx
    t2 = b / safe_dx
    t_enter = jnp.minimum(t1, t2)
    t_exit = jnp.maximum(t1, t2)

    # If dx ~ 0: overlap iff currently overlapping (a>0 and b<0)
    static_overlap = (a > 0) & (b < 0)
    static_t_enter = jnp.where(static_overlap, 0.0, horizon + 10.0)
    static_t_exit = jnp.where(static_overlap, horizon + 10.0, -1.0)

    near_zero = jnp.abs(dx) < eps
    t_enter = jnp.where(near_zero, static_t_enter, t_enter)
    t_exit = jnp.where(near_zero, static_t_exit, t_exit)

    # Effective overlap time within [0, horizon]
    t_lo = jnp.maximum(t_enter, 0.0)
    t_hi = jnp.minimum(t_exit, horizon)
    overlaps = in_lane & (t_lo <= t_hi) & (t_hi >= 0.0)

    big = horizon + 10.0
    ttc_per_car = jnp.where(overlaps, t_lo, big)
    return jnp.min(ttc_per_car)


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    cy = curr[1]

    ttc_cur = _min_ttc(curr, prev, cy,
                       params["lane_band_cur"],
                       params["horizon_cur"],
                       params["x_margin"])
    next_y = cy - params["next_lane_dy"]
    ttc_next = _min_ttc(curr, prev, next_y,
                        params["lane_band_next"],
                        params["horizon_next"],
                        params["x_margin"])
    down_y = cy + params["next_lane_dy"]
    ttc_down = _min_ttc(curr, prev, down_y,
                        params["lane_band_next"],
                        params["horizon_cur"],
                        params["x_margin"])

    near_top = cy < (TOP_BORDER + 12.0)

    # Next lane is "safe enough" -> go UP
    next_safe = ttc_next >= params["horizon_next"]
    action = jnp.where(next_safe | near_top, jnp.int32(UP), jnp.int32(NOOP))

    # Emergency: current lane very imminent collision, lane below clear -> DOWN
    cur_emergency = ttc_cur < params["ttc_emergency"]
    down_safe = ttc_down >= params["horizon_cur"]
    can_down = cy < (BOTTOM_BORDER - 8.0)
    emergency = cur_emergency & down_safe & can_down & (~near_top) & (~next_safe)
    action = jnp.where(emergency, jnp.int32(DOWN), action)

    return action


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
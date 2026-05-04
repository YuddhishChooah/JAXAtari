"""
Auto-generated policy v4
Generated at: 2026-04-28 14:17:57
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

LANE_Y = jnp.array([27.0, 43.0, 59.0, 75.0, 91.0, 107.0, 123.0, 139.0, 155.0, 171.0])


def init_params():
    return {
        "safe_ttc": jnp.array(6.0),         # threat if 0 < ttc < safe_ttc on current lane
        "entry_ttc": jnp.array(8.0),        # stricter horizon for next lane entry
        "entry_margin": jnp.array(2.0),     # required positive gap at next-lane entry
        "approach_eps": jnp.array(0.15),    # min |dx| to count car as moving
        "current_slack": jnp.array(3.0),    # extra x slack on current lane
    }


def _threat_score(curr, prev, lane_idx, params, ttc_horizon, extra_slack):
    """Return min "danger" across cars in given lane index.
    Positive value = safe margin (larger is safer).
    Negative value = threat (more negative is worse).
    """
    cx_curr = curr[8:18]
    cy = curr[18:28]
    cw = curr[28:38]
    active = curr[48:58]
    cx_prev = prev[8:18]
    dx = cx_curr - cx_prev

    target_y = LANE_Y[lane_idx]
    in_lane = (jnp.abs(cy - target_y) < 6.0).astype(jnp.float32) * active

    chicken_cx = CHICKEN_X + CHICKEN_WIDTH * 0.5
    car_cx = cx_curr + cw * 0.5
    half_sum = cw * 0.5 + CHICKEN_WIDTH * 0.5 + extra_slack

    # signed distance: positive means car is to one side, vector to chicken
    signed = chicken_cx - car_cx  # if dx>0 (car moving right), threatening when signed>0

    # approach velocity toward chicken: positive if closing
    approach = dx * jnp.sign(signed)
    closing = approach > params["approach_eps"]

    # current absolute gap (negative if overlapping)
    cur_gap = jnp.abs(signed) - half_sum

    # time-to-contact (frames) using closing speed
    safe_speed = jnp.maximum(approach, 0.01)
    ttc = (jnp.abs(signed) - half_sum) / safe_speed

    # threat if currently overlapping OR (closing and ttc within horizon)
    overlapping = cur_gap < 0.0
    will_hit = closing & (ttc < ttc_horizon) & (ttc > -1.0)
    is_threat = overlapping | will_hit

    # margin: how safe (large positive = very safe)
    margin = jnp.where(is_threat, -ttc_horizon + ttc, cur_gap + 50.0)
    # for non-lane cars: very safe
    margin = jnp.where(in_lane > 0.5, margin, jnp.full_like(margin, 1e3))
    return jnp.min(margin)


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    chicken_y = curr[1]

    # find nearest lane index to chicken
    dists = jnp.abs(LANE_Y - chicken_y)
    cur_lane = jnp.argmin(dists)
    next_lane = jnp.maximum(cur_lane - 1, 0)

    cur_margin = _threat_score(curr, prev, cur_lane, params,
                               params["safe_ttc"], params["current_slack"])
    next_margin = _threat_score(curr, prev, next_lane, params,
                                params["entry_ttc"], params["entry_margin"])

    # commit rule: between lanes -> push UP
    nearest_y = LANE_Y[cur_lane]
    between_lanes = jnp.abs(chicken_y - nearest_y) > 3.0

    near_top = chicken_y < 25.0

    cur_safe = cur_margin > 0.0
    next_safe = next_margin > 0.0

    up_ok = cur_safe & next_safe

    action = jnp.where(
        near_top, UP,
        jnp.where(between_lanes, UP,
                  jnp.where(up_ok, UP, NOOP))
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
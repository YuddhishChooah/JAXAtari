"""
Auto-generated policy v1
Generated at: 2026-04-07 18:33:13
"""

"""
Parametric JAX policy for Atari Freeway.
Uses two stacked frames to estimate car velocities and time safe lane entries.
CMA-ES tunes 6 float parameters controlling safety thresholds and timing.
"""

import jax.numpy as jnp
import jax

# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------
NOOP = 0
UP   = 2
DOWN = 5

# ---------------------------------------------------------------------------
# Observation layout
# ---------------------------------------------------------------------------
FRAME_SIZE   = 44
CHICKEN_X    = 40
CHICKEN_W    = 6
CHICKEN_H    = 8
SCREEN_WIDTH = 160
NUM_LANES    = 10

LANE_Y = jnp.array([23, 39, 55, 71, 87, 103, 119, 135, 151, 167], dtype=jnp.float32)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _car_x(frame, i):
    return frame[4 + 4 * i]

def _car_w(frame, i):
    return frame[6 + 4 * i]

def _chicken_y(frame):
    return frame[1]


def _overlap_1d(ax, aw, bx, bw):
    """Check overlap between segment [ax, ax+aw) and [bx, bx+bw)."""
    return jnp.maximum(0.0, jnp.minimum(ax + aw, bx + bw) - jnp.maximum(ax, bx))


def _projected_safe(prev, curr, lane_idx, params):
    """
    Return 1.0 if the chicken can safely enter `lane_idx` on the next step.
    Uses velocity estimate from two frames to project car position.
    """
    safety_margin = params["safety_margin"]   # extra clearance in pixels
    lookahead     = params["lookahead"]        # how many steps to project forward

    cx   = float(CHICKEN_X)
    cw   = float(CHICKEN_W)

    car_x_prev = _car_x(prev, lane_idx)
    car_x_curr = _car_x(curr, lane_idx)
    car_w_curr = _car_w(curr, lane_idx)
    dx         = car_x_curr - car_x_prev          # velocity estimate

    # Project car position forward by `lookahead` steps (wrap in screen width)
    proj_x = (car_x_curr + dx * lookahead) % SCREEN_WIDTH

    # Also check wrapped-around car approaching from the other side
    proj_x_wrap = (proj_x - jnp.sign(dx) * SCREEN_WIDTH)

    eff_w = car_w_curr + safety_margin

    overlap_main = _overlap_1d(cx, cw, proj_x,      eff_w)
    overlap_wrap = _overlap_1d(cx, cw, proj_x_wrap, eff_w)
    overlap      = jnp.maximum(overlap_main, overlap_wrap)

    return jnp.where(overlap <= 0.0, 1.0, 0.0)


def _find_next_lane(chicken_y):
    """Return index of the next lane above the chicken (0 = top)."""
    # The lane the chicken needs to cross next is the one just above chicken_y.
    # We pick the lane whose y is closest to (chicken_y - lane_height) from above.
    diffs = chicken_y - LANE_Y          # positive means lane is above chicken
    # We want the smallest positive diff (nearest lane above)
    masked = jnp.where(diffs > -8.0, diffs, jnp.inf)
    idx    = jnp.argmin(masked)
    return idx


def _is_in_danger(prev, curr, params):
    """Check if current lane (beneath chicken) has a threatening car."""
    chicken_y   = _chicken_y(curr)
    # Danger: any car overlaps the chicken's current x position this frame
    danger_threshold = params["danger_threshold"]

    def lane_danger(i):
        car_x_curr = _car_x(curr, i)
        car_x_prev = _car_x(prev, i)
        car_w_curr = _car_w(curr, i)
        dx         = car_x_curr - car_x_prev
        proj_x     = (car_x_curr + dx * params["danger_lookahead"]) % SCREEN_WIDTH
        proj_x_w   = proj_x - jnp.sign(dx) * SCREEN_WIDTH
        eff_w      = car_w_curr + danger_threshold
        ov1        = _overlap_1d(float(CHICKEN_X), float(CHICKEN_W), proj_x,   eff_w)
        ov2        = _overlap_1d(float(CHICKEN_X), float(CHICKEN_W), proj_x_w, eff_w)
        lane_y     = LANE_Y[i]
        near       = jnp.abs(lane_y - chicken_y) < 12.0
        return jnp.where(near, jnp.maximum(ov1, ov2), 0.0)

    dangers = jnp.array([lane_danger(i) for i in range(NUM_LANES)])
    return jnp.max(dangers) > 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_params() -> dict:
    return {
        "safety_margin"   : 6.0,   # extra pixel buffer around car width
        "lookahead"       : 2.0,   # steps to project car position
        "danger_threshold": 8.0,   # overlap threshold for danger detection
        "danger_lookahead": 1.0,   # steps ahead for danger check
        "up_bias"         : 0.8,   # unused placeholder kept for CMA-ES dimensionality
        "safe_fraction"   : 0.5,   # unused placeholder (can be repurposed)
    }


def policy(obs_flat: jnp.ndarray, params: dict) -> int:
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    chicken_y  = _chicken_y(curr)
    next_lane  = _find_next_lane(chicken_y)

    safe_ahead = _projected_safe(prev, curr, next_lane, params)
    in_danger  = _is_in_danger(prev, curr, params)

    # Decision tree (JAX-safe):
    #   if in_danger   -> DOWN (escape)
    #   elif safe_ahead -> UP  (advance)
    #   else            -> NOOP (wait for gap)
    action = jnp.where(
        in_danger,
        jnp.int32(DOWN),
        jnp.where(
            safe_ahead > 0.5,
            jnp.int32(UP),
            jnp.int32(NOOP)
        )
    )
    return action


def measure_main(episode_rewards, episode_scores=None) -> float:
    return jnp.sum(episode_rewards)
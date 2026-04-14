"""
Auto-generated policy v2
Generated at: 2026-04-07 18:37:10
"""

"""
Parametric JAX policy for Atari Freeway.
Uses two stacked frames to estimate car velocities and time safe lane entries.
CMA-ES tunes 6 float parameters controlling safety thresholds and timing.

Key improvements over previous version:
- Danger detection no longer triggers DOWN (which wastes time); just waits or moves up.
- Check BOTH current lane AND next lane before moving UP.
- Only hold (NOOP) when truly unsafe; otherwise keep pressing UP.
- Removed DOWN action to reduce collisions from retreating into traffic.
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
    """Overlap between segment [ax, ax+aw) and [bx, bx+bw)."""
    return jnp.maximum(0.0, jnp.minimum(ax + aw, bx + bw) - jnp.maximum(ax, bx))


def _lane_safe(prev, curr, lane_idx, lookahead, margin):
    """
    Return 1.0 if chicken can safely occupy `lane_idx` after `lookahead` steps.
    Checks projected car position with wrap-around.
    """
    cx  = jnp.float32(CHICKEN_X)
    cw  = jnp.float32(CHICKEN_W)

    car_x_prev = _car_x(prev, lane_idx)
    car_x_curr = _car_x(curr, lane_idx)
    car_w_curr = _car_w(curr, lane_idx)
    dx         = car_x_curr - car_x_prev

    proj_x      = (car_x_curr + dx * lookahead) % SCREEN_WIDTH
    proj_x_wrap = proj_x - jnp.sign(dx) * SCREEN_WIDTH

    eff_w = car_w_curr + margin

    ov1 = _overlap_1d(cx, cw, proj_x,      eff_w)
    ov2 = _overlap_1d(cx, cw, proj_x_wrap, eff_w)

    return jnp.where(jnp.maximum(ov1, ov2) <= 0.0, 1.0, 0.0)


def _find_next_lane(chicken_y):
    """
    Return index of the next lane the chicken is about to enter (moving upward).
    This is the lane with smallest positive distance above chicken_y.
    """
    diffs  = chicken_y - LANE_Y          # positive => lane is above
    masked = jnp.where(diffs > -8.0, diffs, jnp.inf)
    return jnp.argmin(masked)


def _current_lane(chicken_y):
    """Return index of the lane closest to current chicken_y."""
    diffs = jnp.abs(LANE_Y - chicken_y)
    return jnp.argmin(diffs)


def _any_lane_danger(prev, curr, chicken_y, lookahead, margin, proximity):
    """
    Return True if any nearby lane has a car projected to overlap the chicken.
    Only checks lanes within `proximity` pixels of chicken_y.
    """
    def lane_threat(i):
        safe_i  = _lane_safe(prev, curr, i, lookahead, margin)
        near_i  = jnp.abs(LANE_Y[i] - chicken_y) < proximity
        return jnp.where(near_i & (safe_i < 0.5), 1.0, 0.0)

    threats = jnp.array([lane_threat(i) for i in range(NUM_LANES)])
    return jnp.max(threats) > 0.5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_params() -> dict:
    return {
        "safety_margin"   : 5.5,   # extra pixel buffer around car width
        "lookahead"       : 1.6,   # steps to project for next-lane check
        "danger_lookahead": 1.3,   # steps to project for current-position danger
        "danger_margin"   : 8.0,   # extra margin for danger detection
        "proximity"       : 12.0,  # how close a lane must be to count as danger
        "safe_fraction"   : 0.5,   # unused placeholder kept for CMA-ES dimensionality
    }


def policy(obs_flat: jnp.ndarray, params: dict) -> int:
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    chicken_y = _chicken_y(curr)
    next_lane = _find_next_lane(chicken_y)

    # Is the next lane safe to step into?
    next_safe = _lane_safe(prev, curr, next_lane,
                           params["lookahead"], params["safety_margin"])

    # Is there immediate danger at the current position?
    in_danger = _any_lane_danger(prev, curr, chicken_y,
                                 params["danger_lookahead"],
                                 params["danger_margin"],
                                 params["proximity"])

    # Decision logic (no DOWN — retreating costs more than waiting):
    #   if next lane is safe AND not in immediate danger -> UP
    #   else -> NOOP (wait for a gap)
    can_move_up = (next_safe > 0.5) & (~in_danger)

    action = jnp.where(
        can_move_up,
        jnp.int32(UP),
        jnp.int32(NOOP)
    )
    return action


def measure_main(episode_rewards, episode_scores=None) -> float:
    return jnp.sum(episode_rewards)
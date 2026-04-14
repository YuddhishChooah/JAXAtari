"""
Auto-generated policy v4
Generated at: 2026-04-07 18:44:09
"""

"""
Parametric JAX policy for Atari Freeway.
Uses two stacked frames to estimate car velocities and time safe lane entries.
CMA-ES tunes 6 float parameters controlling safety thresholds and timing.

Key improvements:
- Reduced hesitation by only checking the *next* lane ahead (not all nearby lanes).
- Separate lookahead for "current danger" vs "next lane entry".
- Still avoids DOWN action to prevent retreating into traffic.
- Bias strongly toward UP; only NOOP when genuinely unsafe.
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
    Projects car position forward with velocity estimate and checks overlap.
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
    Lane must be at or above current y position (within a small tolerance).
    """
    diffs  = chicken_y - LANE_Y          # positive => lane is above chicken
    masked = jnp.where(diffs > -4.0, diffs, jnp.inf)
    return jnp.argmin(masked)


def _current_lane(chicken_y):
    """Return index of the lane closest to current chicken_y."""
    diffs = jnp.abs(LANE_Y - chicken_y)
    return jnp.argmin(diffs)


def _two_lane_safe(prev, curr, chicken_y, lookahead, margin, danger_lookahead, danger_margin):
    """
    Check safety for current lane and next lane above.
    Returns 1.0 only if both are safe.
    """
    cur_lane  = _current_lane(chicken_y)
    next_lane = _find_next_lane(chicken_y)

    cur_safe  = _lane_safe(prev, curr, cur_lane,  danger_lookahead, danger_margin)
    next_safe = _lane_safe(prev, curr, next_lane, lookahead,        margin)

    return jnp.where((cur_safe > 0.5) & (next_safe > 0.5), 1.0, 0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_params() -> dict:
    return {
        "safety_margin"   : 5.0,   # pixel buffer for next-lane check
        "lookahead"       : 1.4,   # steps to project for next-lane check
        "danger_lookahead": 0.8,   # steps to project for current-lane danger
        "danger_margin"   : 6.0,   # pixel buffer for current-lane danger
        "aggression"      : 0.3,   # unused placeholder (kept for CMA-ES dims)
        "safe_fraction"   : 0.4,   # unused placeholder (kept for CMA-ES dims)
    }


def policy(obs_flat: jnp.ndarray, params: dict) -> int:
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    chicken_y = _chicken_y(curr)

    # Check both current lane (am I safe here?) and next lane (safe to enter?)
    both_safe = _two_lane_safe(
        prev, curr, chicken_y,
        params["lookahead"],
        params["safety_margin"],
        params["danger_lookahead"],
        params["danger_margin"],
    )

    action = jnp.where(
        both_safe > 0.5,
        jnp.int32(UP),
        jnp.int32(NOOP)
    )
    return action


def measure_main(episode_rewards, episode_scores=None) -> float:
    return jnp.sum(episode_rewards)
"""
Auto-generated policy v3
Generated at: 2026-04-07 18:41:23
"""

"""
Parametric JAX policy for Atari Freeway.
Uses two stacked frames to estimate car velocities and time safe lane entries.
CMA-ES tunes 6 float parameters controlling safety thresholds and timing.

Key improvements:
- Tighter danger detection: only block UP if car will hit in the very next step.
- Separate lookahead for "will I be hit NOW" vs "is the next lane safe to enter".
- Bias strongly toward UP; NOOP only when truly necessary.
- Project car position including wrap-around more accurately.
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
    Checks projected car position with wrap-around on both sides.
    """
    cx  = jnp.float32(CHICKEN_X)
    cw  = jnp.float32(CHICKEN_W)

    car_x_curr = _car_x(curr, lane_idx)
    car_x_prev = _car_x(prev, lane_idx)
    car_w_curr = _car_w(curr, lane_idx)
    dx         = car_x_curr - car_x_prev

    proj_x = (car_x_curr + dx * lookahead) % SCREEN_WIDTH

    # wrap-around ghost: car can be on the other side of the wrap boundary
    proj_x_lo = proj_x - SCREEN_WIDTH
    proj_x_hi = proj_x + SCREEN_WIDTH

    eff_w = car_w_curr + margin

    ov1 = _overlap_1d(cx, cw, proj_x,    eff_w)
    ov2 = _overlap_1d(cx, cw, proj_x_lo, eff_w)
    ov3 = _overlap_1d(cx, cw, proj_x_hi, eff_w)

    return jnp.where(jnp.maximum(jnp.maximum(ov1, ov2), ov3) <= 0.0, 1.0, 0.0)


def _find_next_lane(chicken_y):
    """
    Return index of the next lane the chicken is about to enter (moving upward).
    Lanes above chicken_y (lower index = higher on screen = smaller Y).
    """
    diffs  = chicken_y - LANE_Y          # positive => lane is above chicken
    masked = jnp.where(diffs > -8.0, diffs, jnp.inf)
    return jnp.argmin(masked)


def _current_lane(chicken_y):
    """Return index of the lane closest to current chicken_y."""
    diffs = jnp.abs(LANE_Y - chicken_y)
    return jnp.argmin(diffs)


def _immediate_danger(prev, curr, chicken_y, danger_lookahead, danger_margin):
    """
    Return True if the chicken's CURRENT lane or the next lane has a car
    about to hit within `danger_lookahead` frames.
    Only checks 2 closest lanes to reduce false positives.
    """
    cur_lane  = _current_lane(chicken_y)
    next_lane = _find_next_lane(chicken_y)

    safe_cur  = _lane_safe(prev, curr, cur_lane,  danger_lookahead, danger_margin)
    safe_next = _lane_safe(prev, curr, next_lane, danger_lookahead, danger_margin)

    return (safe_cur < 0.5) | (safe_next < 0.5)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_params() -> dict:
    return {
        "safety_margin"   : 4.0,   # extra pixel buffer for next-lane entry check
        "lookahead"       : 2.0,   # steps to project for next-lane entry check
        "danger_lookahead": 0.8,   # steps to project for immediate danger check
        "danger_margin"   : 6.0,   # extra margin for immediate danger detection
        "proximity"       : 10.0,  # (unused slot kept for CMA-ES shape compat)
        "safe_fraction"   : 0.5,   # (unused slot kept for CMA-ES shape compat)
    }


def policy(obs_flat: jnp.ndarray, params: dict) -> int:
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    chicken_y = _chicken_y(curr)
    next_lane = _find_next_lane(chicken_y)

    # Is the next lane safe to step into?
    next_safe = _lane_safe(
        prev, curr, next_lane,
        params["lookahead"],
        params["safety_margin"]
    )

    # Is there an immediate collision threat at current or next position?
    in_danger = _immediate_danger(
        prev, curr, chicken_y,
        params["danger_lookahead"],
        params["danger_margin"]
    )

    # Decision: move UP unless next lane is blocked or immediate danger
    # Bias heavily toward UP; NOOP only when both conditions fail.
    # next_safe already accounts for forward projection, so trust it.
    can_move_up = (next_safe > 0.5) & (~in_danger)

    action = jnp.where(
        can_move_up,
        jnp.int32(UP),
        jnp.int32(NOOP)
    )
    return action


def measure_main(episode_rewards, episode_scores=None) -> float:
    return jnp.sum(episode_rewards)
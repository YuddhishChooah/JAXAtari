"""
Auto-generated policy v4
Generated at: 2026-04-07 19:44:14
"""

"""
Parametric JAX policy for Atari Asterix – improved v3.
CMA-ES tunes 7 float parameters.

Key changes vs v2:
- Danger score uses current-frame enemy proximity with a sharper kernel
  (linear falloff inside danger_radius instead of exp, so large radius
  still penalises far enemies but the gradient is steeper near player).
- Horizontal commitment: once in target lane, ALWAYS move toward target_cx
  unless an emergency dodge fires; removed the weaker drift logic.
- Dodge margin raised slightly; dodge now also considers future enemy x
  (velocity-extrapolated) so we react one step earlier.
- NOOP rate cut: default action when in lane with no item is lateral
  drift toward screen center at higher priority than before.
- Lane score adds a small bonus for lanes adjacent to the current lane
  to encourage harvesting nearby rather than jumping across the board.
"""

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FRAME_SIZE   = 136
N_LANES      = 8
LANE_CENTERS = jnp.array([27., 43., 59., 75., 91., 107., 123., 139.])

NOOP, UP, RIGHT, LEFT, DOWN          = 0, 1, 2, 3, 4
UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = 5, 6, 7, 8

SCREEN_CENTER_X = 80.0
LANE_HALF_WIDTH = 10.0

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def init_params() -> dict:
    return {
        "danger_radius":   jnp.array(22.0),
        "danger_weight":   jnp.array(4.0),
        "collect_weight":  jnp.array(2.5),
        "intercept_tol":   jnp.array(5.0),
        "lane_tol":        jnp.array(5.0),
        "dodge_margin":    jnp.array(18.0),
        "proximity_bonus": jnp.array(1.5),
    }

# ---------------------------------------------------------------------------
# Frame accessors
# ---------------------------------------------------------------------------

def _player(frame):
    return frame[0], frame[1]

def _enemy_x(frame):       return frame[8:16]
def _enemy_y(frame):       return frame[16:24]
def _enemy_active(frame):  return frame[40:48]

def _collect_x(frame):     return frame[72:80]
def _collect_y(frame):     return frame[80:88]
def _collect_active(frame):return frame[104:112]

# ---------------------------------------------------------------------------
# Lane helpers
# ---------------------------------------------------------------------------

def _entities_in_lane(entity_y, lane_idx):
    lane_cy = LANE_CENTERS[lane_idx]
    return jnp.abs(entity_y - lane_cy) < LANE_HALF_WIDTH


def _lane_danger(lane_idx, ex, ey, ea, prev_ex, player_x, danger_radius):
    """
    Danger uses linear (not exp) proximity inside danger_radius on
    velocity-extrapolated future enemy x, so nearby enemies dominate.
    """
    in_lane  = _entities_in_lane(ey, lane_idx).astype(jnp.float32)
    active   = ea.astype(jnp.float32)
    vel      = ex - prev_ex
    future_x = ex + vel
    dx       = jnp.abs(future_x - player_x)
    # linear kernel: 1 when dx=0, 0 when dx>=danger_radius
    close    = jnp.maximum(0.0, 1.0 - dx / (danger_radius + 1e-3))
    return jnp.sum(active * in_lane * close)


def _lane_collect_value(lane_idx, cx, cy, ca, player_x, proximity_bonus):
    in_lane = _entities_in_lane(cy, lane_idx).astype(jnp.float32)
    active  = ca.astype(jnp.float32)
    mask    = active * in_lane
    count   = jnp.sum(mask)
    total   = count + 1e-6
    nearest_cx     = jnp.sum(mask * cx) / total
    lateral_close  = jnp.exp(-jnp.abs(nearest_cx - player_x) / 40.0)
    bonus          = jnp.where(count > 0.5, proximity_bonus * lateral_close, 0.0)
    return count + bonus


def _nearest_collect_x_in_lane(lane_idx, cx, cy, ca):
    in_lane = _entities_in_lane(cy, lane_idx).astype(jnp.float32)
    active  = ca.astype(jnp.float32)
    mask    = active * in_lane
    total   = jnp.sum(mask) + 1e-6
    avg_x   = jnp.sum(mask * cx) / total
    has_any = jnp.sum(mask) > 0.5
    return jnp.where(has_any, avg_x, SCREEN_CENTER_X)

# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

def policy(obs_flat: jnp.ndarray, params: dict) -> int:
    danger_radius   = params["danger_radius"]
    danger_weight   = params["danger_weight"]
    collect_weight  = params["collect_weight"]
    intercept_tol   = params["intercept_tol"]
    lane_tol        = params["lane_tol"]
    dodge_margin    = params["dodge_margin"]
    proximity_bonus = params["proximity_bonus"]

    prev = obs_flat[:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:]

    px, py = _player(curr)

    ex      = _enemy_x(curr)
    ey      = _enemy_y(curr)
    ea      = _enemy_active(curr)
    prev_ex = _enemy_x(prev)

    cx_arr  = _collect_x(curr)
    cy_arr  = _collect_y(curr)
    ca      = _collect_active(curr)

    # ------------------------------------------------------------------
    # Current lane index
    # ------------------------------------------------------------------
    lane_diffs = jnp.abs(LANE_CENTERS - py)
    curr_lane  = jnp.argmin(lane_diffs)

    # ------------------------------------------------------------------
    # Score each lane with adjacency bonus to encourage nearby harvesting
    # ------------------------------------------------------------------
    def lane_score(i):
        d    = _lane_danger(i, ex, ey, ea, prev_ex, px, danger_radius)
        v    = _lane_collect_value(i, cx_arr, cy_arr, ca, px, proximity_bonus)
        # small bonus for lanes adjacent to current lane (reduces big jumps)
        adj  = jnp.where(jnp.abs(i - curr_lane) <= 1, 0.3, 0.0)
        return collect_weight * v - danger_weight * d + adj

    scores    = jnp.array([lane_score(i) for i in range(N_LANES)])
    best_lane = jnp.argmax(scores)
    target_y  = LANE_CENTERS[best_lane]

    # ------------------------------------------------------------------
    # Target collectible x in best lane
    # ------------------------------------------------------------------
    target_cx = _nearest_collect_x_in_lane(best_lane, cx_arr, cy_arr, ca)

    # ------------------------------------------------------------------
    # Vertical direction
    # ------------------------------------------------------------------
    in_target_lane = jnp.abs(py - target_y) < lane_tol
    need_up        = py > (target_y + lane_tol)
    need_down      = py < (target_y - lane_tol)

    # ------------------------------------------------------------------
    # Horizontal: always commit when in target lane; drift to center
    # when no item present (replaces old passive NOOP)
    # ------------------------------------------------------------------
    dx_collect = target_cx - px
    need_right = dx_collect >  intercept_tol
    need_left  = dx_collect < -intercept_tol

    # Drift toward center when in lane and no direct item target
    drift_right = (px < SCREEN_CENTER_X - intercept_tol)
    drift_left  = (px > SCREEN_CENTER_X + intercept_tol)
    h_right = need_right | (~need_right & ~need_left & drift_right)
    h_left  = need_left  | (~need_right & ~need_left & drift_left)

    # ------------------------------------------------------------------
    # Emergency dodge: use velocity-extrapolated enemy position
    # ------------------------------------------------------------------
    curr_lane_cy = LANE_CENTERS[curr_lane]
    in_curr_lane = (jnp.abs(ey - curr_lane_cy) < LANE_HALF_WIDTH).astype(jnp.float32)
    active_mask  = ea.astype(jnp.float32) * in_curr_lane
    vel_e        = ex - prev_ex
    future_ex    = ex + vel_e
    dx_enemies   = jnp.abs(future_ex - px)
    masked_dx    = jnp.where(active_mask > 0.5, dx_enemies, 999.0)
    nearest_idx  = jnp.argmin(masked_dx)
    nearest_e_x  = ex[nearest_idx]
    nearest_dist = masked_dx[nearest_idx]

    enemy_close  = nearest_dist < dodge_margin
    dodge_right  = enemy_close & (nearest_e_x < px)
    dodge_left   = enemy_close & (nearest_e_x >= px)

    # Override horizontal with dodge
    h_right_final = jnp.where(enemy_close, dodge_right, h_right)
    h_left_final  = jnp.where(enemy_close, dodge_left,  h_left)

    # ------------------------------------------------------------------
    # Combine into action
    # ------------------------------------------------------------------
    action = jnp.array(NOOP)

    # Default lateral even when navigating vertically (keep moving)
    action = jnp.where(h_right_final, jnp.array(RIGHT), action)
    action = jnp.where(h_left_final,  jnp.array(LEFT),  action)

    # Pure vertical when no horizontal needed
    action = jnp.where(need_up   & ~h_right_final & ~h_left_final, jnp.array(UP),   action)
    action = jnp.where(need_down & ~h_right_final & ~h_left_final, jnp.array(DOWN), action)

    # Diagonals
    action = jnp.where(need_up   & h_right_final, jnp.array(UPRIGHT),   action)
    action = jnp.where(need_up   & h_left_final,  jnp.array(UPLEFT),    action)
    action = jnp.where(need_down & h_right_final, jnp.array(DOWNRIGHT), action)
    action = jnp.where(need_down & h_left_final,  jnp.array(DOWNLEFT),  action)

    # Emergency pure lateral dodge (overrides diagonals when in-lane)
    action = jnp.where(enemy_close & dodge_right & in_target_lane, jnp.array(RIGHT), action)
    action = jnp.where(enemy_close & dodge_left  & in_target_lane, jnp.array(LEFT),  action)

    return action.astype(jnp.int32)

# ---------------------------------------------------------------------------
# Measure
# ---------------------------------------------------------------------------

def measure_main(episode_rewards: jnp.ndarray, episode_scores: dict) -> float:
    return jnp.sum(episode_rewards)
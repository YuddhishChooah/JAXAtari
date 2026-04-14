"""
Auto-generated policy v1
Generated at: 2026-04-07 18:47:47
"""

"""
Parametric JAX policy for Atari Asterix.
CMA-ES tunes 6 float parameters controlling lane selection and lateral movement.
"""

import jax.numpy as jnp
import jax

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FRAME_SIZE = 136
N_LANES = 8
LANE_CENTERS = jnp.array([27., 43., 59., 75., 91., 107., 123., 139.])

# Action aliases
NOOP, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4
UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = 5, 6, 7, 8

# Player x center (screen width ~160, player roughly centered)
SCREEN_CENTER_X = 80.0
PLAYER_SPAWN_X = 76.0

# ---------------------------------------------------------------------------
# Parameter initialisation
# ---------------------------------------------------------------------------

def init_params() -> dict:
    return {
        "danger_radius":   jnp.array(20.0),   # x-distance within which an enemy is dangerous
        "danger_weight":   jnp.array(3.0),    # how much to penalise danger vs reward
        "collect_weight":  jnp.array(2.0),    # reward for active collectible in lane
        "intercept_tol":   jnp.array(6.0),    # x tolerance to count as "aligned" with collectible
        "lane_tol":        jnp.array(5.0),    # y tolerance to count as "in lane"
        "dodge_margin":    jnp.array(12.0),   # x margin to trigger horizontal dodge
    }

# ---------------------------------------------------------------------------
# Frame accessors (operate on a single 136-element frame slice)
# ---------------------------------------------------------------------------

def _player(frame):
    return frame[0], frame[1]  # x, y

def _enemy_x(frame):   return frame[8:16]
def _enemy_y(frame):   return frame[16:24]
def _enemy_active(frame): return frame[40:48]
def _enemy_orient(frame): return frame[64:72]

def _collect_x(frame):      return frame[72:80]
def _collect_y(frame):      return frame[80:88]
def _collect_active(frame): return frame[104:112]

# ---------------------------------------------------------------------------
# Lane helpers
# ---------------------------------------------------------------------------

def _lane_danger(lane_idx, enemy_x, enemy_y, enemy_active, player_x, danger_radius):
    """Scalar danger for one lane: sum of proximity of active enemies."""
    lane_cy = LANE_CENTERS[lane_idx]
    in_lane = (jnp.abs(enemy_y - lane_cy) < 10.0).astype(jnp.float32)
    active   = enemy_active.astype(jnp.float32)
    dx       = jnp.abs(enemy_x - player_x)
    close    = jnp.exp(-dx / (danger_radius + 1e-3))
    return jnp.sum(active * in_lane * close)


def _lane_value(lane_idx, collect_x, collect_y, collect_active, player_x):
    """Scalar value for one lane: count active collectibles nearby."""
    lane_cy = LANE_CENTERS[lane_idx]
    in_lane = (jnp.abs(collect_y - lane_cy) < 10.0).astype(jnp.float32)
    active  = collect_active.astype(jnp.float32)
    return jnp.sum(active * in_lane)


def _nearest_collect_x(lane_idx, collect_x, collect_y, collect_active):
    """X position of the nearest active collectible in this lane, or SCREEN_CENTER_X."""
    lane_cy = LANE_CENTERS[lane_idx]
    in_lane = (jnp.abs(collect_y - lane_cy) < 10.0).astype(jnp.float32)
    active  = collect_active.astype(jnp.float32)
    mask    = active * in_lane
    # weighted average x
    total   = jnp.sum(mask) + 1e-6
    cx      = jnp.sum(mask * collect_x) / total
    has_any = (jnp.sum(mask) > 0.0)
    return jnp.where(has_any, cx, SCREEN_CENTER_X)

# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

def policy(obs_flat: jnp.ndarray, params: dict) -> int:
    danger_radius  = params["danger_radius"]
    danger_weight  = params["danger_weight"]
    collect_weight = params["collect_weight"]
    intercept_tol  = params["intercept_tol"]
    lane_tol       = params["lane_tol"]
    dodge_margin   = params["dodge_margin"]

    prev = obs_flat[:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:]

    px, py      = _player(curr)
    ex          = _enemy_x(curr)
    ey          = _enemy_y(curr)
    ea          = _enemy_active(curr)
    cx_arr      = _collect_x(curr)
    cy_arr      = _collect_y(curr)
    ca          = _collect_active(curr)

    # ------------------------------------------------------------------
    # Score each lane
    # ------------------------------------------------------------------
    def lane_score(i):
        d = _lane_danger(i, ex, ey, ea, px, danger_radius)
        v = _lane_value(i, cx_arr, cy_arr, ca, px)
        return collect_weight * v - danger_weight * d

    scores = jnp.array([lane_score(i) for i in range(N_LANES)])

    # Find best lane
    best_lane = jnp.argmax(scores)
    target_y  = LANE_CENTERS[best_lane]

    # ------------------------------------------------------------------
    # Determine current lane
    # ------------------------------------------------------------------
    lane_diffs = jnp.abs(LANE_CENTERS - py)
    curr_lane  = jnp.argmin(lane_diffs)

    in_target_lane = jnp.abs(py - target_y) < lane_tol

    # ------------------------------------------------------------------
    # Nearest collectible x in best lane
    # ------------------------------------------------------------------
    target_cx = _nearest_collect_x(best_lane, cx_arr, cy_arr, ca)

    # ------------------------------------------------------------------
    # Nearest enemy in current lane for dodge
    # ------------------------------------------------------------------
    curr_lane_cy = LANE_CENTERS[curr_lane]
    in_curr_lane = (jnp.abs(ey - curr_lane_cy) < 10.0).astype(jnp.float32)
    active_mask  = ea.astype(jnp.float32) * in_curr_lane
    # closest enemy
    dx_enemies   = jnp.abs(ex - px)
    # mask inactive with large value
    masked_dx    = jnp.where(active_mask > 0.5, dx_enemies, 999.0)
    nearest_e_idx = jnp.argmin(masked_dx)
    nearest_e_x   = ex[nearest_e_idx]
    nearest_e_dx  = px - nearest_e_x   # positive => enemy is to the left

    enemy_close = (masked_dx[nearest_e_idx] < dodge_margin)

    # ------------------------------------------------------------------
    # Vertical direction
    # ------------------------------------------------------------------
    need_up   = py > (target_y + lane_tol)   # need to move to lower y (up)
    need_down = py < (target_y - lane_tol)   # need to move to higher y (down)

    # ------------------------------------------------------------------
    # Horizontal direction: intercept collectible or dodge
    # ------------------------------------------------------------------
    dx_collect = target_cx - px               # positive => collectible is to the right
    need_right = dx_collect > intercept_tol
    need_left  = dx_collect < -intercept_tol

    # When enemy is dangerously close, dodge away from it
    dodge_right = enemy_close & (nearest_e_dx < 0.0)  # enemy to right → go left? no: nearest_e_dx = px - e_x
    # nearest_e_dx > 0 means enemy is to the left → dodge right
    dodge_left  = enemy_close & (nearest_e_dx > 0.0)

    # Override horizontal if dodging
    h_right = jnp.where(enemy_close, dodge_right, need_right)
    h_left  = jnp.where(enemy_close, dodge_left,  need_left)

    # ------------------------------------------------------------------
    # Combine into action
    # ------------------------------------------------------------------
    # Priority: vertical + horizontal combined
    # UP=1, RIGHT=2, LEFT=3, DOWN=4, UPRIGHT=5, UPLEFT=6, DOWNRIGHT=7, DOWNLEFT=8

    action = jnp.array(NOOP)

    # Vertical needed?
    going_up   = need_up
    going_down = need_down

    # Combined actions
    action = jnp.where(going_up   & h_right, jnp.array(UPRIGHT),   action)
    action = jnp.where(going_up   & h_left,  jnp.array(UPLEFT),    action)
    action = jnp.where(going_up   & ~h_right & ~h_left, jnp.array(UP),   action)
    action = jnp.where(going_down & h_right, jnp.array(DOWNRIGHT),  action)
    action = jnp.where(going_down & h_left,  jnp.array(DOWNLEFT),   action)
    action = jnp.where(going_down & ~h_right & ~h_left, jnp.array(DOWN), action)

    # Already in target lane: move horizontally
    action = jnp.where(in_target_lane & h_right & ~going_up & ~going_down, jnp.array(RIGHT), action)
    action = jnp.where(in_target_lane & h_left  & ~going_up & ~going_down, jnp.array(LEFT),  action)

    # Emergency dodge (override anything when very close enemy)
    action = jnp.where(enemy_close & dodge_right & ~going_up & ~going_down, jnp.array(RIGHT), action)
    action = jnp.where(enemy_close & dodge_left  & ~going_up & ~going_down, jnp.array(LEFT),  action)

    return action.astype(jnp.int32)

# ---------------------------------------------------------------------------
# Measure
# ---------------------------------------------------------------------------

def measure_main(episode_rewards: jnp.ndarray, episode_scores: dict) -> float:
    return jnp.sum(episode_rewards)
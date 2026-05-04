"""
Auto-generated policy v5
Generated at: 2026-04-28 16:53:07
"""

import jax
import jax.numpy as jnp

# Action constants
NOOP = 0
RIGHT = 1
LEFT = 2
DOWN = 3
UPRIGHT = 4
UPLEFT = 5
DOWNRIGHT = 6
DOWNLEFT = 7

FRAME_SIZE = 136
LANE_CENTERS = jnp.array([27.0, 43.0, 59.0, 75.0, 91.0, 107.0, 123.0, 139.0])
VALUE_TABLE = jnp.array([50.0, 100.0, 200.0, 300.0, 0.0, 0.0, 0.0, 0.0])

# Approximate frames it takes to traverse one lane vertically
FRAMES_PER_LANE = 8.0
# Half-widths used as a collision buffer (player ~ 8, enemy ~ 8)
COLLISION_HALF = 10.0


def init_params():
    return {
        "danger_w": jnp.array(1.7042),
        "danger_radius": jnp.array(25.1535),
        "lane_dist_w": jnp.array(39.3979),
        "intercept_w": jnp.array(0.8705),
        "value_w": jnp.array(0.6187),
        "switch_bias": jnp.array(14.6654),
        "danger_thresh": jnp.array(0.5297),
    }


def _split(obs_flat):
    return obs_flat[0:FRAME_SIZE], obs_flat[FRAME_SIZE:2 * FRAME_SIZE]


def _player_lane(py):
    return jnp.argmin(jnp.abs(LANE_CENTERS - py))


def _lane_danger(prev_ex, curr_ex, e_active, ref_x, t_arrival, radius):
    """Project enemy x to arrival time and compute soft+hard danger.

    Returns (soft_danger, hard_unsafe) per lane (each shape [8]).
    """
    edx = curr_ex - prev_ex
    # Evaluate min |enemy_x - ref_x| over t in [0, t_arrival + 4]
    horizon = t_arrival + 4.0
    denom = jnp.where(jnp.abs(edx) < 1e-3, 1e-3, edx)
    t_star = (ref_x - curr_ex) / denom
    t_clamped = jnp.clip(t_star, 0.0, horizon)
    x_at_t = curr_ex + t_clamped * edx
    gap = jnp.abs(x_at_t - ref_x)

    # also evaluate exactly at arrival time (in case enemy passes after)
    x_at_arrive = curr_ex + t_arrival * edx
    gap_arrive = jnp.abs(x_at_arrive - ref_x)
    gap = jnp.minimum(gap, gap_arrive)

    # inactive enemies: push gap large
    gap = gap + (1.0 - e_active) * 1000.0

    soft = jnp.exp(-gap / radius) * e_active
    hard_unsafe = (gap < COLLISION_HALF * 2.0) & (e_active > 0.5)
    return soft, hard_unsafe


def policy(obs_flat, params):
    prev, curr = _split(obs_flat)

    px = curr[0]
    py = curr[1]

    p_ex = prev[8:16]
    c_ex = curr[8:16]
    e_active = curr[40:48]

    p_cx = prev[72:80]
    c_cx = curr[72:80]
    c_active = curr[104:112]
    c_vid = curr[112:120]

    cur_lane = _player_lane(py)
    lane_idx = jnp.arange(8)

    # Predicted collectible x a few frames ahead (interception target)
    cdx = c_cx - p_cx
    pred_cx = c_cx + 3.0 * cdx

    # Per-lane arrival-x = predicted collectible x (or px if no collectible)
    target_x_lane = jnp.where(c_active > 0.5, pred_cx, jnp.full_like(pred_cx, px))

    # Per-lane arrival-time = lanes_to_cross * frames_per_lane
    lane_dist = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)
    t_arrival = lane_dist * FRAMES_PER_LANE

    radius = params["danger_radius"]

    # Per-lane destination danger: enemy in lane i, evaluated at that lane's
    # arrival x and arrival time.
    soft_danger, hard_unsafe = _lane_danger(
        p_ex, c_ex, e_active, target_x_lane, t_arrival, radius
    )

    # Also compute danger at player's current x for current-lane gating
    soft_here, hard_here = _lane_danger(
        p_ex, c_ex, e_active, jnp.full_like(c_ex, px), jnp.zeros_like(c_ex), radius
    )

    # Value
    vid_int = jnp.clip(c_vid.astype(jnp.int32), 0, 7)
    value = VALUE_TABLE[vid_int] * c_active

    # Interception cost: distance from player x to predicted collectible x
    intercept_cost = jnp.abs(pred_cx - px)

    # Lane scoring (soft danger only; hard veto applied separately)
    score = (
        params["value_w"] * value
        - params["danger_w"] * soft_danger * 100.0
        - params["intercept_w"] * intercept_cost
        - params["lane_dist_w"] * lane_dist
    )

    # Stay bonus when current lane is safe AND has a positive item
    cur_soft = soft_here[cur_lane]
    cur_hard = hard_here[cur_lane]
    cur_safe = (cur_soft < params["danger_thresh"]) & (~cur_hard)
    cur_has_value = value[cur_lane] > 0.0
    stay_ok = cur_safe & cur_has_value
    stay_bonus = jnp.where(
        (lane_idx == cur_lane) & stay_ok, params["switch_bias"], 0.0
    )
    score = score + stay_bonus

    # Soft + hard safety gating per lane
    lane_safe = (soft_danger < params["danger_thresh"]) & (~hard_unsafe)
    has_value = (value > 0.0) & lane_safe
    score_value = jnp.where(has_value, score, score - 1e5)

    best_value_lane = jnp.argmax(score_value)
    any_value = jnp.any(has_value)

    # Fallback: safest reachable lane
    safe_score = (
        -params["danger_w"] * soft_danger * 100.0
        - params["lane_dist_w"] * lane_dist
        - jnp.where(hard_unsafe, 1e4, 0.0)
    )
    best_safe_lane = jnp.argmax(safe_score)

    target_lane = jnp.where(any_value, best_value_lane, best_safe_lane)

    tgt_active = c_active[target_lane] > 0.5
    tgt_cx = pred_cx[target_lane]
    hold_x = jnp.array(80.0)
    target_x = jnp.where(tgt_active, tgt_cx, hold_x)

    # Current-lane danger drives evasive override
    cur_dangerous = (cur_soft > params["danger_thresh"]) | cur_hard

    # Enemy-side detection in current lane
    cur_enemy_x = c_ex[cur_lane]
    enemy_on_right = cur_enemy_x > px
    escape_right = ~enemy_on_right

    # Adjacent lane danger for vertical escape choice
    up_lane = jnp.maximum(cur_lane - 1, 0)
    down_lane = jnp.minimum(cur_lane + 1, 7)
    up_bad = (soft_danger[up_lane] > params["danger_thresh"]) | hard_unsafe[up_lane]
    down_bad = (soft_danger[down_lane] > params["danger_thresh"]) | hard_unsafe[down_lane]

    can_up = cur_lane > 0
    can_down = cur_lane < 7
    prefer_up = (~up_bad) & can_up
    prefer_up = prefer_up | (down_bad & can_up)
    prefer_up = prefer_up & can_up

    danger_action = jnp.where(
        prefer_up,
        jnp.where(escape_right, UPRIGHT, UPLEFT),
        jnp.where(escape_right, DOWNRIGHT, DOWNLEFT),
    )
    both_adj_bad = up_bad & down_bad
    horiz_dodge = jnp.where(escape_right, RIGHT, LEFT)
    danger_action = jnp.where(both_adj_bad, horiz_dodge, danger_action)

    # Normal action toward target lane / x
    need_up = target_lane < cur_lane
    need_down = target_lane > cur_lane
    go_right = target_x > px + 2.0
    go_left = target_x < px - 2.0

    # Veto diagonal if destination lane is unsafe
    dest_unsafe = (soft_danger[target_lane] > params["danger_thresh"]) | hard_unsafe[target_lane]
    safe_to_transition = ~dest_unsafe

    up_action = jnp.where(go_left, UPLEFT, UPRIGHT)
    down_action = jnp.where(go_left, DOWNLEFT, DOWNRIGHT)
    horiz_action = jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))

    vert_action = jnp.where(need_up, up_action, down_action)
    need_vert = need_up | need_down
    normal_action = jnp.where(
        need_vert & safe_to_transition, vert_action, horiz_action
    )

    # Patrol when aligned and no active target
    aligned = (~need_vert) & (~go_right) & (~go_left)
    patrol_action = jnp.where(px < 80.0, RIGHT, LEFT)
    normal_action = jnp.where(aligned & ~tgt_active, patrol_action, normal_action)

    action = jnp.where(cur_dangerous, danger_action, normal_action)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
"""
Auto-generated policy v3
Generated at: 2026-04-28 15:33:37
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

# Approximate frames to traverse one lane (used for arrival-x prediction)
FRAMES_PER_LANE = 4.0


def init_params():
    return {
        "danger_w": jnp.array(1.7042),
        "danger_radius": jnp.array(30.0),
        "lane_dist_w": jnp.array(12.0),
        "intercept_w": jnp.array(0.8705),
        "value_w": jnp.array(0.6187),
        "switch_bias": jnp.array(20.0),
        "danger_thresh": jnp.array(0.45),
    }


def _split(obs_flat):
    return obs_flat[0:FRAME_SIZE], obs_flat[FRAME_SIZE:2 * FRAME_SIZE]


def _player_lane(py):
    return jnp.argmin(jnp.abs(LANE_CENTERS - py))


def _projected_min_gap(prev_x, curr_x, active, ref_x, t_lo, t_hi):
    # min |curr + t*dx - ref_x| over t in [t_lo, t_hi]
    dx = curr_x - prev_x
    denom = jnp.where(jnp.abs(dx) < 1e-3, 1e-3, dx)
    t_star = (ref_x - curr_x) / denom
    t_clamped = jnp.clip(t_star, t_lo, t_hi)
    x_at_t = curr_x + t_clamped * dx
    gap = jnp.abs(x_at_t - ref_x)
    gap = gap + (1.0 - active) * 1000.0
    return gap


def _lane_danger_at_arrival(p_ex, c_ex, e_active, arrival_x, t_lo, t_hi, radius):
    # Per-lane danger, evaluated at the arrival x for that lane and over a forward time window
    gap = _projected_min_gap(p_ex, c_ex, e_active, arrival_x, t_lo, t_hi)
    danger = jnp.exp(-gap / radius) * e_active
    return danger


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

    radius = params["danger_radius"]
    thresh = params["danger_thresh"]

    # Lane distance from current lane (in lanes)
    lane_dist = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)

    # Estimated arrival time per lane (frames)
    t_arrival = lane_dist * FRAMES_PER_LANE + 1.0

    # Predicted arrival x: assume player tries to move toward the collectible's predicted x
    # but as a first approximation use current px (lateral motion happens during transit).
    # Use a forward window centered around arrival time.
    cdx = c_cx - p_cx
    pred_cx = c_cx + t_arrival * cdx  # collectible x at predicted arrival time per lane

    # Per-lane danger evaluated at the predicted collectible interception x at arrival
    # window: from now until a few frames after arrival
    t_lo = jnp.maximum(t_arrival - 2.0, 0.0)
    t_hi = t_arrival + 4.0
    dest_danger = _lane_danger_at_arrival(p_ex, c_ex, e_active, pred_cx, t_lo, t_hi, radius)

    # Also danger at player's current x (for staying / horizontal play)
    danger_now = _lane_danger_at_arrival(p_ex, c_ex, e_active, px * jnp.ones(8), 0.0, 6.0, radius)

    # Path danger: for current lane, danger over the next few frames at px
    cur_path_danger = danger_now[cur_lane]

    # Value
    vid_int = jnp.clip(c_vid.astype(jnp.int32), 0, 7)
    value = VALUE_TABLE[vid_int] * c_active

    # Interception cost (how far we'd need to move horizontally on arrival)
    intercept_cost = jnp.abs(pred_cx - px)

    # Lane score: value harvest - danger - intercept - distance
    score = (
        params["value_w"] * value
        - params["danger_w"] * dest_danger * 300.0
        - params["intercept_w"] * intercept_cost
        - params["lane_dist_w"] * lane_dist
    )

    # Stay/commit bonus for current lane if it has an active item and is safe
    cur_safe = dest_danger[cur_lane] < thresh
    cur_has_value = value[cur_lane] > 0.0
    stay_bonus = jnp.where(
        (lane_idx == cur_lane) & cur_safe & cur_has_value,
        params["switch_bias"],
        0.0,
    )
    score = score + stay_bonus

    # Mask: only consider lanes that are active collectibles AND safe at arrival
    safe_mask = dest_danger < thresh
    has_value = (value > 0.0) & safe_mask
    score_value = jnp.where(has_value, score, -1e9)

    best_value_lane = jnp.argmax(score_value)
    any_value = jnp.any(has_value)

    # Fallback: safest lane near current
    safe_score = -params["danger_w"] * dest_danger * 300.0 - params["lane_dist_w"] * lane_dist
    best_safe_lane = jnp.argmax(safe_score)

    target_lane = jnp.where(any_value, best_value_lane, best_safe_lane)

    tgt_active = c_active[target_lane] > 0.5
    tgt_cx = pred_cx[target_lane]
    hold_x = jnp.array(80.0)
    target_x = jnp.where(tgt_active, tgt_cx, hold_x)

    # ---- Action selection ----

    # Current-lane danger: use both danger_now (at px) and immediate dest_danger
    cur_dangerous = (cur_path_danger > thresh) | (dest_danger[cur_lane] > thresh)

    # Side of enemy in current lane
    cur_enemy_x = c_ex[cur_lane]
    enemy_on_right = cur_enemy_x > px
    escape_right = ~enemy_on_right

    # Adjacent-lane danger at arrival (use indices into dest_danger)
    up_lane = jnp.maximum(cur_lane - 1, 0)
    down_lane = jnp.minimum(cur_lane + 1, 7)
    up_danger = dest_danger[up_lane]
    down_danger = dest_danger[down_lane]
    can_up = cur_lane > 0
    can_down = cur_lane < 7

    prefer_up = (up_danger <= down_danger) & can_up
    prefer_up = prefer_up | (~can_down & can_up)

    danger_action = jnp.where(
        prefer_up,
        jnp.where(escape_right, UPRIGHT, UPLEFT),
        jnp.where(escape_right, DOWNRIGHT, DOWNLEFT),
    )

    # If both adjacent lanes are also dangerous, dodge horizontally
    both_adj_bad = (up_danger > thresh) & (down_danger > thresh)
    horiz_dodge = jnp.where(escape_right, RIGHT, LEFT)
    danger_action = jnp.where(both_adj_bad, horiz_dodge, danger_action)

    # Normal action: move toward target lane and target x
    need_up = target_lane < cur_lane
    need_down = target_lane > cur_lane
    need_vert = need_up | need_down
    go_right = target_x > px + 2.0
    go_left = target_x < px - 2.0

    # Abort transition if destination lane danger spikes
    dest_dangerous = dest_danger[target_lane] > thresh
    safe_to_transition = ~dest_dangerous

    up_action = jnp.where(go_left, UPLEFT, UPRIGHT)
    down_action = jnp.where(go_left, DOWNLEFT, DOWNRIGHT)
    horiz_action = jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))

    vert_action = jnp.where(need_up, up_action, down_action)
    normal_action = jnp.where(
        need_vert & safe_to_transition, vert_action, horiz_action
    )

    # If aligned with no item incoming, gentle patrol
    aligned = (~need_vert) & (~go_right) & (~go_left)
    patrol_action = jnp.where(px < 80.0, RIGHT, LEFT)
    normal_action = jnp.where(aligned & ~tgt_active, patrol_action, normal_action)

    action = jnp.where(cur_dangerous, danger_action, normal_action)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
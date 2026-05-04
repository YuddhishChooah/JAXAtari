"""
Auto-generated policy v2
Generated at: 2026-04-28 14:58:00
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


def init_params():
    return {
        "danger_w": jnp.array(1.4748),
        "danger_radius": jnp.array(25.034),
        "lane_dist_w": jnp.array(20.0),
        "intercept_w": jnp.array(0.7771),
        "value_w": jnp.array(0.7041),
        "switch_bias": jnp.array(6.0),
        "danger_thresh": jnp.array(0.55),
    }


def _split(obs_flat):
    return obs_flat[0:FRAME_SIZE], obs_flat[FRAME_SIZE:2 * FRAME_SIZE]


def _player_lane(py):
    return jnp.argmin(jnp.abs(LANE_CENTERS - py))


def _projected_min_gap(prev_x, curr_x, active, px, horizon=6.0):
    # min |x_t - px| over t in [0, horizon]; closed form for linear motion
    dx = curr_x - prev_x
    # parameter t* that minimizes |curr + t*dx - px|
    denom = jnp.where(jnp.abs(dx) < 1e-3, 1e-3, dx)
    t_star = (px - curr_x) / denom
    t_clamped = jnp.clip(t_star, 0.0, horizon)
    x_at_t = curr_x + t_clamped * dx
    gap = jnp.abs(x_at_t - px)
    # if inactive, push gap large
    gap = gap + (1.0 - active) * 1000.0
    return gap, dx


def _danger_at_x(prev_ex, curr_ex, e_active, ref_x, params, horizon):
    gap, dx = _projected_min_gap(prev_ex, curr_ex, e_active, ref_x, horizon)
    # directional weight: enemy moving toward ref_x is more dangerous
    toward = jnp.sign(ref_x - curr_ex) * jnp.sign(dx)
    dir_factor = 1.0 + 0.5 * jnp.clip(toward, 0.0, 1.0)
    radius = params["danger_radius"]
    danger = jnp.exp(-gap / radius) * dir_factor * e_active
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

    # danger evaluated at player's current x with multi-step projection
    danger_now = _danger_at_x(p_ex, c_ex, e_active, px, params, 6.0)
    # danger evaluated at predicted collectible interception x (for transition arrival)
    cdx = c_cx - p_cx
    pred_cx = c_cx + 3.0 * cdx
    # value
    vid_int = jnp.clip(c_vid.astype(jnp.int32), 0, 7)
    value = VALUE_TABLE[vid_int] * c_active

    # interception cost
    intercept_cost = jnp.abs(pred_cx - px)

    # lane-distance / reachability
    lane_dist = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)

    # transition-aware danger: also evaluate enemy danger at the player's x
    # over a forward window; use danger_now for all lanes (already at px)
    # but for adjacent-lane crossings, danger at the lane index matters.
    danger = danger_now

    # base lane score
    score = (
        params["value_w"] * value
        - params["danger_w"] * danger * 300.0
        - params["intercept_w"] * intercept_cost
        - params["lane_dist_w"] * lane_dist
    )

    # stay bonus only when current lane is safe AND has a positive item
    cur_safe = danger[cur_lane] < params["danger_thresh"]
    cur_has_value = value[cur_lane] > 0.0
    stay_ok = cur_safe & cur_has_value
    stay_bonus = jnp.where(
        (lane_idx == cur_lane) & stay_ok, params["switch_bias"], 0.0
    )
    score = score + stay_bonus

    safe_mask = danger < params["danger_thresh"]
    has_value = (value > 0.0) & safe_mask
    score_value = jnp.where(has_value, score, score - 1e5)

    best_value_lane = jnp.argmax(score_value)
    any_value = jnp.any(has_value)

    # fallback safest lane near current
    safe_score = -params["danger_w"] * danger * 300.0 - params["lane_dist_w"] * lane_dist
    best_safe_lane = jnp.argmax(safe_score)

    target_lane = jnp.where(any_value, best_value_lane, best_safe_lane)

    tgt_active = c_active[target_lane] > 0.5
    tgt_cx = pred_cx[target_lane]
    # hold-near-center fallback (low exposure) instead of edge-seeking patrol
    hold_x = jnp.array(80.0)
    target_x = jnp.where(tgt_active, tgt_cx, hold_x)

    # current lane danger gating
    cur_danger = danger[cur_lane]
    cur_dangerous = cur_danger > params["danger_thresh"]

    # enemy in current lane: which side
    cur_enemy_x = c_ex[cur_lane]
    enemy_on_right = cur_enemy_x > px

    # pick escape vertical: compare up vs down lane danger
    up_lane = jnp.maximum(cur_lane - 1, 0)
    down_lane = jnp.minimum(cur_lane + 1, 7)
    up_danger = danger[up_lane]
    down_danger = danger[down_lane]
    can_up = cur_lane > 0
    can_down = cur_lane < 7
    # prefer the safer adjacent lane; tie-break to up
    prefer_up = (up_danger <= down_danger) & can_up
    prefer_up = prefer_up | (~can_down)
    prefer_up = prefer_up & can_up

    # escape direction horizontally: away from enemy
    escape_right = ~enemy_on_right

    danger_action = jnp.where(
        prefer_up,
        jnp.where(escape_right, UPRIGHT, UPLEFT),
        jnp.where(escape_right, DOWNRIGHT, DOWNLEFT),
    )

    # if both adjacent lanes are also dangerous, just dodge horizontally
    both_adj_bad = (up_danger > params["danger_thresh"]) & (down_danger > params["danger_thresh"])
    horiz_dodge = jnp.where(escape_right, RIGHT, LEFT)
    danger_action = jnp.where(both_adj_bad, horiz_dodge, danger_action)

    # normal action
    need_up = target_lane < cur_lane
    need_down = target_lane > cur_lane
    go_right = target_x > px + 2.0
    go_left = target_x < px - 2.0

    # Abort diagonal if destination lane danger spikes
    dest_dangerous = danger[target_lane] > params["danger_thresh"]
    safe_to_transition = ~dest_dangerous

    up_action = jnp.where(go_left, UPLEFT, UPRIGHT)
    down_action = jnp.where(go_left, DOWNLEFT, DOWNRIGHT)
    horiz_action = jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))

    vert_action = jnp.where(need_up, up_action, down_action)
    need_vert = need_up | need_down
    normal_action = jnp.where(
        need_vert & safe_to_transition, vert_action, horiz_action
    )

    # if aligned and idle, patrol slightly to keep harvesting
    aligned = (~need_vert) & (~go_right) & (~go_left)
    patrol_action = jnp.where(px < 80.0, RIGHT, LEFT)
    normal_action = jnp.where(aligned & ~tgt_active, patrol_action, normal_action)

    action = jnp.where(cur_dangerous, danger_action, normal_action)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
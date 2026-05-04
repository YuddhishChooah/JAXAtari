"""
Auto-generated policy v4
Generated at: 2026-04-28 16:13:23
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
HORIZONS = jnp.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
COLLISION_BUFFER = 9.0
PLAYER_STEP = 3.0  # approx player horizontal speed per frame


def init_params():
    return {
        "danger_w": jnp.array(1.7042),
        "danger_radius": jnp.array(25.1535),
        "lane_dist_w": jnp.array(20.0),
        "intercept_w": jnp.array(0.8705),
        "value_w": jnp.array(0.6187),
        "switch_bias": jnp.array(6.0),
        "danger_thresh": jnp.array(0.40),
    }


def _split(obs_flat):
    return obs_flat[0:FRAME_SIZE], obs_flat[FRAME_SIZE:2 * FRAME_SIZE]


def _player_lane(py):
    return jnp.argmin(jnp.abs(LANE_CENTERS - py))


def _danger_at(prev_ex, curr_ex, e_active, e_orient, ref_x, params):
    """Multi-horizon max-danger evaluation at a reference x."""
    dx = curr_ex - prev_ex
    # positions over horizons: shape (8 lanes, H)
    proj = curr_ex[:, None] + dx[:, None] * HORIZONS[None, :]
    gap = jnp.abs(proj - ref_x)  # (8, H)

    # approach-direction multiplier using orientation and dx sign
    # orient: 1 = moving right, 2 = moving left
    moving_right = (e_orient > 0.5) & (e_orient < 1.5)
    moving_left = (e_orient > 1.5) & (e_orient < 2.5)
    enemy_left_of_ref = curr_ex < ref_x
    approaching = (moving_right & enemy_left_of_ref) | (moving_left & ~enemy_left_of_ref)
    receding = (moving_right & ~enemy_left_of_ref) | (moving_left & enemy_left_of_ref)
    dir_factor = jnp.where(approaching, 2.5, jnp.where(receding, 0.4, 1.0))  # (8,)

    radius = params["danger_radius"]
    base = jnp.exp(-gap / radius)  # (8, H)
    # hard collision buffer: any horizon with gap < buffer is danger=1
    hard = (gap < COLLISION_BUFFER).astype(jnp.float32)
    per_h = jnp.maximum(base, hard)
    danger_lane = jnp.max(per_h, axis=1) * dir_factor * e_active
    return jnp.clip(danger_lane, 0.0, 1.5)


def policy(obs_flat, params):
    prev, curr = _split(obs_flat)

    px = curr[0]
    py = curr[1]

    p_ex = prev[8:16]
    c_ex = curr[8:16]
    e_active = curr[40:48]
    e_orient = curr[64:72]

    p_cx = prev[72:80]
    c_cx = curr[72:80]
    c_active = curr[104:112]
    c_vid = curr[112:120]

    cur_lane = _player_lane(py)
    lane_idx = jnp.arange(8)

    # collectible prediction
    cdx = c_cx - p_cx
    pred_cx = c_cx + 4.0 * cdx

    # value
    vid_int = jnp.clip(c_vid.astype(jnp.int32), 0, 7)
    value = VALUE_TABLE[vid_int] * c_active

    # lane distance / reachability
    lane_dist = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)

    # estimate the player's x at the time of arrival in the candidate lane.
    # If targeting the collectible's predicted x, the agent will move toward it.
    # Use a midpoint between current px and predicted target x as the "transit x".
    transit_x = 0.5 * (px + pred_cx)  # (8,)

    # danger at each lane evaluated at *that lane's* transit x
    # Vectorize: compute per-lane danger separately because ref_x varies per lane.
    def danger_for_lane(ref_x):
        return _danger_at(p_ex, c_ex, e_active, e_orient, ref_x, params)

    # full (8 ref_x, 8 lanes) matrix; we take diagonal.
    danger_matrix = jax.vmap(danger_for_lane)(transit_x)  # (8 candidates, 8 lanes)
    danger_at_target = jnp.diagonal(danger_matrix)  # (8,)

    # danger at current player x (used for current-lane and escape decisions)
    danger_at_px = _danger_at(p_ex, c_ex, e_active, e_orient, px, params)

    # transition danger: max along the path lanes between cur_lane and target lane
    # approximate by averaging endpoints — keep simple
    transition_danger = 0.5 * (danger_at_px + danger_at_target)

    # interception cost
    intercept_cost = jnp.abs(pred_cx - px)

    # combined safety check
    danger = jnp.maximum(danger_at_target, danger_at_px[cur_lane] * 0.0 + danger_at_target)

    # active collectible mask
    has_value = (value > 0.0) & (c_active > 0.5)
    safe_lane = transition_danger < params["danger_thresh"]

    # lane score for active safe collectibles
    score = (
        params["value_w"] * value
        - params["danger_w"] * transition_danger * 300.0
        - params["intercept_w"] * intercept_cost
        - params["lane_dist_w"] * lane_dist
    )

    # stay bonus when current lane is safe and has value
    cur_safe = danger_at_px[cur_lane] < params["danger_thresh"]
    stay_ok = cur_safe & has_value[cur_lane]
    stay_bonus = jnp.where(
        (lane_idx == cur_lane) & stay_ok, params["switch_bias"], 0.0
    )
    score = score + stay_bonus

    eligible = has_value & safe_lane
    score_masked = jnp.where(eligible, score, -1e9)
    best_lane = jnp.argmax(score_masked)
    any_eligible = jnp.any(eligible)

    # fallback: safest reachable lane
    safe_score = -params["danger_w"] * danger_at_px * 300.0 - params["lane_dist_w"] * lane_dist
    best_safe_lane = jnp.argmax(safe_score)

    target_lane = jnp.where(any_eligible, best_lane, best_safe_lane)

    tgt_active = c_active[target_lane] > 0.5
    tgt_cx = pred_cx[target_lane]
    hold_x = jnp.array(80.0)
    target_x = jnp.where(tgt_active, tgt_cx, hold_x)

    # ===== action selection =====
    cur_danger = danger_at_px[cur_lane]
    cur_dangerous = cur_danger > params["danger_thresh"]

    cur_enemy_x = c_ex[cur_lane]
    enemy_on_right = cur_enemy_x > px
    escape_right = ~enemy_on_right

    up_lane = jnp.maximum(cur_lane - 1, 0)
    down_lane = jnp.minimum(cur_lane + 1, 7)
    # evaluate adjacent lane danger at the x we'd arrive at (px ± step)
    up_arrive_x = jnp.where(escape_right, px + PLAYER_STEP, px - PLAYER_STEP)
    down_arrive_x = up_arrive_x
    up_danger_arr = _danger_at(p_ex, c_ex, e_active, e_orient, up_arrive_x, params)[up_lane]
    down_danger_arr = _danger_at(p_ex, c_ex, e_active, e_orient, down_arrive_x, params)[down_lane]

    can_up = cur_lane > 0
    can_down = cur_lane < 7
    prefer_up = (up_danger_arr <= down_danger_arr) & can_up
    prefer_up = prefer_up | (~can_down)
    prefer_up = prefer_up & can_up

    danger_action = jnp.where(
        prefer_up,
        jnp.where(escape_right, UPRIGHT, UPLEFT),
        jnp.where(escape_right, DOWNRIGHT, DOWNLEFT),
    )

    both_adj_bad = (up_danger_arr > params["danger_thresh"]) & (down_danger_arr > params["danger_thresh"])
    horiz_dodge = jnp.where(escape_right, RIGHT, LEFT)
    danger_action = jnp.where(both_adj_bad, horiz_dodge, danger_action)

    # normal navigation
    need_up = target_lane < cur_lane
    need_down = target_lane > cur_lane
    go_right = target_x > px + 2.0
    go_left = target_x < px - 2.0

    # only commit a diagonal if destination lane is safe at arrival x
    arrive_x_diag = jnp.where(go_left, px - PLAYER_STEP, px + PLAYER_STEP)
    dest_danger_arrive = _danger_at(p_ex, c_ex, e_active, e_orient, arrive_x_diag, params)[target_lane]
    safe_to_transition = dest_danger_arrive < params["danger_thresh"]

    up_action = jnp.where(go_left, UPLEFT, UPRIGHT)
    down_action = jnp.where(go_left, DOWNLEFT, DOWNRIGHT)
    horiz_action = jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))

    vert_action = jnp.where(need_up, up_action, down_action)
    need_vert = need_up | need_down
    normal_action = jnp.where(
        need_vert & safe_to_transition, vert_action, horiz_action
    )

    # if aligned and idle, patrol slightly to harvest
    aligned = (~need_vert) & (~go_right) & (~go_left)
    patrol_action = jnp.where(px < 80.0, RIGHT, LEFT)
    normal_action = jnp.where(aligned & ~tgt_active, patrol_action, normal_action)

    action = jnp.where(cur_dangerous, danger_action, normal_action)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
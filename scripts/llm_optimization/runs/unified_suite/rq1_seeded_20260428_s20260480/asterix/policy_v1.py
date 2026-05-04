"""
Auto-generated policy v1
Generated at: 2026-04-28 14:28:13
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
# visual_id -> point value
VALUE_TABLE = jnp.array([50.0, 100.0, 200.0, 300.0, 0.0, 0.0, 0.0, 0.0])


def init_params():
    return {
        "danger_w": jnp.array(1.5),
        "danger_radius": jnp.array(25.0),
        "lane_dist_w": jnp.array(40.0),
        "intercept_w": jnp.array(0.6),
        "value_w": jnp.array(1.0),
        "switch_bias": jnp.array(15.0),
        "danger_thresh": jnp.array(0.55),
    }


def _split_frames(obs_flat):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]
    return prev, curr


def _player_lane(py):
    # nearest lane index based on y
    diffs = jnp.abs(LANE_CENTERS - py)
    return jnp.argmin(diffs)


def _lane_danger(curr_ex, prev_ex, ey, e_active, e_orient, px, params):
    # predict next-frame enemy x
    dx = curr_ex - prev_ex
    pred_x = curr_ex + 2.0 * dx
    # signed gap: positive if enemy approaching player from one side
    dist_now = jnp.abs(curr_ex - px)
    dist_pred = jnp.abs(pred_x - px)
    closeness = jnp.minimum(dist_now, dist_pred)
    # danger high when close and active
    radius = params["danger_radius"]
    danger = jnp.exp(-closeness / radius) * e_active
    return danger


def policy(obs_flat, params):
    prev, curr = _split_frames(obs_flat)

    px = curr[0]
    py = curr[1]

    # enemies
    p_ex = prev[8:16]
    c_ex = curr[8:16]
    ey = curr[16:24]
    e_active = curr[40:48]
    e_orient = curr[64:72]

    # collectibles
    p_cx = prev[72:80]
    c_cx = curr[72:80]
    c_vid = curr[112:120]
    c_active = curr[104:112]

    cur_lane = _player_lane(py)
    lane_idx = jnp.arange(8)

    # per-lane danger
    danger = _lane_danger(c_ex, p_ex, ey, e_active, e_orient, px, params)

    # collectible value per lane (vid -> points)
    vid_int = jnp.clip(c_vid.astype(jnp.int32), 0, 7)
    value = VALUE_TABLE[vid_int] * c_active

    # predicted collectible x for interception
    cdx = c_cx - p_cx
    pred_cx = c_cx + 2.0 * cdx
    intercept_cost = jnp.abs(pred_cx - px)

    # lane distance cost
    lane_dist = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)

    # lane score
    score = (
        params["value_w"] * value
        - params["danger_w"] * danger * 300.0
        - params["intercept_w"] * intercept_cost
        - params["lane_dist_w"] * lane_dist
    )

    # bonus for staying in current lane to avoid thrashing
    stay_bonus = jnp.where(lane_idx == cur_lane, params["switch_bias"], 0.0)
    # only give stay bonus if current lane has positive value or is safe
    score = score + stay_bonus

    # mask out very dangerous lanes from being targets unless no other option
    safe_mask = danger < params["danger_thresh"]
    # prefer lanes with positive value and safe
    has_value = (value > 0.0) & safe_mask
    score_value = jnp.where(has_value, score, score - 1e4)

    best_value_lane = jnp.argmax(score_value)
    any_value = jnp.any(has_value)

    # fallback: safest lane near player
    safe_score = -params["danger_w"] * danger * 300.0 - params["lane_dist_w"] * lane_dist
    best_safe_lane = jnp.argmax(safe_score)

    target_lane = jnp.where(any_value, best_value_lane, best_safe_lane)

    # target x: predicted collectible x in target lane if active, else patrol
    tgt_active = c_active[target_lane] > 0.5
    tgt_cx = pred_cx[target_lane]
    # patrol target: opposite side of screen to encourage motion
    patrol_x = jnp.where(px < 80.0, 140.0, 20.0)
    target_x = jnp.where(tgt_active, tgt_cx, patrol_x)

    # current lane danger
    cur_danger = danger[cur_lane]
    cur_dangerous = cur_danger > params["danger_thresh"]

    # nearest enemy in current lane for dodge
    cur_enemy_x = c_ex[cur_lane]
    enemy_on_right = cur_enemy_x > px

    # vertical decision
    need_up = target_lane < cur_lane
    need_down = target_lane > cur_lane
    # horizontal decision toward target_x
    go_right = target_x > px + 2.0
    go_left = target_x < px - 2.0

    # If current lane dangerous, escape diagonally away from enemy and toward safer lane
    # Escape horizontally away from enemy
    escape_right = ~enemy_on_right  # enemy on left -> go right
    # Combine with vertical: prefer moving up if possible
    # Use up unless cur_lane is 0
    can_up = cur_lane > 0
    can_down = cur_lane < 7

    danger_action = jnp.where(
        can_up,
        jnp.where(escape_right, UPRIGHT, UPLEFT),
        jnp.where(escape_right, DOWNRIGHT, DOWNLEFT),
    )

    # Normal action: combine vertical and horizontal
    # up + right
    up_right = UPRIGHT
    up_left = UPLEFT
    down_right = DOWNRIGHT
    down_left = DOWNLEFT

    # pick horizontal preference toward target
    horiz = jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))

    normal_action = jnp.where(
        need_up,
        jnp.where(go_left, up_left, up_right),
        jnp.where(
            need_down,
            jnp.where(go_left, down_left, down_right),
            jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP)),
        ),
    )

    # If aligned and in safe target lane, keep patrolling laterally instead of NOOP
    aligned = (~need_up) & (~need_down) & (~go_right) & (~go_left)
    patrol_action = jnp.where(px < 80.0, RIGHT, LEFT)
    normal_action = jnp.where(aligned, patrol_action, normal_action)

    action = jnp.where(cur_dangerous, danger_action, normal_action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
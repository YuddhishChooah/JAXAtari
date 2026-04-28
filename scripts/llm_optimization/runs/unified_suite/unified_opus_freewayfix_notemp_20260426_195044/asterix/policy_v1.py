"""
Auto-generated policy v1
Generated at: 2026-04-26 20:09:30
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

# Visual_id -> point value for collectibles
VALUE_TABLE = jnp.array([50.0, 100.0, 200.0, 300.0, 0.0, 0.0, 0.0, 0.0])


def init_params():
    return {
        "danger_radius": jnp.array(25.0),
        "danger_weight": jnp.array(2.5),
        "value_weight": jnp.array(1.0),
        "lane_dist_penalty": jnp.array(15.0),
        "intercept_penalty": jnp.array(0.3),
        "switch_bonus": jnp.array(8.0),
        "safe_threshold": jnp.array(40.0),
    }


def _current_lane(py):
    # closest lane index
    diffs = jnp.abs(LANE_CENTERS - py)
    return jnp.argmin(diffs)


def _lane_danger(enemy_x, enemy_active, enemy_dx, player_x, radius):
    # predict enemy x next step
    pred_x = enemy_x + 2.0 * enemy_dx
    dist_now = jnp.abs(enemy_x - player_x)
    dist_pred = jnp.abs(pred_x - player_x)
    dist = jnp.minimum(dist_now, dist_pred)
    # closer = more danger; only if active
    danger = jnp.maximum(0.0, radius - dist) * enemy_active
    return danger


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    px = curr[0]
    py = curr[1]

    enemy_x = curr[8:16]
    enemy_active = curr[40:48]
    prev_enemy_x = prev[8:16]
    enemy_dx = enemy_x - prev_enemy_x

    collect_x = curr[72:80]
    collect_active = curr[104:112]
    collect_vid = curr[112:120].astype(jnp.int32)
    prev_collect_x = prev[72:80]
    collect_dx = collect_x - prev_collect_x

    # lookup value safely
    vid_clipped = jnp.clip(collect_vid, 0, 7)
    collect_value = VALUE_TABLE[vid_clipped] * collect_active

    # current lane
    cur_lane = _current_lane(py)
    lane_idx = jnp.arange(8)

    # per-lane danger
    danger = _lane_danger(enemy_x, enemy_active, enemy_dx,
                          px, params["danger_radius"])

    # intercept distance: where collectible will be
    pred_collect_x = collect_x + 2.0 * collect_dx
    intercept_dist = jnp.abs(pred_collect_x - px)

    # lane distance from current lane
    lane_dist = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)

    # lane score
    score = (params["value_weight"] * collect_value
             - params["danger_weight"] * danger
             - params["lane_dist_penalty"] * lane_dist
             - params["intercept_penalty"] * intercept_dist)

    # bonus for staying in current lane to reduce thrashing
    stay_bonus = jnp.where(lane_idx == cur_lane, params["switch_bonus"], 0.0)
    # only add stay bonus if current lane has an active collectible
    cur_has_item = collect_active[cur_lane] > 0.5
    score = score + jnp.where(cur_has_item, stay_bonus, 0.0)

    # mask: only consider lanes with active positive-value collectibles
    valid = (collect_active > 0.5) & (collect_value > 0.0)
    score = jnp.where(valid, score, -1e9)

    target_lane = jnp.argmax(score)
    any_target = jnp.any(valid)

    # if no target, pick safest nearby lane
    safety_score = -danger - 5.0 * lane_dist
    safe_lane = jnp.argmax(safety_score)
    target_lane = jnp.where(any_target, target_lane, safe_lane)

    # target x within target lane
    target_x = jnp.where(collect_active[target_lane] > 0.5,
                         pred_collect_x[target_lane],
                         px)

    # vertical direction
    lane_diff = target_lane - cur_lane  # >0 means need to go down
    # horizontal direction toward target_x
    x_diff = target_x - px

    # current lane danger check
    cur_danger = danger[cur_lane]
    is_dangerous = cur_danger > params["safe_threshold"]

    # if dangerous and no vertical move planned, escape vertically
    # pick escape direction based on safer adjacent lane
    up_lane = jnp.maximum(cur_lane - 1, 0)
    down_lane = jnp.minimum(cur_lane + 1, 7)
    up_safer = danger[up_lane] < danger[down_lane]

    # determine action
    go_up = lane_diff < 0
    go_down = lane_diff > 0
    go_right = x_diff > 1.0
    go_left = x_diff < -1.0

    # When dangerous with no lane plan, prefer escape
    escape_up = is_dangerous & (lane_diff == 0) & up_safer
    escape_down = is_dangerous & (lane_diff == 0) & (~up_safer)
    go_up = go_up | escape_up
    go_down = go_down | escape_down

    # default horizontal direction for diagonals if no horizontal needed
    # use direction of target item motion or toward x; if aligned, drift right
    horiz_right = go_right | ((~go_left) & (x_diff >= 0))

    # Build action
    action = jnp.where(
        go_up,
        jnp.where(horiz_right, UPRIGHT, UPLEFT),
        jnp.where(
            go_down,
            jnp.where(horiz_right, DOWNRIGHT, DOWNLEFT),
            jnp.where(
                go_right, RIGHT,
                jnp.where(go_left, LEFT,
                          # aligned in target lane: keep patrolling laterally
                          jnp.where(cur_has_item, RIGHT, RIGHT)),
            ),
        ),
    )

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
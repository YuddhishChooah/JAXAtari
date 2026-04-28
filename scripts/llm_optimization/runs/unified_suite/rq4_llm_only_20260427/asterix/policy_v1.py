"""
Auto-generated policy v1
Generated at: 2026-04-27 17:32:39
"""

import jax
import jax.numpy as jnp

FRAME_SIZE = 136
LANE_CENTERS = jnp.array([27.0, 43.0, 59.0, 75.0, 91.0, 107.0, 123.0, 139.0])

# Actions
NOOP = 0
RIGHT = 1
LEFT = 2
DOWN = 3
UPRIGHT = 4
UPLEFT = 5
DOWNRIGHT = 6
DOWNLEFT = 7

# Value mapping for collectibles by visual_id: 0->50,1->100,2->200,3->300,4->0
VALUE_TABLE = jnp.array([50.0, 100.0, 200.0, 300.0, 0.0, 50.0, 50.0, 50.0])


def init_params():
    return {
        "danger_radius": 28.0,    # x-distance under which an enemy threatens
        "danger_weight": 2.5,     # weight of danger in lane score
        "lane_dist_penalty": 35.0,  # cost per lane of distance
        "value_weight": 1.0,      # weight on collectible value
        "x_intercept_penalty": 0.4,  # penalty per pixel of x-distance to item
        "danger_threshold": 18.0,   # if enemy nearer than this in current lane => dodge
    }


def _player_lane(py):
    # Find nearest lane index
    diffs = jnp.abs(LANE_CENTERS - py)
    return jnp.argmin(diffs)


def _lane_value(coll_active, coll_vid):
    vid_idx = jnp.clip(coll_vid.astype(jnp.int32), 0, 7)
    vals = VALUE_TABLE[vid_idx]
    return vals * coll_active


def _lane_danger(player_x, enemy_x, enemy_dx, enemy_active, danger_radius):
    # Project enemy 3 frames ahead
    proj_x = enemy_x + 3.0 * enemy_dx
    dist_now = jnp.abs(enemy_x - player_x)
    dist_proj = jnp.abs(proj_x - player_x)
    dist = jnp.minimum(dist_now, dist_proj)
    # Closer => more danger; clip at danger_radius
    raw = jnp.maximum(0.0, danger_radius - dist) / danger_radius
    return raw * enemy_active


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    px = curr[0]
    py = curr[1]

    enemy_x_curr = curr[8:16]
    enemy_x_prev = prev[8:16]
    enemy_dx = enemy_x_curr - enemy_x_prev
    enemy_active = curr[40:48]

    coll_x_curr = curr[72:80]
    coll_x_prev = prev[72:80]
    coll_dx = coll_x_curr - coll_x_prev
    coll_active = curr[104:112]
    coll_vid = curr[112:120]

    cur_lane = _player_lane(py)

    # Per-lane danger (uses player x for x-intercept threat)
    danger = _lane_danger(px, enemy_x_curr, enemy_dx, enemy_active,
                          params["danger_radius"])

    # Per-lane value
    value = _lane_value(coll_active, coll_vid)

    # Predicted collectible x in 4 frames
    coll_pred_x = coll_x_curr + 4.0 * coll_dx
    x_dist = jnp.abs(coll_pred_x - px)

    lane_idx = jnp.arange(8)
    lane_dist = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)

    # Lane score: prefer active collectibles with high value, low danger,
    # close laterally, close in lanes.
    score = (params["value_weight"] * value
             - params["danger_weight"] * danger * 100.0
             - params["x_intercept_penalty"] * x_dist * coll_active
             - params["lane_dist_penalty"] * lane_dist)

    # Heavily penalize lanes with no active collectible so we chase items
    score = score - 500.0 * (1.0 - coll_active)

    target_lane = jnp.argmax(score)

    # Target x: if target lane has active collectible, use predicted x; else px
    target_has_item = coll_active[target_lane] > 0.5
    target_x = jnp.where(target_has_item, coll_pred_x[target_lane], px)

    # Current lane danger check (using player x)
    cur_danger_dist = jnp.min(
        jnp.where(enemy_active > 0.5,
                  jnp.abs(enemy_x_curr - px),
                  jnp.array(999.0))
    )
    # But only same-lane enemies matter for immediate hit - approximate by
    # checking the enemy in cur_lane
    cur_lane_enemy_active = enemy_active[cur_lane]
    cur_lane_enemy_x = enemy_x_curr[cur_lane]
    cur_lane_enemy_dx = enemy_dx[cur_lane]
    cur_lane_enemy_dist = jnp.abs(cur_lane_enemy_x - px)
    in_danger = (cur_lane_enemy_active > 0.5) & (
        cur_lane_enemy_dist < params["danger_threshold"]
    )

    # Decide vertical direction
    lane_diff = target_lane - cur_lane  # negative => need to go up
    need_up = lane_diff < 0
    need_down = lane_diff > 0

    # Decide horizontal direction
    x_diff = target_x - px
    go_right = x_diff > 2.0
    go_left = x_diff < -2.0

    # Danger override: pick escape direction
    # Escape vertically away from threatening enemy lane via diagonal
    enemy_coming_from_right = cur_lane_enemy_x > px
    # If enemy on right, prefer going left and up/down
    danger_horiz_left = enemy_coming_from_right
    # Up if possible (cur_lane > 0), else down
    can_up = cur_lane > 0
    # Compose action priority

    def normal_action():
        # Combine vertical + horizontal
        a_up_r = jnp.where(go_right, UPRIGHT, jnp.where(go_left, UPLEFT, UPLEFT))
        a_up = jnp.where(go_right, UPRIGHT, UPLEFT)
        a_down = jnp.where(go_right, DOWNRIGHT,
                           jnp.where(go_left, DOWNLEFT, DOWN))
        a_horiz = jnp.where(go_right, RIGHT,
                            jnp.where(go_left, LEFT, NOOP))
        a = jnp.where(need_up, a_up,
                      jnp.where(need_down, a_down, a_horiz))
        # If aligned and no need, do small patrol toward target_x sign
        a = jnp.where((~need_up) & (~need_down) & (~go_right) & (~go_left)
                      & target_has_item, RIGHT, a)
        return a

    def danger_action():
        # Move away horizontally + try to leave lane
        h_left = danger_horiz_left
        a_up = jnp.where(h_left, UPLEFT, UPRIGHT)
        a_dn = jnp.where(h_left, DOWNLEFT, DOWNRIGHT)
        return jnp.where(can_up, a_up, a_dn)

    action = jnp.where(in_danger, danger_action(), normal_action())
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
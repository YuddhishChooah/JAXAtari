"""
Auto-generated policy v2
Generated at: 2026-04-27 17:34:50
"""

import jax
import jax.numpy as jnp

FRAME_SIZE = 136
LANE_CENTERS = jnp.array([27.0, 43.0, 59.0, 75.0, 91.0, 107.0, 123.0, 139.0])

NOOP = 0
RIGHT = 1
LEFT = 2
DOWN = 3
UPRIGHT = 4
UPLEFT = 5
DOWNRIGHT = 6
DOWNLEFT = 7

# visual_id -> point value: 0->50, 1->100, 2->200, 3->300, 4->0
VALUE_TABLE = jnp.array([50.0, 100.0, 200.0, 300.0, 0.0, 50.0, 50.0, 50.0])

SCREEN_W = 160.0


def init_params():
    return {
        "danger_radius": 32.0,
        "danger_threshold": 22.0,
        "danger_weight": 3.0,
        "lane_dist_penalty": 12.0,
        "value_weight": 1.2,
        "x_intercept_penalty": 0.25,
    }


def _player_lane(py):
    diffs = jnp.abs(LANE_CENTERS - py)
    return jnp.argmin(diffs)


def _lane_value(coll_active, coll_vid):
    vid_idx = jnp.clip(coll_vid.astype(jnp.int32), 0, 7)
    return VALUE_TABLE[vid_idx] * coll_active


def _arrival_steps(lane_idx, cur_lane):
    # crude: 4 frames per lane traversed, min 2
    d = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)
    return 2.0 + 4.0 * d


def _lane_danger_at(player_x, enemy_x, enemy_dx, enemy_active, k_steps,
                    danger_radius):
    # Project enemy x at arrival time per-lane
    proj_x = enemy_x + k_steps * enemy_dx
    dist_proj = jnp.abs(proj_x - player_x)
    dist_now = jnp.abs(enemy_x - player_x)
    dist = jnp.minimum(dist_now, dist_proj)
    raw = jnp.maximum(0.0, danger_radius - dist) / danger_radius
    return raw * enemy_active


def _lane_min_enemy_gap(player_x, enemy_x, enemy_dx, enemy_active, k_steps):
    proj_x = enemy_x + k_steps * enemy_dx
    gap_proj = jnp.abs(proj_x - player_x)
    gap_now = jnp.abs(enemy_x - player_x)
    gap = jnp.minimum(gap_now, gap_proj)
    # inactive lanes -> huge gap
    return jnp.where(enemy_active > 0.5, gap, jnp.full_like(gap, 999.0))


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

    lane_idx = jnp.arange(8)
    k_steps = _arrival_steps(lane_idx, cur_lane)

    # Per-lane projected collectible x at arrival
    coll_pred_x = coll_x_curr + k_steps * coll_dx
    # Target x within lane: predicted collectible x if active else player x
    target_x_per_lane = jnp.where(coll_active > 0.5, coll_pred_x, px)

    # Per-lane danger evaluated at arrival, using target_x in that lane
    danger = _lane_danger_at(target_x_per_lane, enemy_x_curr, enemy_dx,
                             enemy_active, k_steps, params["danger_radius"])

    # Hard safety gate: minimum enemy gap at arrival
    enemy_gap = _lane_min_enemy_gap(target_x_per_lane, enemy_x_curr,
                                    enemy_dx, enemy_active, k_steps)
    safe_buffer = params["danger_threshold"]
    safe_lane = enemy_gap > safe_buffer  # bool per lane

    # Per-lane value
    value = _lane_value(coll_active, coll_vid)

    x_dist = jnp.abs(coll_pred_x - px)
    lane_dist = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)

    # Score active+safe collectible lanes
    base_score = (params["value_weight"] * value
                  - params["danger_weight"] * danger * 50.0
                  - params["x_intercept_penalty"] * x_dist
                  - params["lane_dist_penalty"] * lane_dist)

    # Mask: only lanes with active collectible AND safe at arrival
    eligible = (coll_active > 0.5) & safe_lane
    score = jnp.where(eligible, base_score, -1e6)

    has_eligible = jnp.any(eligible)
    target_lane_collect = jnp.argmax(score)

    # Fallback: safest reachable lane (minimize danger + small lane_dist cost)
    fallback_score = -danger * 100.0 - 5.0 * lane_dist
    # Strongly prefer safe lanes
    fallback_score = fallback_score + jnp.where(safe_lane, 50.0, -50.0)
    target_lane_safe = jnp.argmax(fallback_score)

    target_lane = jnp.where(has_eligible, target_lane_collect, target_lane_safe)

    target_has_item = coll_active[target_lane] > 0.5
    target_x = jnp.where(target_has_item, coll_pred_x[target_lane], px)
    # If no item and we're staying in lane, pick a patrol x toward screen center
    patrol_x = jnp.where(px < SCREEN_W * 0.5, px + 20.0, px - 20.0)
    target_x = jnp.where(target_has_item, target_x, patrol_x)

    # Current-lane immediate danger check
    cur_lane_e_active = enemy_active[cur_lane]
    cur_lane_e_x = enemy_x_curr[cur_lane]
    cur_lane_e_dx = enemy_dx[cur_lane]
    proj_e_x = cur_lane_e_x + 3.0 * cur_lane_e_dx
    cur_e_gap = jnp.minimum(jnp.abs(cur_lane_e_x - px),
                            jnp.abs(proj_e_x - px))
    in_danger = (cur_lane_e_active > 0.5) & (cur_e_gap < params["danger_threshold"])

    # Decide directions
    lane_diff = target_lane - cur_lane
    need_up = lane_diff < 0
    need_down = lane_diff > 0

    x_diff = target_x - px
    go_right = x_diff > 2.0
    go_left = x_diff < -2.0

    # Normal action: combine vertical + horizontal preference
    a_up = jnp.where(go_left, UPLEFT, UPRIGHT)  # default UPRIGHT if go_right or neutral
    a_down = jnp.where(go_left, DOWNLEFT,
                       jnp.where(go_right, DOWNRIGHT, DOWN))
    a_horiz = jnp.where(go_right, RIGHT,
                        jnp.where(go_left, LEFT,
                                  # aligned and no vertical need: keep moving laterally
                                  jnp.where(px < SCREEN_W * 0.5, RIGHT, LEFT)))
    normal_a = jnp.where(need_up, a_up,
                         jnp.where(need_down, a_down, a_horiz))

    # Danger escape: choose safer adjacent lane via diagonal away from enemy
    enemy_on_right = cur_lane_e_x > px
    horiz_escape_left = enemy_on_right  # move opposite to enemy

    up_lane = jnp.maximum(cur_lane - 1, 0)
    dn_lane = jnp.minimum(cur_lane + 1, 7)
    up_safe = safe_lane[up_lane] & (cur_lane > 0)
    dn_safe = safe_lane[dn_lane] & (cur_lane < 7)

    esc_up = jnp.where(horiz_escape_left, UPLEFT, UPRIGHT)
    esc_dn = jnp.where(horiz_escape_left, DOWNLEFT, DOWNRIGHT)
    esc_horiz = jnp.where(horiz_escape_left, LEFT, RIGHT)

    danger_a = jnp.where(up_safe, esc_up,
                         jnp.where(dn_safe, esc_dn, esc_horiz))

    action = jnp.where(in_danger, danger_a, normal_a)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
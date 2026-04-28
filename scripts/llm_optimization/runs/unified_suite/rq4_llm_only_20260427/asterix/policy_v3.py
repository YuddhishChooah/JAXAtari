"""
Auto-generated policy v3
Generated at: 2026-04-27 17:37:08
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
        "danger_radius": 26.0,
        "danger_threshold": 18.0,
        "danger_weight": 3.0,
        "lane_dist_penalty": 8.0,
        "value_weight": 1.0,
        "x_intercept_penalty": 0.15,
    }


def _player_lane(py):
    diffs = jnp.abs(LANE_CENTERS - py)
    return jnp.argmin(diffs)


def _lane_value(coll_active, coll_vid):
    vid_idx = jnp.clip(coll_vid.astype(jnp.int32), 0, 7)
    return VALUE_TABLE[vid_idx] * coll_active


def _arrival_steps(lane_idx, cur_lane):
    d = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)
    return 3.0 + 3.0 * d


def _transit_min_gap(player_x, target_x, enemy_x, enemy_dx, enemy_active,
                     k_arrival):
    # Sample at k = 1, k_arrival/2, k_arrival
    k1 = jnp.maximum(k_arrival * 0.33, 1.0)
    k2 = jnp.maximum(k_arrival * 0.66, 1.0)
    k3 = jnp.maximum(k_arrival, 1.0)
    # Player x interpolated linearly toward target_x across transit
    px1 = player_x + (target_x - player_x) * 0.33
    px2 = player_x + (target_x - player_x) * 0.66
    px3 = target_x
    ex1 = enemy_x + k1 * enemy_dx
    ex2 = enemy_x + k2 * enemy_dx
    ex3 = enemy_x + k3 * enemy_dx
    g1 = jnp.abs(ex1 - px1)
    g2 = jnp.abs(ex2 - px2)
    g3 = jnp.abs(ex3 - px3)
    g_now = jnp.abs(enemy_x - player_x)
    gap = jnp.minimum(jnp.minimum(g1, g2), jnp.minimum(g3, g_now))
    return jnp.where(enemy_active > 0.5, gap, jnp.full_like(gap, 999.0))


def _soft_danger(min_gap, danger_radius):
    raw = jnp.maximum(0.0, danger_radius - min_gap) / danger_radius
    return raw


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
    coll_pred_x = jnp.clip(coll_pred_x, 0.0, SCREEN_W)
    target_x_per_lane = jnp.where(coll_active > 0.5, coll_pred_x, px)

    # Transit-window minimum enemy gap per candidate lane
    transit_gap = _transit_min_gap(px, target_x_per_lane, enemy_x_curr,
                                   enemy_dx, enemy_active, k_steps)

    danger = _soft_danger(transit_gap, params["danger_radius"])

    # Hard safety gate coupled to danger_radius
    safe_buffer = params["danger_threshold"]
    # Destination lanes farther away need larger buffer
    extra_buf = jnp.abs(lane_idx - cur_lane).astype(jnp.float32) * 1.5
    safe_lane = transit_gap > (safe_buffer + extra_buf)

    value = _lane_value(coll_active, coll_vid)

    x_dist = jnp.abs(coll_pred_x - px)
    lane_dist = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)

    # Time-discounted value: reward harvesting items reachable soon
    arrival_t = k_steps + x_dist * 0.5
    time_discount = 1.0 / (1.0 + 0.05 * arrival_t)

    base_score = (params["value_weight"] * value * time_discount
                  - params["danger_weight"] * danger * 50.0
                  - params["x_intercept_penalty"] * x_dist
                  - params["lane_dist_penalty"] * lane_dist)

    # Hysteresis bonus for current lane
    stay_bonus = jnp.where(lane_idx == cur_lane, 6.0, 0.0)
    base_score = base_score + stay_bonus

    eligible = (coll_active > 0.5) & safe_lane
    score = jnp.where(eligible, base_score, -1e6)

    has_eligible = jnp.any(eligible)
    target_lane_collect = jnp.argmax(score)

    # Scout fallback: prefer safe lanes near current; bias slightly upward
    # because items often appear above; but only as a tiebreak.
    scout_score = (jnp.where(safe_lane, 30.0, -50.0)
                   - 4.0 * lane_dist
                   - danger * 40.0
                   + jnp.where(enemy_active > 0.5, -5.0, 2.0))
    target_lane_safe = jnp.argmax(scout_score)

    target_lane = jnp.where(has_eligible, target_lane_collect, target_lane_safe)

    target_has_item = coll_active[target_lane] > 0.5
    item_x = coll_pred_x[target_lane]
    # If scouting, drift toward the side with more collectible spawns historically
    # (just use opposite side from nearest enemy in current lane)
    cur_e_x = enemy_x_curr[cur_lane]
    cur_e_active = enemy_active[cur_lane]
    away_x = jnp.where(cur_e_x > px, px - 25.0, px + 25.0)
    away_x = jnp.where(cur_e_active > 0.5, away_x,
                       jnp.where(px < SCREEN_W * 0.5, px + 25.0, px - 25.0))
    target_x = jnp.where(target_has_item, item_x, away_x)
    target_x = jnp.clip(target_x, 5.0, SCREEN_W - 5.0)

    # Immediate danger on current lane
    cur_lane_e_dx = enemy_dx[cur_lane]
    proj_e_x_short = cur_e_x + 2.0 * cur_lane_e_dx
    cur_e_gap = jnp.minimum(jnp.abs(cur_e_x - px),
                            jnp.abs(proj_e_x_short - px))
    in_danger = (cur_e_active > 0.5) & (cur_e_gap < params["danger_threshold"])

    # Direction selection
    lane_diff = target_lane - cur_lane
    need_up = lane_diff < 0
    need_down = lane_diff > 0

    x_diff = target_x - px
    go_right = x_diff > 2.0
    go_left = x_diff < -2.0

    a_up = jnp.where(go_left, UPLEFT, UPRIGHT)
    a_down = jnp.where(go_left, DOWNLEFT,
                       jnp.where(go_right, DOWNRIGHT, DOWN))
    a_horiz = jnp.where(go_right, RIGHT,
                        jnp.where(go_left, LEFT,
                                  jnp.where(px < SCREEN_W * 0.5, RIGHT, LEFT)))
    normal_a = jnp.where(need_up, a_up,
                         jnp.where(need_down, a_down, a_horiz))

    # Danger escape: pick safer adjacent lane via transit_gap
    up_lane = jnp.maximum(cur_lane - 1, 0)
    dn_lane = jnp.minimum(cur_lane + 1, 7)
    up_gap = transit_gap[up_lane]
    dn_gap = transit_gap[dn_lane]
    up_safe = (up_gap > params["danger_threshold"]) & (cur_lane > 0)
    dn_safe = (dn_gap > params["danger_threshold"]) & (cur_lane < 7)

    enemy_on_right = cur_e_x > px
    horiz_escape_left = enemy_on_right

    esc_up = jnp.where(horiz_escape_left, UPLEFT, UPRIGHT)
    esc_dn = jnp.where(horiz_escape_left, DOWNLEFT, DOWNRIGHT)
    esc_horiz = jnp.where(horiz_escape_left, LEFT, RIGHT)

    # Prefer the safer of up/down based on transit_gap
    up_better = up_gap >= dn_gap
    primary_esc = jnp.where(up_better & up_safe, esc_up,
                            jnp.where(dn_safe, esc_dn,
                                      jnp.where(up_safe, esc_up, esc_horiz)))

    action = jnp.where(in_danger, primary_esc, normal_a)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
"""
Auto-generated policy v5
Generated at: 2026-04-27 17:41:52
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

VALUE_TABLE = jnp.array([50.0, 100.0, 200.0, 300.0, 0.0, 50.0, 50.0, 50.0])

SCREEN_W = 160.0


def init_params():
    return {
        "danger_radius": 34.0,
        "danger_threshold": 22.0,
        "danger_weight": 3.0,
        "lane_dist_penalty": 14.0,
        "value_weight": 1.0,
        "x_intercept_penalty": 0.20,
    }


def _player_lane(py):
    return jnp.argmin(jnp.abs(LANE_CENTERS - py))


def _lane_value(coll_active, coll_vid):
    vid_idx = jnp.clip(coll_vid.astype(jnp.int32), 0, 7)
    return VALUE_TABLE[vid_idx] * coll_active


def _arrival_steps(lane_idx, cur_lane):
    d = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)
    return 4.0 + 4.0 * d


def _transit_min_gap(player_x, target_x, enemy_x, enemy_dx, enemy_active,
                     k_arrival):
    # Sample 6 points across [0.2*k, 1.2*k]
    fracs = jnp.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    # Shapes: enemies [8] -> broadcast vs samples [6]
    # Compute per-lane per-sample gap, then min over samples.
    def gap_at(frac):
        k = jnp.maximum(k_arrival * frac, 1.0)
        # player x interpolated; clamp interp to [0,1] for post-arrival
        interp = jnp.minimum(frac, 1.0)
        px_t = player_x + (target_x - player_x) * interp
        ex_t = enemy_x + k * enemy_dx
        return jnp.abs(ex_t - px_t)

    g0 = gap_at(fracs[0])
    g1 = gap_at(fracs[1])
    g2 = gap_at(fracs[2])
    g3 = gap_at(fracs[3])
    g4 = gap_at(fracs[4])
    g5 = gap_at(fracs[5])
    gnow = jnp.abs(enemy_x - player_x)
    gap = jnp.minimum(jnp.minimum(jnp.minimum(g0, g1), jnp.minimum(g2, g3)),
                      jnp.minimum(jnp.minimum(g4, g5), gnow))
    return jnp.where(enemy_active > 0.5, gap, jnp.full_like(gap, 999.0))


def _danger_penalty(min_gap, danger_radius):
    # Smooth exponential-style penalty: large when gap small, ~0 when far.
    # Scaled to ~50 at gap=0, decays past danger_radius.
    raw = jnp.maximum(0.0, danger_radius - min_gap) / danger_radius
    return raw * raw * 80.0


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
    lane_dist = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)
    k_steps = _arrival_steps(lane_idx, cur_lane)

    # Predicted collectible x at arrival
    coll_pred_x = jnp.clip(coll_x_curr + k_steps * coll_dx, 0.0, SCREEN_W)

    # Anti-interception: bias target_x away from in-lane enemy by small offset
    enemy_at_arrival = enemy_x_curr + k_steps * enemy_dx
    away_sign = jnp.sign(coll_pred_x - enemy_at_arrival)
    away_sign = jnp.where(jnp.abs(away_sign) < 0.5, 1.0, away_sign)
    safe_offset = away_sign * 4.0 * enemy_active
    target_x_per_lane = jnp.where(coll_active > 0.5,
                                  jnp.clip(coll_pred_x + safe_offset,
                                           5.0, SCREEN_W - 5.0),
                                  px)

    transit_gap = _transit_min_gap(px, target_x_per_lane, enemy_x_curr,
                                   enemy_dx, enemy_active, k_steps)

    # Buffer scales with lane_dist and closing speed
    closing = jnp.maximum(jnp.abs(enemy_dx), 1.0)
    buf = params["danger_threshold"] + 3.0 * lane_dist + 0.8 * closing
    safe_lane = transit_gap > buf

    danger_pen = _danger_penalty(transit_gap, params["danger_radius"])

    value = _lane_value(coll_active, coll_vid)
    x_dist = jnp.abs(coll_pred_x - px)
    arrival_t = k_steps + x_dist * 0.5
    time_discount = 1.0 / (1.0 + 0.05 * arrival_t)

    base_score = (params["value_weight"] * value * time_discount
                  - params["danger_weight"] * danger_pen
                  - params["x_intercept_penalty"] * x_dist
                  - params["lane_dist_penalty"] * lane_dist)

    # Small stay bonus only if current lane has an active collectible
    cur_has_item = coll_active[cur_lane] > 0.5
    stay_bonus = jnp.where((lane_idx == cur_lane) & cur_has_item, 3.0, 0.0)
    base_score = base_score + stay_bonus

    eligible = (coll_active > 0.5) & safe_lane & (value > 0.5)
    score = jnp.where(eligible, base_score, -1e6)
    has_eligible = jnp.any(eligible)
    target_lane_collect = jnp.argmax(score)

    # Safe fallback: nearest safe lane with no active enemy preferred
    empty_bonus = jnp.where(enemy_active < 0.5, 20.0, 0.0)
    scout_score = (jnp.where(safe_lane, 30.0, -100.0)
                   + empty_bonus
                   - 5.0 * lane_dist
                   - 0.5 * danger_pen)
    target_lane_safe = jnp.argmax(scout_score)

    target_lane = jnp.where(has_eligible, target_lane_collect, target_lane_safe)

    target_has_item = coll_active[target_lane] > 0.5
    item_x = target_x_per_lane[target_lane]
    cur_e_x = enemy_x_curr[cur_lane]
    cur_e_active = enemy_active[cur_lane]
    cur_e_dx = enemy_dx[cur_lane]
    away_x = jnp.where(cur_e_x > px, px - 30.0, px + 30.0)
    away_x = jnp.where(cur_e_active > 0.5, away_x,
                       jnp.where(px < SCREEN_W * 0.5, px + 20.0, px - 20.0))
    target_x = jnp.where(target_has_item, item_x, away_x)
    target_x = jnp.clip(target_x, 5.0, SCREEN_W - 5.0)

    # Immediate danger on current lane: 4-step projection
    proj_e_x = cur_e_x + 4.0 * cur_e_dx
    cur_e_gap = jnp.minimum(jnp.abs(cur_e_x - px), jnp.abs(proj_e_x - px))
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

    # Escape: use post-diagonal x (px shifted away from enemy) for adj lane gaps
    enemy_on_right = cur_e_x > px
    horiz_escape_left = enemy_on_right
    px_esc = jnp.where(horiz_escape_left, px - 6.0, px + 6.0)

    up_lane = jnp.maximum(cur_lane - 1, 0)
    dn_lane = jnp.minimum(cur_lane + 1, 7)
    # Recompute simple gaps for adjacent lanes at escape x
    up_e_x_at = enemy_x_curr[up_lane] + 4.0 * enemy_dx[up_lane]
    dn_e_x_at = enemy_x_curr[dn_lane] + 4.0 * enemy_dx[dn_lane]
    up_gap = jnp.where(enemy_active[up_lane] > 0.5,
                       jnp.minimum(jnp.abs(enemy_x_curr[up_lane] - px_esc),
                                   jnp.abs(up_e_x_at - px_esc)),
                       999.0)
    dn_gap = jnp.where(enemy_active[dn_lane] > 0.5,
                       jnp.minimum(jnp.abs(enemy_x_curr[dn_lane] - px_esc),
                                   jnp.abs(dn_e_x_at - px_esc)),
                       999.0)
    esc_buf = params["danger_threshold"]
    up_safe = (up_gap > esc_buf) & (cur_lane > 0)
    dn_safe = (dn_gap > esc_buf) & (cur_lane < 7)
    up_empty = (enemy_active[up_lane] < 0.5) & (cur_lane > 0)
    dn_empty = (enemy_active[dn_lane] < 0.5) & (cur_lane < 7)

    esc_up = jnp.where(horiz_escape_left, UPLEFT, UPRIGHT)
    esc_dn = jnp.where(horiz_escape_left, DOWNLEFT, DOWNRIGHT)
    esc_horiz = jnp.where(horiz_escape_left, LEFT, RIGHT)

    # Prefer empty adjacent lane, then safer-by-gap, else horizontal dodge
    up_score = jnp.where(up_empty, 1000.0, jnp.where(up_safe, up_gap, -1.0))
    dn_score = jnp.where(dn_empty, 1000.0, jnp.where(dn_safe, dn_gap, -1.0))
    use_up = up_score >= dn_score
    primary_esc = jnp.where(use_up & (up_score > 0), esc_up,
                            jnp.where(dn_score > 0, esc_dn, esc_horiz))

    action = jnp.where(in_danger, primary_esc, normal_a)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
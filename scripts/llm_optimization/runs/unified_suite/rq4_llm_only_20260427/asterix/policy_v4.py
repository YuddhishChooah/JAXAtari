"""
Auto-generated policy v4
Generated at: 2026-04-27 17:39:22
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
        "danger_radius": 28.0,
        "danger_threshold": 22.0,
        "danger_weight": 3.0,
        "lane_dist_penalty": 4.0,
        "value_weight": 1.0,
        "x_intercept_penalty": 0.12,
    }


def _player_lane(py):
    return jnp.argmin(jnp.abs(LANE_CENTERS - py))


def _lane_value(coll_active, coll_vid):
    vid_idx = jnp.clip(coll_vid.astype(jnp.int32), 0, 7)
    return VALUE_TABLE[vid_idx] * coll_active


def _arrival_steps(lane_idx, cur_lane):
    d = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)
    return 3.0 + 3.0 * d


def _transit_min_gap(player_x, target_x, enemy_x, enemy_dx, enemy_active,
                     k_arrival):
    # 5 sample fractions across the transit window (excluding t=0)
    fracs = jnp.array([0.2, 0.4, 0.6, 0.8, 1.0])
    # k samples shape: (5, 8)
    k_samp = fracs[:, None] * jnp.maximum(k_arrival, 1.0)[None, :]
    px_samp = player_x + (target_x - player_x)[None, :] * fracs[:, None]
    ex_samp = enemy_x[None, :] + k_samp * enemy_dx[None, :]
    gaps = jnp.abs(ex_samp - px_samp)
    min_gap = jnp.min(gaps, axis=0)
    return jnp.where(enemy_active > 0.5, min_gap, jnp.full_like(min_gap, 999.0))


def _soft_danger(min_gap, danger_radius):
    return jnp.maximum(0.0, danger_radius - min_gap) / danger_radius


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

    # Predicted collectible x at arrival
    coll_pred_x = jnp.clip(coll_x_curr + k_steps * coll_dx, 0.0, SCREEN_W)
    target_x_per_lane = jnp.where(coll_active > 0.5, coll_pred_x, px)

    # Transit gap (mid- and end-transit only, no g_now)
    transit_gap = _transit_min_gap(px, target_x_per_lane, enemy_x_curr,
                                   enemy_dx, enemy_active, k_steps)
    danger = _soft_danger(transit_gap, params["danger_radius"])

    # Safety buffer scales with transit time, not lane count
    safe_buffer = params["danger_threshold"] + 0.6 * k_steps
    safe_lane = transit_gap > safe_buffer

    # Hard "do not enter" floor: mid-transit gap must clear collision buffer
    hard_clear = transit_gap > 10.0

    value = _lane_value(coll_active, coll_vid)
    x_dist = jnp.abs(coll_pred_x - px)
    lane_dist = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)

    arrival_t = k_steps + x_dist * 0.5
    time_discount = 1.0 / (1.0 + 0.04 * arrival_t)

    base_score = (params["value_weight"] * value * time_discount
                  - params["danger_weight"] * danger * 30.0
                  - params["x_intercept_penalty"] * x_dist
                  - params["lane_dist_penalty"] * lane_dist)

    # Tiny stay bonus only — don't trap the agent
    stay_bonus = jnp.where(lane_idx == cur_lane, 2.0, 0.0)
    base_score = base_score + stay_bonus

    eligible = (coll_active > 0.5) & safe_lane & hard_clear
    score = jnp.where(eligible, base_score, -1e6)

    has_eligible = jnp.any(eligible)
    target_lane_collect = jnp.argmax(score)

    # Safest-lane fallback: pick lane with maximum transit_gap, prefer central
    central_pull = -jnp.abs(lane_idx - 3.5)
    scout_score = (transit_gap
                   - 2.0 * lane_dist
                   + 0.5 * central_pull
                   + jnp.where(safe_lane, 10.0, -20.0))
    target_lane_safe = jnp.argmax(scout_score)

    target_lane = jnp.where(has_eligible, target_lane_collect, target_lane_safe)

    target_has_item = coll_active[target_lane] > 0.5
    item_x = coll_pred_x[target_lane]

    # If scouting, drift toward the center horizontally to maximize spawn coverage
    scout_x = jnp.where(px < SCREEN_W * 0.4, px + 20.0,
                        jnp.where(px > SCREEN_W * 0.6, px - 20.0, px))
    target_x = jnp.where(target_has_item, item_x, scout_x)
    target_x = jnp.clip(target_x, 5.0, SCREEN_W - 5.0)

    # Immediate danger: 5-frame lookahead on current lane
    cur_e_x = enemy_x_curr[cur_lane]
    cur_e_active = enemy_active[cur_lane]
    cur_lane_e_dx = enemy_dx[cur_lane]
    fracs5 = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    proj_gaps = jnp.abs((cur_e_x + fracs5 * cur_lane_e_dx) - px)
    min_proj_gap = jnp.min(proj_gaps)
    in_danger = (cur_e_active > 0.5) & (min_proj_gap < params["danger_threshold"])

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

    # Multi-lane escape: pick safest lane among all 8 by transit_gap, weighted by reachability
    escape_score = transit_gap - 3.0 * lane_dist
    # Don't escape into a clearly unsafe lane
    escape_score = jnp.where(transit_gap > 12.0, escape_score, -1e6)
    escape_lane = jnp.argmax(escape_score)
    esc_diff = escape_lane - cur_lane

    # Horizontal escape direction: away from threat
    enemy_on_right = cur_e_x > px
    esc_horiz = jnp.where(enemy_on_right, LEFT, RIGHT)

    esc_up_a = jnp.where(enemy_on_right, UPLEFT, UPRIGHT)
    esc_dn_a = jnp.where(enemy_on_right, DOWNLEFT, DOWNRIGHT)

    primary_esc = jnp.where(esc_diff < 0, esc_up_a,
                            jnp.where(esc_diff > 0, esc_dn_a, esc_horiz))

    action = jnp.where(in_danger, primary_esc, normal_a)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
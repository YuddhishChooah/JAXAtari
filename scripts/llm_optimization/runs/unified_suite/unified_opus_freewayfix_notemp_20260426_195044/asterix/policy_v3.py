"""
Auto-generated policy v3
Generated at: 2026-04-26 20:24:30
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
        "danger_radius": jnp.array(26.229570388793945),
        "danger_weight": jnp.array(2.3097736835479736),
        "intercept_penalty": jnp.array(0.2331085503101349),
        "lane_dist_penalty": jnp.array(14.997025489807129),
        "safe_threshold": jnp.array(0.45),
        "switch_bonus": jnp.array(8.597784042358398),
        "value_weight": jnp.array(1.8889856338500977),
    }


def _current_lane(py):
    diffs = jnp.abs(LANE_CENTERS - py)
    return jnp.argmin(diffs)


def _danger_at(enemy_x, enemy_active, enemy_dx, ref_x, horizon, radius):
    """Normalized danger in roughly [0, 1+] over a forward time window.
    Uses MAX (closest approach) over horizons rather than MIN."""
    pred_h = enemy_x + horizon * enemy_dx
    pred_half = enemy_x + 0.5 * horizon * enemy_dx
    d0 = jnp.abs(enemy_x - ref_x)
    d1 = jnp.abs(pred_half - ref_x)
    d2 = jnp.abs(pred_h - ref_x)
    # Worst (smallest) distance during transit window
    dist = jnp.minimum(jnp.minimum(d0, d1), d2)
    raw = jnp.exp(-dist / jnp.maximum(radius, 1.0))
    # Approach scaling: speed-weighted, fast closers count more
    speed = jnp.abs(enemy_dx)
    approaching = (jnp.sign(ref_x - enemy_x) * jnp.sign(enemy_dx)) > 0
    approach_mul = 1.0 + jnp.where(approaching, 0.5 + 0.15 * speed, 0.0)
    return raw * enemy_active * approach_mul


def _lane_danger_window(enemy_x, enemy_active, enemy_dx, ref_x, t_arrive,
                        radius):
    """Danger sampled at multiple horizons up to arrival time."""
    h1 = jnp.maximum(t_arrive * 0.4, 2.0)
    h2 = jnp.maximum(t_arrive * 0.8, 4.0)
    h3 = jnp.maximum(t_arrive * 1.2, 6.0)
    d1 = _danger_at(enemy_x, enemy_active, enemy_dx, ref_x, h1, radius)
    d2 = _danger_at(enemy_x, enemy_active, enemy_dx, ref_x, h2, radius)
    d3 = _danger_at(enemy_x, enemy_active, enemy_dx, ref_x, h3, radius)
    return jnp.maximum(jnp.maximum(d1, d2), d3)


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

    vid_clipped = jnp.clip(collect_vid, 0, 7)
    collect_value = VALUE_TABLE[vid_clipped] * collect_active

    cur_lane = _current_lane(py)
    lane_idx = jnp.arange(8)

    radius = params["danger_radius"]

    # Estimate arrival time per lane: lane change frames + horizontal slide
    pred_collect_x = collect_x + 3.0 * collect_dx
    ref_x = jnp.where(collect_active > 0.5, pred_collect_x, px)
    lane_dist = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)
    intercept_dist = jnp.abs(ref_x - px)
    t_arrive = 4.0 * lane_dist + 0.4 * intercept_dist + 2.0

    # Per-lane danger at the reference x over the arrival window
    danger = _lane_danger_window(enemy_x, enemy_active, enemy_dx,
                                 ref_x, t_arrive, radius)

    # Transit danger: for diagonal moves, the intermediate adjacent lane
    # must also be safe at half the arrival time. Use that lane's enemy
    # evaluated at the player's current x (where we'll cross through).
    # For lane i, the "transit lane" is sign-of-(i-cur_lane) step from cur.
    step = jnp.sign(lane_idx - cur_lane).astype(jnp.int32)
    transit_lane = jnp.clip(cur_lane + step, 0, 7)
    transit_danger_all = _lane_danger_window(
        enemy_x, enemy_active, enemy_dx, px, t_arrive * 0.5, radius)
    transit_danger = transit_danger_all[transit_lane]
    # No transit penalty when staying in current lane
    transit_danger = jnp.where(lane_idx == cur_lane, 0.0, transit_danger)

    effective_danger = jnp.maximum(danger, transit_danger)

    # Current lane danger evaluated at player's actual x, short horizon
    cur_danger = _danger_at(enemy_x, enemy_active, enemy_dx, px, 4.0, radius)
    cur_danger_max = jnp.max(cur_danger)

    # Lane scoring (only for active positive-value lanes)
    score = (params["value_weight"] * collect_value
             - params["danger_weight"] * effective_danger * 50.0
             - params["lane_dist_penalty"] * lane_dist
             - params["intercept_penalty"] * intercept_dist)

    # Stay bonus: continue harvesting in current lane if it has an item
    cur_has_item = (collect_active[cur_lane] > 0.5) & (
        collect_value[cur_lane] > 0.0)
    stay_bonus = jnp.where(lane_idx == cur_lane,
                           params["switch_bonus"], 0.0)
    score = score + jnp.where(cur_has_item, stay_bonus, 0.0)

    # Veto: reject lanes whose effective danger exceeds the threshold
    veto = effective_danger > params["safe_threshold"]
    valid = (collect_active > 0.5) & (collect_value > 0.0) & (~veto)
    masked_score = jnp.where(valid, score, -1e9)

    target_lane = jnp.argmax(masked_score)
    any_target = jnp.any(valid)

    # Fallback: among non-vetoed lanes (including empty ones), pick the
    # safest reachable lane. If everything is vetoed, prefer current lane
    # and dodge horizontally.
    safe_mask = ~veto
    fallback_score = (-3.0 * effective_danger - 1.5 * lane_dist
                      + 1.0 * collect_active)
    fallback_score = jnp.where(safe_mask, fallback_score, -1e9)
    safe_lane = jnp.argmax(fallback_score)
    any_safe = jnp.any(safe_mask)
    safe_lane = jnp.where(any_safe, safe_lane, cur_lane)

    target_lane = jnp.where(any_target, target_lane, safe_lane)

    # Target x within target lane
    t_active = collect_active[target_lane] > 0.5
    target_x = jnp.where(t_active, pred_collect_x[target_lane], px)

    lane_diff = target_lane - cur_lane
    x_diff = target_x - px

    go_up = lane_diff < 0
    go_down = lane_diff > 0
    go_right = x_diff > 1.0
    go_left = x_diff < -1.0

    # Emergency escape: if current lane is dangerous now, override toward
    # a safer adjacent lane regardless of prior plan.
    is_dangerous = cur_danger_max > params["safe_threshold"]
    up_lane = jnp.maximum(cur_lane - 1, 0)
    down_lane = jnp.minimum(cur_lane + 1, 7)
    up_safer = effective_danger[up_lane] < effective_danger[down_lane]
    escape_up = is_dangerous & up_safer
    escape_down = is_dangerous & (~up_safer)
    go_up = go_up | escape_up
    go_down = (go_down | escape_down) & (~go_up)

    # Horizontal: aim toward target x; if neutral, drift with collectible
    target_dx = collect_dx[target_lane]
    drift_right = jnp.where(jnp.abs(x_diff) > 1.0,
                            x_diff >= 0,
                            target_dx >= 0)
    horiz_right = drift_right

    # If escaping, bias horizontal away from nearest approaching enemy
    nearest_enemy_x = enemy_x[cur_lane]
    flee_right = nearest_enemy_x < px
    horiz_right = jnp.where(is_dangerous, flee_right, horiz_right)

    diag_action = jnp.where(
        go_up,
        jnp.where(horiz_right, UPRIGHT, UPLEFT),
        jnp.where(horiz_right, DOWNRIGHT, DOWNLEFT),
    )
    horiz_action = jnp.where(go_right, RIGHT,
                             jnp.where(go_left, LEFT,
                                       jnp.where(horiz_right, RIGHT, LEFT)))
    action = jnp.where(go_up | go_down, diag_action, horiz_action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
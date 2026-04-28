"""
Auto-generated policy v5
Generated at: 2026-04-26 20:39:47
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
        "value_weight": jnp.array(1.8889856338500977),
        "lane_dist_penalty": jnp.array(14.997025489807129),
        "intercept_penalty": jnp.array(0.2331085503101349),
        "switch_bonus": jnp.array(8.597784042358398),
        "safe_threshold": jnp.array(40.595794677734375),
    }


def _current_lane(py):
    diffs = jnp.abs(LANE_CENTERS - py)
    return jnp.argmin(diffs)


def _danger_at(enemy_x, enemy_active, enemy_dx, ref_x, horizon, radius):
    # Project enemy x at given horizon (frames ahead).
    pred = enemy_x + horizon * enemy_dx
    d_now = jnp.abs(enemy_x - ref_x)
    d_mid = jnp.abs(enemy_x + 0.5 * horizon * enemy_dx - ref_x)
    d_end = jnp.abs(pred - ref_x)
    dist = jnp.minimum(jnp.minimum(d_now, d_mid), d_end)
    raw = jnp.exp(-dist / jnp.maximum(radius, 1.0))
    # Approach multiplier: enemy moving toward ref_x is more dangerous.
    approach = jnp.sign(ref_x - enemy_x) * jnp.sign(enemy_dx)
    approach_mul = 1.0 + 1.5 * jnp.maximum(approach, 0.0)
    return raw * enemy_active * approach_mul * radius


def _path_danger(enemy_x, enemy_active, enemy_dx, lane_idx, cur_lane,
                 ref_x_per_lane, horizon, radius):
    # Danger at target lane (ref_x_per_lane[i]) at arrival horizon.
    target_d = _danger_at(enemy_x[lane_idx], enemy_active[lane_idx],
                          enemy_dx[lane_idx], ref_x_per_lane[lane_idx],
                          horizon, radius)
    return target_d


def policy(obs_flat, params):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]

    px = curr[0]
    py = curr[1]

    enemy_x = curr[8:16]
    enemy_active = curr[40:48]
    prev_enemy_x = prev[8:16]
    # Smoothed/clamped dx to avoid jitter.
    enemy_dx = jnp.clip(enemy_x - prev_enemy_x, -4.0, 4.0)

    collect_x = curr[72:80]
    collect_active = curr[104:112]
    collect_vid = curr[112:120].astype(jnp.int32)
    prev_collect_x = prev[72:80]
    collect_dx = jnp.clip(collect_x - prev_collect_x, -4.0, 4.0)

    vid_clipped = jnp.clip(collect_vid, 0, 7)
    collect_value = VALUE_TABLE[vid_clipped] * collect_active

    cur_lane = _current_lane(py)
    lane_idx = jnp.arange(8)

    radius = params["danger_radius"]

    # Predict collectible position a few frames ahead (intercept point).
    pred_collect_x = collect_x + 3.0 * collect_dx

    # Per-lane reference x for danger evaluation: collectible's predicted x
    # if the lane has an item; else the player's current x.
    ref_x = jnp.where(collect_active > 0.5, pred_collect_x, px)

    # Estimate arrival horizon per lane: vertical cost + horizontal cost.
    lane_dist = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)
    x_gap = jnp.abs(ref_x - px)
    horizon = 2.0 + 1.5 * lane_dist + 0.15 * x_gap

    # Danger at target lane evaluated at arrival.
    target_danger = _danger_at(enemy_x, enemy_active, enemy_dx,
                               ref_x, horizon, radius)

    # Danger at current lane (right now) -- player at current x.
    cur_danger_arr = _danger_at(enemy_x, enemy_active, enemy_dx,
                                px, 2.0, radius)
    cur_danger = cur_danger_arr[cur_lane]

    # Transit danger: for multi-lane jumps, also check intermediate lane
    # at the player's current x with a short horizon.
    mid_lane = jnp.clip((cur_lane + lane_idx) // 2, 0, 7)
    transit_danger = _danger_at(enemy_x[mid_lane], enemy_active[mid_lane],
                                enemy_dx[mid_lane], px, horizon * 0.5,
                                radius)
    # Path danger = max(target, transit). Same lane => no transit penalty.
    same_lane = lane_idx == cur_lane
    path_danger = jnp.where(same_lane, target_danger,
                            jnp.maximum(target_danger, transit_danger))

    # Intercept distance penalty.
    intercept_dist = jnp.abs(pred_collect_x - px)

    # Lane scoring (only for active positive-value lanes).
    score = (params["value_weight"] * collect_value
             - params["danger_weight"] * path_danger
             - params["lane_dist_penalty"] * lane_dist
             - params["intercept_penalty"] * intercept_dist)

    # Hysteresis: small bonus to current lane to prevent oscillation.
    stay_bonus = jnp.where(lane_idx == cur_lane,
                           params["switch_bonus"], 0.0)
    score = score + stay_bonus

    # Hard veto: dangerous lanes can't be targets. Stricter for diagonals.
    veto_target = path_danger > params["safe_threshold"]
    valid = (collect_active > 0.5) & (collect_value > 0.0) & (~veto_target)
    masked_score = jnp.where(valid, score, -1e9)

    target_lane = jnp.argmax(masked_score)
    any_target = jnp.any(valid)

    # Fallback when no safe value lane: pick lane minimizing danger,
    # close to current lane. Do NOT bias toward any active lane.
    fallback_score = -2.0 * path_danger - 3.0 * lane_dist
    safe_lane = jnp.argmax(fallback_score)
    target_lane = jnp.where(any_target, target_lane, safe_lane)

    # Target x: lead the collectible if the target lane has an item.
    t_active = collect_active[target_lane] > 0.5
    target_x = jnp.where(t_active, pred_collect_x[target_lane], px)

    lane_diff = target_lane - cur_lane
    x_diff = target_x - px

    go_up = lane_diff < 0
    go_down = lane_diff > 0
    go_right = x_diff > 1.0
    go_left = x_diff < -1.0

    # If we want to change lane but the path is unsafe, defer:
    # move horizontally to widen gap from nearest enemy in current lane.
    target_path_danger = path_danger[target_lane]
    unsafe_transit = (target_path_danger > params["safe_threshold"]) & (
        lane_diff != 0)

    # Escape if current lane is dangerous and we are not already moving.
    is_dangerous = cur_danger > params["safe_threshold"]
    up_lane = jnp.maximum(cur_lane - 1, 0)
    down_lane = jnp.minimum(cur_lane + 1, 7)
    up_safer = path_danger[up_lane] < path_danger[down_lane]
    no_vert = (lane_diff == 0)
    escape_up = is_dangerous & no_vert & up_safer
    escape_down = is_dangerous & no_vert & (~up_safer)
    go_up = go_up | escape_up
    go_down = go_down | escape_down

    # If transit unsafe, cancel the vertical commit and dodge horizontally.
    go_up = go_up & (~unsafe_transit)
    go_down = go_down & (~unsafe_transit)

    # Dodge direction in current lane: move away from nearest approaching enemy.
    cur_enemy_x = enemy_x[cur_lane]
    dodge_right = (cur_enemy_x < px)
    horiz_right_default = x_diff >= 0
    target_dx = collect_dx[target_lane]
    drift_right = jnp.where(jnp.abs(x_diff) > 1.0,
                            horiz_right_default,
                            target_dx >= 0)
    horiz_right = jnp.where(unsafe_transit, dodge_right, drift_right)

    # Build action.
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
"""
Auto-generated policy v2
Generated at: 2026-04-26 20:17:08
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
        "danger_radius": jnp.array(25.209314346313477),
        "danger_weight": jnp.array(2.278055429458618),
        "value_weight": jnp.array(1.167771816253662),
        "lane_dist_penalty": jnp.array(14.770901679992676),
        "intercept_penalty": jnp.array(0.502372682094574),
        "switch_bonus": jnp.array(8.284049987792969),
        "safe_threshold": jnp.array(40.5380744934082),
    }


def _current_lane(py):
    diffs = jnp.abs(LANE_CENTERS - py)
    return jnp.argmin(diffs)


def _lane_danger_at(enemy_x, enemy_active, enemy_dx, ref_x, horizon, radius):
    # Project enemies forward over multiple horizons; take worst (closest)
    # approach to ref_x. Approaching enemies count more than receding.
    pred1 = enemy_x + 1.0 * enemy_dx
    pred2 = enemy_x + 0.5 * horizon * enemy_dx
    pred3 = enemy_x + 1.0 * horizon * enemy_dx
    d0 = jnp.abs(enemy_x - ref_x)
    d1 = jnp.abs(pred1 - ref_x)
    d2 = jnp.abs(pred2 - ref_x)
    d3 = jnp.abs(pred3 - ref_x)
    dist = jnp.minimum(jnp.minimum(d0, d1), jnp.minimum(d2, d3))
    # Smooth exponential danger: value in [0, 1] scaled by radius.
    raw = jnp.exp(-dist / jnp.maximum(radius, 1.0))
    # Approach scaling: enemy moving toward ref_x is more dangerous.
    approaching = jnp.sign(ref_x - enemy_x) * jnp.sign(enemy_dx)
    approach_mul = 1.0 + 0.5 * jnp.maximum(approaching, 0.0)
    return raw * enemy_active * approach_mul * radius


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

    # Per-lane danger, evaluated at the player's predicted x when arriving.
    # For target lane, predict where player will be after a couple frames.
    # Use intercept x of the collectible as the reference if active.
    pred_collect_x = collect_x + 2.0 * collect_dx
    ref_x = jnp.where(collect_active > 0.5, pred_collect_x, px)

    danger = _lane_danger_at(enemy_x, enemy_active, enemy_dx,
                             ref_x, 4.0, radius)

    # Current lane danger uses player's actual x
    cur_danger = _lane_danger_at(enemy_x, enemy_active, enemy_dx,
                                 px, 4.0, radius)[cur_lane]

    # Intercept distance (how far we have to slide horizontally)
    intercept_dist = jnp.abs(pred_collect_x - px)

    # Lane distance from current lane
    lane_dist = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)

    # Lane scoring (only for active positive-value lanes)
    score = (params["value_weight"] * collect_value
             - params["danger_weight"] * danger
             - params["lane_dist_penalty"] * lane_dist
             - params["intercept_penalty"] * intercept_dist)

    # Stay bonus: reward continuing to harvest in current lane if it has item
    cur_has_item = (collect_active[cur_lane] > 0.5) & (
        collect_value[cur_lane] > 0.0)
    stay_bonus = jnp.where(lane_idx == cur_lane,
                           params["switch_bonus"], 0.0)
    score = score + jnp.where(cur_has_item, stay_bonus, 0.0)

    # Hard veto: very dangerous lanes are never targets
    veto = danger > params["safe_threshold"]
    valid = (collect_active > 0.5) & (collect_value > 0.0) & (~veto)
    masked_score = jnp.where(valid, score, -1e9)

    target_lane = jnp.argmax(masked_score)
    any_target = jnp.any(valid)

    # Fallback: pick lane with best (low-danger, near) profile, prefer
    # adjacent safe lanes, and bias toward an active (any) collectible lane
    # even if value lookup said 0.
    fallback_score = (-2.0 * danger - 3.0 * lane_dist
                      + 5.0 * collect_active)
    safe_lane = jnp.argmax(fallback_score)
    target_lane = jnp.where(any_target, target_lane, safe_lane)

    # Target x within target lane: lead the collectible
    t_active = collect_active[target_lane] > 0.5
    target_x = jnp.where(t_active, pred_collect_x[target_lane], px)

    # Vertical and horizontal directions
    lane_diff = target_lane - cur_lane
    x_diff = target_x - px

    go_up = lane_diff < 0
    go_down = lane_diff > 0
    go_right = x_diff > 1.0
    go_left = x_diff < -1.0

    # If current lane is dangerous and we are not already moving vertically,
    # escape to safer adjacent lane.
    is_dangerous = cur_danger > params["safe_threshold"]
    up_lane = jnp.maximum(cur_lane - 1, 0)
    down_lane = jnp.minimum(cur_lane + 1, 7)
    up_safer = danger[up_lane] < danger[down_lane]
    no_vert = (lane_diff == 0)
    escape_up = is_dangerous & no_vert & up_safer
    escape_down = is_dangerous & no_vert & (~up_safer)
    go_up = go_up | escape_up
    go_down = go_down | escape_down

    # Horizontal default for diagonals: use intercept direction; if neutral,
    # follow collectible's motion direction in the target lane.
    target_dx = collect_dx[target_lane]
    drift_right = jnp.where(jnp.abs(x_diff) > 1.0,
                            x_diff >= 0,
                            target_dx >= 0)
    horiz_right = drift_right

    # Build action
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
"""
Auto-generated policy v1
Generated at: 2026-04-26 18:27:32
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
# Visual id -> point value (id 4 is zero / penalty)
VALUE_TABLE = jnp.array([50.0, 100.0, 200.0, 300.0, 0.0, 0.0, 0.0, 0.0])


def init_params():
    return {
        "danger_w": jnp.float32(1.5),
        "danger_range": jnp.float32(40.0),
        "value_w": jnp.float32(1.0),
        "lane_dist_w": jnp.float32(20.0),
        "intercept_w": jnp.float32(0.3),
        "danger_thresh": jnp.float32(0.6),
        "switch_bias": jnp.float32(5.0),
    }


def _split_frames(obs_flat):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]
    return prev, curr


def _lane_value(visual_id, active):
    idx = jnp.clip(visual_id.astype(jnp.int32), 0, 7)
    v = VALUE_TABLE[idx]
    return v * active


def _lane_danger(player_x, enemy_x, enemy_dx, enemy_active, danger_range):
    # Project enemy a few frames ahead
    proj = enemy_x + 3.0 * enemy_dx
    dist = jnp.minimum(jnp.abs(enemy_x - player_x), jnp.abs(proj - player_x))
    closeness = jnp.clip(1.0 - dist / danger_range, 0.0, 1.0)
    return closeness * enemy_active


def policy(obs_flat, params):
    prev, curr = _split_frames(obs_flat)

    px = curr[0]
    py = curr[1]

    # Enemies
    e_x_prev = prev[8:16]
    e_x = curr[8:16]
    e_active = curr[40:48]
    e_dx = e_x - e_x_prev

    # Collectibles
    c_x_prev = prev[72:80]
    c_x = curr[72:80]
    c_active = curr[104:112]
    c_vid = curr[112:120]
    c_dx = c_x - c_x_prev

    # Identify current lane by closest lane center to player y
    lane_diffs = jnp.abs(LANE_CENTERS - py)
    cur_lane = jnp.argmin(lane_diffs)

    # Per-lane danger and value
    danger = _lane_danger(px, e_x, e_dx, e_active, params["danger_range"])
    values = _lane_value(c_vid, c_active)

    # Predicted collectible x (a few steps ahead) for interception
    c_proj = c_x + 2.0 * c_dx
    intercept_dist = jnp.abs(c_proj - px)

    lane_idx = jnp.arange(8)
    lane_dist = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)

    # Lane score: prefer high value, low danger, close lane, close intercept
    safe_mask = (danger < params["danger_thresh"]).astype(jnp.float32)
    has_item = (values > 0.0).astype(jnp.float32)

    score = (
        params["value_w"] * values * has_item
        - params["danger_w"] * danger * 100.0
        - params["lane_dist_w"] * lane_dist
        - params["intercept_w"] * intercept_dist
    )
    # Strong bonus for being already safe AND having an item
    score = score + 50.0 * safe_mask * has_item
    # Small bonus to prefer staying in current lane (hysteresis)
    stay_bonus = (lane_idx == cur_lane).astype(jnp.float32) * params["switch_bias"]
    score = score + stay_bonus * has_item

    # Pick best lane that has any item; if none, pick safest reachable lane
    any_item = jnp.any(has_item > 0.0)
    # Fallback score: just safety + closeness
    fallback_score = -params["danger_w"] * danger * 100.0 - params["lane_dist_w"] * lane_dist
    best_item_lane = jnp.argmax(score)
    best_safe_lane = jnp.argmax(fallback_score)
    target_lane = jnp.where(any_item, best_item_lane, best_safe_lane)

    # Target x: collectible projected position if item lane, else player x
    target_has_item = has_item[target_lane] > 0.0
    target_x = jnp.where(target_has_item, c_proj[target_lane], px)

    # Current lane danger -> emergency dodge
    cur_danger = danger[cur_lane]
    cur_dangerous = cur_danger > params["danger_thresh"]

    # Decide vertical direction
    lane_delta = target_lane - cur_lane  # negative -> go up, positive -> go down
    go_up = lane_delta < 0
    go_down = lane_delta > 0
    same_lane = lane_delta == 0

    # Decide horizontal direction toward target_x
    dx_to_target = target_x - px
    go_right = dx_to_target > 2.0
    go_left = dx_to_target < -2.0

    # Combine into action
    # Default: lateral toward target
    horiz_action = jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))

    # Up movement: must be diagonal. Use side that matches horizontal need or safer side.
    # If no strong horizontal preference, pick UPRIGHT by default; flip if left enemy nearer.
    nearest_enemy_side = jnp.sign(px - e_x[cur_lane])  # +1 enemy left, -1 enemy right
    up_action = jnp.where(
        go_right, UPRIGHT,
        jnp.where(go_left, UPLEFT,
                  jnp.where(nearest_enemy_side > 0, UPRIGHT, UPLEFT))
    )

    down_action = jnp.where(
        go_right, DOWNRIGHT,
        jnp.where(go_left, DOWNLEFT,
                  jnp.where(nearest_enemy_side > 0, DOWNRIGHT, DOWNLEFT))
    )

    action = jnp.where(go_up, up_action,
                       jnp.where(go_down, down_action, horiz_action))

    # Emergency: current lane dangerous and we're staying. Force diagonal escape.
    escape_up = jnp.where(nearest_enemy_side > 0, UPRIGHT, UPLEFT)
    escape_down = jnp.where(nearest_enemy_side > 0, DOWNRIGHT, DOWNLEFT)
    # Prefer up unless at top
    escape = jnp.where(cur_lane > 0, escape_up, escape_down)
    action = jnp.where(cur_dangerous & same_lane, escape, action)

    # Avoid pure NOOP when in safe lane with no item: patrol horizontally
    no_item_safe = same_lane & (~target_has_item) & (~cur_dangerous)
    patrol = jnp.where(px < 76.0, RIGHT, LEFT)
    action = jnp.where(no_item_safe & (action == NOOP), patrol, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
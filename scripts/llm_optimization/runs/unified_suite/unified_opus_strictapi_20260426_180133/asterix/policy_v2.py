"""
Auto-generated policy v2
Generated at: 2026-04-26 18:34:36
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
SCREEN_MID = 76.0


def init_params():
    return {
        "danger_range": jnp.float32(45.0),
        "danger_thresh": jnp.float32(0.35),
        "transit_margin": jnp.float32(8.0),
        "value_w": jnp.float32(1.0),
        "lane_dist_w": jnp.float32(0.6),
        "intercept_w": jnp.float32(0.05),
        "stay_bias": jnp.float32(0.4),
    }


def _split_frames(obs_flat):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]
    return prev, curr


def _value_of(visual_id, active):
    idx = jnp.clip(visual_id.astype(jnp.int32), 0, 7)
    return VALUE_TABLE[idx] * active


def _transit_danger(player_x, e_x, e_dx, e_active, transit_steps,
                    danger_range, transit_margin):
    # Evaluate enemy positions at multiple horizons across the transit window.
    # transit_steps: per-lane number of frames to reach that lane.
    # We sample horizons 0,1,2,...,K and take worst-case clearance.
    # Use fixed sample set with masks based on transit_steps.
    horizons = jnp.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])  # (H,)
    # Shape: (lanes, H)
    e_x_b = e_x[:, None]
    e_dx_b = jnp.clip(e_dx, -4.0, 4.0)[:, None]
    proj = e_x_b + horizons[None, :] * e_dx_b
    dist = jnp.abs(proj - player_x) - transit_margin
    closeness = jnp.clip(1.0 - dist / danger_range, 0.0, 1.0)
    # Mask horizons beyond per-lane transit time + small buffer
    max_h = transit_steps[:, None] + 2.0
    mask = (horizons[None, :] <= max_h).astype(jnp.float32)
    closeness = closeness * mask
    worst = jnp.max(closeness, axis=1)
    return worst * e_active


def policy(obs_flat, params):
    prev, curr = _split_frames(obs_flat)

    px = curr[0]
    py = curr[1]

    e_x_prev = prev[8:16]
    e_x = curr[8:16]
    e_active = curr[40:48]
    e_dx = e_x - e_x_prev

    c_x_prev = prev[72:80]
    c_x = curr[72:80]
    c_active = curr[104:112]
    c_vid = curr[112:120]
    c_dx = jnp.clip(c_x - c_x_prev, -4.0, 4.0)

    lane_idx = jnp.arange(8)
    cur_lane = jnp.argmin(jnp.abs(LANE_CENTERS - py))
    lane_dist = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)

    # Transit time ~ 4 frames per lane step (rough)
    transit_steps = lane_dist * 4.0

    danger = _transit_danger(
        px, e_x, e_dx, e_active, transit_steps,
        params["danger_range"], params["transit_margin"]
    )

    values = _value_of(c_vid, c_active)
    # Project collectible to arrival time
    c_proj = c_x + transit_steps * c_dx
    intercept_dist = jnp.abs(c_proj - px)

    has_item = (values > 0.0).astype(jnp.float32)
    safe_mask = (danger < params["danger_thresh"]).astype(jnp.float32)

    # Hard veto: only safe + active item lanes are item candidates
    item_candidate = has_item * safe_mask

    # Value normalized to ~[0,1]
    norm_value = values / 300.0
    stay = (lane_idx == cur_lane).astype(jnp.float32) * params["stay_bias"]

    item_score = (
        params["value_w"] * norm_value
        - params["lane_dist_w"] * lane_dist
        - params["intercept_w"] * intercept_dist
        + stay
    )
    # Mask non-candidates with very low score
    item_score = jnp.where(item_candidate > 0.0, item_score, -1e6)

    any_candidate = jnp.any(item_candidate > 0.0)

    # Fallback: safest reachable lane (low danger, close)
    fallback_score = -2.0 * danger - params["lane_dist_w"] * lane_dist
    best_item_lane = jnp.argmax(item_score)
    best_safe_lane = jnp.argmax(fallback_score)
    target_lane = jnp.where(any_candidate, best_item_lane, best_safe_lane)

    target_has_item = has_item[target_lane] > 0.0
    target_x = jnp.where(target_has_item, c_proj[target_lane], px)

    cur_danger = danger[cur_lane]
    cur_dangerous = cur_danger > params["danger_thresh"]

    # Vertical decision
    lane_delta = target_lane - cur_lane
    go_up = lane_delta < 0
    go_down = lane_delta > 0
    same_lane = lane_delta == 0

    # Horizontal need
    dx_to_target = target_x - px
    go_right = dx_to_target > 2.0
    go_left = dx_to_target < -2.0

    # Check destination-lane entry clearance: only proceed diagonally if dest safe
    dest_danger = danger[target_lane]
    dest_safe = dest_danger < params["danger_thresh"]

    horiz_action = jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))

    # Determine side preference: away from nearest enemy in cur lane
    cur_enemy_x = e_x[cur_lane]
    enemy_side = jnp.sign(px - cur_enemy_x)  # +1 enemy on left, -1 enemy on right
    safer_side_right = enemy_side >= 0  # bool: prefer moving right is safer

    up_action = jnp.where(
        go_right, UPRIGHT,
        jnp.where(go_left, UPLEFT,
                  jnp.where(safer_side_right, UPRIGHT, UPLEFT))
    )
    down_action = jnp.where(
        go_right, DOWNRIGHT,
        jnp.where(go_left, DOWNLEFT,
                  jnp.where(safer_side_right, DOWNRIGHT, DOWNLEFT))
    )

    # If destination unsafe, do horizontal repositioning instead of diagonal
    safe_to_transit = dest_safe
    vert_action = jnp.where(go_up, up_action, down_action)
    action = jnp.where(
        same_lane, horiz_action,
        jnp.where(safe_to_transit, vert_action, horiz_action)
    )

    # Emergency: current lane dangerous -> escape diagonally to safer adjacent lane
    up_lane_danger = jnp.where(cur_lane > 0, danger[jnp.maximum(cur_lane - 1, 0)], 1.0)
    down_lane_danger = jnp.where(cur_lane < 7, danger[jnp.minimum(cur_lane + 1, 7)], 1.0)
    escape_up_pref = up_lane_danger <= down_lane_danger
    escape_up = jnp.where(safer_side_right, UPRIGHT, UPLEFT)
    escape_down = jnp.where(safer_side_right, DOWNRIGHT, DOWNLEFT)
    escape = jnp.where(escape_up_pref, escape_up, escape_down)
    # Only escape diagonally if that escape lane is actually safe; otherwise dodge horizontally
    escape_lane_safe = jnp.where(escape_up_pref,
                                 up_lane_danger < params["danger_thresh"],
                                 down_lane_danger < params["danger_thresh"])
    horiz_dodge = jnp.where(safer_side_right, RIGHT, LEFT)
    emergency_action = jnp.where(escape_lane_safe, escape, horiz_dodge)
    action = jnp.where(cur_dangerous, emergency_action, action)

    # Anti-idle: if NOOP in safe lane, move toward nearest active item across any lane
    # Compute side of nearest active collectible
    any_item_anywhere = jnp.any(has_item > 0.0)
    # weight by value, prefer nearest in lane distance
    item_weight = values - 1e3 * lane_dist
    item_weight = jnp.where(has_item > 0.0, item_weight, -1e9)
    best_any = jnp.argmax(item_weight)
    item_side_x = c_proj[best_any]
    patrol_side = jnp.where(item_side_x > px, RIGHT, LEFT)
    fallback_patrol = jnp.where(any_item_anywhere, patrol_side,
                                jnp.where(px < SCREEN_MID, RIGHT, LEFT))
    action = jnp.where(action == NOOP, fallback_patrol, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
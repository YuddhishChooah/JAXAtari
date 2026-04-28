"""
Auto-generated policy v4
Generated at: 2026-04-26 18:49:17
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
VALUE_TABLE = jnp.array([50.0, 100.0, 200.0, 300.0, 0.0, 0.0, 0.0, 0.0])
SCREEN_MID = 76.0


def init_params():
    return {
        "danger_range": jnp.float32(45.5),
        "danger_thresh": jnp.float32(0.82),
        "transit_margin": jnp.float32(8.0),
        "value_w": jnp.float32(1.23),
        "lane_dist_w": jnp.float32(0.85),
        "intercept_w": jnp.float32(0.30),
        "stay_bias": jnp.float32(0.20),
    }


def _split_frames(obs_flat):
    prev = obs_flat[0:FRAME_SIZE]
    curr = obs_flat[FRAME_SIZE:2 * FRAME_SIZE]
    return prev, curr


def _value_of(visual_id, active):
    idx = jnp.clip(visual_id.astype(jnp.int32), 0, 7)
    return VALUE_TABLE[idx] * active


def _directional_danger(player_x, e_x, e_dx, e_active, transit_steps,
                        danger_range, transit_margin):
    """Per-lane danger weighted by approach direction."""
    horizons = jnp.array([0.0, 3.0, 6.0, 10.0, 14.0, 20.0])  # (H,)
    e_dx_c = jnp.clip(e_dx, -4.0, 4.0)
    proj = e_x[:, None] + horizons[None, :] * e_dx_c[:, None]  # (L,H)

    dist = jnp.abs(proj - player_x) - transit_margin
    closeness = jnp.clip(1.0 - dist / danger_range, 0.0, 1.0)

    # Approach factor: enemy moving toward player x
    to_player = jnp.sign(player_x - e_x)  # +1 if player to right of enemy
    approach = jnp.clip(to_player * e_dx_c, -1.0, 4.0)
    # 0.4 baseline so stationary enemies still register; full when approaching
    approach_w = 0.35 + 0.65 * jnp.clip(approach / 2.0, 0.0, 1.0)

    closeness = closeness * approach_w[:, None]

    # Mask horizons by transit time + dwell
    dwell = 8.0
    max_h = transit_steps[:, None] + dwell
    min_h = jnp.maximum(transit_steps[:, None] - 2.0, 0.0)
    mask = ((horizons[None, :] <= max_h) & (horizons[None, :] >= min_h)).astype(jnp.float32)
    closeness = closeness * mask
    worst = jnp.max(closeness, axis=1)
    return worst * e_active


def _path_danger(cur_lane, target_lane, lane_danger):
    lane_idx = jnp.arange(8)
    lo = jnp.minimum(cur_lane, target_lane)
    hi = jnp.maximum(cur_lane, target_lane)
    in_path = (lane_idx >= lo) & (lane_idx <= hi)
    masked = jnp.where(in_path, lane_danger, 0.0)
    return jnp.max(masked)


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
    transit_steps = lane_dist * 4.0

    danger = _directional_danger(
        px, e_x, e_dx, e_active, transit_steps,
        params["danger_range"], params["transit_margin"]
    )

    values = _value_of(c_vid, c_active)
    has_item = (values > 0.0).astype(jnp.float32)

    # Project collectible x to arrival time
    c_proj = c_x + transit_steps * c_dx
    intercept_dist = jnp.abs(c_proj - px)
    # Normalized intercept (screen ~160 px wide)
    intercept_norm = intercept_dist / 80.0

    norm_value = values / 300.0

    # Reachability-weighted value: divide by (1 + lane_dist) so far high-value
    # items still beat near low-value ones when safe.
    reach_value = norm_value / (1.0 + params["lane_dist_w"] * lane_dist)

    # Stay bonus only when current lane has an active item
    cur_lane_oh = (lane_idx == cur_lane).astype(jnp.float32)
    stay = cur_lane_oh * has_item * params["stay_bias"]

    # Alignment bonus: persistence — prefer lanes whose item is near px already
    aligned = (jnp.abs(c_x - px) < 12.0).astype(jnp.float32) * has_item * 0.15

    # Hard veto only at very high closeness
    hard_veto = (danger > 0.92).astype(jnp.float32)
    item_candidate = has_item * (1.0 - hard_veto)

    danger_w = 1.5 + 2.5 * params["danger_thresh"]

    item_score = (
        params["value_w"] * reach_value
        - params["intercept_w"] * intercept_norm
        - danger_w * danger
        + stay
        + aligned
    )
    item_score = jnp.where(item_candidate > 0.0, item_score, -1e6)

    any_candidate = jnp.any(item_candidate > 0.0)

    # Fallback: safest reachable lane
    fallback_score = -3.0 * danger - 0.3 * lane_dist
    best_item_lane = jnp.argmax(item_score)
    best_safe_lane = jnp.argmax(fallback_score)
    target_lane = jnp.where(any_candidate, best_item_lane, best_safe_lane)

    target_has_item = has_item[target_lane] > 0.0
    target_x = jnp.where(target_has_item, c_proj[target_lane], px)

    cur_danger = danger[cur_lane]
    cur_dangerous = cur_danger > params["danger_thresh"]

    # Path danger: stricter for destination, looser for transit
    path_d = _path_danger(cur_lane, target_lane, danger)
    dest_danger = danger[target_lane]
    transit_safe = (path_d < 0.85) & (dest_danger < 0.7)

    lane_delta = target_lane - cur_lane
    go_up = lane_delta < 0
    go_down = lane_delta > 0
    same_lane = lane_delta == 0

    dx_to_target = target_x - px
    go_right = dx_to_target > 1.5
    go_left = dx_to_target < -1.5

    horiz_action = jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))

    cur_enemy_x = e_x[cur_lane]
    enemy_side = jnp.sign(px - cur_enemy_x)
    safer_side_right = enemy_side >= 0

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

    vert_action = jnp.where(go_up, up_action, down_action)
    action = jnp.where(
        same_lane, horiz_action,
        jnp.where(transit_safe, vert_action, horiz_action)
    )

    # Emergency escape
    up_lane_danger = jnp.where(cur_lane > 0, danger[jnp.maximum(cur_lane - 1, 0)], 1.0)
    down_lane_danger = jnp.where(cur_lane < 7, danger[jnp.minimum(cur_lane + 1, 7)], 1.0)

    up_clearly_safe = (up_lane_danger < 0.5 * cur_danger) & (up_lane_danger < 0.6)
    down_clearly_safe = (down_lane_danger < 0.5 * cur_danger) & (down_lane_danger < 0.6)
    escape_up_pref = up_lane_danger <= down_lane_danger

    escape_up = jnp.where(safer_side_right, UPRIGHT, UPLEFT)
    escape_down = jnp.where(safer_side_right, DOWNRIGHT, DOWNLEFT)
    diag_escape = jnp.where(escape_up_pref, escape_up, escape_down)
    diag_safe = jnp.where(escape_up_pref, up_clearly_safe, down_clearly_safe)

    horiz_dodge = jnp.where(safer_side_right, RIGHT, LEFT)
    emergency_action = jnp.where(diag_safe, diag_escape, horiz_dodge)
    action = jnp.where(cur_dangerous, emergency_action, action)

    # Anti-idle: head toward best safe item across lanes
    item_weight = reach_value - 1.5 * danger
    item_weight = jnp.where(has_item > 0.0, item_weight, -1e9)
    best_any = jnp.argmax(item_weight)
    any_item_anywhere = jnp.any(has_item > 0.0)
    item_side_x = c_proj[best_any]
    patrol_side = jnp.where(item_side_x > px, RIGHT, LEFT)
    fallback_patrol = jnp.where(any_item_anywhere, patrol_side,
                                jnp.where(px < SCREEN_MID, RIGHT, LEFT))
    action = jnp.where(action == NOOP, fallback_patrol, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
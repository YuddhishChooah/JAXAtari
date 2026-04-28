"""
Auto-generated policy v4
Generated at: 2026-04-26 20:32:08
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
        "safe_threshold": jnp.array(0.55),
        "switch_bonus": jnp.array(8.597784042358398),
        "value_weight": jnp.array(1.8889856338500977),
    }


def _current_lane(py):
    diffs = jnp.abs(LANE_CENTERS - py)
    return jnp.argmin(diffs)


def _danger_at_ref(enemy_x, enemy_active, enemy_dx, ref_x, radius):
    """Normalized danger in [0,1] for a single lane, given a reference x.

    Samples enemy projected positions over horizons {0,1,2,3,4} frames and
    takes the closest approach to ref_x. Approach speed amplifies threat.
    """
    horizons = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # shape: (H, L)
    proj = enemy_x[None, :] + horizons[:, None] * enemy_dx[None, :]
    dist = jnp.min(jnp.abs(proj - ref_x[None, :]), axis=0)
    raw = jnp.exp(-dist / jnp.maximum(radius, 1.0))
    # Closing speed amplification (capped)
    closing = jnp.sign(ref_x - enemy_x) * enemy_dx
    closing_pos = jnp.maximum(closing, 0.0)
    approach_mul = 1.0 + jnp.minimum(closing_pos * 0.4, 1.5)
    return raw * enemy_active * approach_mul


def _transit_danger(enemy_x, enemy_active, enemy_dx, px, target_x, radius):
    """Worst-case normalized danger along the transit corridor to target_x."""
    # Sample 4 ref points along path: 25%, 50%, 75%, 100%
    refs = jnp.stack([
        px + 0.25 * (target_x - px),
        px + 0.50 * (target_x - px),
        px + 0.75 * (target_x - px),
        target_x,
    ])
    # For each ref compute per-lane danger; aggregate by max over refs.
    def per_ref(rx):
        return _danger_at_ref(enemy_x, enemy_active, enemy_dx,
                              jnp.full_like(enemy_x, rx), radius)
    d_all = jax.vmap(per_ref)(refs)  # (4, L)
    return jnp.max(d_all, axis=0)


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
    safe_thr = params["safe_threshold"]

    # Project collectible interception x (small lead)
    pred_collect_x = collect_x + 2.0 * collect_dx

    # Danger at the destination intercept point for each lane
    dest_danger = _danger_at_ref(enemy_x, enemy_active, enemy_dx,
                                 pred_collect_x, radius)

    # Transit danger: worst case along path from current px to target x
    transit = _transit_danger(enemy_x, enemy_active, enemy_dx,
                              px, pred_collect_x, radius)

    # Combined danger per lane (worst of the two)
    lane_danger = jnp.maximum(dest_danger, transit)

    # Current lane danger at player's current x
    cur_lane_danger_vec = _danger_at_ref(enemy_x, enemy_active, enemy_dx,
                                         jnp.full_like(enemy_x, px), radius)
    cur_danger = cur_lane_danger_vec[cur_lane]

    # Intercept distance and lane distance
    intercept_dist = jnp.abs(pred_collect_x - px)
    lane_dist = jnp.abs(lane_idx - cur_lane).astype(jnp.float32)

    # Lane scoring (only for active positive-value lanes)
    score = (params["value_weight"] * collect_value
             - params["danger_weight"] * lane_danger * 50.0
             - params["lane_dist_penalty"] * lane_dist
             - params["intercept_penalty"] * intercept_dist)

    # Stay bonus when current lane has a positive-value item
    cur_has_item = (collect_active[cur_lane] > 0.5) & (
        collect_value[cur_lane] > 0.0)
    stay_bonus = jnp.where(lane_idx == cur_lane,
                           params["switch_bonus"], 0.0)
    score = score + jnp.where(cur_has_item, stay_bonus, 0.0)

    # Veto highly dangerous lanes (normalized danger threshold)
    veto = lane_danger > safe_thr
    valid = (collect_active > 0.5) & (collect_value > 0.0) & (~veto)
    masked_score = jnp.where(valid, score, -1e9)

    target_lane = jnp.argmax(masked_score)
    any_target = jnp.any(valid)

    # Fallback: safest reachable lane (no collision-prone choices)
    # Prefer adjacent low-danger lanes; do not weight inactive lanes by value.
    fallback_score = -3.0 * lane_danger - 2.0 * lane_dist
    safe_lane = jnp.argmax(fallback_score)
    target_lane = jnp.where(any_target, target_lane, safe_lane)

    # Target x within target lane: lead the collectible if active
    t_active = collect_active[target_lane] > 0.5
    target_x = jnp.where(t_active, pred_collect_x[target_lane], px)

    # Vertical and horizontal directions
    lane_diff = target_lane - cur_lane
    x_diff = target_x - px

    go_up = lane_diff < 0
    go_down = lane_diff > 0
    go_right = x_diff > 1.0
    go_left = x_diff < -1.0

    # Re-evaluate destination safety every frame: if destination lane became
    # too dangerous, abort vertical move and stay horizontal in current lane.
    dest_unsafe = lane_danger[target_lane] > safe_thr
    abort_vert = dest_unsafe & ~any_target
    go_up = go_up & ~abort_vert
    go_down = go_down & ~abort_vert

    # Escape: if current lane dangerous, pick safer adjacent lane
    is_dangerous = cur_danger > safe_thr
    up_lane = jnp.maximum(cur_lane - 1, 0)
    down_lane = jnp.minimum(cur_lane + 1, 7)
    up_d = lane_danger[up_lane]
    dn_d = lane_danger[down_lane]
    up_safer = up_d < dn_d
    no_vert = (lane_diff == 0)
    escape_up = is_dangerous & no_vert & up_safer & (up_d < safe_thr)
    escape_down = is_dangerous & no_vert & (~up_safer) & (dn_d < safe_thr)
    go_up = go_up | escape_up
    go_down = go_down | escape_down

    # Horizontal default direction
    target_dx = collect_dx[target_lane]
    drift_right = jnp.where(jnp.abs(x_diff) > 1.0,
                            x_diff >= 0,
                            target_dx >= 0)
    horiz_right = drift_right

    # If both adjacent lanes are unsafe and we'd otherwise go vertical,
    # convert to a horizontal dodge instead of a diagonal into danger.
    both_adj_unsafe = (up_d > safe_thr) & (dn_d > safe_thr)
    suppress_vert = both_adj_unsafe & is_dangerous
    go_up = go_up & ~suppress_vert
    go_down = go_down & ~suppress_vert

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
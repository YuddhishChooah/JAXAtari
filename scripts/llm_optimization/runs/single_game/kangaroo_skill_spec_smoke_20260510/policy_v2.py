"""
Auto-generated policy v2
Generated at: 2026-05-10 02:08:48
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Observation index aliases
# ---------------------------------------------------------------------------
PX, PY, PW, PH, PA = 0, 1, 2, 3, 4
PORI = 7

LAD_X0, LAD_X1 = 168, 188
LAD_Y0, LAD_Y1 = 188, 208
LAD_W0, LAD_W1 = 208, 228
LAD_H0, LAD_H1 = 228, 248
LAD_A0, LAD_A1 = 248, 268

MON_X0, MON_X1 = 376, 380
MON_Y0, MON_Y1 = 380, 384
MON_A0, MON_A1 = 392, 396

COC_X0, COC_X1 = 408, 412
COC_Y0, COC_Y1 = 412, 416
COC_A0, COC_A1 = 424, 428

FCOC_X, FCOC_Y, FCOC_A = 368, 369, 372

CHILD_X, CHILD_Y = 360, 361

# Action constants
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT = 6, 7
DOWNRIGHT, DOWNLEFT = 8, 9
UPFIRE = 10
RIGHTFIRE, LEFTFIRE = 11, 12


# ---------------------------------------------------------------------------
# init_params
# ---------------------------------------------------------------------------
def init_params():
    return {
        "reach_y_tol": jnp.float32(20.0),
        "min_climb_gain": jnp.float32(8.0),
        "center_band_w": jnp.float32(10.0),
        "overlap_frac_min": jnp.float32(0.35),
        "punch_dx": jnp.float32(22.0),
        "row_y_tol": jnp.float32(10.0),
        "danger_r": jnp.float32(12.0),
        "dismount_margin": jnp.float32(6.0),
    }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _player_geom(obs):
    px = obs[PX]
    py = obs[PY]
    pw = obs[PW]
    ph = obs[PH]
    pcx = px + 0.5 * pw
    pby = py + ph
    return px, py, pw, ph, pcx, pby


def _ladders(obs):
    lx = obs[LAD_X0:LAD_X1]
    ly = obs[LAD_Y0:LAD_Y1]
    lw = obs[LAD_W0:LAD_W1]
    lh = obs[LAD_H0:LAD_H1]
    la = obs[LAD_A0:LAD_A1]
    lcx = lx + 0.5 * lw
    lby = ly + lh
    return lx, ly, lw, lh, la, lcx, lby


def _overlap_fraction(px, pw, lx, lw):
    left = jnp.maximum(px, lx)
    right = jnp.minimum(px + pw, lx + lw)
    overlap = jnp.maximum(0.0, right - left)
    return overlap / jnp.maximum(1.0, pw)


# ---------------------------------------------------------------------------
# Ladder selection: filter by reach + upward + active, then nearest in x
# Reference y for reach uses a virtual feet level so we can also call this
# from a "post-climb" virtual platform.
# ---------------------------------------------------------------------------
def _select_ladder_at(obs, params, ref_pby, ref_py, ref_pcx):
    lx, ly, lw, lh, la, lcx, lby = _ladders(obs)
    reach = jnp.abs(lby - ref_pby) < params["reach_y_tol"]
    upward = ly < (ref_py - params["min_climb_gain"])
    active = la > 0.5
    valid = reach & upward & active
    dx = jnp.abs(lcx - ref_pcx)
    score = jnp.where(valid, dx, jnp.float32(1e6))
    idx = jnp.argmin(score)
    any_valid = jnp.any(valid)
    return idx, any_valid


# ---------------------------------------------------------------------------
# Threats: nearest active same-row monkey -> signed dx (preserves stable-200)
# ---------------------------------------------------------------------------
def _threats(obs, params):
    _, py, _, ph, pcx, _ = _player_geom(obs)
    pcy = py + 0.5 * ph
    mx = obs[MON_X0:MON_X1]
    my = obs[MON_Y0:MON_Y1]
    ma = obs[MON_A0:MON_A1]
    mcy = my + 8.0
    same_row = jnp.abs(mcy - pcy) < params["row_y_tol"]
    valid = (ma > 0.5) & same_row
    dx = mx - pcx
    score = jnp.where(valid, jnp.abs(dx), jnp.float32(1e6))
    idx = jnp.argmin(score)
    any_monkey = jnp.any(valid)
    monkey_sign_dx = dx[idx]
    return any_monkey, monkey_sign_dx


# ---------------------------------------------------------------------------
# Falling coconut danger (overhead)
# ---------------------------------------------------------------------------
def _falling_overhead(obs, params):
    px, py, pw, ph, pcx, _ = _player_geom(obs)
    fa = obs[FCOC_A]
    fx = obs[FCOC_X]
    fy = obs[FCOC_Y]
    near_x = jnp.abs(fx - pcx) < params["danger_r"]
    above = (fy < py + ph) & (fy > py - 40.0)
    return (fa > 0.5) & near_x & above, fx - pcx


def _thrown_coconut_near(obs, params):
    _, py, _, ph, pcx, _ = _player_geom(obs)
    pcy = py + 0.5 * ph
    cx = obs[COC_X0:COC_X1]
    cy = obs[COC_Y0:COC_Y1]
    ca = obs[COC_A0:COC_A1]
    near_x = jnp.abs(cx - pcx) < params["danger_r"]
    near_y = jnp.abs(cy - pcy) < 10.0
    return jnp.any((ca > 0.5) & near_x & near_y)


# ---------------------------------------------------------------------------
# Move toward x
# ---------------------------------------------------------------------------
def _move_toward_x(target_dx):
    return jnp.where(target_dx >= 0.0, jnp.int32(RIGHT), jnp.int32(LEFT))


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
def policy(obs_flat, params):
    obs = obs_flat
    px, py, pw, ph, pcx, pby = _player_geom(obs)
    lx, ly, lw, lh, la, lcx, lby = _ladders(obs)

    # Clamp overlap_frac_min into safe (0,1] range so search cannot disable climb.
    ov_min = jnp.clip(params["overlap_frac_min"], 0.15, 0.95)
    band = jnp.maximum(params["center_band_w"], jnp.float32(6.0))

    # --- Primary ladder selection from current platform ---
    sel_idx, has_ladder = _select_ladder_at(obs, params, pby, py, pcx)
    safe_idx = jnp.maximum(sel_idx, 0)

    sel_lx = lx[safe_idx]
    sel_ly = ly[safe_idx]
    sel_lw = lw[safe_idx]
    sel_lh = lh[safe_idx]
    sel_lcx = lcx[safe_idx]
    sel_lby = sel_ly + sel_lh

    overlap = _overlap_fraction(px, pw, sel_lx, sel_lw)
    centered = jnp.abs(sel_lcx - pcx) < band
    # Soft column predicate: either real overlap OR center within band.
    on_column = has_ladder & ((overlap >= ov_min) | centered)
    can_climb_up = py > (sel_ly + params["dismount_margin"])
    near_top = py <= (sel_ly + params["dismount_margin"])

    target_dx = sel_lcx - pcx

    # --- Threats (preserved stable-200 contract) ---
    any_monkey, monkey_sign_dx = _threats(obs, params)
    punch_in_range = any_monkey & (jnp.abs(monkey_sign_dx) < params["punch_dx"])
    punch_action = jnp.where(monkey_sign_dx >= 0.0,
                             jnp.int32(RIGHTFIRE),
                             jnp.int32(LEFTFIRE))

    # --- Hazards ---
    fc_overhead, fc_dx = _falling_overhead(obs, params)
    thrown = _thrown_coconut_near(obs, params)

    # --- Default traverse action toward selected ladder ---
    traverse_action = _move_toward_x(target_dx)

    # --- Climb branch: on column with room above ---
    climb_action = jnp.int32(UP)

    # --- Post-climb dismount: at/near top of selected ladder ---
    # Reselect ladder from the upper platform (virtual feet at sel_ly + ph).
    virt_pby = sel_ly + ph
    virt_py = sel_ly
    next_idx, has_next = _select_ladder_at(obs, params, virt_pby, virt_py, pcx)
    next_safe = jnp.maximum(next_idx, 0)
    next_lcx = lcx[next_safe]
    # If we found a next ladder, dismount toward it; else toward child x.
    child_x_v = obs[CHILD_X]
    fallback_dx = child_x_v - pcx
    dismount_dx = jnp.where(has_next, next_lcx - pcx, fallback_dx)
    dismount_action = _move_toward_x(dismount_dx)

    # --- Hazard escape ---
    # If a falling coconut is overhead and we're on a ladder column, climbing up
    # is fine only if it actually clears the column; otherwise step sideways
    # away from the coconut x.
    dodge_dir = jnp.where(fc_dx >= 0.0, jnp.int32(LEFT), jnp.int32(RIGHT))
    # If at top with hazard overhead, step out laterally.
    hazard_action = jnp.where(near_top, dodge_dir, dodge_dir)
    hazard_action = jnp.where(thrown, dodge_dir, hazard_action)
    hazard_active = fc_overhead | thrown

    # --- Fallback: no ladder reachable, probe toward child ---
    fallback_action = _move_toward_x(child_x_v - pcx)

    # --- Compose decision (priority: hazard > near_top dismount > climb > traverse > fallback)
    action = jnp.where(has_ladder, traverse_action, fallback_action)
    action = jnp.where(on_column & can_climb_up, climb_action, action)
    action = jnp.where(on_column & near_top, dismount_action, action)
    action = jnp.where(hazard_active, hazard_action, action)

    # --- PRESERVATION CONTRACT: punch override (do NOT gate on on_column) ---
    action = jnp.where(punch_in_range, punch_action, action)

    return action.astype(jnp.int32)


# ---------------------------------------------------------------------------
# measure_main
# ---------------------------------------------------------------------------
def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)


# ---------------------------------------------------------------------------
# dense_reward
# ---------------------------------------------------------------------------
def dense_reward(obs_history, actions, rewards, total_reward, active_mask):
    mask = active_mask.astype(jnp.float32)
    msum = jnp.maximum(1.0, jnp.sum(mask))

    py = obs_history[:, PY]
    ph = obs_history[:, PH]
    px = obs_history[:, PX]
    pw = obs_history[:, PW]
    pcx = px + 0.5 * pw
    pby = py + ph

    lx = obs_history[:, LAD_X0:LAD_X1]
    ly = obs_history[:, LAD_Y0:LAD_Y1]
    lw = obs_history[:, LAD_W0:LAD_W1]
    lh = obs_history[:, LAD_H0:LAD_H1]
    la = obs_history[:, LAD_A0:LAD_A1]
    lcx = lx + 0.5 * lw
    lby = ly + lh

    # --- Reachable ladder per step ---
    reach = jnp.abs(lby - pby[:, None]) < 22.0
    upward = ly < (py[:, None] - 6.0)
    active_l = la > 0.5
    valid = reach & upward & active_l
    any_reach = jnp.any(valid, axis=1).astype(jnp.float32)
    reach_rate = jnp.sum(any_reach * mask) / msum

    # --- Soft alignment to nearest reachable ladder (center band) ---
    dx_to_lcx = jnp.abs(lcx - pcx[:, None])
    dx_to_lcx_v = jnp.where(valid, dx_to_lcx, jnp.float32(1e6))
    min_dx = jnp.min(dx_to_lcx_v, axis=1)
    aligned = (min_dx < 12.0).astype(jnp.float32) * any_reach
    align_rate = jnp.sum(aligned * mask) / msum

    # --- Column overlap (hard) per step ---
    left = jnp.maximum(px[:, None], lx)
    right = jnp.minimum(px[:, None] + pw[:, None], lx + lw)
    overlap = jnp.maximum(0.0, right - left) / jnp.maximum(1.0, pw[:, None])
    overlap = overlap * active_l.astype(jnp.float32)
    best_overlap = jnp.max(overlap, axis=1)
    soft_col = ((best_overlap >= 0.30) | (min_dx < 10.0)).astype(jnp.float32)

    # --- Upward progress: bounded height gain ---
    py_masked = jnp.where(mask > 0.5, py, jnp.float32(1e6))
    min_py = jnp.min(py_masked)
    start_py = py[0]
    height_gain = jnp.clip(start_py - min_py, 0.0, 80.0)

    # --- Climb dy: gated on soft column predicate ---
    dy = jnp.zeros_like(py)
    dy = dy.at[1:].set(py[:-1] - py[1:])
    valid_climb = (soft_col[1:] > 0.5) & (mask[1:] > 0.5)
    climb_dy = jnp.sum(jnp.where(valid_climb, jnp.clip(dy[1:], 0.0, 4.0), 0.0))
    climb_dy = jnp.clip(climb_dy, 0.0, 60.0)

    # --- UP-without-overlap penalty ---
    up_acts = ((actions == UP) | (actions == UPRIGHT) | (actions == UPLEFT)
               | (actions == UPFIRE)).astype(jnp.float32)
    bad_up = up_acts * (1.0 - soft_col) * mask
    bad_up_rate = jnp.clip(jnp.sum(bad_up) / msum, 0.0, 1.0)

    # --- FIRE usage ---
    fire_acts = ((actions == FIRE) | (actions == RIGHTFIRE)
                 | (actions == LEFTFIRE) | (actions == UPFIRE)).astype(jnp.float32)
    fire_rate = jnp.sum(fire_acts * mask) / msum

    # First-reward preservation pressure
    got_reward = (total_reward > 0.0).astype(jnp.float32)
    no_fire = (jnp.sum(fire_acts * mask) < 1.0).astype(jnp.float32)
    scoreless_no_fire = no_fire * (1.0 - got_reward)

    # Punch farming: lots of FIRE without any height gain
    no_height = (height_gain < 4.0).astype(jnp.float32)
    punch_farm_pen = jnp.clip(fire_rate * no_height, 0.0, 1.0)

    # Horizontal-only attractor: large x_range with no height gain
    px_masked = jnp.where(mask > 0.5, px, px[0])
    x_range = jnp.max(px_masked) - jnp.min(px_masked)
    horiz_only_pen = jnp.clip((x_range / 120.0) * no_height, 0.0, 1.0)

    # Upward milestones (post-200 transition pressure)
    milestone_1 = (height_gain >= 10.0).astype(jnp.float32)
    milestone_2 = (height_gain >= 24.0).astype(jnp.float32)
    milestone_3 = (height_gain >= 40.0).astype(jnp.float32)

    aux = (
        15.0 * reach_rate
        + 20.0 * align_rate
        + 3.0 * climb_dy            # up to 180
        + 2.0 * height_gain         # up to 160
        + 60.0 * milestone_1
        + 90.0 * milestone_2
        + 120.0 * milestone_3
        + 25.0 * got_reward         # preserve first-reward pressure (reduced)
        - 60.0 * bad_up_rate
        - 50.0 * punch_farm_pen
        - 40.0 * horiz_only_pen
        - 30.0 * scoreless_no_fire
    )
    aux = jnp.clip(aux, -200.0, 600.0)
    return aux
"""
Auto-generated policy v1
Generated at: 2026-05-10 02:05:56
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

# Action constants
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT = 6, 7
RIGHTFIRE, LEFTFIRE = 11, 12


# ---------------------------------------------------------------------------
# init_params
# ---------------------------------------------------------------------------
def init_params():
    return {
        "reach_y_tol": jnp.float32(20.0),
        "min_climb_gain": jnp.float32(8.0),
        "center_band_w": jnp.float32(5.0),
        "overlap_frac_min": jnp.float32(0.45),
        "punch_range": jnp.float32(22.0),
        "row_y_tol": jnp.float32(10.0),
        "hazard_x_tol": jnp.float32(10.0),
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
# Skills
# ---------------------------------------------------------------------------
def select_reachable_ladder(obs, params):
    """Return index of best reachable upward ladder, or -1."""
    _, py, _, _, pcx, pby = _player_geom(obs)
    lx, ly, lw, lh, la, lcx, lby = _ladders(obs)

    reach = jnp.abs(lby - pby) < params["reach_y_tol"]
    upward = ly < (py - params["min_climb_gain"])
    active = la > 0.5
    valid = reach & upward & active

    dx = jnp.abs(lcx - pcx)
    # large penalty for invalid ladders
    score = jnp.where(valid, dx, jnp.float32(1e6))
    idx = jnp.argmin(score)
    any_valid = jnp.any(valid)
    return jnp.where(any_valid, idx, -1), any_valid


def monkey_in_punch_range(obs, params):
    """Return (threat, dx_signed) for nearest same-row monkey."""
    _, py, _, ph, pcx, _ = _player_geom(obs)
    pcy = py + 0.5 * ph
    mx = obs[MON_X0:MON_X1]
    my = obs[MON_Y0:MON_Y1]
    ma = obs[MON_A0:MON_A1]
    mcy = my + 8.0  # approx center
    same_row = jnp.abs(mcy - pcy) < params["row_y_tol"]
    dx = mx - pcx
    in_range = jnp.abs(dx) < params["punch_range"]
    valid = (ma > 0.5) & same_row & in_range
    score = jnp.where(valid, jnp.abs(dx), jnp.float32(1e6))
    idx = jnp.argmin(score)
    threat = jnp.any(valid)
    sgn_dx = dx[idx]
    return threat, sgn_dx


def hazard_near_player(obs, params):
    """Falling coconut nearby above/at player."""
    px, py, pw, ph, pcx, _ = _player_geom(obs)
    fa = obs[FCOC_A]
    fx = obs[FCOC_X]
    fy = obs[FCOC_Y]
    near_x = jnp.abs(fx - pcx) < params["hazard_x_tol"]
    in_y = (fy > py - 30.0) & (fy < py + ph + 5.0)
    falling = (fa > 0.5) & near_x & in_y

    # thrown coconuts
    cx = obs[COC_X0:COC_X1]
    cy = obs[COC_Y0:COC_Y1]
    ca = obs[COC_A0:COC_A1]
    near_cx = jnp.abs(cx - pcx) < params["hazard_x_tol"]
    near_cy = jnp.abs(cy - (py + 0.5 * ph)) < 12.0
    thrown = jnp.any((ca > 0.5) & near_cx & near_cy)

    return falling | thrown, jnp.where(fx < pcx, -1.0, 1.0)


def horizontal_action_toward(target_dx_sign, with_fire):
    """sign>0 means target right of player."""
    go_right = target_dx_sign > 0
    base = jnp.where(go_right, RIGHT, LEFT)
    fire_act = jnp.where(go_right, RIGHTFIRE, LEFTFIRE)
    return jnp.where(with_fire, fire_act, base)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
def policy(obs_flat, params):
    obs = obs_flat
    px, py, pw, ph, pcx, pby = _player_geom(obs)
    lx, ly, lw, lh, la, lcx, lby = _ladders(obs)

    sel_idx, has_ladder = select_reachable_ladder(obs, params)
    safe_idx = jnp.maximum(sel_idx, 0)

    sel_lx = lx[safe_idx]
    sel_ly = ly[safe_idx]
    sel_lw = lw[safe_idx]
    sel_lh = lh[safe_idx]
    sel_lcx = lcx[safe_idx]
    sel_lby = sel_ly + sel_lh

    overlap = _overlap_fraction(px, pw, sel_lx, sel_lw)
    centered = jnp.abs(sel_lcx - pcx) < params["center_band_w"]
    on_column = (overlap >= params["overlap_frac_min"]) & centered
    can_climb_up = py > (sel_ly + params["dismount_margin"])
    same_platform = jnp.abs(sel_lby - pby) < params["reach_y_tol"]

    target_dx = sel_lcx - pcx
    target_sign = jnp.where(target_dx >= 0.0, 1.0, -1.0)

    monkey_threat, mdx = monkey_in_punch_range(obs, params)
    # punch only when along the route to the selected ladder (same direction)
    same_dir = (mdx * target_dx) >= 0.0
    punch_now = monkey_threat & same_dir & has_ladder & same_platform & (~on_column)

    hazard, haz_sign = hazard_near_player(obs, params)

    # --- Decision tree ---
    # 1) Hazard: dodge horizontally away (still toward ladder if possible)
    dodge_sign = -haz_sign  # move opposite hazard
    dodge_action = jnp.where(dodge_sign > 0, RIGHT, LEFT)

    # 2) On ladder column with room: climb
    climb_action = jnp.int32(UP)

    # 3) At top: dismount toward open side (use target sign if valid, else right)
    dismount_action = jnp.where(target_sign > 0, RIGHT, LEFT)
    on_top = on_column & (~can_climb_up)

    # 4) Traverse toward selected ladder
    traverse_action = horizontal_action_toward(target_sign, punch_now)

    # 5) Fallback: if no ladder reachable, do small probe right
    fallback_action = jnp.int32(NOOP)

    action = jnp.where(
        hazard,
        dodge_action,
        jnp.where(
            on_column & can_climb_up,
            climb_action,
            jnp.where(
                on_top,
                dismount_action,
                jnp.where(
                    has_ladder,
                    traverse_action,
                    fallback_action,
                ),
            ),
        ),
    )
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
    T = obs_history.shape[0]

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

    # --- Reachable ladder indicator per step ---
    pby_b = pby[:, None]
    py_b = py[:, None]
    reach = jnp.abs(lby - pby_b) < 20.0
    upward = ly < (py_b - 8.0)
    active_l = la > 0.5
    valid = reach & upward & active_l
    any_reach = jnp.any(valid, axis=1).astype(jnp.float32)
    reach_term = jnp.clip(jnp.sum(any_reach * mask) / jnp.maximum(1.0, jnp.sum(mask)), 0.0, 1.0)

    # --- Column overlap fraction per step (best ladder) ---
    left = jnp.maximum(px[:, None], lx)
    right = jnp.minimum(px[:, None] + pw[:, None], lx + lw)
    overlap = jnp.maximum(0.0, right - left) / jnp.maximum(1.0, pw[:, None])
    overlap = overlap * active_l.astype(jnp.float32)
    best_overlap = jnp.max(overlap, axis=1)
    on_col = (best_overlap >= 0.45).astype(jnp.float32)

    # --- Upward progress (height gain): drop in player_y ---
    py_masked = jnp.where(mask > 0.5, py, jnp.float32(1e6))
    min_py = jnp.min(py_masked)
    start_py = py[0]
    height_gain = jnp.clip(start_py - min_py, 0.0, 80.0)  # bounded

    # --- Climb dy: rewarded only when on column ---
    dy = jnp.zeros_like(py)
    dy = dy.at[1:].set(py[:-1] - py[1:])  # positive when moving up
    valid_climb = (on_col[1:] > 0.5) & (mask[1:] > 0.5)
    climb_dy = jnp.sum(jnp.where(valid_climb, jnp.clip(dy[1:], 0.0, 4.0), 0.0))
    climb_dy = jnp.clip(climb_dy, 0.0, 60.0)

    # --- UP-without-overlap penalty ---
    up_acts = (actions == UP) | (actions == UPRIGHT) | (actions == UPLEFT)
    up_acts = up_acts.astype(jnp.float32)
    bad_up = up_acts * (1.0 - on_col) * mask
    bad_up_rate = jnp.sum(bad_up) / jnp.maximum(1.0, jnp.sum(mask))
    bad_up_pen = jnp.clip(bad_up_rate, 0.0, 1.0)

    # --- FIRE / punch usage: reward presence of any FIRE that yielded reward
    fire_acts = (actions == FIRE) | (actions == RIGHTFIRE) | (actions == LEFTFIRE)
    fire_acts_f = fire_acts.astype(jnp.float32) * mask
    fire_rate = jnp.sum(fire_acts_f) / jnp.maximum(1.0, jnp.sum(mask))

    # Penalize punch-farming: many fires but no upward progress
    no_height = (height_gain < 4.0).astype(jnp.float32)
    punch_farm_pen = jnp.clip(fire_rate * no_height, 0.0, 1.0)

    # --- First reward preservation: reward earning any positive reward early ---
    got_reward = (total_reward > 0.0).astype(jnp.float32)

    # --- Scoreless + no-FIRE: must be punished
    no_fire = (jnp.sum(fire_acts_f) < 1.0).astype(jnp.float32)
    scoreless_no_fire = no_fire * (1.0 - got_reward)

    # --- Combine, scaled to tens of points but bounded ---
    aux = (
        30.0 * reach_term
        + 2.0 * climb_dy            # up to 120
        + 1.5 * height_gain          # up to 120
        + 40.0 * got_reward
        - 50.0 * bad_up_pen
        - 40.0 * punch_farm_pen
        - 30.0 * scoreless_no_fire
    )
    aux = jnp.clip(aux, -150.0, 300.0)
    return aux
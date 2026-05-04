"""
Auto-generated policy v2
Generated at: 2026-05-01 14:53:36
"""

"""
Kangaroo policy v7
Fixes post-200 stall by:
- center-band on-ladder test (no 1-pixel x-overlap latch),
- explicit dismount phase that steps clear of the column before re-selecting,
- next-ladder selector evaluated against the *upper* platform when near top,
- removing on-column NOOP-freeze in coconut hazards (commit to climb/exit),
- preserving the working RIGHTFIRE first-reward punch branch.
"""

import jax
import jax.numpy as jnp

NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = 6, 7, 8, 9
UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE = 10, 11, 12, 13


def init_params():
    return {
        "reach_tol": jnp.array(15.5),
        "reach_up": jnp.array(28.0),
        "align_tol": jnp.array(6.0),
        "center_band": jnp.array(4.0),    # tight on-column test
        "danger_r": jnp.array(18.2),
        "punch_dx": jnp.array(19.5),
        "dismount_clear": jnp.array(10.0),  # x-distance to be considered "off column"
        "height_weight": jnp.array(1.1),
    }


def _ladders(obs):
    lx = jax.lax.dynamic_slice(obs, (168,), (20,))
    ly = jax.lax.dynamic_slice(obs, (188,), (20,))
    lw = jax.lax.dynamic_slice(obs, (208,), (20,))
    lh = jax.lax.dynamic_slice(obs, (228,), (20,))
    la = jax.lax.dynamic_slice(obs, (248,), (20,))
    return lx, ly, lw, lh, la


def _move_toward_x(dx, tol):
    right = dx > tol
    left = dx < -tol
    return jnp.where(right, RIGHT, jnp.where(left, LEFT, NOOP))


def _is_on_ladder(obs, pcx, py, pby, center_band):
    """Center-band test: feet within ladder vertical span and pcx near ladder center."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh
    centered = jnp.abs(pcx - lcx) < center_band
    vertically_in = (pby >= lty - 2.0) & (py <= lby + 4.0)
    active = la > 0.5
    on = centered & vertically_in & active
    any_on = jnp.any(on)
    top_for = jnp.where(on, lty, jnp.array(1e6))
    nearest_top = jnp.min(top_for)
    return any_on, nearest_top


def _select_reachable(obs, pcx, pby_eval, py_eval, reach_tol, reach_up,
                      height_weight, exclude_cx, exclude_cw):
    """Best upward ladder reachable from a given platform reference (pby_eval, py_eval)."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh

    diff = lby - pby_eval
    reachable = (diff > -reach_up) & (diff < reach_tol)
    useful = lty < (py_eval - 4.0)
    active = la > 0.5

    # Exclude a specific column (the one we're climbing/just climbed)
    not_excluded = jnp.abs(lcx - exclude_cx) > exclude_cw

    valid = reachable & useful & active & not_excluded
    dx = jnp.abs(lcx - pcx)
    height_gap = jnp.maximum(py_eval - lty, 0.0)
    cost = jnp.where(valid, dx - height_weight * height_gap, jnp.array(1e6))
    idx = jnp.argmin(cost)
    has_valid = jnp.any(valid)

    fb_valid = useful & active & not_excluded
    fb_cost = jnp.where(fb_valid, dx, jnp.array(1e6))
    fb_idx = jnp.argmin(fb_cost)
    has_fb = jnp.any(fb_valid)

    sel_idx = jnp.where(has_valid, idx, fb_idx)
    return lcx[sel_idx], lty[sel_idx], has_valid, has_fb


def _threats(obs, pcx, py, danger_r):
    mx = jax.lax.dynamic_slice(obs, (376,), (4,))
    my = jax.lax.dynamic_slice(obs, (380,), (4,))
    ma = jax.lax.dynamic_slice(obs, (392,), (4,))
    cx = jax.lax.dynamic_slice(obs, (408,), (4,))
    cy = jax.lax.dynamic_slice(obs, (412,), (4,))
    ca = jax.lax.dynamic_slice(obs, (424,), (4,))
    fcx = obs[368]
    fcy = obs[369]
    fca = obs[372]

    mdx = mx - pcx
    mdy = my - py
    monkey_near = (jnp.abs(mdx) < danger_r) & (jnp.abs(mdy) < danger_r) & (ma > 0.5)
    any_monkey = jnp.any(monkey_near)
    mcost = jnp.where(monkey_near, jnp.abs(mdx), jnp.array(1e6))
    midx = jnp.argmin(mcost)
    monkey_sign_dx = mdx[midx]

    overhead_fc = (
        (jnp.abs(fcx - pcx) < danger_r)
        & (fcy < py)
        & (py - fcy < danger_r * 2.5)
        & (fca > 0.5)
    )
    fc_dx = fcx - pcx

    tdx = cx - pcx
    tdy = cy - py
    thrown_near = (jnp.abs(tdx) < danger_r) & (jnp.abs(tdy) < danger_r) & (ca > 0.5)
    any_thrown = jnp.any(thrown_near)

    return any_monkey, monkey_sign_dx, overhead_fc, fc_dx, any_thrown


def policy(obs_flat, params):
    px = obs_flat[0]
    py = obs_flat[1]
    pw = obs_flat[2]
    ph = obs_flat[3]
    pcx = px + pw * 0.5
    pby = py + ph
    child_x = obs_flat[360]

    reach_tol = params["reach_tol"]
    reach_up = params["reach_up"]
    align_tol = params["align_tol"]
    center_band = params["center_band"]
    danger_r = params["danger_r"]
    punch_dx = params["punch_dx"]
    dismount_clear = params["dismount_clear"]
    height_weight = params["height_weight"]

    # --- Ladder state via center-band test ---
    on_column, on_top_y = _is_on_ladder(obs_flat, pcx, py, pby, center_band)
    near_top = on_column & ((py - on_top_y) < 6.0)

    # --- Primary ladder selection (reachable from current platform) ---
    # Exclude current column (use pcx as exclusion center if on_column).
    excl_cx = jnp.where(on_column, pcx, jnp.array(-1e6))
    excl_cw = jnp.where(on_column, center_band + 4.0, jnp.array(0.0))

    lcx, lty, has_valid, has_fb = _select_reachable(
        obs_flat, pcx, pby, py, reach_tol, reach_up,
        height_weight, excl_cx, excl_cw,
    )

    # --- Next-ladder selection from the UPPER platform (used near top) ---
    # Evaluate reach against where the player will stand after dismount:
    #   py_eval ≈ on_top_y, pby_eval ≈ on_top_y + ph
    py_upper = on_top_y
    pby_upper = on_top_y + ph
    nlcx, nlty, n_has_valid, n_has_fb = _select_reachable(
        obs_flat, pcx, pby_upper, py_upper, reach_tol, reach_up,
        height_weight, pcx, center_band + 4.0,
    )

    # --- Off-ladder horizontal seek toward primary target ---
    child_dir_x = pcx + jnp.sign(child_x - pcx) * 30.0
    target_x = jnp.where(has_valid | has_fb, lcx, child_dir_x)
    horiz_action = _move_toward_x(target_x - pcx, 1.0)

    aligned_with_target = (jnp.abs(lcx - pcx) < align_tol) & (has_valid | has_fb)

    # --- Threats ---
    any_monkey, monkey_sign_dx, overhead_fc, fc_dx, any_thrown = _threats(
        obs_flat, pcx, py, danger_r
    )

    # === Build action ===
    action = horiz_action

    # Aligned with reachable ladder while off-column -> climb
    action = jnp.where(aligned_with_target & ~on_column, UP, action)

    # On column and not yet near top -> keep climbing
    action = jnp.where(on_column & ~near_top, UP, action)

    # Near top -> dismount horizontally toward NEXT ladder (upper-platform selection)
    next_target = jnp.where(n_has_valid | n_has_fb, nlcx, child_dir_x)
    dismount_dx = next_target - pcx
    dismount_dir = _move_toward_x(dismount_dx, 1.0)
    # If next target equals current column or dx is 0, force a default sidestep.
    default_side = jnp.where(child_x < pcx, LEFT, RIGHT)
    dismount_dir = jnp.where(jnp.abs(dismount_dx) < 1.0, default_side, dismount_dir)
    action = jnp.where(near_top, dismount_dir, action)

    # Punch nearby monkey (preserve first-reward branch; allow even on column off-top)
    punch_action = jnp.where(monkey_sign_dx > 0, RIGHTFIRE, LEFTFIRE)
    punch_in_range = any_monkey & (jnp.abs(monkey_sign_dx) < punch_dx)
    action = jnp.where(punch_in_range, punch_action, action)

    # Off-ladder hazard avoidance
    action = jnp.where(any_thrown & ~on_column, FIRE, action)
    fc_step = jnp.where(fc_dx > 0, LEFT, RIGHT)
    action = jnp.where(overhead_fc & ~on_column, fc_step, action)

    # On-ladder hazards: COMMIT, do not freeze.
    # If overhead falling coconut while climbing and not yet near top, keep climbing past it.
    # If near top with hazard, prioritize stepping out of column.
    on_col_hazard = (any_thrown | overhead_fc) & on_column
    action = jnp.where(on_col_hazard & near_top, dismount_dir, action)
    # (off-top on-column hazard keeps the UP set above; no freeze branch)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
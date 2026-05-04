"""
Auto-generated policy v2
Generated at: 2026-05-01 13:50:39
"""

"""
Kangaroo policy v7
Fixes post-200 stall by:
- decoupling 'on ladder column' from bottom-below test (center-band only),
- recomputing next-ladder selection from the post-climb platform height,
- excluding the just-used ladder column with a wide pad on dismount,
- replacing on-ladder NOOP-under-hazard with active evasion (dismount/descend),
- tightening climb x-overlap to remove 1-pixel spurious climbs.
"""

import jax
import jax.numpy as jnp

NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = 6, 7, 8, 9
UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE = 10, 11, 12, 13


def init_params():
    return {
        "reach_tol": jnp.array(15.5),
        "reach_up": jnp.array(28.2),
        "align_tol": jnp.array(6.1),
        "danger_r": jnp.array(18.2),
        "punch_dx": jnp.array(19.5),
        "height_weight": jnp.array(1.12),
        "col_band": jnp.array(5.0),     # half-width band around ladder center for on-column
        "dismount_pad": jnp.array(14.0),  # extra exclusion around just-used column
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


def _on_ladder_column(obs, pcx, py, pby, col_band):
    """Center-band on-column test, decoupled from bottom-below."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh
    centered = jnp.abs(pcx - lcx) < col_band
    # Player vertically inside the ladder span (with small slack).
    vertical_in = (py >= lty - 6.0) & (py <= lby + 4.0)
    active = la > 0.5
    on = centered & vertical_in & active
    any_on = jnp.any(on)
    top_for = jnp.where(on, lty, jnp.array(1e6))
    cx_for = jnp.where(on, lcx, jnp.array(1e6))
    nearest_top = jnp.min(top_for)
    cur_lcx = jnp.min(cx_for)  # if any_on, this is the active column center
    return any_on, nearest_top, cur_lcx


def _select_best(obs, pcx, ref_pby, ref_py, reach_tol, reach_up,
                 height_weight, exclude_cx, exclude_pad):
    """Best upward ladder reachable from a hypothetical (ref_py, ref_pby).
    Excludes ladders whose center is within exclude_pad of exclude_cx.
    """
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh

    diff = lby - ref_pby
    reachable = (diff > -reach_up) & (diff < reach_tol)
    useful = lty < (ref_py - 4.0)
    active = la > 0.5
    excluded = jnp.abs(lcx - exclude_cx) < exclude_pad

    valid = reachable & useful & active & ~excluded
    dx = jnp.abs(lcx - pcx)
    height_gap = jnp.maximum(ref_py - lty, 0.0)
    cost = jnp.where(valid, dx - height_weight * height_gap, jnp.array(1e6))
    idx = jnp.argmin(cost)
    has_valid = jnp.any(valid)

    fb_valid = useful & active & ~excluded
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
    danger_r = params["danger_r"]
    punch_dx = params["punch_dx"]
    height_weight = params["height_weight"]
    col_band = params["col_band"]
    dismount_pad = params["dismount_pad"]

    # On-ladder-column test (center-band, decoupled from bottom).
    on_column, on_top_y, cur_col_cx = _on_ladder_column(
        obs_flat, pcx, py, pby, col_band
    )
    near_top = on_column & ((py - on_top_y) < 10.0)

    # Primary selection: from current platform, exclude current column (if any).
    excl_cx = jnp.where(on_column, cur_col_cx, jnp.array(-1e6))
    lcx, lty, has_valid, has_fb = _select_best(
        obs_flat, pcx, pby, py,
        reach_tol, reach_up, height_weight,
        excl_cx, dismount_pad,
    )

    # Post-climb selection: evaluate reachability assuming we are on the
    # NEW upper platform (top of current ladder).
    post_py = on_top_y - 2.0
    post_pby = post_py + ph
    d_lcx, d_lty, d_has_valid, d_has_fb = _select_best(
        obs_flat, pcx, post_pby, post_py,
        reach_tol, reach_up, height_weight,
        excl_cx, dismount_pad,
    )

    child_dir_x = pcx + jnp.sign(child_x - pcx) * 30.0
    target_x = jnp.where(has_valid | has_fb, lcx, child_dir_x)
    dx_to_target = target_x - pcx

    # Tight alignment: require player center inside ladder band, not slack overlap.
    aligned = has_valid & (jnp.abs(lcx - pcx) < align_tol)
    horiz_action = _move_toward_x(dx_to_target, 1.0)

    # Threats
    any_monkey, monkey_sign_dx, overhead_fc, fc_dx, any_thrown = _threats(
        obs_flat, pcx, py, danger_r
    )

    # --- Build action ---
    action = horiz_action

    # Aligned with reachable ladder -> climb
    action = jnp.where(aligned, UP, action)

    # On column and not near top -> keep climbing
    action = jnp.where(on_column & ~near_top, UP, action)

    # Near top of column -> dismount toward post-climb next ladder.
    dismount_target = jnp.where(d_has_valid | d_has_fb, d_lcx, child_dir_x)
    dismount_dir = _move_toward_x(dismount_target - pcx, 1.0)
    dismount_dir = jnp.where(
        dismount_dir == NOOP,
        jnp.where(child_x < pcx, LEFT, RIGHT),
        dismount_dir,
    )
    action = jnp.where(near_top, dismount_dir, action)

    # Punch nearby monkey (off ladder only, to avoid stalling on column).
    punch_action = jnp.where(monkey_sign_dx > 0, RIGHTFIRE, LEFTFIRE)
    punch_in_range = any_monkey & (jnp.abs(monkey_sign_dx) < punch_dx) & ~on_column
    action = jnp.where(punch_in_range, punch_action, action)

    # Active evasion under hazards.
    # Off ladder + thrown coconut nearby -> jump.
    action = jnp.where(any_thrown & ~on_column, FIRE, action)

    # On ladder + hazard:
    #   if near top -> commit to dismount direction (already set, but force it),
    #   else -> step out using dismount_dir if available, else descend.
    on_col_hazard = (any_thrown | overhead_fc) & on_column
    evade_dir = jnp.where(d_has_valid | d_has_fb, dismount_dir, DOWN)
    action = jnp.where(on_col_hazard & near_top, dismount_dir, action)
    action = jnp.where(on_col_hazard & ~near_top, evade_dir, action)

    # Off-ladder overhead falling coconut: sidestep.
    fc_step = jnp.where(fc_dx > 0, LEFT, RIGHT)
    action = jnp.where(overhead_fc & ~on_column, fc_step, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
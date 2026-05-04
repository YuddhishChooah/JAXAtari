"""
Auto-generated policy v4
Generated at: 2026-05-01 15:05:43
"""

"""
Kangaroo policy v7
Preserves the stable 200-point first-reward route (RIGHTFIRE punch + climb at x~132)
and fixes the post-200 stall by:
- tighter center-band on-ladder test (no more 1-pixel overlap UP),
- a genuinely different dismount selector (excludes current column by width,
  drops the strict useful-upward filter),
- on-ladder hazard handling that climbs through instead of freezing,
- forcing horizontal commit after near-top.
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
        "column_band": jnp.array(4.5),   # center-band half-width for on-ladder test
        "danger_r": jnp.array(18.2),
        "punch_dx": jnp.array(19.5),
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


def _is_on_ladder(obs, pcx, py, pby, band):
    """Center-band test: player center is within `band` of ladder center,
    feet are between ladder top and ladder bottom (with a small margin)."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh
    centered = jnp.abs(pcx - lcx) < band
    vert_ok = (pby > lty - 2.0) & (pby < lby + 6.0)
    active = la > 0.5
    on = centered & vert_ok & active
    any_on = jnp.any(on)
    top_for = jnp.where(on, lty, jnp.array(1e6))
    cx_for = jnp.where(on, lcx, jnp.array(1e6))
    nearest_top = jnp.min(top_for)
    cur_lcx = jnp.where(any_on, cx_for[jnp.argmin(top_for)], jnp.array(-1e6))
    return any_on, nearest_top, cur_lcx


def _select_ladder(obs, pcx, pby, py, reach_tol, reach_up, height_weight,
                   exclude_lcx, exclude_w, require_upward):
    """Best ladder. exclude_lcx is the center-x of a ladder to avoid
    (use -1e6 to disable). require_upward gates the lty<py-4 filter."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh

    diff = lby - pby
    reachable = (diff > -reach_up) & (diff < reach_tol)

    upward = lty < (py - 4.0)
    useful = jnp.where(require_upward, upward, jnp.ones_like(upward, dtype=bool))
    active = la > 0.5

    not_excluded = jnp.abs(lcx - exclude_lcx) > exclude_w

    valid = reachable & useful & active & not_excluded
    dx = jnp.abs(lcx - pcx)
    height_gap = jnp.maximum(py - lty, 0.0)
    height_gap = jnp.minimum(height_gap, 60.0)
    cost = jnp.where(valid, dx - height_weight * height_gap, jnp.array(1e6))
    idx = jnp.argmin(cost)
    has_valid = jnp.any(valid)

    fb_valid = active & not_excluded & useful
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
    column_band = params["column_band"]
    danger_r = params["danger_r"]
    punch_dx = params["punch_dx"]
    height_weight = params["height_weight"]

    # Center-band on-ladder test (much stricter than tiny x-overlap).
    on_column, on_top_y, cur_lcx = _is_on_ladder(
        obs_flat, pcx, py, pby, column_band
    )
    near_top = on_column & ((py - on_top_y) < (ph + 2.0))

    # Primary ladder: upward, reachable, do not re-pick the column we are on.
    no_excl = jnp.array(-1e6)
    p_lcx, p_lty, p_has_valid, p_has_fb = _select_ladder(
        obs_flat, pcx, pby, py, reach_tol, reach_up, height_weight,
        exclude_lcx=jnp.where(on_column, cur_lcx, no_excl),
        exclude_w=jnp.array(8.0),
        require_upward=jnp.array(True),
    )

    # Dismount ladder: at near-top, exclude current column by a wide margin and
    # drop the strict upward-only filter so a side ladder is also acceptable.
    d_lcx, d_lty, d_has_valid, d_has_fb = _select_ladder(
        obs_flat, pcx, pby, py, reach_tol, reach_up, height_weight,
        exclude_lcx=cur_lcx,
        exclude_w=jnp.array(20.0),
        require_upward=jnp.array(False),
    )

    # Approach phase: head toward the primary target.
    child_dir_x = pcx + jnp.sign(child_x - pcx) * 30.0
    target_x = jnp.where(p_has_valid | p_has_fb, p_lcx, child_dir_x)
    approach = _move_toward_x(target_x - pcx, 1.0)
    aligned = p_has_valid & (jnp.abs(p_lcx - pcx) < align_tol)

    # Threats
    any_monkey, monkey_sign_dx, overhead_fc, fc_dx, any_thrown = _threats(
        obs_flat, pcx, py, danger_r
    )

    # --- Build action ---
    action = approach

    # Aligned with reachable ladder bottom -> step up.
    action = jnp.where(aligned, UP, action)

    # On column and not near top -> keep climbing.
    action = jnp.where(on_column & ~near_top, UP, action)

    # Near top of column -> commit horizontally to next ladder (or child dir).
    dismount_target = jnp.where(d_has_valid | d_has_fb, d_lcx, child_dir_x)
    dismount_dir = _move_toward_x(dismount_target - pcx, 1.0)
    fallback_lr = jnp.where(child_x < pcx, LEFT, RIGHT)
    dismount_dir = jnp.where(dismount_dir == NOOP, fallback_lr, dismount_dir)
    action = jnp.where(near_top, dismount_dir, action)

    # Punch nearby monkey (preserves the first-reward RIGHTFIRE branch).
    punch_action = jnp.where(monkey_sign_dx > 0, RIGHTFIRE, LEFTFIRE)
    punch_in_range = any_monkey & (jnp.abs(monkey_sign_dx) < punch_dx)
    action = jnp.where(punch_in_range, punch_action, action)

    # Thrown coconut nearby off-ladder -> jump.
    action = jnp.where(any_thrown & ~on_column, FIRE, action)
    # Thrown coconut on ladder -> climb through (UP), do not freeze.
    action = jnp.where(any_thrown & on_column & ~near_top, UP, action)

    # Overhead falling coconut: sidestep off-column; on-column climb through.
    fc_step = jnp.where(fc_dx > 0, LEFT, RIGHT)
    action = jnp.where(overhead_fc & ~on_column, fc_step, action)
    action = jnp.where(overhead_fc & on_column & ~near_top, UP, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
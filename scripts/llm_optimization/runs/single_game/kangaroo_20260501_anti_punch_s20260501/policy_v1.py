"""
Auto-generated policy v1
Generated at: 2026-05-01 13:44:43
"""

"""
Kangaroo policy v8
Preserves the first-reward route (RIGHTFIRE punch near x=132) and the
asymmetric ladder reach test, but fixes the post-200 stall by:
- using a parametric center-band test for on-ladder column (not 1-pixel overlap),
- computing a genuinely different dismount target using a post-climb pby surrogate,
- committing to a horizontal dismount with a wider near_top window,
- replacing on-ladder freeze under hazard with a directional dismount/descend.
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
        "column_band": jnp.array(5.0),     # center-band half-width for "on ladder"
        "near_top_tol": jnp.array(14.0),   # widened so dismount fires reliably
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


def _on_ladder_column(obs, pcx, py, pby, column_band):
    """Center-band test: |pcx - ladder_center| < column_band."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh
    centered = jnp.abs(pcx - lcx) < column_band
    top_above = lty < py + 2.0
    bottom_below = lby >= pby - 6.0
    active = la > 0.5
    on = centered & top_above & bottom_below & active
    any_on = jnp.any(on)
    top_for = jnp.where(on, lty, jnp.array(1e6))
    nearest_top = jnp.min(top_for)
    return any_on, nearest_top


def _select_reachable(obs, pcx, ref_pby, py_for_useful, reach_tol, reach_up,
                      height_weight, exclude_cx, exclude_band):
    """Select best upward ladder reachable from ref_pby.

    exclude_cx/exclude_band let us drop the current ladder column when picking
    a dismount target.
    """
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh

    diff = lby - ref_pby
    reachable = (diff > -reach_up) & (diff < reach_tol)
    useful = lty < (py_for_useful - 4.0)
    active = la > 0.5
    not_excluded = jnp.abs(lcx - exclude_cx) > exclude_band

    valid = reachable & useful & active & not_excluded
    dx = jnp.abs(lcx - pcx)
    height_gap = jnp.maximum(py_for_useful - lty, 0.0)
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
    column_band = params["column_band"]
    near_top_tol = params["near_top_tol"]
    danger_r = params["danger_r"]
    punch_dx = params["punch_dx"]
    height_weight = params["height_weight"]

    # On-ladder via center-band (not 1-pixel overlap).
    on_column, on_top_y = _on_ladder_column(obs_flat, pcx, py, pby, column_band)
    near_top = on_column & ((py - on_top_y) < near_top_tol)

    # Primary selection from current platform (use real pby).
    prim_lcx, prim_lty, prim_valid, prim_fb = _select_reachable(
        obs_flat, pcx, pby, py, reach_tol, reach_up, height_weight,
        exclude_cx=jnp.array(-1e4), exclude_band=jnp.array(0.0),
    )

    # Dismount selection: simulate standing on the platform at the top of the
    # current ladder. Exclude the current ladder column so we pick a NEW one.
    post_pby = on_top_y + ph + 2.0
    post_py = on_top_y - 1.0
    dis_lcx, dis_lty, dis_valid, dis_fb = _select_reachable(
        obs_flat, pcx, post_pby, post_py, reach_tol, reach_up, height_weight,
        exclude_cx=pcx, exclude_band=column_band + 2.0,
    )

    # Horizontal toward primary target.
    child_dir_x = pcx + jnp.sign(child_x - pcx) * 30.0
    target_x = jnp.where(prim_valid | prim_fb, prim_lcx, child_dir_x)
    horiz_action = _move_toward_x(target_x - pcx, 1.0)

    aligned = prim_valid & (jnp.abs(prim_lcx - pcx) < align_tol)

    # Threats
    any_monkey, monkey_sign_dx, overhead_fc, fc_dx, any_thrown = _threats(
        obs_flat, pcx, py, danger_r
    )

    # --- Build action ---
    action = horiz_action

    # Aligned with reachable ladder bottom -> climb
    action = jnp.where(aligned, UP, action)

    # On column and not near top -> keep climbing
    action = jnp.where(on_column & ~near_top, UP, action)

    # Near top of ladder -> commit to horizontal dismount.
    dismount_target = jnp.where(
        dis_valid | dis_fb, dis_lcx,
        jnp.where(child_x < pcx, pcx - 30.0, pcx + 30.0),
    )
    dismount_dir = _move_toward_x(dismount_target - pcx, 0.5)
    # Force a direction even if exactly aligned: bias toward child.
    dismount_dir = jnp.where(
        dismount_dir == NOOP,
        jnp.where(child_x < pcx, LEFT, RIGHT),
        dismount_dir,
    )
    action = jnp.where(near_top, dismount_dir, action)

    # Punch nearby monkey (preserve first-reward route).
    punch_action = jnp.where(monkey_sign_dx > 0, RIGHTFIRE, LEFTFIRE)
    punch_in_range = any_monkey & (jnp.abs(monkey_sign_dx) < punch_dx)
    action = jnp.where(punch_in_range, punch_action, action)

    # Thrown coconut nearby off-ladder -> jump.
    action = jnp.where(any_thrown & ~on_column, FIRE, action)
    # Thrown coconut while on ladder -> dismount sideways (don't freeze).
    on_lad_hazard_dir = jnp.where(child_x < pcx, LEFT, RIGHT)
    action = jnp.where(
        any_thrown & on_column,
        on_lad_hazard_dir,
        action,
    )

    # Overhead falling coconut: sidestep off column.
    fc_step = jnp.where(fc_dx > 0, LEFT, RIGHT)
    action = jnp.where(overhead_fc & ~on_column, fc_step, action)
    # On column with falling coconut overhead -> step off sideways too.
    action = jnp.where(overhead_fc & on_column, fc_step, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
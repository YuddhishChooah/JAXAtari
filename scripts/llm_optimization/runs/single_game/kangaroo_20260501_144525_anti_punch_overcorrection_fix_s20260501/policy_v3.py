"""
Auto-generated policy v3
Generated at: 2026-05-01 14:59:34
"""

"""
Kangaroo policy v7
Post-200 fix:
- Differentiate dismount selector using upper-platform y reference.
- Replace coarse on-column x-band with parametric center-band overlap.
- Step OFF the ladder column when near top with overhead hazard, instead of NOOP.
- Cost function prefers closest reachable ladder; height only as tie-breaker.
- Preserve the first-reward RIGHTFIRE punch branch.
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
        "center_band": jnp.array(5.0),   # half-width for on-column center test
        "danger_r": jnp.array(18.2),
        "punch_dx": jnp.array(19.5),
        "next_plat_dy": jnp.array(20.0), # offset added to on_top_y for next-plat feet
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
    """Center-band overlap test: player center within ladder center +/- band."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh
    centered = jnp.abs(pcx - lcx) < center_band
    top_above = lty < py + 2.0
    bottom_below = lby >= pby - 6.0
    active = la > 0.5
    on = centered & top_above & bottom_below & active
    any_on = jnp.any(on)
    top_for = jnp.where(on, lty, jnp.array(1e6))
    nearest_top = jnp.min(top_for)
    return any_on, nearest_top


def _select_ladder(obs, pcx, ref_pby, ref_py, reach_tol, reach_up, exclude_band):
    """Pick closest reachable upward active ladder.
    ref_pby/ref_py are the reference feet/top y to test reachability against.
    exclude_band excludes ladders whose center is within band of pcx (current column).
    """
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh

    diff = lby - ref_pby
    reachable = (diff > -reach_up) & (diff < reach_tol)
    useful = lty < (ref_py - 4.0)
    active = la > 0.5
    not_current = jnp.abs(lcx - pcx) > exclude_band

    valid = reachable & useful & active & not_current
    dx = jnp.abs(lcx - pcx)
    height_gap = jnp.maximum(ref_py - lty, 0.0)
    # Closest first; small height tie-breaker.
    cost = jnp.where(valid, dx - 0.05 * height_gap, jnp.array(1e6))
    idx = jnp.argmin(cost)
    has_valid = jnp.any(valid)

    # Fallback ignoring reach: nearest upward active ladder not in current column.
    fb_valid = useful & active & not_current
    fb_cost = jnp.where(fb_valid, dx, jnp.array(1e6))
    fb_idx = jnp.argmin(fb_cost)
    has_fb = jnp.any(fb_valid)

    sel = jnp.where(has_valid, idx, fb_idx)
    return lcx[sel], lty[sel], has_valid, has_fb


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
    next_plat_dy = params["next_plat_dy"]

    # On-ladder via center-band overlap (no 1-pixel latch).
    on_column, on_top_y = _is_on_ladder(obs_flat, pcx, py, pby, center_band)
    near_top = on_column & ((py - on_top_y) < 10.0)

    # Primary target: from current platform.
    p_lcx, p_lty, p_has_valid, p_has_fb = _select_ladder(
        obs_flat, pcx, pby, py, reach_tol, reach_up, exclude_band=center_band
    )

    # Dismount/next target: evaluate from predicted upper-platform feet.
    upper_pby = on_top_y + next_plat_dy
    upper_py = on_top_y + next_plat_dy - ph
    n_lcx, n_lty, n_has_valid, n_has_fb = _select_ladder(
        obs_flat, pcx, upper_pby, upper_py, reach_tol, reach_up,
        exclude_band=center_band,
    )

    # Default horizontal target = primary ladder; if none, drift toward child.
    child_dir_x = pcx + jnp.sign(child_x - pcx) * 30.0
    target_x = jnp.where(p_has_valid | p_has_fb, p_lcx, child_dir_x)
    horiz_action = _move_toward_x(target_x - pcx, 1.0)

    aligned = p_has_valid & (jnp.abs(p_lcx - pcx) < align_tol)

    # Threats
    any_monkey, monkey_sign_dx, overhead_fc, fc_dx, any_thrown = _threats(
        obs_flat, pcx, py, danger_r
    )

    # --- Build action ---
    action = horiz_action

    # Aligned with reachable ladder bottom -> climb.
    action = jnp.where(aligned, UP, action)

    # On column and not near top -> keep climbing.
    action = jnp.where(on_column & ~near_top, UP, action)

    # Near top -> dismount toward next upper-platform ladder.
    dismount_target = jnp.where(n_has_valid | n_has_fb, n_lcx, child_dir_x)
    dismount_dx = dismount_target - pcx
    dismount_dir = _move_toward_x(dismount_dx, 1.0)
    # If exactly aligned (NOOP), pick a default side (toward child) so we exit column.
    dismount_dir = jnp.where(
        dismount_dir == NOOP,
        jnp.where(child_x < pcx, LEFT, RIGHT),
        dismount_dir,
    )
    action = jnp.where(near_top, dismount_dir, action)

    # Punch nearby monkey (preserve first-reward branch; allow even on column off-top).
    punch_action = jnp.where(monkey_sign_dx > 0, RIGHTFIRE, LEFTFIRE)
    punch_in_range = any_monkey & (jnp.abs(monkey_sign_dx) < punch_dx)
    action = jnp.where(punch_in_range & ~on_column, punch_action, action)

    # Off-ladder hazards: jump over thrown coconut.
    action = jnp.where(any_thrown & ~on_column, FIRE, action)

    # Overhead falling coconut off-column -> sidestep.
    fc_step = jnp.where(fc_dx > 0, LEFT, RIGHT)
    action = jnp.where(overhead_fc & ~on_column, fc_step, action)

    # On-column hazard: do NOT freeze. If near top, force dismount step.
    # If not near top, briefly step out of center band toward dismount side.
    on_col_hazard = on_column & (overhead_fc | any_thrown)
    escape_dir = jnp.where(child_x < pcx, LEFT, RIGHT)
    action = jnp.where(on_col_hazard & near_top, dismount_dir, action)
    action = jnp.where(on_col_hazard & ~near_top, escape_dir, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
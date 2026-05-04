"""
Auto-generated policy v5
Generated at: 2026-05-01 14:09:03
"""

"""
Kangaroo policy v7
Fix: post-200 dismount and next-ladder commit.
- Tighter on-ladder predicate (center-band, not pad overlap).
- Distinct dismount selector that excludes the current ladder column with
  a wide exclusion radius.
- Parametric near-top window using feet vs ladder-top.
- On-ladder hazard retreats sideways/downward instead of freezing.
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
        "center_band": jnp.array(4.0),     # tighter on-ladder x test
        "near_top_tol": jnp.array(10.0),   # feet within this of ladder top => topped out
        "danger_r": jnp.array(18.0),
        "punch_dx": jnp.array(19.5),
        "dismount_excl": jnp.array(20.0),  # exclude current column when picking next ladder
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


def _on_ladder_state(obs, pcx, py, pby, center_band, near_top_tol):
    """Detect being inside a ladder column using a centered band."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh
    active = la > 0.5

    centered = jnp.abs(pcx - lcx) < center_band
    vertically_in = (pby >= lty - 2.0) & (py <= lby + 2.0)
    on = centered & vertically_in & active

    any_on = jnp.any(on)
    top_for = jnp.where(on, lty, jnp.array(1e6))
    nearest_top = jnp.min(top_for)
    col_x_for = jnp.where(on, lcx, jnp.array(1e6))
    col_x = jnp.min(col_x_for)

    # topped out: feet at or above ladder top within tolerance
    topped = any_on & (pby <= nearest_top + near_top_tol)
    return any_on, topped, nearest_top, col_x


def _select_ladder(obs, pcx, pby, py, reach_tol, reach_up, exclude_x, exclude_r):
    """Pick best upward, reachable, active ladder, excluding a column near exclude_x."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh

    diff = lby - pby
    reachable = (diff > -reach_up) & (diff < reach_tol)
    useful = lty < (py - 4.0)
    active = la > 0.5
    not_excluded = jnp.abs(lcx - exclude_x) > exclude_r

    valid = reachable & useful & active & not_excluded
    dx = jnp.abs(lcx - pcx)
    cost = jnp.where(valid, dx, jnp.array(1e6))
    idx = jnp.argmin(cost)
    has_valid = jnp.any(valid)

    # Fallback: any upward active ladder not in excluded column.
    fb_valid = useful & active & not_excluded
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
    tcost = jnp.where(thrown_near, jnp.abs(tdx), jnp.array(1e6))
    tidx = jnp.argmin(tcost)
    thrown_sign_dx = tdx[tidx]

    return any_monkey, monkey_sign_dx, overhead_fc, fc_dx, any_thrown, thrown_sign_dx


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
    near_top_tol = params["near_top_tol"]
    danger_r = params["danger_r"]
    punch_dx = params["punch_dx"]
    dismount_excl = params["dismount_excl"]

    # On-ladder detection (tight, centered).
    on_column, topped, on_top_y, col_x = _on_ladder_state(
        obs_flat, pcx, py, pby, center_band, near_top_tol
    )

    # Primary ladder (used when not on a ladder yet). No exclusion.
    p_lcx, p_lty, p_has_valid, p_has_fb = _select_ladder(
        obs_flat, pcx, pby, py, reach_tol, reach_up,
        exclude_x=jnp.array(-1e4), exclude_r=jnp.array(0.0),
    )

    # Dismount/next-ladder target: exclude the column we are currently in.
    d_lcx, d_lty, d_has_valid, d_has_fb = _select_ladder(
        obs_flat, pcx, pby, py, reach_tol, reach_up,
        exclude_x=col_x, exclude_r=dismount_excl,
    )

    # Child-biased fallback if nothing useful is available.
    child_dir_x = pcx + jnp.sign(child_x - pcx) * 30.0

    # Threats
    (any_monkey, monkey_sign_dx, overhead_fc, fc_dx,
     any_thrown, thrown_sign_dx) = _threats(obs_flat, pcx, py, danger_r)

    # ---- Build action ----

    # Default: approach primary ladder if known, else drift toward child.
    approach_target = jnp.where(p_has_valid | p_has_fb, p_lcx, child_dir_x)
    action = _move_toward_x(approach_target - pcx, 1.0)

    # Aligned with reachable primary ladder (and not yet on column) -> climb.
    aligned = p_has_valid & (jnp.abs(p_lcx - pcx) < align_tol) & ~on_column
    action = jnp.where(aligned, UP, action)

    # On column and not topped out -> keep climbing.
    action = jnp.where(on_column & ~topped, UP, action)

    # Topped out -> commit to a *different* ladder column (or child fallback).
    dismount_target = jnp.where(d_has_valid | d_has_fb, d_lcx, child_dir_x)
    dismount_dir = _move_toward_x(dismount_target - pcx, 1.0)
    dismount_dir = jnp.where(
        dismount_dir == NOOP,
        jnp.where(child_x < pcx, LEFT, RIGHT),
        dismount_dir,
    )
    action = jnp.where(topped, dismount_dir, action)

    # Punch nearby monkey when off-ladder.
    punch_action = jnp.where(monkey_sign_dx > 0, RIGHTFIRE, LEFTFIRE)
    punch_in_range = any_monkey & (jnp.abs(monkey_sign_dx) < punch_dx) & ~on_column
    action = jnp.where(punch_in_range, punch_action, action)

    # Thrown coconut nearby off-ladder -> jump.
    action = jnp.where(any_thrown & ~on_column, FIRE, action)

    # Thrown coconut while on ladder -> retreat away horizontally
    # (descend column slightly via DOWN if topped, else step opposite).
    retreat_dir = jnp.where(thrown_sign_dx > 0, LEFT, RIGHT)
    on_ladder_threat = any_thrown & on_column & ~topped
    action = jnp.where(on_ladder_threat, DOWN, action)
    action = jnp.where(any_thrown & topped, retreat_dir, action)

    # Overhead falling coconut: sidestep when off column;
    # when on column mid-climb, descend to clear the column.
    fc_step = jnp.where(fc_dx > 0, LEFT, RIGHT)
    action = jnp.where(overhead_fc & ~on_column, fc_step, action)
    action = jnp.where(overhead_fc & on_column & ~topped, DOWN, action)
    action = jnp.where(overhead_fc & topped, fc_step, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
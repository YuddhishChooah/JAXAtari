"""
Auto-generated policy v5
Generated at: 2026-05-01 15:12:01
"""

"""
Kangaroo policy v7
Preserves the stable 200-point first-reward route (RIGHTFIRE punch near the
first ladder) and fixes the post-200 stall by:
- using a center-band on-ladder test (no 1-pixel UP latch),
- a *distinct* dismount selector that excludes the current column with a
  larger pad so the chosen "next ladder" is provably different,
- replacing on-ladder hazard NOOP with a step-down + sideways move,
- clipping the height bias and adding an x-distance ceiling on cost.
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
        "align_tol": jnp.array(6.1),
        "center_band": jnp.array(4.5),     # tighter on-ladder gate
        "danger_r": jnp.array(18.2),
        "punch_dx": jnp.array(19.5),
        "height_weight": jnp.array(1.1),
        "dismount_pad": jnp.array(10.0),   # extra exclusion for next-ladder
    }


# ---------- perception helpers ----------

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
    """Center-band on-ladder test (no tiny-overlap latch)."""
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
    # current column index (-1 if none); use argmax of bool mask, guarded by any_on
    col_idx = jnp.argmax(on.astype(jnp.int32))
    return any_on, nearest_top, col_idx, lcx, lw


def _reachable_ladder_score(obs, pcx, pby, py, reach_tol, reach_up,
                            height_weight, exclude_lcx, exclude_pad):
    """Best upward ladder, excluding a given column center."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh

    diff = lby - pby
    reachable = (diff > -reach_up) & (diff < reach_tol)
    useful = lty < (py - 4.0)
    active = la > 0.5

    # Exclude the column near exclude_lcx by horizontal center distance
    not_excluded = jnp.abs(lcx - exclude_lcx) > exclude_pad

    valid = reachable & useful & active & not_excluded
    dx = jnp.abs(lcx - pcx)
    height_gap = jnp.clip(py - lty, 0.0, 60.0)  # clip excessive upward bias
    # cost: prefer near in x, mild upward bias, with x ceiling
    x_pen = jnp.where(dx > 80.0, 1e3, 0.0)
    cost = jnp.where(valid,
                     dx - height_weight * height_gap + x_pen,
                     jnp.array(1e6))
    idx = jnp.argmin(cost)
    has_valid = jnp.any(valid)

    # fallback: any upward active ladder not excluded (ignore reach)
    fb_valid = useful & active & not_excluded
    fb_cost = jnp.where(fb_valid, dx, jnp.array(1e6))
    fb_idx = jnp.argmin(fb_cost)
    has_fb = jnp.any(fb_valid)

    sel_idx = jnp.where(has_valid, idx, fb_idx)
    return lcx[sel_idx], lty[sel_idx], has_valid, has_fb


def _danger_near_player(obs, pcx, py, danger_r):
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


# ---------- main policy ----------

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
    height_weight = params["height_weight"]
    dismount_pad = params["dismount_pad"]

    # On a ladder column? (center-band test)
    on_column, on_top_y, col_idx, all_lcx, all_lw = _is_on_ladder(
        obs_flat, pcx, py, pby, center_band
    )
    near_top = on_column & ((py - on_top_y) < 6.0)

    # Current column center (used to exclude from dismount selector)
    cur_lcx = jnp.where(on_column, all_lcx[col_idx], jnp.array(-1e3))

    # Primary selector: from current platform, exclude the column we are in
    # (use a small pad keyed off ladder width)
    lcx, lty, has_valid, has_fb = _reachable_ladder_score(
        obs_flat, pcx, pby, py, reach_tol, reach_up, height_weight,
        cur_lcx, jnp.array(2.0)
    )

    # Dismount selector: exclude current column with a *larger* pad so that
    # the chosen next ladder is provably different from the one we climbed.
    d_lcx, d_lty, d_has_valid, d_has_fb = _reachable_ladder_score(
        obs_flat, pcx, pby, py, reach_tol, reach_up, height_weight,
        cur_lcx, dismount_pad
    )

    # Child-biased fallback if nothing useful
    child_dir_x = pcx + jnp.sign(child_x - pcx) * 30.0
    target_x = jnp.where(has_valid | has_fb, lcx, child_dir_x)
    dx_to_target = target_x - pcx

    aligned = has_valid & (jnp.abs(lcx - pcx) < align_tol)
    horiz_action = _move_toward_x(dx_to_target, 1.0)

    # Threats
    (any_monkey, monkey_sign_dx, overhead_fc, fc_dx,
     any_thrown, thrown_sign_dx) = _danger_near_player(
        obs_flat, pcx, py, danger_r
    )

    # ---------- assemble action ----------
    action = horiz_action

    # Aligned with reachable ladder -> climb
    action = jnp.where(aligned, UP, action)

    # On column and not near top -> keep climbing
    action = jnp.where(on_column & ~near_top, UP, action)

    # Near top -> dismount toward next ladder (or child if none)
    dismount_target = jnp.where(d_has_valid | d_has_fb, d_lcx, child_dir_x)
    raw_dismount = _move_toward_x(dismount_target - pcx, 1.0)
    dismount_dir = jnp.where(
        raw_dismount == NOOP,
        jnp.where(child_x < pcx, LEFT, RIGHT),
        raw_dismount,
    )
    action = jnp.where(near_top, dismount_dir, action)

    # Punch nearby monkey (preserve first-reward route)
    punch_action = jnp.where(monkey_sign_dx > 0, RIGHTFIRE, LEFTFIRE)
    punch_in_range = any_monkey & (jnp.abs(monkey_sign_dx) < punch_dx)
    action = jnp.where(punch_in_range, punch_action, action)

    # Thrown coconut nearby off-ladder -> jump
    action = jnp.where(any_thrown & ~on_column, FIRE, action)

    # Thrown coconut while on ladder -> bail off-column (DOWN + away horizontally)
    bail_dir = jnp.where(thrown_sign_dx > 0, DOWNLEFT, DOWNRIGHT)
    action = jnp.where(any_thrown & on_column & ~near_top, bail_dir, action)

    # Overhead falling coconut: sidestep off column, bail down on column
    fc_step = jnp.where(fc_dx > 0, LEFT, RIGHT)
    action = jnp.where(overhead_fc & ~on_column, fc_step, action)
    fc_bail = jnp.where(fc_dx > 0, DOWNLEFT, DOWNRIGHT)
    action = jnp.where(overhead_fc & on_column & ~near_top, fc_bail, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
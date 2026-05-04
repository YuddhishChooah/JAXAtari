"""
Auto-generated policy v6
Generated at: 2026-04-30 14:18:43
"""

"""
Kangaroo policy v4
Fixes chained climbs by:
- asymmetric reach window (ladder bottoms slightly below feet still count),
- geometry-driven fallback target (nearest upward ladder by x, ignoring reach),
- dismount direction from a second ladder selection that ignores on_column,
- on-ladder hazard handling (pause/punch instead of climbing into coconut).
"""

import jax
import jax.numpy as jnp

NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = 6, 7, 8, 9
UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE = 10, 11, 12, 13


def init_params():
    return {
        "reach_tol": jnp.array(15.7),
        "reach_up": jnp.array(28.0),     # extra tolerance for ladder bottoms above feet
        "align_tol": jnp.array(6.3),
        "on_pad": jnp.array(3.2),
        "danger_r": jnp.array(18.2),
        "punch_dx": jnp.array(19.8),
        "height_weight": jnp.array(1.06),
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


def _on_ladder_column(obs, pcx, py, pby, pad):
    lx, ly, lw, lh, la = _ladders(obs)
    lty = ly
    lby = ly + lh
    in_x = (pcx >= lx - pad) & (pcx <= lx + lw + pad)
    top_above = lty < py + 2.0
    bottom_below = lby >= pby - 4.0
    active = la > 0.5
    on = in_x & top_above & bottom_below & active
    any_on = jnp.any(on)
    top_for = jnp.where(on, lty, jnp.array(1e6))
    nearest_top = jnp.min(top_for)
    return any_on, nearest_top


def _select_best(obs, pcx, pby, py, reach_tol, reach_up, height_weight, exclude_pad):
    """Best upward ladder. Returns (lcx, lty, has_valid, has_fallback, fb_lcx)."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh

    # Asymmetric reach: ladder bottom may be a bit above feet (we step onto it)
    # or slightly below feet (transient on dismount).
    diff = lby - pby  # positive: ladder bottom below feet
    reachable = (diff > -reach_up) & (diff < reach_tol)

    useful = lty < (py - 4.0)
    active = la > 0.5

    # Optional exclusion: ladder column currently containing player (used for primary)
    in_col = (pcx >= lx - exclude_pad) & (pcx <= lx + lw + exclude_pad)

    valid = reachable & useful & active & ~in_col
    dx = jnp.abs(lcx - pcx)
    height_gap = jnp.maximum(py - lty, 0.0)
    cost = jnp.where(valid, dx - height_weight * height_gap, jnp.array(1e6))
    idx = jnp.argmin(cost)
    has_valid = jnp.any(valid)

    # Fallback: nearest upward active ladder regardless of reach,
    # and not the column we are currently in.
    fb_valid = useful & active & ~in_col
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
    on_pad = params["on_pad"]
    danger_r = params["danger_r"]
    punch_dx = params["punch_dx"]
    height_weight = params["height_weight"]

    # On a ladder column?
    on_column, on_top_y = _on_ladder_column(obs_flat, pcx, py, pby, on_pad)
    near_top = on_column & ((py - on_top_y) < 8.0)

    # Primary selection excludes current column; fallback ignores reach.
    lcx, lty, has_valid, has_fb = _select_best(
        obs_flat, pcx, pby, py, reach_tol, reach_up, height_weight, on_pad
    )

    # Dismount target: same selector but exclude_pad=0 so we still avoid current col.
    d_lcx, d_lty, d_has_valid, d_has_fb = _select_best(
        obs_flat, pcx, pby, py, reach_tol, reach_up, height_weight, on_pad
    )

    # Child-biased fallback if absolutely nothing useful is found
    child_dir_x = pcx + jnp.sign(child_x - pcx) * 30.0
    target_x = jnp.where(has_valid | has_fb, lcx, child_dir_x)
    dx_to_target = target_x - pcx

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

    # Near top of column -> dismount toward next ladder lcx (or child if none)
    dismount_target = jnp.where(d_has_valid | d_has_fb, d_lcx, child_dir_x)
    dismount_dir = _move_toward_x(dismount_target - pcx, 1.0)
    dismount_dir = jnp.where(
        dismount_dir == NOOP,
        jnp.where(child_x < pcx, LEFT, RIGHT),
        dismount_dir,
    )
    action = jnp.where(near_top, dismount_dir, action)

    # Punch nearby monkey
    punch_action = jnp.where(monkey_sign_dx > 0, RIGHTFIRE, LEFTFIRE)
    punch_in_range = any_monkey & (jnp.abs(monkey_sign_dx) < punch_dx)
    action = jnp.where(punch_in_range, punch_action, action)

    # Thrown coconut nearby off-ladder -> jump
    action = jnp.where(any_thrown & ~on_column, FIRE, action)
    # Thrown coconut while on ladder -> hold position (NOOP) to let it pass
    action = jnp.where(any_thrown & on_column & ~near_top, NOOP, action)

    # Overhead falling coconut: sidestep off column, pause on column
    fc_step = jnp.where(fc_dx > 0, LEFT, RIGHT)
    action = jnp.where(overhead_fc & ~on_column, fc_step, action)
    action = jnp.where(overhead_fc & on_column & ~near_top, NOOP, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
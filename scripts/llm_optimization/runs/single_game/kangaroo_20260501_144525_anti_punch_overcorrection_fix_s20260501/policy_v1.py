"""
Auto-generated policy v1
Generated at: 2026-05-01 14:47:23
"""

"""
Kangaroo policy v8
Fixes post-200 stall at top of first ladder by:
- stricter on-column test (center band + feet strictly inside ladder),
- distinct dismount selector with larger exclusion of current column,
- active hazard evasion on ladder (descend or jump off, not NOOP),
- UPFIRE to clear ladder lip when safe,
- preserved first-reward RIGHTFIRE punch path.
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
        "col_band": jnp.array(5.0),
        "danger_r": jnp.array(18.2),
        "punch_dx": jnp.array(19.5),
        "height_weight": jnp.array(1.12),
        "dismount_pad": jnp.array(14.0),
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
    """Strict: player center within col_band of ladder center AND feet inside ladder vertical span."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh
    center_ok = jnp.abs(pcx - lcx) < col_band
    feet_inside = (pby > lty + 2.0) & (pby < lby + 6.0)
    active = la > 0.5
    on = center_ok & feet_inside & active
    any_on = jnp.any(on)
    top_for = jnp.where(on, lty, jnp.array(1e6))
    nearest_top = jnp.min(top_for)
    return any_on, nearest_top


def _select_best(obs, pcx, pby, py, reach_tol, reach_up, height_weight, exclude_pad):
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh

    diff = lby - pby
    reachable = (diff > -reach_up) & (diff < reach_tol)
    useful = lty < (py - 6.0)
    active = la > 0.5
    in_col = jnp.abs(pcx - lcx) < exclude_pad

    valid = reachable & useful & active & ~in_col
    dx = jnp.abs(lcx - pcx)
    height_gap = jnp.maximum(py - lty, 0.0)
    cost = jnp.where(valid, dx - height_weight * height_gap, jnp.array(1e6))
    idx = jnp.argmin(cost)
    has_valid = jnp.any(valid)

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
    col_band = params["col_band"]
    danger_r = params["danger_r"]
    punch_dx = params["punch_dx"]
    height_weight = params["height_weight"]
    dismount_pad = params["dismount_pad"]

    on_column, on_top_y = _on_ladder_column(obs_flat, pcx, py, pby, col_band)
    near_top = on_column & ((py - on_top_y) < 10.0) & ((py - on_top_y) > -2.0)

    # Primary: pick reachable next ladder, exclude current column lightly.
    lcx, lty, has_valid, has_fb = _select_best(
        obs_flat, pcx, pby, py, reach_tol, reach_up, height_weight, col_band
    )

    # Dismount: must pick a *different* ladder column, so use larger exclusion.
    d_lcx, d_lty, d_has_valid, d_has_fb = _select_best(
        obs_flat, pcx, pby, py, reach_tol, reach_up, height_weight, dismount_pad
    )

    child_dir_x = pcx + jnp.sign(child_x - pcx) * 30.0
    target_x = jnp.where(has_valid | has_fb, lcx, child_dir_x)
    dx_to_target = target_x - pcx

    aligned = has_valid & (jnp.abs(lcx - pcx) < align_tol)
    horiz_action = _move_toward_x(dx_to_target, 1.0)

    any_monkey, monkey_sign_dx, overhead_fc, fc_dx, any_thrown = _threats(
        obs_flat, pcx, py, danger_r
    )

    # --- Build action ---
    action = horiz_action

    # Aligned with reachable ladder (off-column) -> climb
    action = jnp.where(aligned & ~on_column, UP, action)

    # On column and not near top -> climb
    action = jnp.where(on_column & ~near_top, UP, action)

    # Near top -> dismount toward different next ladder; UPFIRE if safe to clear lip
    dismount_target = jnp.where(d_has_valid | d_has_fb, d_lcx, child_dir_x)
    dismount_dx = dismount_target - pcx
    dismount_dir = _move_toward_x(dismount_dx, 1.0)
    dismount_dir = jnp.where(
        dismount_dir == NOOP,
        jnp.where(child_x < pcx, LEFT, RIGHT),
        dismount_dir,
    )
    safe_top = near_top & ~any_thrown & ~overhead_fc
    action = jnp.where(near_top, dismount_dir, action)
    # Use UPFIRE only when very close to top and safe, to clear the lip.
    very_top = on_column & ((py - on_top_y) < 4.0) & ((py - on_top_y) > -2.0)
    action = jnp.where(very_top & safe_top, UPFIRE, action)

    # Punch nearby monkey (preserve first-reward branch; allow even on column)
    punch_action = jnp.where(monkey_sign_dx > 0, RIGHTFIRE, LEFTFIRE)
    punch_in_range = any_monkey & (jnp.abs(monkey_sign_dx) < punch_dx)
    action = jnp.where(punch_in_range, punch_action, action)

    # Thrown coconut off-ladder -> jump
    action = jnp.where(any_thrown & ~on_column, FIRE, action)
    # Thrown coconut on ladder -> descend (active evasion, not NOOP)
    action = jnp.where(any_thrown & on_column & ~near_top, DOWN, action)
    # Thrown coconut on ladder near top -> jump off in dismount direction
    jump_dir = jnp.where(dismount_dx > 0, DOWNRIGHT, DOWNLEFT)
    action = jnp.where(any_thrown & near_top, jump_dir, action)

    # Overhead falling coconut: sidestep off column; descend on column
    fc_step = jnp.where(fc_dx > 0, LEFT, RIGHT)
    action = jnp.where(overhead_fc & ~on_column, fc_step, action)
    action = jnp.where(overhead_fc & on_column & ~near_top, DOWN, action)
    action = jnp.where(overhead_fc & near_top, dismount_dir, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
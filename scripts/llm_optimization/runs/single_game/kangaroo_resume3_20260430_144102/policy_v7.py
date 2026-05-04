"""
Auto-generated policy v7
Generated at: 2026-04-30 14:43:40
"""

"""
Kangaroo policy v7
Fixes post-200 stall by:
- hard-excluding the just-climbed ladder column via min_next_dx,
- tightening on_ladder_column (strict x containment, ladder extends below feet),
- replacing on-ladder coconut NOOP with active dismount,
- biasing next-ladder selection toward the child's side once above start band.
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
        "danger_r": jnp.array(18.2),
        "punch_dx": jnp.array(19.5),
        "height_weight": jnp.array(1.12),
        "min_next_dx": jnp.array(20.0),
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


def _on_ladder_column(obs, pcx, py, pby):
    """Strict: pcx inside [lx, lx+lw], ladder extends below feet, top above head."""
    lx, ly, lw, lh, la = _ladders(obs)
    lty = ly
    lby = ly + lh
    in_x = (pcx >= lx) & (pcx <= lx + lw)
    extends_below = lby > pby - 2.0
    top_above = lty < py - 2.0
    active = la > 0.5
    on = in_x & extends_below & top_above & active
    any_on = jnp.any(on)
    top_for = jnp.where(on, lty, jnp.array(1e6))
    cx_for = jnp.where(on, lx + lw * 0.5, jnp.array(1e6))
    nearest_top = jnp.min(top_for)
    col_cx = jnp.min(cx_for)
    return any_on, nearest_top, col_cx


def _select_ladder(obs, pcx, pby, py, child_x,
                   reach_tol, reach_up, height_weight,
                   exclude_cx, min_next_dx):
    """Pick best upward reachable ladder, excluding any near exclude_cx."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh

    diff = lby - pby
    reachable = (diff > -reach_up) & (diff < reach_tol)
    useful = lty < (py - 4.0)
    active = la > 0.5
    far_enough = jnp.abs(lcx - exclude_cx) > min_next_dx

    # child-side bias: ladders on child's side preferred
    child_side = jnp.sign(child_x - pcx)
    lad_side = jnp.sign(lcx - pcx)
    side_bonus = jnp.where(lad_side == child_side, -8.0, 0.0)

    valid = reachable & useful & active & far_enough
    dx = jnp.abs(lcx - pcx)
    height_gap = jnp.maximum(py - lty, 0.0)
    cost = jnp.where(valid, dx - height_weight * height_gap + side_bonus,
                     jnp.array(1e6))
    idx = jnp.argmin(cost)
    has_valid = jnp.any(valid)

    # Fallback: any active upward ladder far enough from exclude_cx
    fb_valid = useful & active & far_enough
    fb_cost = jnp.where(fb_valid, dx + side_bonus, jnp.array(1e6))
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
    danger_r = params["danger_r"]
    punch_dx = params["punch_dx"]
    height_weight = params["height_weight"]
    min_next_dx = params["min_next_dx"]

    on_column, on_top_y, col_cx = _on_ladder_column(obs_flat, pcx, py, pby)
    near_top = on_column & ((py - on_top_y) < 6.0)

    # Primary target: when not on column, exclude_cx = pcx (no exclusion effect).
    # When on column, exclude col_cx so we don't re-pick it.
    exclude_cx = jnp.where(on_column, col_cx, jnp.array(-1e6))
    # When off column, allow current column; use a distant exclude.
    lcx, lty, has_valid, has_fb = _select_ladder(
        obs_flat, pcx, pby, py, child_x,
        reach_tol, reach_up, height_weight,
        exclude_cx, jnp.where(on_column, min_next_dx, jnp.array(0.0))
    )

    # Child-biased fallback target if absolutely no ladder
    child_dir_x = pcx + jnp.sign(child_x - pcx) * 30.0
    target_x = jnp.where(has_valid | has_fb, lcx, child_dir_x)
    dx_to_target = target_x - pcx

    aligned = has_valid & (jnp.abs(lcx - pcx) < align_tol) & ~on_column
    horiz_action = _move_toward_x(dx_to_target, 1.0)

    any_monkey, monkey_sign_dx, overhead_fc, fc_dx, any_thrown, thrown_sign_dx = (
        _threats(obs_flat, pcx, py, danger_r)
    )

    # --- Build action ---
    action = horiz_action

    # Off column, aligned with reachable ladder -> climb
    action = jnp.where(aligned, UP, action)

    # On column and not near top -> keep climbing
    action = jnp.where(on_column & ~near_top, UP, action)

    # Near top of column -> commit to horizontal dismount toward next target
    dismount_dir = _move_toward_x(target_x - pcx, 0.5)
    dismount_dir = jnp.where(
        dismount_dir == NOOP,
        jnp.where(child_x < pcx, LEFT, RIGHT),
        dismount_dir,
    )
    action = jnp.where(near_top, dismount_dir, action)

    # Punch nearby monkey (off ladder)
    punch_action = jnp.where(monkey_sign_dx > 0, RIGHTFIRE, LEFTFIRE)
    punch_in_range = any_monkey & (jnp.abs(monkey_sign_dx) < punch_dx) & ~on_column
    action = jnp.where(punch_in_range, punch_action, action)

    # Thrown coconut nearby off-ladder -> jump
    action = jnp.where(any_thrown & ~on_column, FIRE, action)
    # Thrown coconut while on ladder -> actively leave column toward dismount dir
    on_ladder_evade = jnp.where(thrown_sign_dx > 0, DOWNLEFT, DOWNRIGHT)
    action = jnp.where(any_thrown & on_column & ~near_top, on_ladder_evade, action)
    action = jnp.where(any_thrown & near_top, dismount_dir, action)

    # Overhead falling coconut: sidestep off column, dismount on column
    fc_step = jnp.where(fc_dx > 0, LEFT, RIGHT)
    action = jnp.where(overhead_fc & ~on_column, fc_step, action)
    fc_evade_on = jnp.where(fc_dx > 0, DOWNLEFT, DOWNRIGHT)
    action = jnp.where(overhead_fc & on_column & ~near_top, fc_evade_on, action)
    action = jnp.where(overhead_fc & near_top, dismount_dir, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
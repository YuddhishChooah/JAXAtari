"""
Auto-generated policy v5
Generated at: 2026-04-30 14:14:32
"""

"""
Kangaroo policy v4
Focus: chain second-ladder climbs after dismount.
- Two-tier ladder selection: strict reach, then relaxed reach.
- Memoryless exclusion of just-climbed ladder via lty >= py - margin.
- Directional dismount toward next candidate ladder.
- Directed walk toward nearest active upward ladder when none reachable.
- Reduced height bias so x-distance dominates next-ladder choice.
"""

import jax
import jax.numpy as jnp

NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = 6, 7, 8, 9
UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE = 10, 11, 12, 13


def init_params():
    return {
        "reach_tol": jnp.array(16.0),
        "reach_relax": jnp.array(1.8),
        "align_tol": jnp.array(6.0),
        "on_pad": jnp.array(3.0),
        "danger_r": jnp.array(18.0),
        "punch_dx": jnp.array(20.0),
        "height_weight": jnp.array(0.3),
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


def _score_ladders(obs, pcx, pby, py, reach_tol, height_weight):
    """Return per-ladder cost under strict and relaxed reach windows."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh

    bottom_diff = jnp.abs(lby - pby)
    active = la > 0.5
    # useful: ladder top must be meaningfully above current head
    useful = lty < (py - 6.0)
    # exclude just-climbed: its top is at/above current y (we already used it)
    not_just_climbed = lty < (py - 2.0)

    base_valid = active & useful & not_just_climbed

    dx = jnp.abs(lcx - pcx)
    height_gap = jnp.maximum(py - lty, 0.0)
    # cap height bonus so x-distance dominates next-ladder picks
    capped_h = jnp.minimum(height_gap, 30.0)
    cost_base = dx - height_weight * capped_h

    return lcx, lty, lby, dx, bottom_diff, base_valid, cost_base


def _select_target(lcx, dx, bottom_diff, base_valid, cost_base,
                   reach_tol, reach_relax):
    strict = base_valid & (bottom_diff < reach_tol)
    relaxed = base_valid & (bottom_diff < reach_tol * reach_relax)

    cost_s = jnp.where(strict, cost_base, jnp.array(1e6))
    cost_r = jnp.where(relaxed, cost_base, jnp.array(1e6))
    cost_any = jnp.where(base_valid, dx, jnp.array(1e6))

    idx_s = jnp.argmin(cost_s)
    idx_r = jnp.argmin(cost_r)
    idx_a = jnp.argmin(cost_any)

    has_s = jnp.any(strict)
    has_r = jnp.any(relaxed)
    has_a = jnp.any(base_valid)

    idx = jnp.where(has_s, idx_s, jnp.where(has_r, idx_r, idx_a))
    return idx, has_s, has_r, has_a


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

    overhead_fc = (jnp.abs(fcx - pcx) < danger_r) & (fcy < py) & (py - fcy < danger_r * 2.0) & (fca > 0.5)
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

    reach_tol = params["reach_tol"]
    reach_relax = params["reach_relax"]
    align_tol = params["align_tol"]
    on_pad = params["on_pad"]
    danger_r = params["danger_r"]
    punch_dx = params["punch_dx"]
    height_weight = params["height_weight"]

    # Currently inside a ladder column?
    on_column, on_top_y = _on_ladder_column(obs_flat, pcx, py, pby, on_pad)
    near_top = on_column & ((py - on_top_y) < 8.0)

    # Score and pick next ladder
    lcx_arr, lty_arr, lby_arr, dx_arr, bdiff_arr, base_valid, cost_base = \
        _score_ladders(obs_flat, pcx, pby, py, reach_tol, height_weight)
    idx, has_strict, has_relaxed, has_any = _select_target(
        lcx_arr, dx_arr, bdiff_arr, base_valid, cost_base, reach_tol, reach_relax
    )
    target_lcx = lcx_arr[idx]

    # Has reachable ladder (strict OR relaxed)
    has_reach = has_strict | has_relaxed

    # Fallback: walk toward any active upward ladder if none reachable
    walk_target = jnp.where(has_any, target_lcx, jnp.array(80.0))
    target_x = jnp.where(has_reach, target_lcx, walk_target)
    dx_to_target = target_x - pcx

    aligned = has_reach & (jnp.abs(target_lcx - pcx) < align_tol)

    horiz_action = _move_toward_x(dx_to_target, 1.0)

    # Threats
    any_monkey, monkey_sign_dx, overhead_fc, fc_dx, any_thrown, thrown_sign_dx = \
        _threats(obs_flat, pcx, py, danger_r)

    # --- Compose action by priority (low -> high) ---
    action = horiz_action

    # Aligned at ladder bottom -> climb
    action = jnp.where(aligned, UP, action)

    # Inside column and not at top -> keep climbing
    action = jnp.where(on_column & ~near_top, UP, action)

    # At top -> dismount toward next candidate (or center if none)
    dismount_target = jnp.where(has_any, target_lcx, jnp.array(80.0))
    dismount_dx = dismount_target - pcx
    dismount_dir = _move_toward_x(dismount_dx, 1.0)
    dismount_dir = jnp.where(dismount_dir == NOOP, RIGHT, dismount_dir)
    action = jnp.where(near_top, dismount_dir, action)

    # Punch nearby monkey
    punch_action = jnp.where(monkey_sign_dx > 0, RIGHTFIRE, LEFTFIRE)
    punch_in_range = any_monkey & (jnp.abs(monkey_sign_dx) < punch_dx)
    action = jnp.where(punch_in_range, punch_action, action)

    # Thrown coconut: step away (perpendicular to its dx)
    dodge_dir = jnp.where(thrown_sign_dx > 0, LEFT, RIGHT)
    action = jnp.where(any_thrown & ~on_column, dodge_dir, action)

    # Overhead falling coconut: sidestep off-column, UPFIRE on-column
    fc_step = jnp.where(fc_dx > 0, LEFT, RIGHT)
    action = jnp.where(overhead_fc & ~on_column, fc_step, action)
    action = jnp.where(overhead_fc & on_column, UPFIRE, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
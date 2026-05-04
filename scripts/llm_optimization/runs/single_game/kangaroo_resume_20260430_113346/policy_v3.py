"""
Auto-generated policy v3
Generated at: 2026-04-30 11:37:56
"""

"""
Kangaroo policy v3
Fixes: chain ladder climbs across platforms, exclude just-climbed ladder,
prefer taller useful ladders, stronger dismount, FIRE-jump to bridge gaps,
proactive punching of nearby monkeys.
"""

import jax
import jax.numpy as jnp

# Action constants
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = 6, 7, 8, 9
UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE = 10, 11, 12, 13


def init_params():
    return {
        "reach_tol": jnp.array(16.0),       # symmetric |lby - pby| < reach_tol
        "align_tol": jnp.array(6.0),        # x tol to start climbing
        "on_pad": jnp.array(3.0),           # x padding around ladder column
        "danger_r": jnp.array(18.0),        # hazard radius (used for both dx and dy)
        "punch_dx": jnp.array(20.0),        # punch range (>= danger_r so it fires)
        "height_weight": jnp.array(0.6),    # cost: dx - height_weight * height_gap
        "jump_gap": jnp.array(22.0),        # FIRE-jump if ladder is borderline reachable
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


def _select_ladder(obs, pcx, pby, py, on_column, reach_tol, height_weight, on_pad):
    """Pick best reachable upward ladder, excluding the column the player is in."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh

    # Symmetric reach window (slightly looser below feet)
    bottom_diff = jnp.abs(lby - pby)
    reachable = bottom_diff < reach_tol

    # Borderline reachable (for FIRE-jump bridging)
    borderline = bottom_diff < (reach_tol * 1.5)

    # Useful: top is above current head (real upward progress)
    useful = lty < (py - 4.0)
    active = la > 0.5

    # Exclude the ladder whose column currently contains the player
    in_col = (pcx >= lx - on_pad) & (pcx <= lx + lw + on_pad)
    not_current = ~(in_col & on_column)

    valid = reachable & useful & active & not_current
    valid_jump = borderline & useful & active & not_current

    dx = jnp.abs(lcx - pcx)
    height_gap = jnp.maximum(py - lty, 0.0)
    cost_base = dx - height_weight * height_gap

    cost = jnp.where(valid, cost_base, jnp.array(1e6))
    idx = jnp.argmin(cost)

    cost_jump = jnp.where(valid_jump, cost_base, jnp.array(1e6))
    jdx = jnp.argmin(cost_jump)

    has_valid = jnp.any(valid)
    has_jump = jnp.any(valid_jump)

    # Prefer strict reachable; fall back to borderline target
    sel_idx = jnp.where(has_valid, idx, jdx)
    return lcx[sel_idx], lty[sel_idx], lby[sel_idx], has_valid, has_jump


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

    return any_monkey, monkey_sign_dx, overhead_fc, fc_dx, any_thrown


def policy(obs_flat, params):
    px = obs_flat[0]
    py = obs_flat[1]
    pw = obs_flat[2]
    ph = obs_flat[3]
    orient = obs_flat[7]
    pcx = px + pw * 0.5
    pby = py + ph

    reach_tol = params["reach_tol"]
    align_tol = params["align_tol"]
    on_pad = params["on_pad"]
    danger_r = params["danger_r"]
    punch_dx = params["punch_dx"]
    height_weight = params["height_weight"]
    jump_gap = params["jump_gap"]

    # Are we currently inside a ladder column?
    on_column, on_top_y = _on_ladder_column(obs_flat, pcx, py, pby, on_pad)
    near_top = on_column & ((py - on_top_y) < 8.0)

    # Pick next ladder, excluding current column
    lcx, lty, lby, has_valid, has_jump = _select_ladder(
        obs_flat, pcx, pby, py, on_column, reach_tol, height_weight, on_pad
    )

    # Sweep fallback: deterministic horizontal sweep based on x position
    # Move right unless near right edge; this avoids aimless drift toward child_x.
    sweep_target = jnp.where(pcx < 80.0, pcx + 40.0, pcx - 40.0)
    target_x = jnp.where(has_valid | has_jump, lcx, sweep_target)
    dx_to_target = target_x - pcx

    # Aligned with selected ladder bottom -> climb
    aligned = (has_valid | has_jump) & (jnp.abs(lcx - pcx) < align_tol)

    # FIRE-jump to bridge a borderline gap when aligned but not strictly reachable
    needs_jump = aligned & ~has_valid & has_jump

    # Horizontal traversal action
    horiz_action = _move_toward_x(dx_to_target, 1.0)

    # Threats
    any_monkey, monkey_sign_dx, overhead_fc, fc_dx, any_thrown = _threats(
        obs_flat, pcx, py, danger_r
    )

    # --- Build action with priority order ---
    # Default: horizontal traversal toward target
    action = horiz_action

    # If aligned with ladder at ground -> climb
    action = jnp.where(aligned & has_valid, UP, action)

    # If on column and not near top -> keep climbing
    action = jnp.where(on_column & ~near_top, UP, action)

    # Near top of column -> dismount horizontally (force a step off column)
    dismount_dir = _move_toward_x(dx_to_target, 1.0)
    dismount_dir = jnp.where(dismount_dir == NOOP, RIGHT, dismount_dir)
    action = jnp.where(near_top, dismount_dir, action)

    # Punch nearby monkey (works on or off column)
    punch_action = jnp.where(orient > 180.0, LEFTFIRE, RIGHTFIRE)
    # Override with monkey direction if known
    punch_action = jnp.where(monkey_sign_dx > 0, RIGHTFIRE, LEFTFIRE)
    punch_in_range = any_monkey & (jnp.abs(monkey_sign_dx) < punch_dx)
    action = jnp.where(punch_in_range, punch_action, action)

    # Thrown coconut nearby and not climbing -> jump
    action = jnp.where(any_thrown & ~on_column, FIRE, action)

    # Overhead falling coconut: sidestep if off column, UPFIRE if on column
    fc_step = jnp.where(fc_dx > 0, LEFT, RIGHT)
    action = jnp.where(overhead_fc & ~on_column, fc_step, action)
    action = jnp.where(overhead_fc & on_column, UPFIRE, action)

    # FIRE-jump to bridge ladder reach gap (only when no immediate hazard handled)
    action = jnp.where(needs_jump & ~any_thrown & ~overhead_fc, FIRE, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
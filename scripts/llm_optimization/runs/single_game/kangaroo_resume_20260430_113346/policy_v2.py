"""
Auto-generated policy v2
Generated at: 2026-04-29 16:08:17
"""

import jax
import jax.numpy as jnp

# Action constants
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = 6, 7, 8, 9
UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE = 10, 11, 12, 13


def init_params():
    return {
        "ladder_reach_tol": jnp.array(14.0),    # ladder bottom near feet tolerance
        "ladder_align_tol": jnp.array(5.0),     # x tol to start climbing from ground
        "on_ladder_pad": jnp.array(3.0),        # x padding around ladder column
        "danger_dx": jnp.array(16.0),           # horizontal danger distance
        "danger_dy": jnp.array(14.0),           # vertical danger distance
        "punch_dx": jnp.array(14.0),            # punch range
        "upward_bias": jnp.array(10.0),         # ladder must be at least this much higher
    }


def _move_toward_x(dx, tol):
    # Returns RIGHT/LEFT/NOOP given signed dx and a small deadband tol
    right = dx > tol
    left = dx < -tol
    a = jnp.where(right, RIGHT, jnp.where(left, LEFT, NOOP))
    return a


def _ladders(obs):
    lx = jax.lax.dynamic_slice(obs, (168,), (20,))
    ly = jax.lax.dynamic_slice(obs, (188,), (20,))
    lw = jax.lax.dynamic_slice(obs, (208,), (20,))
    lh = jax.lax.dynamic_slice(obs, (228,), (20,))
    la = jax.lax.dynamic_slice(obs, (248,), (20,))
    return lx, ly, lw, lh, la


def _select_ladder(obs, pcx, pby, py, reach_tol, upward_bias):
    """Pick a ladder whose bottom is near or below feet and whose top is well above."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lby = ly + lh
    lty = ly

    # Asymmetric reach: ladder bottom at or below feet (allow small overshoot)
    bottom_ok = (lby >= pby - reach_tol * 0.5) & (lby <= pby + reach_tol)
    # Useful: top is above current player y by more than upward_bias
    useful = lty < (py - upward_bias)
    active = la > 0.5
    valid = bottom_ok & useful & active

    dx = jnp.abs(lcx - pcx)
    height_gap = jnp.maximum(py - lty, 0.0)
    # Smooth cost: prefer close in x, with mild preference for taller climbs
    cost = jnp.where(valid, dx - 0.05 * height_gap, jnp.array(1e6))
    idx = jnp.argmin(cost)
    has_valid = jnp.any(valid)
    return lcx[idx], lty[idx], lby[idx], has_valid


def _on_ladder_column(obs, pcx, py, pby, pad):
    """True if player x is inside any active ladder column with top above and bottom below feet."""
    lx, ly, lw, lh, la = _ladders(obs)
    lty = ly
    lby = ly + lh
    in_x = (pcx >= lx - pad) & (pcx <= lx + lw + pad)
    top_above = lty < py + 2.0  # ladder extends above the player head
    bottom_at_or_below = lby >= pby - 4.0
    active = la > 0.5
    on = in_x & top_above & bottom_at_or_below & active
    any_on = jnp.any(on)
    # Also report the top-y of the ladder we're on (min top among matches)
    top_for = jnp.where(on, lty, jnp.array(1e6))
    nearest_top = jnp.min(top_for)
    return any_on, nearest_top


def _threats(obs, pcx, py, danger_dx, danger_dy):
    mx = jax.lax.dynamic_slice(obs, (376,), (4,))
    my = jax.lax.dynamic_slice(obs, (380,), (4,))
    ma = jax.lax.dynamic_slice(obs, (392,), (4,))
    cx = jax.lax.dynamic_slice(obs, (408,), (4,))
    cy = jax.lax.dynamic_slice(obs, (412,), (4,))
    ca = jax.lax.dynamic_slice(obs, (424,), (4,))
    fcx = obs[368]
    fcy = obs[369]
    fca = obs[372]

    # Monkey punch detection
    mdx = mx - pcx
    mdy = my - py
    monkey_near = (jnp.abs(mdx) < danger_dx) & (jnp.abs(mdy) < danger_dy) & (ma > 0.5)
    any_monkey = jnp.any(monkey_near)
    mcost = jnp.where(monkey_near, jnp.abs(mdx), jnp.array(1e6))
    midx = jnp.argmin(mcost)
    monkey_sign_dx = mdx[midx]

    # Overhead falling coconut: x near player, y above and within reach
    overhead_fc = (jnp.abs(fcx - pcx) < danger_dx) & (fcy < py) & (py - fcy < danger_dy * 2.0) & (fca > 0.5)

    # Thrown coconuts close in both axes
    tdx = cx - pcx
    tdy = cy - py
    thrown_near = (jnp.abs(tdx) < danger_dx) & (jnp.abs(tdy) < danger_dy) & (ca > 0.5)
    any_thrown = jnp.any(thrown_near)
    tcost = jnp.where(thrown_near, jnp.abs(tdx), jnp.array(1e6))
    tidx = jnp.argmin(tcost)
    thrown_sign_dx = tdx[tidx]

    return any_monkey, monkey_sign_dx, overhead_fc, any_thrown, thrown_sign_dx


def policy(obs_flat, params):
    px = obs_flat[0]
    py = obs_flat[1]
    pw = obs_flat[2]
    ph = obs_flat[3]
    pcx = px + pw * 0.5
    pby = py + ph

    reach_tol = params["ladder_reach_tol"]
    align_tol = params["ladder_align_tol"]
    on_pad = params["on_ladder_pad"]
    danger_dx = params["danger_dx"]
    danger_dy = params["danger_dy"]
    punch_dx = params["punch_dx"]
    upward_bias = params["upward_bias"]

    # Currently on a ladder column? (state inferred from geometry)
    on_column, on_top_y = _on_ladder_column(obs_flat, pcx, py, pby, on_pad)
    # If close to ladder top, we should dismount horizontally
    near_top = on_column & ((py - on_top_y) < 6.0)

    # Pick a target ladder for traversal
    lcx, lty, lby, has_valid = _select_ladder(obs_flat, pcx, pby, py, reach_tol, upward_bias)

    # Fallback target if no valid ladder: drift toward child x to explore upper level
    child_x = obs_flat[360]
    child_active = obs_flat[364]
    fallback_x = jnp.where(child_active > 0.5, child_x, pcx + 30.0)
    target_x = jnp.where(has_valid, lcx, fallback_x)

    dx_to_target = target_x - pcx

    # --- Action priorities ---
    # 1) Climbing: if on ladder column and not at top, force UP.
    climb_action = UP

    # 2) Aligned with ladder bottom on ground: start climbing.
    aligned_with_ladder = has_valid & (jnp.abs(lcx - pcx) < align_tol)

    # 3) Horizontal traversal toward target.
    horiz_action = _move_toward_x(dx_to_target, 1.0)

    # Threats
    any_monkey, monkey_sign_dx, overhead_fc, any_thrown, thrown_sign_dx = _threats(
        obs_flat, pcx, py, danger_dx, danger_dy
    )

    # Default base action
    base = horiz_action
    base = jnp.where(aligned_with_ladder, climb_action, base)
    base = jnp.where(on_column & ~near_top, climb_action, base)
    # When near top and on column, prefer dismount toward target
    dismount = _move_toward_x(dx_to_target, 1.0)
    dismount = jnp.where(dismount == NOOP, RIGHT, dismount)
    base = jnp.where(near_top, dismount, base)

    # Punch monkey if in range
    punch_in_range = any_monkey & (jnp.abs(monkey_sign_dx) < punch_dx)
    punch_action = jnp.where(monkey_sign_dx > 0, RIGHTFIRE, LEFTFIRE)

    # Overhead falling coconut: if on column, UPFIRE to keep climbing-punch; else jump aside
    fc_on_climb = overhead_fc & on_column
    fc_off_climb = overhead_fc & ~on_column
    fc_action_climb = UPFIRE
    # sidestep: move away from coconut x
    fc_dx = obs_flat[368] - pcx
    fc_action_step = jnp.where(fc_dx > 0, LEFT, RIGHT)

    # Thrown coconut nearby off ladder: jump (FIRE) to dodge
    thrown_off_climb = any_thrown & ~on_column
    thrown_action = FIRE

    action = base
    action = jnp.where(thrown_off_climb, thrown_action, action)
    action = jnp.where(fc_off_climb, fc_action_step, action)
    action = jnp.where(fc_on_climb, fc_action_climb, action)
    action = jnp.where(punch_in_range & ~on_column, punch_action, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
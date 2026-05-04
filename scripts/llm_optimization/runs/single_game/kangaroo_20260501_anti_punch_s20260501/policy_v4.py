"""
Auto-generated policy v4
Generated at: 2026-05-01 14:02:49
"""

"""
Kangaroo policy v7
Fixes post-200 stall by:
- proper center-band on-ladder test (not 1px overlap),
- recent-column x-exclusion so just-climbed ladder is not reselected,
- post-climb forced horizontal commitment toward a different ladder,
- preserved first-reward route via unchanged starting reach/punch logic.
"""

import jax
import jax.numpy as jnp

NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = 6, 7, 8, 9
UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE = 10, 11, 12, 13


def init_params():
    return {
        "reach_tol": jnp.array(15.5),
        "align_tol": jnp.array(6.0),
        "danger_r": jnp.array(18.2),
        "punch_dx": jnp.array(19.5),
        "center_band": jnp.array(4.0),    # how close pcx must be to ladder center to count as on-ladder
        "recent_pad": jnp.array(16.0),    # x-exclusion radius around current x for next-ladder pick
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


def _is_on_ladder(obs, pcx, py, pby, center_band):
    """Center-band test. Player is on ladder iff its center is within
    [lcx-center_band, lcx+center_band] AND vertical overlap exists."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh
    near_center = jnp.abs(pcx - lcx) < center_band
    vertical_in = (pby >= lty - 2.0) & (py <= lby + 2.0)
    active = la > 0.5
    on = near_center & vertical_in & active
    any_on = jnp.any(on)
    top_for = jnp.where(on, lty, jnp.array(1e6))
    nearest_top = jnp.min(top_for)
    return any_on, nearest_top


def _reachable_ladder(obs, pcx, pby, py, reach_tol, height_weight, exclude_x, exclude_pad):
    """Pick best upward reachable ladder, excluding an x-band around exclude_x."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh

    diff = jnp.abs(lby - pby)
    reachable = diff < reach_tol
    useful = lty < (py - 4.0)
    active = la > 0.5
    in_excl = jnp.abs(lcx - exclude_x) < exclude_pad

    valid = reachable & useful & active & ~in_excl
    dx = jnp.abs(lcx - pcx)
    height_gap = jnp.maximum(py - lty, 0.0)
    cost = jnp.where(valid, dx - height_weight * height_gap, jnp.array(1e6))
    idx = jnp.argmin(cost)
    has_valid = jnp.any(valid)
    return lcx[idx], lty[idx], has_valid


def _next_platform_ladder(obs, pcx, py, exclude_x, exclude_pad):
    """Any upward active ladder excluding x-band around exclude_x. No reach test."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    useful = lty < (py - 4.0)
    active = la > 0.5
    in_excl = jnp.abs(lcx - exclude_x) < exclude_pad
    valid = useful & active & ~in_excl
    dx = jnp.abs(lcx - pcx)
    cost = jnp.where(valid, dx, jnp.array(1e6))
    idx = jnp.argmin(cost)
    has_valid = jnp.any(valid)
    return lcx[idx], has_valid


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
    align_tol = params["align_tol"]
    danger_r = params["danger_r"]
    punch_dx = params["punch_dx"]
    center_band = params["center_band"]
    recent_pad = params["recent_pad"]
    height_weight = params["height_weight"]

    # On-ladder test using center-band (not 1px overlap)
    on_ladder, on_top_y = _is_on_ladder(obs_flat, pcx, py, pby, center_band)
    near_top = on_ladder & ((py - on_top_y) < 6.0)

    # Primary: reachable ladder, no exclusion (to preserve first-reward path)
    lcx, lty, has_valid = _reachable_ladder(
        obs_flat, pcx, pby, py, reach_tol, height_weight,
        exclude_x=jnp.array(-1000.0), exclude_pad=jnp.array(0.0),
    )

    # Post-climb selection: exclude current x-band so we don't pick the just-used ladder
    next_lcx, has_next = _next_platform_ladder(
        obs_flat, pcx, py, exclude_x=pcx, exclude_pad=recent_pad,
    )

    # Default horizontal action toward primary ladder
    target_x = jnp.where(has_valid, lcx, pcx + jnp.sign(child_x - pcx) * 30.0)
    horiz_action = _move_toward_x(target_x - pcx, 1.0)

    aligned = has_valid & (jnp.abs(lcx - pcx) < align_tol)

    # Threats
    any_monkey, monkey_sign_dx, overhead_fc, fc_dx, any_thrown, thrown_sign_dx = _threats(
        obs_flat, pcx, py, danger_r
    )

    # --- Build action ---
    action = horiz_action

    # Aligned with reachable ladder bottom -> climb
    action = jnp.where(aligned, UP, action)

    # On ladder column and not near top -> keep climbing
    action = jnp.where(on_ladder & ~near_top, UP, action)

    # Near top of ladder -> dismount toward next ladder (excluding current col)
    dismount_x = jnp.where(has_next, next_lcx, pcx + jnp.sign(child_x - pcx) * 40.0)
    dismount_dir = _move_toward_x(dismount_x - pcx, 1.0)
    dismount_dir = jnp.where(
        dismount_dir == NOOP,
        jnp.where(child_x < pcx, LEFT, RIGHT),
        dismount_dir,
    )
    action = jnp.where(near_top, dismount_dir, action)

    # POST-CLIMB COMMIT: when not on ladder and a next-ladder target exists,
    # bias horizontal action toward it (overrides re-targeting just-used ladder).
    post_climb_dir = _move_toward_x(next_lcx - pcx, 1.0)
    use_post = (~on_ladder) & has_next & (~aligned)
    action = jnp.where(use_post, post_climb_dir, action)

    # Punch nearby monkey (off-ladder only to avoid wasting climb frames)
    punch_action = jnp.where(monkey_sign_dx > 0, RIGHTFIRE, LEFTFIRE)
    punch_in_range = any_monkey & (jnp.abs(monkey_sign_dx) < punch_dx) & ~on_ladder
    action = jnp.where(punch_in_range, punch_action, action)

    # Thrown coconut: off-ladder commit escape away from it
    escape_dir = jnp.where(thrown_sign_dx > 0, LEFT, RIGHT)
    action = jnp.where(any_thrown & ~on_ladder, escape_dir, action)
    # Thrown coconut on ladder -> hold
    action = jnp.where(any_thrown & on_ladder & ~near_top, NOOP, action)

    # Overhead falling coconut: sidestep when off ladder, pause on ladder
    fc_step = jnp.where(fc_dx > 0, LEFT, RIGHT)
    action = jnp.where(overhead_fc & ~on_ladder, fc_step, action)
    action = jnp.where(overhead_fc & on_ladder & ~near_top, NOOP, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
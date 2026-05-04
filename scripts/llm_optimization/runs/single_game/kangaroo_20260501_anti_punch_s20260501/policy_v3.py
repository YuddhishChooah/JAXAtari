"""
Auto-generated policy v3
Generated at: 2026-05-01 13:56:48
"""

"""
Kangaroo policy v7
Fixes post-200 stall by:
- stricter center-band on-ladder test (no 1-pixel phantom climbs),
- distinct dismount selector that excludes the just-climbed ladder
  by requiring the next ladder's top to be above current py by a margin,
- gated near-top latch: prefer horizontal traversal until aligned with
  a different ladder column,
- coconut-on-ladder response steps off instead of freezing,
- preserves first-reward route (only post-climb behavior is changed).
"""

import jax
import jax.numpy as jnp

NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = 6, 7, 8, 9
UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE = 10, 11, 12, 13

START_Y = 148.0


def init_params():
    return {
        "reach_tol": jnp.array(16.0),
        "align_tol": jnp.array(5.0),
        "center_band": jnp.array(3.0),
        "danger_r": jnp.array(18.2),
        "punch_dx": jnp.array(19.5),
        "height_weight": jnp.array(1.1),
        "feet_window": jnp.array(28.0),
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


def _is_on_ladder(obs, pcx, py, pby, center_band, feet_window):
    """Strict center-band on-ladder test."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh
    centered = jnp.abs(pcx - lcx) < center_band
    feet_in = (pby > lty - 4.0) & (pby < lby + feet_window)
    active = la > 0.5
    on = centered & feet_in & active
    any_on = jnp.any(on)
    top_for = jnp.where(on, lty, jnp.array(1e6))
    nearest_top = jnp.min(top_for)
    return any_on, nearest_top


def _select_ladder(obs, pcx, pby, py, reach_tol, height_weight, exclude_current_top_y):
    """Pick best reachable upward ladder, excluding the one we just climbed.

    exclude_current_top_y: if a ladder's top is near py (i.e., the ladder we are
    standing on top of), exclude it. Set to a large negative value to disable.
    """
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh

    # Reachable: ladder bottom near our feet (within reach_tol)
    diff = lby - pby
    reachable = jnp.abs(diff) < reach_tol

    # Useful: top is meaningfully above us
    useful = lty < (py - 6.0)
    active = la > 0.5

    # Exclude the ladder we just dismounted from: its top is at ~ py.
    not_just_climbed = jnp.abs(lty - exclude_current_top_y) > 6.0

    valid = reachable & useful & active & not_just_climbed
    dx = jnp.abs(lcx - pcx)
    height_gap = jnp.maximum(py - lty, 0.0)
    cost = jnp.where(valid, dx - height_weight * height_gap, jnp.array(1e6))
    idx = jnp.argmin(cost)
    has_valid = jnp.any(valid)

    # Fallback: nearest useful active ladder (still excluding just-climbed)
    fb_valid = useful & active & not_just_climbed
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
    align_tol = params["align_tol"]
    center_band = params["center_band"]
    danger_r = params["danger_r"]
    punch_dx = params["punch_dx"]
    height_weight = params["height_weight"]
    feet_window = params["feet_window"]

    # Strict on-ladder test (center-band, not pixel overlap)
    on_column, on_top_y = _is_on_ladder(
        obs_flat, pcx, py, pby, center_band, feet_window
    )
    near_top = on_column & ((py - on_top_y) < 10.0)

    # Are we on the starting platform? Preserve the working first-reward route.
    on_start_platform = py > (START_Y - 4.0)

    # Primary selector: don't exclude any ladder (start route relies on this).
    big_neg = jnp.array(-1e6)
    p_lcx, p_lty, p_has_valid, p_has_fb = _select_ladder(
        obs_flat, pcx, pby, py, reach_tol, height_weight, big_neg
    )

    # Dismount selector: exclude the ladder we just climbed (top near current py).
    d_lcx, d_lty, d_has_valid, d_has_fb = _select_ladder(
        obs_flat, pcx, pby, py, reach_tol, height_weight, py
    )

    # Default horizontal target = primary ladder (or child-bias fallback)
    child_bias = pcx + jnp.sign(child_x - pcx) * 30.0
    primary_target = jnp.where(p_has_valid | p_has_fb, p_lcx, child_bias)
    horiz_action = _move_toward_x(primary_target - pcx, 1.0)

    # Aligned with reachable primary ladder -> climb
    aligned_primary = p_has_valid & (jnp.abs(p_lcx - pcx) < align_tol)

    # Threats
    any_monkey, monkey_sign_dx, overhead_fc, fc_dx, any_thrown = _threats(
        obs_flat, pcx, py, danger_r
    )

    # --- Build action ---
    action = horiz_action

    # Climb when aligned with a reachable ladder
    action = jnp.where(aligned_primary, UP, action)

    # On a ladder column: keep climbing unless near top
    action = jnp.where(on_column & ~near_top, UP, action)

    # Near top: traverse toward the *next* ladder (excluding just-climbed)
    dismount_target = jnp.where(d_has_valid | d_has_fb, d_lcx, child_bias)
    dismount_dir = _move_toward_x(dismount_target - pcx, 1.0)
    dismount_dir = jnp.where(
        dismount_dir == NOOP,
        jnp.where(child_x < pcx, LEFT, RIGHT),
        dismount_dir,
    )
    action = jnp.where(near_top, dismount_dir, action)

    # Post-climb latch: if we are above the start platform AND not on a ladder,
    # commit to horizontal traversal to dismount target until aligned.
    above_start = py < (START_Y - 6.0)
    on_upper_platform = above_start & ~on_column
    upper_target = jnp.where(d_has_valid | d_has_fb, d_lcx, child_bias)
    upper_dir = _move_toward_x(upper_target - pcx, 1.0)
    aligned_upper = d_has_valid & (jnp.abs(d_lcx - pcx) < align_tol)
    # On upper platform, walk to next ladder; only climb when aligned with it.
    action = jnp.where(on_upper_platform & ~aligned_upper, upper_dir, action)
    action = jnp.where(on_upper_platform & aligned_upper, UP, action)

    # Punch nearby monkey
    punch_action = jnp.where(monkey_sign_dx > 0, RIGHTFIRE, LEFTFIRE)
    punch_in_range = any_monkey & (jnp.abs(monkey_sign_dx) < punch_dx)
    action = jnp.where(punch_in_range, punch_action, action)

    # Thrown coconut nearby off-ladder -> jump
    action = jnp.where(any_thrown & ~on_column, FIRE, action)
    # Thrown coconut while on ladder -> step off in the safe direction
    step_off_thrown = jnp.where(pcx > 80.0, LEFT, RIGHT)
    action = jnp.where(any_thrown & on_column, step_off_thrown, action)

    # Overhead falling coconut: sidestep off column or step off ladder
    fc_step = jnp.where(fc_dx > 0, LEFT, RIGHT)
    action = jnp.where(overhead_fc & ~on_column, fc_step, action)
    action = jnp.where(overhead_fc & on_column, fc_step, action)

    # Preserve start-platform behavior: never let upper-platform branches override
    # when we are still on the starting platform.
    start_action = jnp.where(aligned_primary, UP, horiz_action)
    start_action = jnp.where(punch_in_range, punch_action, start_action)
    start_action = jnp.where(any_thrown, FIRE, start_action)
    start_action = jnp.where(overhead_fc, fc_step, start_action)
    action = jnp.where(on_start_platform, start_action, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
"""Second manual policy_v6 patch for Kangaroo ladder perception.

Patch 1 fixed the premature climb trigger by requiring real player/ladder
overlap, but it then got stuck at the ladder top because route selection still
chose the same ladder column as the next target. This patch keeps the stricter
climb perception and adds a separate route-exclusion tolerance so dismount
planning ignores ladders in the current x-column.
"""

import jax.numpy as jnp

from scripts.llm_optimization.experiments.kangaroo.policy_v6_overlap_patch import (
    DOWN,
    DOWNFIRE,
    DOWNLEFT,
    DOWNRIGHT,
    FIRE,
    LEFT,
    LEFTFIRE,
    NOOP,
    RIGHT,
    RIGHTFIRE,
    UP,
    UPFIRE,
    UPLEFT,
    UPRIGHT,
    _ladder_match_mask,
    _ladders,
    _move_toward_x,
    _on_ladder_column,
    _target_ladder_aligned,
    _threats,
    init_params as _base_init_params,
    measure_main,
)


def init_params():
    params = dict(_base_init_params())
    # Route planning uses this wider x-column exclusion only to avoid selecting
    # the current ladder stack as the dismount target. Climb detection remains
    # controlled by min_ladder_overlap_frac and ladder_center_tol.
    params["route_exclude_x_tol"] = jnp.array(14.0)
    return params


def _select_best(
    obs,
    px,
    pw,
    pcx,
    pby,
    py,
    reach_tol,
    reach_up,
    height_weight,
    min_overlap_frac,
    center_tol,
    bottom_slack,
    route_exclude_x_tol,
):
    """Best upward ladder, excluding the current ladder stack for routing."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh

    diff = lby - pby
    reachable = (diff > -reach_up) & (diff < reach_tol)
    useful = lty < (py - 4.0)
    active = la > 0.5
    current_ladder = _ladder_match_mask(
        obs, px, pw, pcx, py, pby, min_overlap_frac, center_tol, bottom_slack
    )
    current_route_column = jnp.abs(lcx - pcx) <= route_exclude_x_tol
    exclude_for_route = current_ladder | current_route_column

    valid = reachable & useful & active & ~exclude_for_route
    dx = jnp.abs(lcx - pcx)
    height_gap = jnp.maximum(py - lty, 0.0)
    cost = jnp.where(valid, dx - height_weight * height_gap, jnp.array(1e6))
    idx = jnp.argmin(cost)
    has_valid = jnp.any(valid)

    fb_valid = useful & active & ~exclude_for_route
    fb_cost = jnp.where(fb_valid, dx, jnp.array(1e6))
    fb_idx = jnp.argmin(fb_cost)
    has_fb = jnp.any(fb_valid)

    sel_idx = jnp.where(has_valid, idx, fb_idx)
    return lcx[sel_idx], lty[sel_idx], has_valid, has_fb


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
    danger_r = params["danger_r"]
    punch_dx = params["punch_dx"]
    height_weight = params["height_weight"]
    min_overlap_frac = params["min_ladder_overlap_frac"]
    center_tol = params["ladder_center_tol"]
    bottom_slack = params["ladder_bottom_slack"]
    route_exclude_x_tol = params["route_exclude_x_tol"]

    on_column, on_top_y = _on_ladder_column(
        obs_flat, px, pw, pcx, py, pby, min_overlap_frac, center_tol, bottom_slack
    )
    near_top = on_column & ((py - on_top_y) < 8.0)

    lcx, lty, has_valid, has_fb = _select_best(
        obs_flat,
        px,
        pw,
        pcx,
        pby,
        py,
        reach_tol,
        reach_up,
        height_weight,
        min_overlap_frac,
        center_tol,
        bottom_slack,
        route_exclude_x_tol,
    )
    d_lcx, d_lty, d_has_valid, d_has_fb = _select_best(
        obs_flat,
        px,
        pw,
        pcx,
        pby,
        py,
        reach_tol,
        reach_up,
        height_weight,
        min_overlap_frac,
        center_tol,
        bottom_slack,
        route_exclude_x_tol,
    )

    child_dir_x = pcx + jnp.sign(child_x - pcx) * 30.0
    target_x = jnp.where(has_valid | has_fb, lcx, child_dir_x)
    dx_to_target = target_x - pcx

    aligned = has_valid & _target_ladder_aligned(
        obs_flat, px, pw, pcx, lcx, min_overlap_frac, center_tol
    )
    horiz_action = _move_toward_x(dx_to_target, 1.0)

    any_monkey, monkey_sign_dx, overhead_fc, fc_dx, any_thrown = _threats(
        obs_flat, pcx, py, danger_r
    )

    action = horiz_action
    action = jnp.where(aligned, UP, action)
    action = jnp.where(on_column & ~near_top, UP, action)

    dismount_target = jnp.where(d_has_valid | d_has_fb, d_lcx, child_dir_x)
    dismount_dir = _move_toward_x(dismount_target - pcx, 1.0)
    dismount_dir = jnp.where(
        dismount_dir == NOOP,
        jnp.where(child_x < pcx, LEFT, RIGHT),
        dismount_dir,
    )
    action = jnp.where(near_top, dismount_dir, action)

    punch_action = jnp.where(monkey_sign_dx > 0, RIGHTFIRE, LEFTFIRE)
    punch_in_range = any_monkey & (jnp.abs(monkey_sign_dx) < punch_dx)
    action = jnp.where(punch_in_range, punch_action, action)

    action = jnp.where(any_thrown & ~on_column, FIRE, action)
    action = jnp.where(any_thrown & on_column & ~near_top, NOOP, action)

    fc_step = jnp.where(fc_dx > 0, LEFT, RIGHT)
    action = jnp.where(overhead_fc & ~on_column, fc_step, action)
    action = jnp.where(overhead_fc & on_column & ~near_top, NOOP, action)

    return action.astype(jnp.int32)

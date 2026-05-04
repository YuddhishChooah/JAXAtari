#!/usr/bin/env python
"""Render a Kangaroo rollout with policy-decision diagnostics.

The normal renderer shows what the agent does. This renderer adds a right-side
panel explaining why the saved policy chose each action: current action,
decision overrides, ladder-overlap measurements, and hazard/punch predicates.

The displayed "stack" is a logical policy-decision stack, not Python's runtime
call stack. The generated policies use JAX array operations, so all helper
branches are evaluated every step and then combined with jnp.where.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

import jaxatari
from jaxatari.wrappers import AtariWrapper, FlattenObservationWrapper, ObjectCentricWrapper

from render_policy_video import (
    ACTION_NAMES,
    PROJECT_ROOT,
    find_latest_kangaroo_run,
    jax_scalar_to_int,
    load_policy_module,
    load_run_artifacts,
    unwrap_render_state,
)


def as_float_params(params: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in params.items()}


def move_toward_x(dx: float, tol: float) -> int:
    if dx > tol:
        return 3  # RIGHT
    if dx < -tol:
        return 4  # LEFT
    return 0  # NOOP


def horizontal_overlap(left_a: float, right_a: float, left_b: float, right_b: float) -> float:
    return max(0.0, min(right_a, right_b) - max(left_a, left_b))


def ladder_arrays(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        obs[168:188],
        obs[188:208],
        obs[208:228],
        obs[228:248],
        obs[248:268],
    )


def select_best(
    obs: np.ndarray,
    *,
    px: float,
    pw: float,
    pcx: float,
    pby: float,
    py: float,
    reach_tol: float,
    reach_up: float,
    height_weight: float,
    exclude_pad: float,
    min_overlap_frac: float | None = None,
    center_tol: float | None = None,
    bottom_slack: float = 4.0,
    route_exclude_x_tol: float | None = None,
) -> dict[str, Any]:
    lx, ly, lw, lh, la = ladder_arrays(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh
    diff = lby - pby
    reachable = (diff > -reach_up) & (diff < reach_tol)
    useful = lty < (py - 4.0)
    active = la > 0.5
    if min_overlap_frac is None or center_tol is None:
        in_col = (pcx >= lx - exclude_pad) & (pcx <= lx + lw + exclude_pad)
    else:
        overlap_frac = np.array(
            [
                horizontal_overlap(px, px + pw, float(left), float(left + width))
                / max(min(float(pw), float(width)), 1.0)
                for left, width in zip(lx, lw)
            ],
            dtype=float,
        )
        in_col = (
            (overlap_frac >= min_overlap_frac)
            & (np.abs(lcx - pcx) <= center_tol)
            & (lty < py + 2.0)
            & (lby >= pby - bottom_slack)
            & active
        )
        if route_exclude_x_tol is not None:
            in_col = in_col | (np.abs(lcx - pcx) <= route_exclude_x_tol)

    valid = reachable & useful & active & ~in_col
    dx = np.abs(lcx - pcx)
    height_gap = np.maximum(py - lty, 0.0)
    cost = np.where(valid, dx - height_weight * height_gap, 1e6)
    valid_idx = int(np.argmin(cost))
    has_valid = bool(np.any(valid))

    fallback_valid = useful & active & ~in_col
    fallback_cost = np.where(fallback_valid, dx, 1e6)
    fallback_idx = int(np.argmin(fallback_cost))
    has_fallback = bool(np.any(fallback_valid))
    selected_idx = valid_idx if has_valid else fallback_idx

    return {
        "selected_index": selected_idx,
        "selected_lcx": float(lcx[selected_idx]),
        "selected_lty": float(lty[selected_idx]),
        "selected_lx": float(lx[selected_idx]),
        "selected_ly": float(ly[selected_idx]),
        "selected_lw": float(lw[selected_idx]),
        "selected_lh": float(lh[selected_idx]),
        "has_valid": has_valid,
        "has_fallback": has_fallback,
        "selected_reachable": bool(reachable[selected_idx]),
        "selected_useful": bool(useful[selected_idx]),
        "selected_in_col": bool(in_col[selected_idx]),
        "selected_bottom_diff": float(diff[selected_idx]),
        "selected_center_dx": float(lcx[selected_idx] - pcx),
    }


def on_ladder_column(
    obs: np.ndarray,
    *,
    px: float,
    py: float,
    pw: float,
    ph: float,
    pcx: float,
    pby: float,
    pad: float,
    min_overlap_frac: float | None = None,
    center_tol: float | None = None,
    bottom_slack: float = 4.0,
) -> dict[str, Any]:
    lx, ly, lw, lh, la = ladder_arrays(obs)
    lty = ly
    lby = ly + lh
    if min_overlap_frac is None or center_tol is None:
        in_x = (pcx >= lx - pad) & (pcx <= lx + lw + pad)
    else:
        overlap_frac = np.array(
            [
                horizontal_overlap(px, px + pw, float(left), float(left + width))
                / max(min(float(pw), float(width)), 1.0)
                for left, width in zip(lx, lw)
            ],
            dtype=float,
        )
        in_x = (overlap_frac >= min_overlap_frac) & (np.abs((lx + lw * 0.5) - pcx) <= center_tol)
    top_above = lty < py + 2.0
    bottom_below = lby >= pby - bottom_slack
    active = la > 0.5
    on = in_x & top_above & bottom_below & active
    any_on = bool(np.any(on))
    top_for = np.where(on, lty, 1e6)
    nearest_top = float(np.min(top_for))
    selected_idx = int(np.argmin(top_for)) if any_on else -1

    if selected_idx >= 0:
        overlap_px = horizontal_overlap(px, px + pw, float(lx[selected_idx]), float(lx[selected_idx] + lw[selected_idx]))
        overlap_player_frac = overlap_px / max(float(pw), 1.0)
        overlap_ladder_frac = overlap_px / max(float(lw[selected_idx]), 1.0)
        ladder = {
            "index": selected_idx,
            "x": float(lx[selected_idx]),
            "y": float(ly[selected_idx]),
            "w": float(lw[selected_idx]),
            "h": float(lh[selected_idx]),
            "center_dx": float((lx[selected_idx] + lw[selected_idx] * 0.5) - pcx),
            "bottom_diff": float(lby[selected_idx] - pby),
            "bbox_overlap_px": float(overlap_px),
            "bbox_overlap_player_frac": float(overlap_player_frac),
            "bbox_overlap_ladder_frac": float(overlap_ladder_frac),
        }
    else:
        ladder = None

    return {
        "on_column": any_on,
        "nearest_top": nearest_top,
        "selected_index": selected_idx,
        "ladder": ladder,
        "policy_in_x_count": int(np.sum(in_x & active)),
    }


def threat_debug(obs: np.ndarray, *, pcx: float, py: float, danger_r: float) -> dict[str, Any]:
    mx = obs[376:380]
    my = obs[380:384]
    ma = obs[392:396]
    cx = obs[408:412]
    cy = obs[412:416]
    ca = obs[424:428]
    fcx = float(obs[368])
    fcy = float(obs[369])
    fca = float(obs[372])

    mdx = mx - pcx
    mdy = my - py
    monkey_near = (np.abs(mdx) < danger_r) & (np.abs(mdy) < danger_r) & (ma > 0.5)
    any_monkey = bool(np.any(monkey_near))
    monkey_cost = np.where(monkey_near, np.abs(mdx), 1e6)
    monkey_idx = int(np.argmin(monkey_cost))
    monkey_sign_dx = float(mdx[monkey_idx])

    thrown_dx = cx - pcx
    thrown_dy = cy - py
    thrown_near = (np.abs(thrown_dx) < danger_r) & (np.abs(thrown_dy) < danger_r) & (ca > 0.5)
    any_thrown = bool(np.any(thrown_near))

    overhead_fc = bool((abs(fcx - pcx) < danger_r) and (fcy < py) and (py - fcy < danger_r * 2.5) and (fca > 0.5))

    return {
        "any_monkey": any_monkey,
        "monkey_index": monkey_idx,
        "monkey_dx": monkey_sign_dx,
        "monkey_abs_dx": abs(monkey_sign_dx),
        "any_thrown": any_thrown,
        "overhead_falling_coconut": overhead_fc,
        "falling_coconut_dx": fcx - pcx,
        "falling_coconut_y": fcy,
    }


def shaped_select_reachable_ladder(obs: np.ndarray, *, pcx: float, pby: float, reach_tol: float) -> dict[str, Any]:
    lx, ly, lw, lh, la = ladder_arrays(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh
    active = la > 0.5
    reach = (np.abs(lby - pby) < reach_tol) & (lty < pby - 4.0) & active
    score = np.where(reach, np.abs(lcx - pcx), 1e6)
    idx = int(np.argmin(score))
    return {
        "selected_index": idx,
        "selected_lcx": float(lcx[idx]),
        "selected_lty": float(lty[idx]),
        "selected_lx": float(lx[idx]),
        "selected_ly": float(ly[idx]),
        "selected_lw": float(lw[idx]),
        "selected_lh": float(lh[idx]),
        "selected_reachable": bool(reach[idx]),
        "selected_useful": bool(lty[idx] < pby - 4.0),
        "selected_in_col": False,
        "selected_bottom_diff": float(lby[idx] - pby),
        "selected_center_dx": float(lcx[idx] - pcx),
        "has_valid": bool(np.any(reach)),
        "has_fallback": False,
    }


def shaped_on_column_ladder(
    obs: np.ndarray,
    *,
    px: float,
    py: float,
    pw: float,
    pcx: float,
    pby: float,
    col_frac: float,
) -> dict[str, Any]:
    lx, ly, lw, lh, la = ladder_arrays(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh
    active = la > 0.5
    half_width = lw * (col_frac + 0.1)
    in_x = (np.abs(lcx - pcx) < half_width) & active
    vert_ok = (py > lty - 4.0) & (py < lby + 4.0)
    on = in_x & vert_ok
    score = np.where(on, np.abs(lcx - pcx), 1e6)
    idx = int(np.argmin(score))
    any_on = bool(np.any(on))
    nearest_top = float(lty[idx]) if any_on else 1e6

    if any_on:
        overlap_px = horizontal_overlap(px, px + pw, float(lx[idx]), float(lx[idx] + lw[idx]))
        ladder = {
            "index": idx,
            "x": float(lx[idx]),
            "y": float(ly[idx]),
            "w": float(lw[idx]),
            "h": float(lh[idx]),
            "center_dx": float(lcx[idx] - pcx),
            "bottom_diff": float(lby[idx] - pby),
            "bbox_overlap_px": float(overlap_px),
            "bbox_overlap_player_frac": float(overlap_px / max(pw, 1.0)),
            "bbox_overlap_ladder_frac": float(overlap_px / max(float(lw[idx]), 1.0)),
            "half_width": float(half_width[idx]),
            "vert_ok": bool(vert_ok[idx]),
        }
    else:
        ladder = None

    return {
        "on_column": any_on,
        "nearest_top": nearest_top,
        "selected_index": idx if any_on else -1,
        "ladder": ladder,
        "policy_in_x_count": int(np.sum(in_x)),
    }


def shaped_select_next_ladder_from_top(
    obs: np.ndarray, *, pcx: float, current_ladder_top: float, reach_tol: float
) -> dict[str, Any]:
    lx, ly, lw, lh, la = ladder_arrays(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh
    active = la > 0.5
    reach = (np.abs(lby - current_ladder_top) < reach_tol + 6.0) & (lty < current_ladder_top - 4.0) & active
    score = np.where(reach, np.abs(lcx - pcx), 1e6)
    idx = int(np.argmin(score))
    return {
        "has_next": bool(np.any(reach)),
        "selected_index": idx,
        "selected_lcx": float(lcx[idx]),
        "selected_center_dx": float(lcx[idx] - pcx),
        "selected_bottom_diff": float(lby[idx] - current_ladder_top),
        "selected_reachable": bool(reach[idx]),
    }


def shaped_threat_debug(
    obs: np.ndarray,
    *,
    pcx: float,
    pcy: float,
    punch_dx: float,
    punch_dy: float,
    danger_r: float,
) -> dict[str, Any]:
    mx = obs[376:380]
    my = obs[380:384]
    ma = obs[392:396]
    mdx = mx - pcx
    mdy = my - pcy
    monkey_dist = np.where(ma > 0.5, np.abs(mdx) + np.abs(mdy), 1e6)
    monkey_idx = int(np.argmin(monkey_dist))
    any_monkey = bool(ma[monkey_idx] > 0.5)
    monkey_dx = float(mdx[monkey_idx])
    monkey_dy = float(mdy[monkey_idx])
    monkey_in_punch = any_monkey and abs(monkey_dx) < punch_dx and abs(monkey_dy) < punch_dy

    cx = obs[408:412]
    cy = obs[412:416]
    ca = obs[424:428]
    coconut_dist = np.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2)
    thrown_near = (ca > 0.5) & (coconut_dist < danger_r)
    any_thrown = bool(np.any(thrown_near))

    fcx = float(obs[368])
    fcy = float(obs[369])
    fca = float(obs[372])
    falling_dist = ((fcx - pcx) ** 2 + (fcy - pcy) ** 2) ** 0.5
    falling_near = bool(fca > 0.5 and falling_dist < danger_r)

    if any_thrown:
        threat_idx = int(np.argmin(np.where(ca > 0.5, coconut_dist, 1e6)))
        threat_x = float(cx[threat_idx])
    else:
        threat_x = fcx

    return {
        "any_monkey": any_monkey,
        "monkey_index": monkey_idx,
        "monkey_dx": monkey_dx,
        "monkey_dy": monkey_dy,
        "monkey_abs_dx": abs(monkey_dx),
        "monkey_abs_dy": abs(monkey_dy),
        "punch_in_range": bool(monkey_in_punch),
        "any_thrown": any_thrown,
        "falling_near": falling_near,
        "overhead_falling_coconut": falling_near,
        "coconut_danger": bool(any_thrown or falling_near),
        "threat_x": threat_x,
        "falling_coconut_dx": fcx - pcx,
        "falling_coconut_y": fcy,
    }


def explain_shaped_policy_v2_step(obs_flat: jnp.ndarray, params: dict[str, float], actual_action: int) -> dict[str, Any]:
    obs = np.asarray(obs_flat, dtype=float)
    px = float(obs[0])
    py = float(obs[1])
    pw = float(obs[2])
    ph = float(obs[3])
    pcx = px + pw * 0.5
    pby = py + ph
    pcy = py + ph * 0.5

    reach_tol = params["reach_tol"]
    align_tol = params["align_tol"]
    top_tol = params["top_tol"]
    punch_dx = params["punch_dx"]
    punch_dy = params["punch_dy"]
    danger_r = params["danger_r"]
    col_frac = params["col_frac"]

    reachable = shaped_select_reachable_ladder(obs, pcx=pcx, pby=pby, reach_tol=reach_tol)
    on_info = shaped_on_column_ladder(
        obs, px=px, py=py, pw=pw, pcx=pcx, pby=pby, col_frac=col_frac
    )
    use_on = on_info["on_column"]
    target = on_info["ladder"] if use_on and on_info["ladder"] is not None else reachable
    target_lcx = float(target["center_dx"] + pcx if "center_dx" in target else target["selected_lcx"])
    target_lty = float(target["y"] if "y" in target else target["selected_lty"])
    target_lby = float(target["y"] + target["h"] if "y" in target else target_lty + target["selected_lh"])

    align_dx = target_lcx - pcx
    near_top = bool(use_on and py < target_lty + top_tol)
    above_bottom = bool(py < target_lby - 2.0)
    climbing = bool(use_on and above_bottom and not near_top)
    aligned = bool(abs(align_dx) < align_tol and (reachable["has_valid"] or use_on))

    threats = shaped_threat_debug(
        obs, pcx=pcx, pcy=pcy, punch_dx=punch_dx, punch_dy=punch_dy, danger_r=danger_r
    )
    punch_action = 11 if threats["monkey_dx"] > 0 else 12
    dodge_action = 4 if threats["threat_x"] > pcx else 3
    next_ladder = shaped_select_next_ladder_from_top(
        obs, pcx=pcx, current_ladder_top=target_lty, reach_tol=reach_tol
    )
    if next_ladder["has_next"]:
        dismount_action = 3 if next_ladder["selected_center_dx"] > 0 else 4
    else:
        dismount_action = 3

    approach_action = move_toward_x(align_dx, align_tol)
    if aligned:
        approach_action = 2

    calls = [
        "policy",
        "_select_reachable_ladder",
        "_on_column_ladder",
        "_nearest_monkey",
        "_coconut_danger",
        "_select_next_ladder_from_top",
        "_move_toward_x",
        "decision_tree",
    ]
    overrides = [
        f"reachable={reachable['has_valid']} on_column={use_on}",
        f"approach:{ACTION_NAMES.get(approach_action, approach_action)}",
    ]

    if threats["punch_in_range"]:
        action = punch_action
        final_reason = "monkey_in_punch_priority"
        overrides.append(f"monkey_in_punch_priority:{ACTION_NAMES.get(action, action)}")
    elif climbing:
        action = 2
        final_reason = "on_column_ladder_climb"
        overrides.append("on_column_ladder_climb:UP")
    elif near_top:
        action = dismount_action
        final_reason = "near_top_dismount"
        overrides.append(f"near_top_dismount:{ACTION_NAMES.get(action, action)}")
    elif threats["coconut_danger"]:
        action = dodge_action
        final_reason = "coconut_danger_dodge"
        overrides.append(f"coconut_danger_dodge:{ACTION_NAMES.get(action, action)}")
    elif reachable["has_valid"] or use_on:
        action = approach_action
        final_reason = "approach_reachable_ladder"
        overrides.append(f"approach_reachable_ladder:{ACTION_NAMES.get(action, action)}")
    else:
        action = 3
        final_reason = "fallback_right"
        overrides.append("fallback_right:RIGHT")

    if use_on and on_info["ladder"] is not None:
        primary = {
            "selected_index": int(on_info["ladder"]["index"]),
            "selected_lcx": float(on_info["ladder"]["x"] + on_info["ladder"]["w"] * 0.5),
            "selected_lty": float(on_info["ladder"]["y"]),
            "selected_lx": float(on_info["ladder"]["x"]),
            "selected_ly": float(on_info["ladder"]["y"]),
            "selected_lw": float(on_info["ladder"]["w"]),
            "selected_lh": float(on_info["ladder"]["h"]),
            "selected_reachable": reachable["selected_reachable"],
            "selected_useful": True,
            "selected_in_col": True,
            "selected_bottom_diff": float(on_info["ladder"]["bottom_diff"]),
            "selected_center_dx": float(on_info["ladder"]["center_dx"]),
            "has_valid": True,
            "has_fallback": False,
        }
    else:
        primary = reachable

    selected_overlap_frac = None
    if on_info["ladder"] is not None:
        selected_overlap_frac = on_info["ladder"]["bbox_overlap_player_frac"]

    return {
        "player": {"x": px, "y": py, "w": pw, "h": ph, "center_x": pcx, "bottom_y": pby},
        "actual_action": actual_action,
        "explained_action": int(action),
        "actual_action_name": ACTION_NAMES.get(actual_action, "UNKNOWN"),
        "explained_action_name": ACTION_NAMES.get(int(action), "UNKNOWN"),
        "matches_policy": int(action) == actual_action,
        "logical_stack": calls,
        "decision_overrides": overrides,
        "final_reason": final_reason,
        "primary_ladder": primary,
        "on_ladder_column": on_info,
        "near_top": near_top,
        "aligned": aligned,
        "selected_overlap_frac": selected_overlap_frac,
        "target_x": float(target_lcx),
        "dx_to_target": float(align_dx),
        "threats": threats,
        "punch_in_range": threats["punch_in_range"],
        "policy_family": "shaped_policy_v2",
        "diagnostics": {
            "above_bottom": above_bottom,
            "climbing": climbing,
            "dismount": next_ladder,
        },
        "params": {
            "reach_tol": reach_tol,
            "reach_up": 0.0,
            "align_tol": align_tol,
            "on_pad": 0.0,
            "danger_r": danger_r,
            "punch_dx": punch_dx,
            "punch_dy": punch_dy,
            "height_weight": 0.0,
            "top_tol": top_tol,
            "col_frac": col_frac,
            "min_ladder_overlap_frac": None,
            "ladder_center_tol": None,
            "ladder_bottom_slack": 4.0,
            "route_exclude_x_tol": None,
        },
    }


def explain_policy_step(obs_flat: jnp.ndarray, params: dict[str, float], actual_action: int) -> dict[str, Any]:
    if "col_frac" in params and "punch_dy" in params and "reach_up" not in params:
        return explain_shaped_policy_v2_step(obs_flat, params, actual_action)

    obs = np.asarray(obs_flat, dtype=float)
    px = float(obs[0])
    py = float(obs[1])
    pw = float(obs[2])
    ph = float(obs[3])
    pcx = px + pw * 0.5
    pby = py + ph
    child_x = float(obs[360])

    reach_tol = params["reach_tol"]
    reach_up = params["reach_up"]
    align_tol = params["align_tol"]
    strict_ladder = "min_ladder_overlap_frac" in params
    on_pad = params.get("on_pad", 0.0)
    danger_r = params["danger_r"]
    punch_dx = params["punch_dx"]
    height_weight = params["height_weight"]
    min_overlap_frac = params.get("min_ladder_overlap_frac")
    center_tol = params.get("ladder_center_tol")
    bottom_slack = params.get("ladder_bottom_slack", 4.0)
    route_exclude_x_tol = params.get("route_exclude_x_tol")

    on_info = on_ladder_column(
        obs,
        px=px,
        py=py,
        pw=pw,
        ph=ph,
        pcx=pcx,
        pby=pby,
        pad=on_pad,
        min_overlap_frac=min_overlap_frac,
        center_tol=center_tol,
        bottom_slack=bottom_slack,
    )
    on_column = on_info["on_column"]
    near_top = bool(on_column and ((py - float(on_info["nearest_top"])) < 8.0))

    primary = select_best(
        obs,
        px=px,
        pw=pw,
        pcx=pcx,
        pby=pby,
        py=py,
        reach_tol=reach_tol,
        reach_up=reach_up,
        height_weight=height_weight,
        exclude_pad=on_pad,
        min_overlap_frac=min_overlap_frac,
        center_tol=center_tol,
        bottom_slack=bottom_slack,
        route_exclude_x_tol=route_exclude_x_tol,
    )
    dismount = select_best(
        obs,
        px=px,
        pw=pw,
        pcx=pcx,
        pby=pby,
        py=py,
        reach_tol=reach_tol,
        reach_up=reach_up,
        height_weight=height_weight,
        exclude_pad=on_pad,
        min_overlap_frac=min_overlap_frac,
        center_tol=center_tol,
        bottom_slack=bottom_slack,
        route_exclude_x_tol=route_exclude_x_tol,
    )

    child_dir_x = pcx + np.sign(child_x - pcx) * 30.0
    target_x = primary["selected_lcx"] if (primary["has_valid"] or primary["has_fallback"]) else child_dir_x
    dx_to_target = target_x - pcx
    if strict_ladder:
        overlap_px = horizontal_overlap(
            px,
            px + pw,
            primary["selected_lx"],
            primary["selected_lx"] + primary["selected_lw"],
        )
        selected_overlap_frac = overlap_px / max(min(pw, primary["selected_lw"]), 1.0)
        aligned = bool(
            primary["has_valid"]
            and selected_overlap_frac >= float(min_overlap_frac)
            and abs(primary["selected_lcx"] - pcx) <= float(center_tol)
        )
    else:
        selected_overlap_frac = None
        aligned = bool(primary["has_valid"] and abs(primary["selected_lcx"] - pcx) < align_tol)
    action = move_toward_x(dx_to_target, 1.0)

    calls = ["policy"]
    if strict_ladder:
        calls.extend(["_ladder_overlap_fraction", "_is_on_ladder"])
    else:
        calls.append("_on_ladder_column")
    calls.extend(["_select_best(primary)", "_select_best(dismount)", "_threats"])
    overrides = [f"path_planning:{ACTION_NAMES.get(action, action)}"]
    final_reason = "path_planning"

    if aligned:
        action = 2
        final_reason = "aligned_reachable_ladder"
        overrides.append("aligned_reachable_ladder:UP")

    if on_column and not near_top:
        action = 2
        final_reason = "on_ladder_column_climb"
        overrides.append("on_ladder_column_climb:UP")

    dismount_target = dismount["selected_lcx"] if (dismount["has_valid"] or dismount["has_fallback"]) else child_dir_x
    dismount_dir = move_toward_x(dismount_target - pcx, 1.0)
    if dismount_dir == 0:
        dismount_dir = 4 if child_x < pcx else 3
    if near_top:
        action = dismount_dir
        final_reason = "near_top_dismount"
        overrides.append(f"near_top_dismount:{ACTION_NAMES.get(action, action)}")

    threats = threat_debug(obs, pcx=pcx, py=py, danger_r=danger_r)
    punch_action = 11 if threats["monkey_dx"] > 0 else 12
    punch_in_range = bool(threats["any_monkey"] and threats["monkey_abs_dx"] < punch_dx)
    if punch_in_range:
        action = punch_action
        final_reason = "punch_monkey_priority"
        overrides.append(f"punch_monkey_priority:{ACTION_NAMES.get(action, action)}")

    if threats["any_thrown"] and not on_column:
        action = 1
        final_reason = "jump_thrown_coconut"
        overrides.append("jump_thrown_coconut:FIRE")
    if threats["any_thrown"] and on_column and not near_top:
        action = 0
        final_reason = "hold_on_ladder_for_thrown_coconut"
        overrides.append("hold_on_ladder_for_thrown_coconut:NOOP")

    fc_step = 4 if threats["falling_coconut_dx"] > 0 else 3
    if threats["overhead_falling_coconut"] and not on_column:
        action = fc_step
        final_reason = "sidestep_falling_coconut"
        overrides.append(f"sidestep_falling_coconut:{ACTION_NAMES.get(action, action)}")
    if threats["overhead_falling_coconut"] and on_column and not near_top:
        action = 0
        final_reason = "hold_on_ladder_for_falling_coconut"
        overrides.append("hold_on_ladder_for_falling_coconut:NOOP")

    return {
        "player": {"x": px, "y": py, "w": pw, "h": ph, "center_x": pcx, "bottom_y": pby},
        "actual_action": actual_action,
        "explained_action": int(action),
        "actual_action_name": ACTION_NAMES.get(actual_action, "UNKNOWN"),
        "explained_action_name": ACTION_NAMES.get(int(action), "UNKNOWN"),
        "matches_policy": int(action) == actual_action,
        "logical_stack": calls,
        "decision_overrides": overrides,
        "final_reason": final_reason,
        "primary_ladder": primary,
        "on_ladder_column": on_info,
        "near_top": near_top,
        "aligned": aligned,
        "selected_overlap_frac": selected_overlap_frac,
        "target_x": float(target_x),
        "dx_to_target": float(dx_to_target),
        "threats": threats,
        "punch_in_range": punch_in_range,
        "params": {
            "reach_tol": reach_tol,
            "reach_up": reach_up,
            "align_tol": align_tol,
            "on_pad": on_pad,
            "danger_r": danger_r,
            "punch_dx": punch_dx,
            "height_weight": height_weight,
            "min_ladder_overlap_frac": min_overlap_frac,
            "ladder_center_tol": center_tol,
            "ladder_bottom_slack": bottom_slack,
            "route_exclude_x_tol": route_exclude_x_tol,
        },
    }


def load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_box(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], scale: int, color: tuple[int, int, int], width: int = 2) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle((x0 * scale, y0 * scale, x1 * scale, y1 * scale), outline=color, width=width)


def wrap_lines(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def compose_debug_frame(
    game_frame: np.ndarray,
    debug: dict[str, Any],
    *,
    step: int,
    reward: float,
    score: int,
    lives: int,
    scale: int,
    panel_width: int,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> np.ndarray:
    def ladder_box(row: dict[str, Any]) -> tuple[float, float, float, float]:
        if "x" in row:
            return row["x"], row["y"], row["x"] + row["w"], row["y"] + row["h"]
        return (
            row["selected_lx"],
            row["selected_ly"],
            row["selected_lx"] + row["selected_lw"],
            row["selected_ly"] + row["selected_lh"],
        )

    image = Image.fromarray(game_frame)
    draw = ImageDraw.Draw(image)
    player = debug["player"]
    draw_box(
        draw,
        (player["x"], player["y"], player["x"] + player["w"], player["y"] + player["h"]),
        scale,
        (50, 220, 255),
        width=2,
    )
    ladder = debug["on_ladder_column"]["ladder"] or debug["primary_ladder"]
    if ladder:
        draw_box(
            draw,
            ladder_box(ladder),
            scale,
            (255, 230, 80),
            width=2,
        )

    canvas = Image.new("RGB", (image.width + panel_width, image.height), (20, 22, 24))
    canvas.paste(image, (0, 0))
    panel = ImageDraw.Draw(canvas)
    x = image.width + 14
    y = 12

    def line(text: str, fill: tuple[int, int, int] = (235, 235, 235), *, small: bool = False) -> None:
        nonlocal y
        panel.text((x, y), text, fill=fill, font=small_font if small else font)
        y += 16 if small else 20

    def block(title: str, rows: list[str]) -> None:
        nonlocal y
        y += 6
        line(title, (120, 190, 255))
        for row in rows:
            for wrapped in wrap_lines(row, 42):
                line(wrapped, (225, 225, 225), small=True)

    action_color = (140, 255, 140) if debug["matches_policy"] else (255, 120, 120)
    line(f"step {step}", (255, 255, 255))
    line(f"action {debug['actual_action']}:{debug['actual_action_name']}", action_color)
    line(f"reason {debug['final_reason']}", (255, 220, 120), small=True)
    line(f"reward {reward:.1f}  score {score}  lives {lives}", small=True)
    line(f"pos x={player['x']:.1f} y={player['y']:.1f}", small=True)

    block("logical stack", debug["logical_stack"])
    block("decision overrides", debug["decision_overrides"][-5:])

    on_info = debug["on_ladder_column"]
    on_ladder = on_info["ladder"] or {}
    primary = debug["primary_ladder"]
    overlap_rows = [
        f"on_column={on_info['on_column']} near_top={debug['near_top']}",
        f"aligned={debug['aligned']} policy_in_x_count={on_info['policy_in_x_count']}",
    ]
    if on_ladder:
        overlap_rows.extend(
            [
                f"on_ladder_idx={on_ladder['index']} center_dx={on_ladder['center_dx']:.1f}",
                f"bbox_overlap_px={on_ladder['bbox_overlap_px']:.1f}",
                f"overlap/player={on_ladder['bbox_overlap_player_frac']:.2f}",
                f"overlap/ladder={on_ladder['bbox_overlap_ladder_frac']:.2f}",
            ]
        )
    overlap_rows.extend(
        [
            f"target_idx={primary['selected_index']} dx={primary['selected_center_dx']:.1f}",
            f"bottom_diff={primary['selected_bottom_diff']:.1f}",
            f"reachable={primary['selected_reachable']} useful={primary['selected_useful']}",
        ]
    )
    block("ladder checks", overlap_rows)

    threats = debug["threats"]
    block(
        "priority checks",
        [
            f"punch_in_range={debug['punch_in_range']} monkey_dx={threats['monkey_dx']:.1f}",
            f"thrown_near={threats['any_thrown']}",
            f"falling_overhead={threats['overhead_falling_coconut']}",
            f"falling_dx={threats['falling_coconut_dx']:.1f}",
        ],
    )

    block(
        "params",
        [
            (
                f"align_tol={debug['params']['align_tol']:.2f} "
                + (
                    f"on_pad={debug['params']['on_pad']:.2f}"
                    if debug["params"].get("min_ladder_overlap_frac") is None
                    else f"min_overlap={debug['params']['min_ladder_overlap_frac']:.2f}"
                )
            ),
            f"reach_tol={debug['params']['reach_tol']:.2f} reach_up={debug['params']['reach_up']:.2f}",
            f"punch_dx={debug['params']['punch_dx']:.2f} danger_r={debug['params']['danger_r']:.2f}",
            (
                ""
                if debug["params"].get("min_ladder_overlap_frac") is None
                else f"center_tol={debug['params']['ladder_center_tol']:.2f} bottom_slack={debug['params']['ladder_bottom_slack']:.2f}"
            ),
            (
                ""
                if debug["params"].get("route_exclude_x_tol") is None
                else f"route_exclude_x_tol={debug['params']['route_exclude_x_tol']:.2f}"
            ),
        ],
    )
    return np.asarray(canvas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Kangaroo policy with decision diagnostics")
    parser.add_argument("--seed", type=int, default=20260480)
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help=(
            "Replay the indexed subkey from jax.random.split(PRNGKey(seed), split_count). "
            "Use this to reproduce a vectorized evaluation episode."
        ),
    )
    parser.add_argument(
        "--split-count",
        type=int,
        default=128,
        help="Number of subkeys to generate when --episode-index is set.",
    )
    parser.add_argument("--max-steps", type=int, default=850)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--panel-width", type=int, default=560)
    parser.add_argument("--policy-dir", type=str, default=None)
    parser.add_argument("--policy-path", type=str, default=None)
    parser.add_argument("--history-index", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--trace-output", type=str, default=None)
    parser.add_argument(
        "--trace-only",
        action="store_true",
        help="Write the JSON decision trace without rendering a GIF/MP4.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy_dir = Path(args.policy_dir) if args.policy_dir else find_latest_kangaroo_run()
    if not policy_dir.is_absolute():
        policy_dir = PROJECT_ROOT / policy_dir

    policy_path_override = Path(args.policy_path) if args.policy_path else None
    policy_path, params_raw, frame_stack = load_run_artifacts(
        policy_dir,
        policy_path_override=policy_path_override,
        history_index=args.history_index,
    )
    policy_module = load_policy_module(policy_path)
    params_float = as_float_params(params_raw)
    params = {key: jnp.float32(value) for key, value in params_float.items()}

    out_dir = PROJECT_ROOT / "scripts" / "llm_optimization" / "analysis" / "videos"
    output_path = Path(args.output) if args.output else out_dir / f"{policy_dir.name}_{policy_path.stem}_debug.gif"
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.trace_only and output_path.suffix.lower() not in {".mp4", ".gif"}:
        raise ValueError("--output must end in .mp4 or .gif")
    if not args.trace_only and output_path.suffix.lower() == ".mp4" and cv2 is None:
        raise RuntimeError("OpenCV is not installed; use a .gif output path")

    trace_path = Path(args.trace_output) if args.trace_output else (
        PROJECT_ROOT
        / "scripts"
        / "llm_optimization"
        / "analysis"
        / "traces"
        / f"{policy_dir.name}_{policy_path.stem}_debug_trace.json"
    )
    if not trace_path.is_absolute():
        trace_path = PROJECT_ROOT / trace_path
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    base_env = jaxatari.make("kangaroo")
    wrapped_env = FlattenObservationWrapper(
        ObjectCentricWrapper(
            AtariWrapper(base_env, episodic_life=True),
            frame_stack_size=frame_stack,
            frame_skip=1,
            clip_reward=False,
        )
    )
    reset_key = jrandom.PRNGKey(args.seed)
    if args.episode_index is not None:
        if args.episode_index < 0 or args.episode_index >= args.split_count:
            raise ValueError("--episode-index must be within [0, split_count)")
        reset_key = jrandom.split(reset_key, args.split_count)[args.episode_index]
    obs, state = wrapped_env.reset(reset_key)
    obs_size = obs.shape[-1]
    obs_flat = jnp.asarray(obs[-obs_size:] if obs.shape[0] > obs_size else obs, dtype=jnp.float32)

    font = load_font(14)
    small_font = load_font(12)
    writer = None
    gif_frames: list[Image.Image] = []
    trace_records: list[dict[str, Any]] = []
    action_counts: Counter[int] = Counter()
    total_reward = 0.0
    reward_events = 0
    done = False
    step = 0

    try:
        while step < args.max_steps and not done:
            raw_state = unwrap_render_state(state)
            score = jax_scalar_to_int(getattr(raw_state, "score", None))
            lives = jax_scalar_to_int(getattr(raw_state, "lives", None))
            action = int(jnp.clip(policy_module.policy(obs_flat, params), 0, wrapped_env.action_space().n - 1))
            debug = explain_policy_step(obs_flat, params_float, action)
            action_counts[action] += 1

            if not args.trace_only and step % max(1, args.frame_stride) == 0:
                game_frame = np.asarray(base_env.render(raw_state), dtype=np.uint8)
                game_frame = np.repeat(np.repeat(game_frame, args.scale, axis=0), args.scale, axis=1)
                frame = compose_debug_frame(
                    game_frame,
                    debug,
                    step=step,
                    reward=0.0,
                    score=score,
                    lives=lives,
                    scale=args.scale,
                    panel_width=args.panel_width,
                    font=font,
                    small_font=small_font,
                )
                if output_path.suffix.lower() == ".mp4":
                    if writer is None:
                        height, width = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            args.fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Failed to open writer for {output_path}")
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                else:
                    gif_frames.append(Image.fromarray(frame))

            next_obs, next_state, reward, terminated, truncated, _ = wrapped_env.step(state, action)
            reward_float = float(reward)
            total_reward += reward_float
            if reward_float != 0.0:
                reward_events += 1

            record = {
                "step": step,
                "action": action,
                "action_name": ACTION_NAMES.get(action, "UNKNOWN"),
                "reward": reward_float,
                "score_before": score,
                "lives_before": lives,
                "debug": debug,
            }
            trace_records.append(record)

            obs_flat = jnp.asarray(
                next_obs[-obs_size:] if next_obs.shape[0] > obs_size else next_obs,
                dtype=jnp.float32,
            )
            state = next_state
            done = bool(terminated) or bool(truncated)
            step += 1
    finally:
        if writer is not None:
            writer.release()

    if not args.trace_only and output_path.suffix.lower() == ".gif":
        if not gif_frames:
            raise RuntimeError("No GIF frames were recorded")
        gif_frames[0].save(
            output_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=max(1, int(1000 / max(args.fps, 1))),
            loop=0,
        )

    payload = {
        "policy_dir": str(policy_dir.relative_to(PROJECT_ROOT)),
        "policy_path": str(policy_path.relative_to(PROJECT_ROOT)),
        "seed": args.seed,
        "episode_index": args.episode_index,
        "split_count": args.split_count if args.episode_index is not None else None,
        "max_steps": args.max_steps,
        "frame_stride": args.frame_stride,
        "output_video": None if args.trace_only else str(output_path.relative_to(PROJECT_ROOT)),
        "summary": {
            "steps": step,
            "done": done,
            "total_reward": total_reward,
            "reward_events": reward_events,
            "action_counts": {
                f"{action}:{ACTION_NAMES.get(action, 'UNKNOWN')}": count
                for action, count in sorted(action_counts.items())
            },
            "mismatched_explanations": sum(0 if row["debug"]["matches_policy"] else 1 for row in trace_records),
        },
        "records": trace_records,
    }
    trace_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.trace_only:
        print("Skipped debug video (--trace-only)")
    else:
        print(f"Saved debug video: {output_path}")
    print(f"Saved debug trace: {trace_path}")
    print(
        f"Summary: steps={step}, total_reward={total_reward:.1f}, "
        f"reward_events={reward_events}, done={done}"
    )
    print(
        "Action counts: "
        + str({f"{action}:{ACTION_NAMES.get(action, 'UNKNOWN')}": count for action, count in sorted(action_counts.items())})
    )
    print(f"Mismatched explanations: {payload['summary']['mismatched_explanations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

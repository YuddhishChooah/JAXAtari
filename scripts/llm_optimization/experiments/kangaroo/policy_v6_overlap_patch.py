"""Manual policy_v6 patch for testing stricter ladder perception.

This keeps policy_v6's high-level route logic intact, but changes the low-level
"am I on the ladder?" perception helper so one pixel of x-overlap is not enough
to trigger UP. The new overlap and center thresholds are numeric parameters so
CMA-ES can tune them without another LLM call.
"""

import jax
import jax.numpy as jnp

NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = 6, 7, 8, 9
UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE = 10, 11, 12, 13


def init_params():
    return {
        "reach_tol": jnp.array(15.497392654418945),
        "reach_up": jnp.array(28.213350296020508),
        "align_tol": jnp.array(6.084083080291748),
        "danger_r": jnp.array(18.24340057373047),
        "punch_dx": jnp.array(19.512571334838867),
        "height_weight": jnp.array(1.118639349937439),
        # New perception parameters. With 8px player/ladder widths, 0.50 means
        # at least 4px overlap, avoiding the previous 1px false-positive climb.
        "min_ladder_overlap_frac": jnp.array(0.50),
        "ladder_center_tol": jnp.array(4.0),
        "ladder_bottom_slack": jnp.array(4.0),
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


def _ladder_overlap_fraction(px, pw, lx, lw):
    player_left = px
    player_right = px + pw
    ladder_left = lx
    ladder_right = lx + lw
    overlap = jnp.maximum(
        0.0,
        jnp.minimum(player_right, ladder_right) - jnp.maximum(player_left, ladder_left),
    )
    denom = jnp.maximum(jnp.minimum(pw, lw), 1.0)
    return overlap / denom


def _ladder_match_mask(obs, px, pw, pcx, py, pby, min_overlap_frac, center_tol, bottom_slack):
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh
    overlap_frac = _ladder_overlap_fraction(px, pw, lx, lw)
    centered = jnp.abs(lcx - pcx) <= center_tol
    enough_overlap = overlap_frac >= min_overlap_frac
    top_above = lty < py + 2.0
    bottom_below = lby >= pby - bottom_slack
    active = la > 0.5
    return enough_overlap & centered & top_above & bottom_below & active


def _on_ladder_column(obs, px, pw, pcx, py, pby, min_overlap_frac, center_tol, bottom_slack):
    lx, ly, lw, lh, _ = _ladders(obs)
    lty = ly
    on = _ladder_match_mask(
        obs, px, pw, pcx, py, pby, min_overlap_frac, center_tol, bottom_slack
    )
    any_on = jnp.any(on)
    top_for = jnp.where(on, lty, jnp.array(1e6))
    nearest_top = jnp.min(top_for)
    return any_on, nearest_top


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
):
    """Best upward ladder, excluding only a robustly matched current ladder."""
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

    valid = reachable & useful & active & ~current_ladder
    dx = jnp.abs(lcx - pcx)
    height_gap = jnp.maximum(py - lty, 0.0)
    cost = jnp.where(valid, dx - height_weight * height_gap, jnp.array(1e6))
    idx = jnp.argmin(cost)
    has_valid = jnp.any(valid)

    fb_valid = useful & active & ~current_ladder
    fb_cost = jnp.where(fb_valid, dx, jnp.array(1e6))
    fb_idx = jnp.argmin(fb_cost)
    has_fb = jnp.any(fb_valid)

    sel_idx = jnp.where(has_valid, idx, fb_idx)
    return lcx[sel_idx], lty[sel_idx], has_valid, has_fb


def _target_ladder_aligned(obs, px, pw, pcx, target_x, min_overlap_frac, center_tol):
    lx, _, lw, _, la = _ladders(obs)
    lcx = lx + lw * 0.5
    same_target = jnp.abs(lcx - target_x) < 0.5
    overlap_frac = _ladder_overlap_fraction(px, pw, lx, lw)
    overlap_ok = overlap_frac >= min_overlap_frac
    center_ok = jnp.abs(lcx - pcx) <= center_tol
    target_ok = same_target & overlap_ok & center_ok & (la > 0.5)
    return jnp.any(target_ok)


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
    danger_r = params["danger_r"]
    punch_dx = params["punch_dx"]
    height_weight = params["height_weight"]
    min_overlap_frac = params["min_ladder_overlap_frac"]
    center_tol = params["ladder_center_tol"]
    bottom_slack = params["ladder_bottom_slack"]

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


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)

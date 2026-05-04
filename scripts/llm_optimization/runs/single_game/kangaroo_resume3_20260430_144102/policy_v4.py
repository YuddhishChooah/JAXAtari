"""
Auto-generated policy v4
Generated at: 2026-04-30 11:40:37
"""

"""
Kangaroo policy v4
Key fixes vs v3:
- Asymmetric reach window (ladder bottom can be well below feet).
- Use player feet (pby) consistently for reach + useful tests.
- Persistent ladder targeting: if no strict-reachable ladder, fall back to
  the closest active+useful ladder by x (not a blind sweep).
- Dismount direction picked from the next candidate ladder's lcx.
- On-ladder hazard response: pause/descend if hazard near while climbing.
- Child-direction tiebreaker only as last resort.
"""

import jax
import jax.numpy as jnp

# Action constants
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = 6, 7, 8, 9
UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE = 10, 11, 12, 13


def init_params():
    return {
        "reach_below": jnp.array(28.0),   # ladder bottom may sit this far below feet
        "reach_above": jnp.array(10.0),   # ladder bottom may sit this far above feet
        "align_tol": jnp.array(6.0),      # x tol to start climbing
        "danger_r": jnp.array(18.0),      # hazard radius
        "punch_dx": jnp.array(20.0),      # punch range
        "height_weight": jnp.array(0.4),  # cost: dx - hw * min(height_gap, 40)
        "on_pad": jnp.array(3.0),         # ladder-column x padding
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
    bottom_below = lby >= pby - 6.0
    active = la > 0.5
    on = in_x & top_above & bottom_below & active
    any_on = jnp.any(on)
    top_for = jnp.where(on, lty, jnp.array(1e6))
    nearest_top = jnp.min(top_for)
    return any_on, nearest_top


def _select_ladder(obs, pcx, pby, on_column, on_pad,
                   reach_below, reach_above, height_weight):
    """Two-tier ladder selection: strict reachable, then closest useful."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lty = ly
    lby = ly + lh

    # Reach: ladder bottom can be below feet (reach_below) or slightly above (reach_above)
    below = lby - pby   # positive if ladder bottom is below feet
    reachable = (below > -reach_above) & (below < reach_below)

    # Useful: top above feet (real upward progress)
    useful = lty < (pby - 8.0)
    active = la > 0.5

    # Exclude column the player currently stands in
    in_col = (pcx >= lx - on_pad) & (pcx <= lx + lw + on_pad)
    not_current = ~(in_col & on_column)

    base_valid = useful & active & not_current
    strict = reachable & base_valid

    dx = jnp.abs(lcx - pcx)
    height_gap = jnp.minimum(jnp.maximum(pby - lty, 0.0), 40.0)
    cost_base = dx - height_weight * height_gap

    # Tier 1: strict reachable
    cost_strict = jnp.where(strict, cost_base, jnp.array(1e6))
    sidx = jnp.argmin(cost_strict)
    has_strict = jnp.any(strict)

    # Tier 2: closest active+useful by x (no reach filter)
    cost_loose = jnp.where(base_valid, dx, jnp.array(1e6))
    lidx = jnp.argmin(cost_loose)
    has_loose = jnp.any(base_valid)

    sel_idx = jnp.where(has_strict, sidx, lidx)
    has_any = has_strict | has_loose

    return lcx[sel_idx], lty[sel_idx], lby[sel_idx], has_strict, has_any


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

    overhead_fc = (jnp.abs(fcx - pcx) < danger_r) & (fcy < py) \
                  & (py - fcy < danger_r * 2.0) & (fca > 0.5)
    fc_dx = fcx - pcx

    tdx = cx - pcx
    tdy = cy - py
    thrown_near = (jnp.abs(tdx) < danger_r) & (jnp.abs(tdy) < danger_r * 1.5) & (ca > 0.5)
    any_thrown = jnp.any(thrown_near)

    # Hazard near while climbing: anything within danger_r in y
    hazard_climbing = any_monkey | any_thrown | overhead_fc

    return any_monkey, monkey_sign_dx, overhead_fc, fc_dx, any_thrown, hazard_climbing


def policy(obs_flat, params):
    px = obs_flat[0]
    py = obs_flat[1]
    pw = obs_flat[2]
    ph = obs_flat[3]
    orient = obs_flat[7]
    pcx = px + pw * 0.5
    pby = py + ph

    child_x = obs_flat[360]

    reach_below = params["reach_below"]
    reach_above = params["reach_above"]
    align_tol = params["align_tol"]
    danger_r = params["danger_r"]
    punch_dx = params["punch_dx"]
    height_weight = params["height_weight"]
    on_pad = params["on_pad"]

    # Are we currently inside a ladder column?
    on_column, on_top_y = _on_ladder_column(obs_flat, pcx, py, pby, on_pad)
    near_top = on_column & ((py - on_top_y) < 6.0)

    # Pick next ladder, excluding current column
    lcx, lty, lby, has_strict, has_any = _select_ladder(
        obs_flat, pcx, pby, on_column, on_pad,
        reach_below, reach_above, height_weight,
    )

    # Target: selected ladder if any, else child-direction nudge
    child_dir_x = pcx + jnp.where(child_x > pcx, 30.0, -30.0)
    target_x = jnp.where(has_any, lcx, child_dir_x)
    dx_to_target = target_x - pcx

    aligned_strict = has_strict & (jnp.abs(lcx - pcx) < align_tol)

    horiz_action = _move_toward_x(dx_to_target, 1.0)

    any_monkey, monkey_sign_dx, overhead_fc, fc_dx, any_thrown, hazard_climbing = _threats(
        obs_flat, pcx, py, danger_r
    )

    # --- Build action with priority order ---
    action = horiz_action

    # If aligned with strict-reachable ladder -> climb
    action = jnp.where(aligned_strict, UP, action)

    # If on column and not near top -> keep climbing (unless hazard)
    climb_safe = on_column & ~near_top & ~hazard_climbing
    action = jnp.where(climb_safe, UP, action)

    # On column with hazard nearby -> pause (NOOP) instead of climbing into it
    action = jnp.where(on_column & ~near_top & hazard_climbing, NOOP, action)

    # Near top -> dismount toward next ladder lcx (or child direction)
    dismount_dx = jnp.where(has_any, lcx - pcx, child_x - pcx)
    dismount_dir = jnp.where(dismount_dx > 0, RIGHT, LEFT)
    action = jnp.where(near_top, dismount_dir, action)

    # Punch nearby monkey
    punch_action = jnp.where(monkey_sign_dx > 0, RIGHTFIRE, LEFTFIRE)
    punch_in_range = any_monkey & (jnp.abs(monkey_sign_dx) < punch_dx) & ~on_column
    action = jnp.where(punch_in_range, punch_action, action)

    # Thrown coconut nearby and not climbing -> jump
    action = jnp.where(any_thrown & ~on_column, FIRE, action)

    # Overhead falling coconut: sidestep if off column
    fc_step = jnp.where(fc_dx > 0, LEFT, RIGHT)
    action = jnp.where(overhead_fc & ~on_column, fc_step, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
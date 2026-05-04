"""
Auto-generated policy v2
Generated at: 2026-05-01 23:06:57
"""

import jax
import jax.numpy as jnp


# ---- observation aliases ----
P_X, P_Y, P_W, P_H = 0, 1, 2, 3
P_ORI = 7

LAD_X0 = 168
LAD_Y0 = 188
LAD_W0 = 208
LAD_H0 = 228
LAD_A0 = 248

MON_X0 = 376
MON_Y0 = 380
MON_A0 = 392

COCO_X0 = 408
COCO_Y0 = 412
COCO_A0 = 424

FCOC_X, FCOC_Y, FCOC_A = 368, 369, 372

# Actions
NOOP = 0
FIRE = 1
UP = 2
RIGHT = 3
LEFT = 4
DOWN = 5
UPRIGHT = 6
UPLEFT = 7
RIGHTFIRE = 11
LEFTFIRE = 12


def init_params():
    return {
        "reach_tol":  jnp.float32(14.0),
        "align_tol":  jnp.float32(4.0),
        "top_tol":    jnp.float32(8.0),
        "punch_dx":   jnp.float32(18.0),
        "punch_dy":   jnp.float32(20.0),
        "danger_r":   jnp.float32(20.0),
        "col_frac":   jnp.float32(0.5),
    }


def _ladders(obs):
    lx = jax.lax.dynamic_slice(obs, (LAD_X0,), (20,))
    ly = jax.lax.dynamic_slice(obs, (LAD_Y0,), (20,))
    lw = jax.lax.dynamic_slice(obs, (LAD_W0,), (20,))
    lh = jax.lax.dynamic_slice(obs, (LAD_H0,), (20,))
    la = jax.lax.dynamic_slice(obs, (LAD_A0,), (20,))
    return lx, ly, lw, lh, la


def _select_reachable_ladder(obs, pcx, pby, reach_tol):
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lby = ly + lh
    lty = ly
    reach = (jnp.abs(lby - pby) < reach_tol) & (lty < pby - 4.0) & (la > 0.5)
    big = jnp.float32(1e6)
    score = jnp.where(reach, jnp.abs(lcx - pcx), big)
    idx = jnp.argmin(score)
    any_reach = jnp.any(reach)
    return any_reach, lcx[idx], lty[idx], lby[idx], lw[idx]


def _on_column_ladder(obs, pcx, pby, py, col_frac):
    """Find a ladder whose column currently contains the player (latching)."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lby = ly + lh
    lty = ly
    half = lw * (col_frac + 0.1)
    in_col = (jnp.abs(lcx - pcx) < half) & (la > 0.5)
    # vertical: player is between ladder top and ladder bottom (with margin)
    vert_ok = (py > lty - 4.0) & (py < lby + 4.0)
    on = in_col & vert_ok
    big = jnp.float32(1e6)
    score = jnp.where(on, jnp.abs(lcx - pcx), big)
    idx = jnp.argmin(score)
    any_on = jnp.any(on)
    return any_on, lcx[idx], lty[idx], lby[idx], lw[idx]


def _select_next_ladder_from_top(obs, pcx, lty_cur, reach_tol):
    """When dismounting, pick a ladder reachable from the new platform near lty_cur."""
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lby = ly + lh
    lty = ly
    # new platform bottom roughly equals lty_cur; reachable ladder has its bottom near lty_cur
    reach = (jnp.abs(lby - lty_cur) < reach_tol + 6.0) & (lty < lty_cur - 4.0) & (la > 0.5)
    big = jnp.float32(1e6)
    score = jnp.where(reach, jnp.abs(lcx - pcx), big)
    idx = jnp.argmin(score)
    return jnp.any(reach), lcx[idx]


def _nearest_monkey(obs, px, py):
    mx = jax.lax.dynamic_slice(obs, (MON_X0,), (4,))
    my = jax.lax.dynamic_slice(obs, (MON_Y0,), (4,))
    ma = jax.lax.dynamic_slice(obs, (MON_A0,), (4,))
    big = jnp.float32(1e6)
    dx = mx - px
    dy = my - py
    dist = jnp.where(ma > 0.5, jnp.abs(dx) + jnp.abs(dy), big)
    idx = jnp.argmin(dist)
    return ma[idx] > 0.5, dx[idx], dy[idx]


def _coconut_danger(obs, px, py, r):
    cx = jax.lax.dynamic_slice(obs, (COCO_X0,), (4,))
    cy = jax.lax.dynamic_slice(obs, (COCO_Y0,), (4,))
    ca = jax.lax.dynamic_slice(obs, (COCO_A0,), (4,))
    d = jnp.sqrt((cx - px) ** 2 + (cy - py) ** 2)
    near = (ca > 0.5) & (d < r)
    any_near = jnp.any(near)

    fx, fy, fa = obs[FCOC_X], obs[FCOC_Y], obs[FCOC_A]
    fd = jnp.sqrt((fx - px) ** 2 + (fy - py) ** 2)
    f_near = (fa > 0.5) & (fd < r)

    threat_x = jnp.where(any_near,
                         cx[jnp.argmin(jnp.where(ca > 0.5, d, 1e6))],
                         fx)
    return any_near | f_near, threat_x


def _move_toward_x(dx, deadband):
    return jnp.where(jnp.abs(dx) < deadband,
                     NOOP,
                     jnp.where(dx > 0, RIGHT, LEFT))


def policy(obs_flat, params):
    obs = obs_flat.astype(jnp.float32)

    px = obs[P_X]
    py = obs[P_Y]
    pw = obs[P_W]
    ph = obs[P_H]
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

    # --- ladder selections ---
    any_reach, r_lcx, r_lty, r_lby, r_lw = _select_reachable_ladder(obs, pcx, pby, reach_tol)
    any_on, c_lcx, c_lty, c_lby, c_lw = _on_column_ladder(obs, pcx, pby, py, col_frac)

    # If currently on a ladder column, latch onto it; else use reachable target
    use_on = any_on
    tgt_lcx = jnp.where(use_on, c_lcx, r_lcx)
    tgt_lty = jnp.where(use_on, c_lty, r_lty)
    tgt_lby = jnp.where(use_on, c_lby, r_lby)

    align_dx = tgt_lcx - pcx
    near_top = use_on & (py < tgt_lty + top_tol)
    above_bottom = py < tgt_lby - 2.0

    # Climb whenever on column, above bottom, not at top.
    climbing = use_on & above_bottom & ~near_top

    # --- threats ---
    has_mon, mdx, mdy = _nearest_monkey(obs, pcx, pcy)
    monkey_in_punch = has_mon & (jnp.abs(mdx) < punch_dx) & (jnp.abs(mdy) < punch_dy)
    punch_action = jnp.where(mdx > 0, RIGHTFIRE, LEFTFIRE)

    coco_danger, threat_x = _coconut_danger(obs, pcx, pcy, danger_r)
    dodge_action = jnp.where(threat_x > pcx, LEFT, RIGHT)

    # --- dismount: select next ladder from new platform ---
    has_next, next_lcx = _select_next_ladder_from_top(obs, pcx, tgt_lty, reach_tol)
    dismount_dx = next_lcx - pcx
    dismount_action = jnp.where(has_next,
                                jnp.where(dismount_dx > 0, RIGHT, LEFT),
                                RIGHT)

    # --- horizontal approach to ladder with deadband ---
    approach_action = _move_toward_x(align_dx, align_tol)
    # If in deadband and reachable, force UP onto column
    approach_action = jnp.where(
        (jnp.abs(align_dx) < align_tol) & (any_reach | use_on),
        UP,
        approach_action,
    )

    no_ladder_action = RIGHT

    # --- decision tree ---
    # Priority:
    #  1. punch nearby monkey (preserves stable-200 first reward)
    #  2. climb (when on column and not at top)
    #  3. dismount when near top
    #  4. dodge coconut (only when not on a useful climb)
    #  5. approach reachable ladder
    #  6. fallback: move right
    action = jnp.where(
        monkey_in_punch,
        punch_action,
        jnp.where(
            climbing,
            UP,
            jnp.where(
                near_top,
                dismount_action,
                jnp.where(
                    coco_danger,
                    dodge_action,
                    jnp.where(any_reach | any_on, approach_action, no_ladder_action),
                ),
            ),
        ),
    )

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
"""API-free Kangaroo patch based on shaped-fresh policy_v2.

The only behavioral change is a narrow guard around monkey punching:
RIGHTFIRE/LEFTFIRE keeps top priority for the first lower-platform reward, but
it is not allowed to override ladder-top dismount behavior once the player is
already on a ladder column and near the ladder top.
"""

import jax
import jax.numpy as jnp


P_X, P_Y, P_W, P_H = 0, 1, 2, 3

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

NOOP = 0
UP = 2
RIGHT = 3
LEFT = 4
RIGHTFIRE = 11
LEFTFIRE = 12


def init_params():
    # Tuned best_params from:
    # runs/single_game/kangaroo_20260501_223920_shaped_fresh_s20260501
    return {
        "reach_tol": jnp.float32(14.236533164978027),
        "align_tol": jnp.float32(4.5095696449279785),
        "top_tol": jnp.float32(7.8007354736328125),
        "punch_dx": jnp.float32(17.71430778503418),
        "punch_dy": jnp.float32(19.727657318115234),
        "danger_r": jnp.float32(19.939918518066406),
        "col_frac": jnp.float32(0.7089129090309143),
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
    score = jnp.where(reach, jnp.abs(lcx - pcx), jnp.float32(1e6))
    idx = jnp.argmin(score)
    return jnp.any(reach), lcx[idx], lty[idx], lby[idx], lw[idx]


def _on_column_ladder(obs, pcx, pby, py, col_frac):
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lby = ly + lh
    lty = ly
    half = lw * (col_frac + 0.1)
    in_col = (jnp.abs(lcx - pcx) < half) & (la > 0.5)
    vert_ok = (py > lty - 4.0) & (py < lby + 4.0)
    on = in_col & vert_ok
    score = jnp.where(on, jnp.abs(lcx - pcx), jnp.float32(1e6))
    idx = jnp.argmin(score)
    return jnp.any(on), lcx[idx], lty[idx], lby[idx], lw[idx]


def _select_next_ladder_from_top(obs, pcx, lty_cur, reach_tol):
    lx, ly, lw, lh, la = _ladders(obs)
    lcx = lx + lw * 0.5
    lby = ly + lh
    lty = ly
    reach = (jnp.abs(lby - lty_cur) < reach_tol + 6.0) & (lty < lty_cur - 4.0) & (la > 0.5)
    score = jnp.where(reach, jnp.abs(lcx - pcx), jnp.float32(1e6))
    idx = jnp.argmin(score)
    return jnp.any(reach), lcx[idx]


def _nearest_monkey(obs, px, py):
    mx = jax.lax.dynamic_slice(obs, (MON_X0,), (4,))
    my = jax.lax.dynamic_slice(obs, (MON_Y0,), (4,))
    ma = jax.lax.dynamic_slice(obs, (MON_A0,), (4,))
    dx = mx - px
    dy = my - py
    dist = jnp.where(ma > 0.5, jnp.abs(dx) + jnp.abs(dy), jnp.float32(1e6))
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
    falling_near = (fa > 0.5) & (fd < r)

    threat_x = jnp.where(
        any_near,
        cx[jnp.argmin(jnp.where(ca > 0.5, d, jnp.float32(1e6)))],
        fx,
    )
    return any_near | falling_near, threat_x


def _move_toward_x(dx, deadband):
    return jnp.where(jnp.abs(dx) < deadband, NOOP, jnp.where(dx > 0, RIGHT, LEFT))


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

    any_reach, r_lcx, r_lty, r_lby, _ = _select_reachable_ladder(obs, pcx, pby, reach_tol)
    any_on, c_lcx, c_lty, c_lby, _ = _on_column_ladder(obs, pcx, pby, py, col_frac)

    tgt_lcx = jnp.where(any_on, c_lcx, r_lcx)
    tgt_lty = jnp.where(any_on, c_lty, r_lty)
    tgt_lby = jnp.where(any_on, c_lby, r_lby)

    align_dx = tgt_lcx - pcx
    near_top = any_on & (py < tgt_lty + top_tol)
    above_bottom = py < tgt_lby - 2.0
    climbing = any_on & above_bottom & ~near_top

    has_mon, mdx, mdy = _nearest_monkey(obs, pcx, pcy)
    monkey_in_punch = has_mon & (jnp.abs(mdx) < punch_dx) & (jnp.abs(mdy) < punch_dy)
    # This is the patch: do not let repeated punching override route progress
    # once the player is already at the ladder-top/dismount state.
    monkey_in_punch = monkey_in_punch & ~(any_on & near_top)
    punch_action = jnp.where(mdx > 0, RIGHTFIRE, LEFTFIRE)

    coco_danger, threat_x = _coconut_danger(obs, pcx, pcy, danger_r)
    dodge_action = jnp.where(threat_x > pcx, LEFT, RIGHT)

    has_next, next_lcx = _select_next_ladder_from_top(obs, pcx, tgt_lty, reach_tol)
    dismount_dx = next_lcx - pcx
    dismount_action = jnp.where(has_next, jnp.where(dismount_dx > 0, RIGHT, LEFT), RIGHT)

    approach_action = _move_toward_x(align_dx, align_tol)
    approach_action = jnp.where(
        (jnp.abs(align_dx) < align_tol) & (any_reach | any_on),
        UP,
        approach_action,
    )

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
                    jnp.where(any_reach | any_on, approach_action, RIGHT),
                ),
            ),
        ),
    )
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)

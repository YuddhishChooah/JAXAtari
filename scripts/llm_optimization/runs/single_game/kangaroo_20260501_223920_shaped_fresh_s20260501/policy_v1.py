"""
Auto-generated policy v1
Generated at: 2026-05-01 23:01:14
"""

import jax
import jax.numpy as jnp


# ---- observation aliases ----
P_X, P_Y, P_W, P_H = 0, 1, 2, 3
P_ORI = 7

LAD_X0, LAD_X1 = 168, 188
LAD_Y0, LAD_Y1 = 188, 208
LAD_W0, LAD_W1 = 208, 228
LAD_H0, LAD_H1 = 228, 248
LAD_A0, LAD_A1 = 248, 268

MON_X0, MON_X1 = 376, 380
MON_Y0, MON_Y1 = 380, 384
MON_A0, MON_A1 = 392, 396

COCO_X0, COCO_X1 = 408, 412
COCO_Y0, COCO_Y1 = 412, 416
COCO_A0, COCO_A1 = 424, 428

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
        "reach_tol": jnp.float32(14.0),     # |ladder_bottom - player_bottom| tolerance
        "align_tol": jnp.float32(4.0),      # |ladder_center - player_center| to start climbing
        "top_tol":   jnp.float32(8.0),      # near top of ladder -> dismount
        "punch_dx":  jnp.float32(16.0),     # x-range to punch a monkey
        "danger_r":  jnp.float32(20.0),     # avoid coconut radius
        "climb_bias": jnp.float32(2.0),     # small bias to prefer climbing
    }


def _select_reachable_ladder(obs, player_cx, player_by, reach_tol):
    lx = jax.lax.dynamic_slice(obs, (LAD_X0,), (20,))
    ly = jax.lax.dynamic_slice(obs, (LAD_Y0,), (20,))
    lw = jax.lax.dynamic_slice(obs, (LAD_W0,), (20,))
    lh = jax.lax.dynamic_slice(obs, (LAD_H0,), (20,))
    la = jax.lax.dynamic_slice(obs, (LAD_A0,), (20,))

    lcx = lx + lw * 0.5
    lby = ly + lh
    lty = ly

    # reachable: ladder bottom near player's feet, ladder goes upward
    reach = (jnp.abs(lby - player_by) < reach_tol) & (lty < player_by - 4.0) & (la > 0.5)

    # Among reachable ladders, prefer the one closest in x
    dx = jnp.abs(lcx - player_cx)
    big = jnp.float32(1e6)
    score = jnp.where(reach, dx, big)
    idx = jnp.argmin(score)
    any_reach = jnp.any(reach)

    return any_reach, lcx[idx], lty[idx], lby[idx], lw[idx]


def _nearest_monkey(obs, px, py):
    mx = jax.lax.dynamic_slice(obs, (MON_X0,), (4,))
    my = jax.lax.dynamic_slice(obs, (MON_Y0,), (4,))
    ma = jax.lax.dynamic_slice(obs, (MON_A0,), (4,))
    big = jnp.float32(1e6)
    dx = mx - px
    dy = jnp.abs(my - py)
    dist = jnp.where(ma > 0.5, jnp.abs(dx) + dy, big)
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

    # sign of avoidance: dodge in opposite x direction of nearest active threat
    threat_x = jnp.where(any_near, cx[jnp.argmin(jnp.where(ca > 0.5, d, 1e6))], fx)
    return any_near | f_near, threat_x


def policy(obs_flat, params):
    obs = obs_flat.astype(jnp.float32)

    px = obs[P_X]
    py = obs[P_Y]
    pw = obs[P_W]
    ph = obs[P_H]
    pcx = px + pw * 0.5
    pby = py + ph

    reach_tol = params["reach_tol"]
    align_tol = params["align_tol"]
    top_tol = params["top_tol"]
    punch_dx = params["punch_dx"]
    danger_r = params["danger_r"]

    any_reach, lcx, lty, lby, lw = _select_reachable_ladder(obs, pcx, pby, reach_tol)

    # alignment to ladder center
    align_dx = lcx - pcx
    aligned = jnp.abs(align_dx) < align_tol

    # are we currently climbing? require strong center-band overlap, not 1px
    on_column = (jnp.abs(align_dx) < (lw * 0.4 + 1.0)) & any_reach
    above_bottom = py < lby - 2.0
    near_top = py < lty + top_tol
    climbing = on_column & above_bottom & ~near_top

    # threat detection
    has_mon, mdx, mdy = _nearest_monkey(obs, pcx, py + ph * 0.5)
    monkey_in_punch = has_mon & (jnp.abs(mdx) < punch_dx) & (mdy < 12.0)

    coco_danger, threat_x = _coconut_danger(obs, pcx, py + ph * 0.5, danger_r)

    # ---- decision tree ----
    # 1. climbing has priority once aligned and not at top
    # 2. punch nearby monkey
    # 3. dodge coconut
    # 4. move horizontally toward selected ladder
    # 5. if at top / no reachable ladder, sweep right toward next route

    # default fallback action
    dismount_action = jnp.where(near_top & on_column, RIGHT, RIGHT)

    # horizontal movement toward ladder
    move_right = align_dx > 0
    move_to_ladder = jnp.where(move_right, RIGHT, LEFT)

    # punch direction: face monkey
    punch_action = jnp.where(mdx > 0, RIGHTFIRE, LEFTFIRE)

    # dodge: move opposite of threat
    dodge_action = jnp.where(threat_x > pcx, LEFT, RIGHT)

    # If on ladder column and aligned and not at top -> UP
    climb_action = UP

    # No reachable ladder: try moving right (general progression direction)
    no_ladder_action = RIGHT

    action = jnp.where(
        climbing & aligned,
        climb_action,
        jnp.where(
            monkey_in_punch,
            punch_action,
            jnp.where(
                coco_danger,
                dodge_action,
                jnp.where(
                    near_top & on_column,
                    dismount_action,
                    jnp.where(any_reach, move_to_ladder, no_ladder_action),
                ),
            ),
        ),
    )

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
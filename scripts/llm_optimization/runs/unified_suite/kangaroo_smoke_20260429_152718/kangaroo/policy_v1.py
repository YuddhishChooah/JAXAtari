"""
Auto-generated policy v1
Generated at: 2026-04-29 15:28:00
"""

import jax
import jax.numpy as jnp


# observation index constants
PLAYER_X = 0
PLAYER_Y = 1
PLAYER_ORIENT = 7

LADDER_X_START, LADDER_X_END = 168, 188
LADDER_Y_START, LADDER_Y_END = 188, 208
LADDER_W_START, LADDER_W_END = 208, 228
LADDER_ACT_START, LADDER_ACT_END = 248, 268

CHILD_X = 360
CHILD_Y = 361

MONKEY_X_START, MONKEY_X_END = 376, 380
MONKEY_Y_START, MONKEY_Y_END = 380, 384
MONKEY_ACT_START, MONKEY_ACT_END = 392, 396

COCO_X_START, COCO_X_END = 408, 412
COCO_Y_START, COCO_Y_END = 412, 416
COCO_ACT_START, COCO_ACT_END = 424, 428

FALL_COCO_X = 368
FALL_COCO_Y = 369
FALL_COCO_ACT = 372

# actions
NOOP, FIRE = 0, 1
UP, RIGHT, LEFT, DOWN = 2, 3, 4, 5


def init_params():
    return {
        "ladder_align_tol": jnp.float32(8.0),
        "ladder_y_bias": jnp.float32(1.5),
        "danger_x": jnp.float32(20.0),
        "danger_y": jnp.float32(16.0),
        "punch_dist": jnp.float32(18.0),
        "climb_y_margin": jnp.float32(4.0),
    }


def _best_ladder(px, py, lx, ly, lw, lact, y_bias):
    # prefer active ladders that are above (or near) the player; weighted by dx + bias*dy_above
    cx = lx + lw * 0.5
    dx = jnp.abs(cx - px)
    dy_above = jnp.maximum(py - ly, 0.0)  # how much ladder reaches above player
    # cost: small if ladder is close in x and extends above us
    cost = dx + y_bias * (50.0 - jnp.minimum(dy_above, 50.0))
    # mask inactive
    cost = jnp.where(lact > 0.5, cost, 1e6)
    idx = jnp.argmin(cost)
    return lx[idx], ly[idx], lw[idx], lact[idx]


def _nearest_threat(px, py, xs, ys, acts):
    dx = xs - px
    dy = ys - py
    dist = jnp.abs(dx) + jnp.abs(dy)
    dist = jnp.where(acts > 0.5, dist, 1e6)
    idx = jnp.argmin(dist)
    return xs[idx], ys[idx], acts[idx], dist[idx]


def policy(obs_flat, params):
    px = obs_flat[PLAYER_X]
    py = obs_flat[PLAYER_Y]

    lx = obs_flat[LADDER_X_START:LADDER_X_END]
    ly = obs_flat[LADDER_Y_START:LADDER_Y_END]
    lw = obs_flat[LADDER_W_START:LADDER_W_END]
    lact = obs_flat[LADDER_ACT_START:LADDER_ACT_END]

    mx = obs_flat[MONKEY_X_START:MONKEY_X_END]
    my = obs_flat[MONKEY_Y_START:MONKEY_Y_END]
    mact = obs_flat[MONKEY_ACT_START:MONKEY_ACT_END]

    cx_ar = obs_flat[COCO_X_START:COCO_X_END]
    cy_ar = obs_flat[COCO_Y_START:COCO_Y_END]
    cact = obs_flat[COCO_ACT_START:COCO_ACT_END]

    fcx = obs_flat[FALL_COCO_X]
    fcy = obs_flat[FALL_COCO_Y]
    fcact = obs_flat[FALL_COCO_ACT]

    # pick best ladder
    blx, bly, blw, blact = _best_ladder(px, py, lx, ly, lw, lact, params["ladder_y_bias"])
    ladder_cx = blx + blw * 0.5
    dx_lad = ladder_cx - px
    aligned = jnp.abs(dx_lad) < params["ladder_align_tol"]
    have_ladder = blact > 0.5

    # threats
    mxn, myn, _, mdist = _nearest_threat(px, py, mx, my, mact)
    monkey_dx = mxn - px
    monkey_dy = jnp.abs(myn - py)
    monkey_close = (jnp.abs(monkey_dx) < params["punch_dist"]) & (monkey_dy < params["danger_y"]) & (mdist < 1e5)

    # any thrown coconut nearby
    coc_dx = jnp.abs(cx_ar - px)
    coc_dy = jnp.abs(cy_ar - py)
    coc_near = jnp.any((cact > 0.5) & (coc_dx < params["danger_x"]) & (coc_dy < params["danger_y"]))

    # falling coconut
    fall_dx = jnp.abs(fcx - px)
    fall_dy = jnp.abs(fcy - py)
    fall_near = (fcact > 0.5) & (fall_dx < params["danger_x"]) & (fall_dy < params["danger_y"] * 2.0)

    # default action: move toward best ladder, then climb
    move_horiz = jnp.where(dx_lad > 0, RIGHT, LEFT)

    # if aligned with ladder and ladder extends above us, climb up
    can_climb = aligned & have_ladder & (bly < py + params["climb_y_margin"])
    action = jnp.where(can_climb, UP, move_horiz)

    # if no usable ladder, drift toward child x
    child_x = obs_flat[CHILD_X]
    child_dx = child_x - px
    no_ladder_action = jnp.where(jnp.abs(child_dx) > 4.0,
                                 jnp.where(child_dx > 0, RIGHT, LEFT),
                                 UP)
    action = jnp.where(have_ladder, action, no_ladder_action)

    # punch monkey if very close
    punch_action = jnp.where(monkey_dx > 0,
                             jnp.where(jnp.abs(monkey_dx) < params["punch_dist"] * 0.6, FIRE, RIGHT),
                             jnp.where(jnp.abs(monkey_dx) < params["punch_dist"] * 0.6, FIRE, LEFT))
    action = jnp.where(monkey_close, FIRE, action)

    # dodge thrown coconut by jumping (FIRE) or stepping
    action = jnp.where(coc_near & ~can_climb, FIRE, action)

    # falling coconut: step sideways away from ladder/under-coconut
    dodge_dir = jnp.where(fcx > px, LEFT, RIGHT)
    action = jnp.where(fall_near & ~can_climb, dodge_dir, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
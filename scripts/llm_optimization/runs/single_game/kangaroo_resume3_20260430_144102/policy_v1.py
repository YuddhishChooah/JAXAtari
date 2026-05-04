"""
Auto-generated policy v1
Generated at: 2026-04-29 16:05:16
"""

import jax
import jax.numpy as jnp

# Action constants
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = 6, 7, 8, 9
UPFIRE, RIGHTFIRE, LEFTFIRE = 10, 11, 12


def init_params():
    return {
        "ladder_reach_tol": jnp.array(12.0),   # tol for ladder bottom near player feet
        "ladder_align_tol": jnp.array(4.0),    # x tol to start climbing
        "danger_dx": jnp.array(16.0),          # horizontal danger distance
        "danger_dy": jnp.array(12.0),          # vertical danger distance
        "punch_dx": jnp.array(14.0),           # punch range
        "upward_bias": jnp.array(8.0),         # ladder must be at least this much higher
    }


def _select_ladder(obs, player_cx, player_by, reach_tol, upward_bias):
    lx = jax.lax.dynamic_slice(obs, (168,), (20,))
    ly = jax.lax.dynamic_slice(obs, (188,), (20,))
    lw = jax.lax.dynamic_slice(obs, (208,), (20,))
    lh = jax.lax.dynamic_slice(obs, (228,), (20,))
    la = jax.lax.dynamic_slice(obs, (248,), (20,))

    lcx = lx + lw * 0.5
    lby = ly + lh
    lty = ly

    # Reachable: ladder bottom within reach_tol below/above player feet
    reach_err = jnp.abs(lby - player_by)
    reachable = (reach_err < reach_tol) & (la > 0.5)
    # Useful: top above player (smaller y)
    useful = lty < (player_by - upward_bias)
    valid = reachable & useful

    # Score: prefer valid ladders, then closest in x
    dx = jnp.abs(lcx - player_cx)
    big = jnp.array(1e6)
    cost = jnp.where(valid, dx, big)
    idx = jnp.argmin(cost)
    has_valid = jnp.any(valid)
    return lcx[idx], lty[idx], lby[idx], has_valid


def _nearest_threat(obs, px, py, danger_dx, danger_dy):
    # Monkeys
    mx = jax.lax.dynamic_slice(obs, (376,), (4,))
    my = jax.lax.dynamic_slice(obs, (380,), (4,))
    ma = jax.lax.dynamic_slice(obs, (392,), (4,))
    # Thrown coconuts
    cx = jax.lax.dynamic_slice(obs, (408,), (4,))
    cy = jax.lax.dynamic_slice(obs, (412,), (4,))
    ca = jax.lax.dynamic_slice(obs, (424,), (4,))
    # Falling coconut
    fcx = obs[368]
    fcy = obs[369]
    fca = obs[372]

    all_x = jnp.concatenate([mx, cx, jnp.array([fcx])])
    all_y = jnp.concatenate([my, cy, jnp.array([fcy])])
    all_a = jnp.concatenate([ma, ca, jnp.array([fca])])

    ddx = all_x - px
    ddy = all_y - py
    near = (jnp.abs(ddx) < danger_dx) & (jnp.abs(ddy) < danger_dy) & (all_a > 0.5)
    # Identify monkey threat for punching: monkey + within punch range
    mdx = mx - px
    mdy = my - py
    monkey_near = (jnp.abs(mdx) < danger_dx) & (jnp.abs(mdy) < danger_dy) & (ma > 0.5)
    any_threat = jnp.any(near)
    any_monkey = jnp.any(monkey_near)

    # signed dx of nearest monkey (for facing)
    mcost = jnp.where(monkey_near, jnp.abs(mdx), jnp.array(1e6))
    midx = jnp.argmin(mcost)
    monkey_sign_dx = mdx[midx]
    return any_threat, any_monkey, monkey_sign_dx


def policy(obs_flat, params):
    px = obs_flat[0]
    py = obs_flat[1]
    pw = obs_flat[2]
    ph = obs_flat[3]
    pcx = px + pw * 0.5
    pby = py + ph

    reach_tol = params["ladder_reach_tol"]
    align_tol = params["ladder_align_tol"]
    danger_dx = params["danger_dx"]
    danger_dy = params["danger_dy"]
    punch_dx = params["punch_dx"]
    upward_bias = params["upward_bias"]

    lcx, lty, lby, has_valid = _select_ladder(obs_flat, pcx, pby, reach_tol, upward_bias)

    # Child as fallback target (for x movement when no valid ladder)
    child_x = obs_flat[360]
    child_active = obs_flat[364]
    fallback_x = jnp.where(child_active > 0.5, child_x, pcx)
    target_x = jnp.where(has_valid, lcx, fallback_x)

    dx_to_target = target_x - pcx
    aligned = jnp.abs(dx_to_target) < align_tol

    # Are we currently on a ladder x-band? If aligned and a valid ladder exists, climb.
    on_ladder_band = aligned & has_valid

    # Threats
    any_threat, any_monkey, monkey_sign_dx = _nearest_threat(
        obs_flat, pcx, py, danger_dx, danger_dy
    )

    # Default action: move toward target_x; climb if aligned with valid ladder
    move_right = dx_to_target > 0
    horiz_action = jnp.where(move_right, RIGHT, LEFT)
    climb_action = UP

    base_action = jnp.where(on_ladder_band, climb_action, horiz_action)

    # If a monkey is within punch range, face and punch
    punch_in_range = any_monkey & (jnp.abs(monkey_sign_dx) < punch_dx)
    punch_action = jnp.where(monkey_sign_dx > 0, RIGHTFIRE, LEFTFIRE)

    # If general threat (coconut) and not on ladder band, try jumping (FIRE)
    threat_action = jnp.where(on_ladder_band, UP, FIRE)

    action = base_action
    action = jnp.where(any_threat & ~punch_in_range, threat_action, action)
    action = jnp.where(punch_in_range, punch_action, action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
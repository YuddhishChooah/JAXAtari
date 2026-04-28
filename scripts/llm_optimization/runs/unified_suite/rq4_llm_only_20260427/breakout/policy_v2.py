"""
Auto-generated policy v2
Generated at: 2026-04-27 17:44:38
"""

import jax
import jax.numpy as jnp


# Action ids
NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3

# Observation indices
PLAYER_X = 0
PLAYER_W = 2
BALL_X = 8
BALL_Y = 9
BALL_W = 10
BALL_ACTIVE = 12
BALL_STATE = 14
BALL_ORIENT = 15
LIVES = 124

# Approximate playfield horizontal bounds (Atari Breakout)
WALL_LEFT = 8.0
WALL_RIGHT = 152.0
PADDLE_Y = 189.0  # approx y of paddle top


def init_params():
    return {
        "dead_zone": jnp.float32(1.5),
        "commit_zone": jnp.float32(4.0),
        "k_lead": jnp.float32(0.55),
        "anchor_x": jnp.float32(80.0),
        "anchor_pull": jnp.float32(0.25),
        "descend_gain": jnp.float32(1.0),
    }


def _paddle_center(obs):
    return obs[PLAYER_X] + obs[PLAYER_W] * 0.5


def _move_action(err, dead_zone):
    go_right = err > dead_zone
    go_left = err < -dead_zone
    return jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))


def _reflect(x, lo, hi):
    # Fold x back into [lo, hi] using triangle-wave reflection.
    span = hi - lo
    span = jnp.maximum(span, 1.0)
    t = (x - lo) / span
    # reflect using abs of fractional part of (t mod 2 - 1)
    m = jnp.mod(t, 2.0)
    r = 1.0 - jnp.abs(m - 1.0)
    return lo + r * span


def _ball_dir_x(obs):
    # ball.orientation is often a small int encoding direction.
    # Treat positive => moving right, negative => moving left, 0 => unknown.
    o = obs[BALL_ORIENT]
    s = obs[BALL_STATE]
    # Combine: prefer orientation sign; fall back to (ball_x - anchor) sign.
    raw = o + 0.1 * s
    sgn = jnp.sign(raw)
    # If sign is zero, use a weak fallback toward play-area center.
    fallback = jnp.sign(80.0 - obs[BALL_X])
    return jnp.where(jnp.abs(sgn) < 0.5, fallback, sgn)


def _ball_descending(obs):
    # Heuristic: ball is descending in lower half of screen.
    # Without history, approximate "descending" as ball_y past a threshold
    # OR orientation/state hints. We use ball_y > 90 as descending region.
    return obs[BALL_Y] > 90.0


def policy(obs_flat, params):
    obs = obs_flat.astype(jnp.float32)

    paddle_cx = _paddle_center(obs)
    ball_x = obs[BALL_X]
    ball_y = obs[BALL_Y]
    ball_active = obs[BALL_ACTIVE] > 0.5

    dir_x = _ball_dir_x(obs)
    descending = _ball_descending(obs)

    # Predictive landing x: project ball forward to paddle height.
    # Vertical distance to paddle, clipped to non-negative.
    dy = jnp.maximum(PADDLE_Y - ball_y, 0.0)

    # Lead magnitude grows with descent gain when descending.
    gain = jnp.where(descending, params["descend_gain"], 0.3)
    predicted = ball_x + params["k_lead"] * gain * dir_x * dy

    # Reflect predicted x off side walls so corner trajectories fold correctly.
    predicted = _reflect(predicted, WALL_LEFT, WALL_RIGHT)

    # Anchor blend: anchor dominates only when ball is ascending (not descending).
    # When descending, commit fully to predicted landing.
    anchor_w = jnp.where(descending, 0.0, params["anchor_pull"])
    target = (1.0 - anchor_w) * predicted + anchor_w * params["anchor_x"]

    err = target - paddle_cx

    # Hysteresis: use larger dead zone when already close, smaller when far.
    near = jnp.abs(err) < params["commit_zone"]
    dz = jnp.where(near, params["commit_zone"], params["dead_zone"])
    move = _move_action(err, dz)

    # FIRE when ball not active (waiting to launch).
    action = jnp.where(ball_active, move, FIRE)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
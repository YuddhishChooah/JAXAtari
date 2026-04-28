"""
Auto-generated policy v3
Generated at: 2026-04-27 17:45:45
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
PLAYER_Y = 1
PLAYER_W = 2
BALL_X = 8
BALL_Y = 9
BALL_ACTIVE = 12
BALL_STATE = 14
BALL_ORIENT = 15
BLOCKS_START = 16
BLOCKS_END = 124
LIVES = 124


def init_params():
    return {
        "dead_zone": jnp.float32(1.5),
        "descend_gain": jnp.float32(0.55),
        "ascend_dead_zone": jnp.float32(6.0),
        "vx_scale": jnp.float32(1.1),
        "corner_bias": jnp.float32(3.5),
        "ascend_anchor": jnp.float32(80.0),
    }


def _paddle_center(obs):
    return obs[PLAYER_X] + obs[PLAYER_W] * 0.5


def _move_action(err, dead_zone):
    go_right = err > dead_zone
    go_left = err < -dead_zone
    return jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))


def _ball_vx_sign(obs):
    # Use orientation field as a direction proxy. Many JAXAtari builds encode
    # direction sign in orientation (negative => left, positive => right).
    # Fall back to (ball_x - 80) sign so we still get a usable signal.
    o = obs[BALL_ORIENT]
    s = obs[BALL_STATE]
    raw = o + 0.5 * s
    sign = jnp.sign(raw)
    # If sign is zero, derive from ball position relative to field center.
    fallback = jnp.sign(obs[BALL_X] - 80.0)
    return jnp.where(jnp.abs(sign) < 0.5, fallback, sign)


def _ball_descending(obs):
    # In Atari Breakout the paddle sits near y ~ 189. The ball spends ascending
    # frames moving upward (decreasing y) and descending frames moving down.
    # Without a previous frame we proxy descent using state/orientation parity
    # combined with vertical position: if the ball is in the lower half it is
    # almost always descending toward the paddle.
    return obs[BALL_Y] > 95.0


def _block_side_bias(obs, corner_bias):
    # Count remaining blocks on the left vs right half of the brick wall.
    blocks = jax.lax.dynamic_slice(obs, (BLOCKS_START,), (BLOCKS_END - BLOCKS_START,))
    # 108 blocks: assume row-major 6 rows x 18 cols. Left half = cols 0..8.
    blocks2d = blocks.reshape((6, 18))
    left = jnp.sum(blocks2d[:, :9])
    right = jnp.sum(blocks2d[:, 9:])
    # Bias toward the side that still has more blocks, so we hit the ball with
    # an offset that sends it into the denser side.
    diff = right - left
    side = jnp.tanh(diff * 0.25)  # in [-1, 1]
    return side * corner_bias


def policy(obs_flat, params):
    obs = obs_flat.astype(jnp.float32)

    paddle_cx = _paddle_center(obs)
    paddle_y = obs[PLAYER_Y]
    ball_x = obs[BALL_X]
    ball_y = obs[BALL_Y]
    ball_active = obs[BALL_ACTIVE] > 0.5

    vx_sign = _ball_vx_sign(obs)
    descending = _ball_descending(obs)

    # Vertical distance to paddle plane, normalized.
    dy = jnp.maximum(paddle_y - ball_y, 1.0)
    # Approx time-to-impact proxy ~ dy (since |vy| ~ const). Scale into pixels.
    lead_pixels = params["vx_scale"] * vx_sign * dy * 0.45

    # Predicted intercept x. Clamp into play area so we don't aim at walls.
    predicted_x = jnp.clip(ball_x + lead_pixels, 8.0, 152.0)

    # Corner bias: nudge target so paddle hits ball off-center toward denser side.
    bias = _block_side_bias(obs, params["corner_bias"])

    # Descending: aim predicted intercept + bias, tight dead zone, high gain.
    target_descend = predicted_x + bias
    err_descend = (target_descend - paddle_cx) * params["descend_gain"] / 0.5
    move_descend = _move_action(err_descend, params["dead_zone"])

    # Ascending: drift gently toward an anchor with a wide dead zone to stop jitter.
    err_ascend = params["ascend_anchor"] - paddle_cx
    move_ascend = _move_action(err_ascend, params["ascend_dead_zone"])

    move = jnp.where(descending, move_descend, move_ascend)

    # FIRE when ball is not active (launch / after life loss).
    action = jnp.where(ball_active, move, FIRE)
    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
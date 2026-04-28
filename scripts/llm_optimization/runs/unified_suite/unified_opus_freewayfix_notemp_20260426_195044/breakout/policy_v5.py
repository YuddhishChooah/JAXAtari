"""
Auto-generated policy v5
Generated at: 2026-04-26 20:58:45
"""

import jax
import jax.numpy as jnp

# Action constants
NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3

# Observation indices
PLAYER_X = 0
BALL_X = 8
BALL_Y = 9
BALL_ACTIVE = 12
BLOCKS_START = 16
BLOCKS_END = 124

PADDLE_HALF = 8.0
SCREEN_CENTER = 80.0  # approximate horizontal center


def init_params():
    return {
        "aim_offset": jnp.array(6.0),       # off-center hit magnitude (pixels)
        "descent_y": jnp.array(140.0),      # ball y threshold: below = descending/near
        "dead_zone_desc": jnp.array(1.5),   # tight when ball is coming down
        "dead_zone_asc": jnp.array(6.0),    # loose when ball is going up
        "side_bias_gain": jnp.array(4.0),   # extra aim shift toward dense side
        "fire_bias": jnp.array(0.5),
    }


def _select_move(err, dead_zone):
    go_right = err > dead_zone
    go_left = err < -dead_zone
    return jnp.where(go_right, RIGHT, jnp.where(go_left, LEFT, NOOP))


def _block_side_bias(obs_flat):
    # Returns +1 if right side has more bricks, -1 if left, 0 if equal-ish.
    blocks = jax.lax.dynamic_slice(obs_flat, (BLOCKS_START,), (108,))
    # 108 = 6 rows * 18 cols. Split by column index parity of position within row.
    # Simpler: split flat array in halves by column position. Each row has 18 cols;
    # left half = cols 0..8, right half = cols 9..17.
    blocks_2d = blocks.reshape((6, 18))
    left_count = jnp.sum(blocks_2d[:, :9])
    right_count = jnp.sum(blocks_2d[:, 9:])
    diff = right_count - left_count
    # Normalize to [-1, 1]
    total = left_count + right_count + 1.0
    return diff / total


def policy(obs_flat, params):
    paddle_x = obs_flat[PLAYER_X]
    ball_x = obs_flat[BALL_X]
    ball_y = obs_flat[BALL_Y]
    ball_active = obs_flat[BALL_ACTIVE]

    paddle_center = paddle_x + PADDLE_HALF

    # Side bias from remaining blocks: aim toward denser side.
    side_bias = _block_side_bias(obs_flat)  # in [-1, 1], positive = right denser

    # Descent gate: when ball y is past threshold, treat as descending/imminent.
    descending = ball_y > params["descent_y"]

    # Off-center striking: shift target opposite to side bias so paddle hits ball
    # with the side that deflects ball TOWARD the dense side.
    # If dense side is right (side_bias > 0), we want ball to go right after hit,
    # which means hitting it with the right portion of the paddle, i.e. paddle
    # should be slightly LEFT of ball -> target shifted LEFT -> negative offset.
    aim_shift = -params["aim_offset"] * jnp.sign(side_bias) \
                - params["side_bias_gain"] * side_bias

    # Only apply aim shift on descent; on ascent, just track loosely.
    target_x = ball_x + jnp.where(descending, aim_shift, 0.0)

    err = target_x - paddle_center

    dead_zone = jnp.where(descending,
                          params["dead_zone_desc"],
                          params["dead_zone_asc"])

    move_action = _select_move(err, dead_zone)

    need_fire = ball_active < params["fire_bias"]
    action = jnp.where(need_fire, FIRE, move_action)

    return action.astype(jnp.int32)


def measure_main(episode_rewards, episode_scores):
    return jnp.sum(episode_rewards)
"""
Language Guided Policy Search (LeGPS) loop for JAXAtari games.

This module implements the unified thesis pipeline:
1. Generate a parametric policy structure with an LLM
2. Optimize the policy's numeric parameters with CMA-ES, or skip search for LLM-only ablations
3. Evaluate the tuned controller on the base game task
4. Feed back measured behavior for conservative iterative revision
"""

import os
import sys
import json
import time
import argparse
import importlib.util
import inspect
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Callable, List
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import lax, vmap

# Gaussian Process for Bayesian Optimization
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import minimize

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jaxatari
from jaxatari.wrappers import AtariWrapper, ObjectCentricWrapper, FlattenObservationWrapper

# Import Logger infrastructure
from scripts.llm_optimization.core.logger import (
    create_logger, AblationConfig, format_metrics_for_llm, get_all_metric_descriptions
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for the LLM optimization loop."""
    # LLM settings
    llm_provider: str = "anthropic"  # "openai" or "anthropic"
    llm_model: str = "claude-opus-4-7"
    api_key: Optional[str] = None
    max_tokens: int = 8192
    temperature: Optional[float] = None
    
    # Evaluation settings - MASSIVE PARALLELISM
    num_parallel_envs: int = 1024  # Run 1024 environments in parallel!
    max_steps_per_episode: int = 2000
    num_eval_episodes: int = 1024  # Evaluate over 1024 episodes
    frame_stack_size: int = 0  # 0 = use game default
    search_max_steps: int = 0  # 0 = use max_steps_per_episode
    
    # Parameter search settings - CMA-ES
    optimizer: str = "cma-es"  # "none", "random", "cma-es", or "bayes"
    num_param_samples: int = 64  # Population size for CMA-ES
    param_perturbation_scale: float = 0.3  # Initial sigma for CMA-ES
    cma_es_generations: int = 20  # Number of CMA-ES generations per iteration
    
    # Optimization loop settings
    max_iterations: int = 10
    target_score: float = 15.0  # Win by 15 points on average
    diagnostic_refinement: bool = True  # Ask for a diagnosis before improvement rewrites.
    resume_from_results: bool = False  # Continue from output_dir/optimization_results.json.
    resume_best_params_for_cma: bool = True  # Seed CMA-ES from the previous best params when compatible.
    refresh_resume_metrics: bool = True  # Re-evaluate resumed best with current logger before prompting.
    stop_on_strong_best: bool = True  # Preserve a strong best by stopping after a worse rewrite.
    
    # File paths
    output_dir: str = "scripts/llm_optimization/runs/single_game/pong"
    
    # Debug settings
    verbose: bool = True
    save_intermediate: bool = True
    seed: int = 42


@dataclass(frozen=True)
class GamePromptSpec:
    """Structured per-game metadata for the unified prompt pipeline."""
    environment_description: str
    design_principles: str
    failure_modes: str
    improvement_guidelines: str
    benchmark_context: str = ""


# ============================================================================
# Pong Environment Description (for LLM prompt)
# ============================================================================

PONG_ENVIRONMENT_DESCRIPTION = """
## Pong Environment Description

You are generating a policy for the Atari Pong game implemented in JAX.

### Observation Space
The policy receives a flattened 1D array with 52 features because Pong uses `frame_stack=2`.
Think of it as two object-centric structs concatenated together:

```python
FRAME_SIZE = 26

prev = obs_flat[0:26]
curr = obs_flat[26:52]

prev.player.x = prev[0]
prev.player.y = prev[1]
prev.player.w = prev[2]
prev.player.h = prev[3]
prev.player.active = prev[4]
prev.player.visual_id = prev[5]
prev.player.state = prev[6]
prev.player.orientation = prev[7]

prev.enemy.x = prev[8]
prev.enemy.y = prev[9]
prev.enemy.w = prev[10]
prev.enemy.h = prev[11]
prev.enemy.active = prev[12]
prev.enemy.visual_id = prev[13]
prev.enemy.state = prev[14]
prev.enemy.orientation = prev[15]

prev.ball.x = prev[16]
prev.ball.y = prev[17]
prev.ball.w = prev[18]
prev.ball.h = prev[19]
prev.ball.active = prev[20]
prev.ball.visual_id = prev[21]
prev.ball.state = prev[22]
prev.ball.orientation = prev[23]

prev.score_player = prev[24]
prev.score_enemy = prev[25]

curr.player.x = curr[0]
curr.player.y = curr[1]
curr.player.w = curr[2]
curr.player.h = curr[3]
curr.player.active = curr[4]
curr.player.visual_id = curr[5]
curr.player.state = curr[6]
curr.player.orientation = curr[7]

curr.enemy.x = curr[8]
curr.enemy.y = curr[9]
curr.enemy.w = curr[10]
curr.enemy.h = curr[11]
curr.enemy.active = curr[12]
curr.enemy.visual_id = curr[13]
curr.enemy.state = curr[14]
curr.enemy.orientation = curr[15]

curr.ball.x = curr[16]
curr.ball.y = curr[17]
curr.ball.w = curr[18]
curr.ball.h = curr[19]
curr.ball.active = curr[20]
curr.ball.visual_id = curr[21]
curr.ball.state = curr[22]
curr.ball.orientation = curr[23]

curr.score_player = curr[24]
curr.score_enemy = curr[25]

ball_dx = curr.ball.x - prev.ball.x
ball_dy = curr.ball.y - prev.ball.y
```

Important: do not use the obsolete 28-feature / 14-feature-frame layout. In this current wrapper, the current frame starts at `obs_flat[26]`.

### Useful Constants
You may define named constants in the generated module so you do not have to remember raw numbers:

```python
NOOP = 0
FIRE = 1
MOVE_UP = 2
MOVE_DOWN = 3
FIRE_UP = 4
FIRE_DOWN = 5

SCREEN_WIDTH = 160
SCREEN_HEIGHT = 210
PLAYFIELD_TOP = 24
PLAYFIELD_BOTTOM = 194
PLAYER_X = 140
ENEMY_X = 16
PADDLE_WIDTH = 4
PADDLE_HEIGHT = 16
BALL_WIDTH = 2
BALL_HEIGHT = 4
PADDLE_CENTER_OFFSET = 8
```

### Action Semantics
The wrapped environment expects compact action indices:
- `0`: `NOOP`
- `1`: `FIRE` without paddle movement
- `2`: move paddle UP
- `3`: move paddle DOWN
- `4`: move paddle UP with FIRE
- `5`: move paddle DOWN with FIRE

Important:
- Y increases downward.
- The player controls the RIGHT paddle.
- `2` and `4` move the paddle upward.
- `3` and `5` move the paddle downward.
- Do not use raw ALE action ids like `11` or `12`; in this wrapper they are not valid movement actions.

### Game Dynamics
- The player controls the RIGHT paddle at x ~= 140.
- The enemy controls the LEFT paddle at x ~= 16.
- The ball bounces between paddles and walls.
- Score a point when the opponent misses the ball.
- Game ends when either player reaches 21 points.
- In our fresh follow-up run, FIRE-augmented Pong outperformed movement-only Pong.

### Key Insights for a Good Policy
1. This run targets FULL MATCHES, so a perfect policy should approach `+21`.
2. Use `prev` and `curr` to compute `ball_dx` and `ball_dy`.
3. Track the predicted intercept Y position, not just the current ball Y.
4. A dead zone helps prevent paddle jitter.
5. Hitting the ball off-center can create steep "spiky" returns that are hard for the enemy to catch.
6. FIRE-combo actions `4` and `5` are allowed, but movement correctness matters more than FIRE usage.
"""


# ============================================================================
# Kangaroo Environment Description and Prompts
# ============================================================================

KANGAROO_ENVIRONMENT_DESCRIPTION = """
## Kangaroo Environment Description

You are generating a policy for the Atari Kangaroo game implemented in JAX.

### Game Objective
Control a mother kangaroo to rescue her child (Joey) at the top of the screen.
Navigate platforms using ladders and jumping. Avoid or punch monkeys and coconuts.

### Observation Space
The observation is a flattened 1D array with 111 features:

Index  | Feature              | Description
-------|----------------------|------------------------------------------
0      | player_x             | Player X position (0-160)
1      | player_y             | Player Y position (0-210, lower=higher on screen)
2      | player_o             | Orientation (-1=left, 1=right)
3-42   | platform_positions   | 20 platforms with (x,y) coordinates
43-82  | ladder_positions     | 20 ladders with (x,y) coordinates
83-88  | fruit_positions      | 3 fruits with (x,y), -1 if not present
89-90  | bell_position        | Bell location (x,y) at top
91-92  | child_position       | Joey's location (x,y) - your goal!
93-94  | falling_coco         | Falling coconut (x,y), -1 if none
95-102 | coco_positions       | 4 thrown coconuts with (x,y)
103-110| monkey_positions     | 4 monkeys with (x,y)

### Action Space (18 actions, focus on these):
- Action 0: NOOP
- Action 1: FIRE (Jump/Punch)
- Action 2: UP (Climb ladder)
- Action 3: RIGHT (Move right)
- Action 4: LEFT (Move left)
- Action 5: DOWN (Climb down/Crouch)
- Action 6: UPRIGHT
- Action 7: UPLEFT

### Key Insights:
1. Lower player_y = higher on screen (goal: minimize player_y to reach child)
2. Ladders are key - detect when player is aligned with a ladder
3. Monkeys throw coconuts - avoid or punch them
4. Bell at top signals level completion
"""

# ============================================================================
# Freeway Environment Description
# ============================================================================

# Stronger Freeway variant: object-centric + two-frame temporal reasoning.
FREEWAY_ENVIRONMENT_DESCRIPTION_V2 = """
## Freeway Environment Description

You are generating a policy for the Atari Freeway game implemented in JAX.

### Game Objective
Guide a chicken from the bottom of the screen to the top, avoiding cars.

### Observation Space
For the stronger Freeway setup, use `frame_stack=2`.
The flattened observation has 176 features = previous frame (88) + current frame (88).

```python
FRAME_SIZE = 88
prev = obs_flat[0:88]
curr = obs_flat[88:176]

prev.chicken.x = prev[0]
prev.chicken.y = prev[1]
curr.chicken.x = curr[0]
curr.chicken.y = curr[1]

# Cars are stored as 10-element arrays, not interleaved structs.
car_i_x = frame[8 + i]
car_i_y = frame[18 + i]
car_i_w = frame[28 + i]
car_i_h = frame[38 + i]
car_i_active = frame[48 + i]

# Velocity estimate for lane i
car_i_dx = curr[8 + i] - prev[8 + i]
```

Important: do not use the obsolete 88-feature / 44-feature-frame layout. In this current wrapper, the current frame starts at `obs_flat[88]`.

### Useful Constants
```python
NOOP = 0
UP = 1
DOWN = 2

SCREEN_WIDTH = 160
SCREEN_HEIGHT = 210
CHICKEN_X = 44
CHICKEN_WIDTH = 6
CHICKEN_HEIGHT = 8
TOP_BORDER = 15
BOTTOM_BORDER = 180
LANE_Y = [27, 43, 59, 75, 91, 107, 123, 139, 155, 171]
LANE_DIRECTION = [-1, -1, -1, -1, -1, +1, +1, +1, +1, +1]
```

### Key Insights
1. Chicken x is fixed near 44, only y changes.
2. Each car stays in a fixed lane, but lane traffic moves with different speeds.
3. A static one-frame distance check is too conservative.
4. A stronger policy should estimate lane-wise motion from the last two frames and time safe lane entries.
5. `UP` should still dominate, but only when the projected overlap for the next lane is safe enough.
6. Valid compact actions are `0=NOOP`, `1=UP`, `2=DOWN`; avoid other actions.
7. Score is strongly horizon-dependent; optimize crossing speed / crossings per 1000 frames, not only raw crossings.
8. Collision is a time penalty, not death. Overly cautious waiting usually loses more score than occasional contact.
9. Strong policies are usually UP/NOOP controllers: move UP whenever the current lane and immediate next lane are sufficiently safe, otherwise briefly wait.
10. DOWN should be rare. It is mainly useful as an emergency escape if the current lane is about to collide and the lower lane is immediately safe.
11. Do not require several future lanes to be safe at once. That is too conservative and causes long stalls.
12. A compact projected-overlap check on the current lane and the next lane is often better than complex TTC trees over many lanes.
"""


# ============================================================================
# Breakout Environment Description
# ============================================================================

BREAKOUT_ENVIRONMENT_DESCRIPTION = """
## Breakout Environment Description

You are generating a policy for the Atari Breakout game implemented in JAX.

### Game Objective
Control a paddle to bounce a ball and break bricks. Score points by destroying bricks.
Lose a life if the ball falls below the paddle.

### Observation Space (126 features)
The observation is a flattened 1D array. In the current local JAXAtari runtime, Breakout exposes this object-centric layout:

```python
# Player object (8 scalars)
player.x = obs_flat[0]
player.y = obs_flat[1]
player.w = obs_flat[2]
player.h = obs_flat[3]
player.active = obs_flat[4]
player.visual_id = obs_flat[5]
player.state = obs_flat[6]
player.orientation = obs_flat[7]

# Ball object (8 scalars)
ball.x = obs_flat[8]
ball.y = obs_flat[9]
ball.w = obs_flat[10]
ball.h = obs_flat[11]
ball.active = obs_flat[12]
ball.visual_id = obs_flat[13]
ball.state = obs_flat[14]
ball.orientation = obs_flat[15]

# Blocks
blocks = obs_flat[16:124]   # 108 entries, 1=present, 0=broken

# Scalars
lives = obs_flat[124]
score = obs_flat[125]
```

### Action Space
Valid action indices for this wrapped environment are:
- `0`: NOOP
- `1`: FIRE
- `2`: RIGHT
- `3`: LEFT

Do NOT use Atari raw action ids like 4 for LEFT. This environment uses the compact minimal action set above.

### Key Insights
1. Paddle center is approximately `player.x + 8`.
2. Ball launch may require `FIRE` when the game or a life reset is waiting to start.
3. The main control problem is horizontal interception, not vertical planning.
4. The ball is most dangerous when it is descending toward the paddle, so policies should react more strongly in that regime.
5. A compact controller should track ball x, optionally use ball y to weight urgency, and avoid oscillating left/right every frame.
6. The policy must use only a few tunable parameters so CMA-ES can calibrate it effectively.
"""


# ============================================================================
# Asterix Environment Description
# ============================================================================

ASTERIX_ENVIRONMENT_DESCRIPTION = """
## Asterix Environment Description

You are generating a policy for the Atari Asterix game implemented in JAX.

### Game Objective
Control Asterix across 8 horizontal lanes. Collect items for points and avoid enemy collisions.
The episode reward is the score increase, so a good policy should maximize item collection while surviving.

### Benchmark Context
- Random baseline is about 210
- DQN is about 6012
- Human is about 8503

So scores in the low hundreds are weak. A strong policy should keep collecting continuously and aim well beyond 1000.

### Observation Space
The policy receives a flattened 1D array with 272 features because Asterix uses `frame_stack=2`.
Each frame has 136 features in this exact order:

```python
FRAME_SIZE = 136
prev = obs_flat[0:136]
curr = obs_flat[136:272]

# Player object (8 scalars)
player.x = frame[0]
player.y = frame[1]
player.w = frame[2]
player.h = frame[3]
player.active = frame[4]
player.visual_id = frame[5]
player.state = frame[6]
player.orientation = frame[7]

# Enemy group (8 arrays of length 8)
enemies.x = frame[8:16]
enemies.y = frame[16:24]
enemies.w = frame[24:32]
enemies.h = frame[32:40]
enemies.active = frame[40:48]
enemies.visual_id = frame[48:56]
enemies.state = frame[56:64]
enemies.orientation = frame[64:72]

# Collectible group (8 arrays of length 8)
collect.x = frame[72:80]
collect.y = frame[80:88]
collect.w = frame[88:96]
collect.h = frame[96:104]
collect.active = frame[104:112]
collect.visual_id = frame[112:120]
collect.state = frame[120:128]
collect.orientation = frame[128:136]
```

The observation does NOT include score, lives, or timers directly. Those appear only in logger feedback after evaluation.

With two stacked frames you can estimate lane-wise motion:
```python
enemy_dx_lane0 = curr[8] - prev[8]
collect_dx_lane0 = curr[72] - prev[72]
```

### Action Space
Valid action indices for this wrapped environment are:
- `0`: NOOP
- `1`: RIGHT
- `2`: LEFT
- `3`: DOWN
- `4`: UPRIGHT
- `5`: UPLEFT
- `6`: DOWNRIGHT
- `7`: DOWNLEFT
- `8`: DOWNLEFT-like duplicate; avoid it unless necessary

Important: use this local wrapped action mapping exactly as written above. It was verified through the same wrapper path used by the optimizer. Do NOT use the base source `ACTION_SET` declaration as the policy mapping. In the optimizer-facing wrapped environment, there is no pure `UP`; upward lane changes come from `UPRIGHT=4` or `UPLEFT=5`.

### Useful Facts
1. The player moves between horizontal lanes and can also move laterally within a lane.
2. Each lane may contain at most one main moving entity at a time, typically either an enemy or a collectible moving left/right.
3. Enemy collisions cost lives and trigger respawn / hit timers internally, but those timers are not in the observation.
4. Collectibles increase score; `collect.visual_id` distinguishes item types.
   In the Asterix phase, the approximate point mapping is `0 -> 50`, `1 -> 100`, `2 -> 200`, `3 -> 300`, `4 -> 0`.
   Do not treat all collectibles as equal-value targets.
5. Lower `player.y` means higher on the screen.
6. Enemy orientation is encoded as 0=stationary, 1=moving right, 2=moving left.
7. Lane centers are approximately at y = [27, 43, 59, 75, 91, 107, 123, 139].
8. Pure survival is not enough; high score comes from repeatedly entering safe collectible lanes and moving laterally to intercept items.
9. There is no pure upward action in this wrapped action set. Upward progress must use `UPRIGHT` or `UPLEFT` depending on which side is safer or more collectible-rich.
10. High score comes from value-weighted harvesting. A 300-point item should usually beat a 50-point item when lane danger is comparable.
11. A policy that idles in a safe lane is weak. When a positive-value item is visible and danger is acceptable, the agent should actively intercept it rather than waiting.
12. The object arrays are lane-aligned by index: entry `i` in `enemies.*` or `collect.*` corresponds directly to lane `i`. Do not try to rediscover lane identity from the object y-coordinates.
13. The player starts near the middle lanes. Initial collectibles often appear in several other lanes, so a policy that mostly moves horizontally in the starting lane will score poorly.
14. Every positive-value collectible matters for score density. Do not ignore 50-point items with a high hard threshold.
15. If the current lane is dangerous, diagonal escape to a safer adjacent lane is usually better than staying in the same lane and only dodging horizontally.
16. Keep danger thresholds on a consistent scale. Do not multiply danger by a weight and then compare it against an unweighted threshold times the same weight.
17. Do not infer that "upper-lane bias" is the main fix just because `avg_stage_index` is high. If `topmost_stage_index` is already near 0, the agent can reach top lanes; the stronger fix is usually reachability-weighted safe collectible selection and fewer deaths.

### Recommended Compact Controller Pattern
For Asterix, a strong symbolic baseline should usually follow this order:
1. Predict enemy and collectible x positions from the two stacked frames.
2. Compute a per-lane danger score from projected enemy distance to the player.
3. Mark active collectible lanes as attractive only when the lane is safe enough.
4. Score active safe collectible lanes by collectible value, x-interception distance, and lane-distance/reachability cost from the current lane.
5. Choose the best active safe collectible lane. Do not let an empty lane beat a reachable safe item just because of stay bonus or patrol behavior.
6. If no active safe collectible is available, move toward the safest nearby lane or patrol laterally; do not optimize inactive lanes as if they contained rewards.
7. A blanket upper-lane bias is usually weaker than a lane-distance penalty plus active safe collectible targeting.
"""


# ============================================================================
# Unified Prompt Framework
# ============================================================================

UNIFIED_SYSTEM_PROMPT = """
You are an expert agent that writes compact parametric JAX policies for Atari-style games.

Your job is to produce a complete Python module that:
1. defines a small tunable parameter dictionary,
2. maps the provided flat observation to a valid raw game action,
3. uses only JAX-safe control flow,
4. is simple enough for derivative-free optimization such as CMA-ES.

The method is:
- the LLM proposes policy structure,
- CMA-ES tunes only the numeric parameters inside that structure.

So optimize for:
- simple logic,
- 3-8 tunable float parameters,
- readable constants and aliases,
- robust action semantics,
- behavior that is easy for black-box search to refine.

Output discipline:
- Return only valid Python module code, with no surrounding explanation.
- Prefer a few short helper functions over one very long policy body.
- Avoid deeply nested parentheses and over-nested `jnp.where` expressions.
- Before finalizing, check that the module is syntactically valid and that parentheses/brackets are balanced.
""".strip()

LLM_ONLY_SYSTEM_PROMPT = """
You are an expert agent that writes compact self-contained JAX policies for Atari-style games.

Your job is to produce a complete Python module that:
1. defines a small parameter dictionary with fixed values chosen by the LLM,
2. maps the provided flat observation to a valid raw game action,
3. uses only JAX-safe control flow,
4. is strong without any numerical optimizer tuning the parameters afterward.

This is an ablation setting:
- the LLM proposes both policy structure and numeric constants,
- no CMA-ES or other inner-loop optimizer will tune the parameter values,
- the policy will be evaluated exactly with `init_params()` values.

So optimize for:
- simple logic,
- carefully chosen numeric constants,
- readable constants and aliases,
- robust action semantics,
- behavior that does not rely on later parameter search.

Output discipline:
- Return only valid Python module code, with no surrounding explanation.
- Prefer a few short helper functions over one very long policy body.
- Avoid deeply nested parentheses and over-nested `jnp.where` expressions.
- Before finalizing, check that the module is syntactically valid and that parentheses/brackets are balanced.
""".strip()

UNIFIED_INITIAL_PROMPT = """
You are generating an initial parametric policy for Atari {game_name}.

{environment_description}

## Shared Method Objective
{method_objective}

## Common Design Rules
1. Prefer 3-8 tunable float parameters.
2. Use the object-centric observation exactly as documented.
3. Prefer simple score-seeking logic over complex hand-crafted heuristics.
4. Use named constants and helper aliases instead of unexplained magic indices.
5. {method_design_rule}

## Game-Specific Design Principles
{design_principles}

## Common Failure Modes To Avoid
{failure_modes}

## Requirements
Provide a complete Python module with:

### 1. init_params() -> dict
Return 3-8 tunable float parameters.

### 2. policy(obs_flat: jnp.ndarray, params: dict) -> int
Return one valid raw action from the game-specific action space described above.
This signature is mandatory. Do not add a third `state` argument, `init_state()`,
or any memoryful API because the optimizer calls exactly `policy(obs_flat, params)`.

### 3. measure_main(episode_rewards: jnp.ndarray, episode_scores: dict) -> float
Return `jnp.sum(episode_rewards)`.

## JAX Constraints
1. Use `jax.lax.cond` or `jnp.where` for branching; no Python `if` in the traced policy.
2. Do not use Python `int()` or `float()` inside `policy`.
3. Keep outputs as valid integer actions.
4. Keep the code concise and stable under tracing.
5. Prefer short helper functions if that reduces syntax risk or nested control flow.
6. Do not use mutable or recurrent state. Estimate velocity from stacked observations only.

{benchmark_context}

Generate the policy now.
""".strip()

UNIFIED_IMPROVEMENT_PROMPT = """
You are improving a parametric policy for Atari {game_name}.

{environment_description}

## Current Performance
- Average Return: {avg_return:.2f}
- Average Player Score: {avg_player_score:.2f}
- Average Enemy Score: {avg_enemy_score:.2f}
- Win Rate: {win_rate:.2%}
- Best Parameters: {best_params}

{benchmark_context}

## Current Policy Code
```python
{previous_code}
```

## Feedback
{feedback}

## Shared Improvement Goal
{improvement_objective}
Do not add complexity unless it directly addresses a concrete failure mode.

## Revision Strategy
{revision_strategy}

## Allowed Scope Of Change
{change_budget}

## Game-Specific Improvement Guidelines
{improvement_guidelines}

## Common Constraints
1. Keep parameter count in the 3-8 range unless there is a very strong reason not to.
2. Avoid prompt-specific gimmicks or brittle hard-coded cases.
3. Preserve valid raw game actions.
4. Keep the policy JAX-safe and black-box-optimization friendly.
5. If the current policy is already strong, prefer minimal edits to helper logic or thresholds over a full rewrite.
6. Preserve any clearly working sub-logic unless feedback points to a specific failure.
7. The policy function signature must remain exactly `policy(obs_flat, params)`.
8. Do not add `state`, `init_state()`, or memoryful policy APIs.

Generate the improved policy now.
""".strip()

UNIFIED_DIAGNOSTIC_SYSTEM_PROMPT = """
You are an expert reinforcement-learning policy auditor.
Your job is to diagnose why a compact parametric JAX policy is underperforming.
Return concise technical analysis only. Do not write policy code.
""".strip()

UNIFIED_DIAGNOSTIC_PROMPT = """
Diagnose why the current Atari {game_name} parametric policy is underperforming.

{environment_description}

## Current Performance
- Average Return: {avg_return:.2f}
- Average Player Score: {avg_player_score:.2f}
- Average Enemy Score: {avg_enemy_score:.2f}
- Win Rate: {win_rate:.2%}
- Best Parameters: {best_params}

{benchmark_context}

## Current Policy Code
```python
{previous_code}
```

## Measured Feedback
{feedback}

{failed_candidate_context}

## Required Output
Return a compact diagnosis with exactly these headings:
1. Root cause
2. Evidence from metrics
3. Policy logic causing it
4. What to preserve
5. What to change next

Do not produce code. Focus on actionable structural changes that preserve a small parameterized policy.
""".strip()

UNIFIED_DIAGNOSTIC_REWRITE_PROMPT = """
You are improving a parametric policy for Atari {game_name}.

Use only the task context below, the current policy, and the diagnosis. Ignore any prior conversation or implicit context.

{environment_description}

## Current Performance
- Average Return: {avg_return:.2f}
- Average Player Score: {avg_player_score:.2f}
- Average Enemy Score: {avg_enemy_score:.2f}
- Win Rate: {win_rate:.2%}
- Best Parameters: {best_params}

{benchmark_context}

## Current Policy Code
```python
{previous_code}
```

## Diagnosis Of Why It Is Suboptimal
{diagnosis}

{failed_candidate_context}

## Rewrite Objective
Generate a complete replacement Python module that directly fixes the diagnosed bottleneck.
{rewrite_objective}

## Rewrite Strategy
{rewrite_strategy}

## Common Constraints
1. Keep parameter count in the 3-8 range unless there is a strong reason not to.
2. Preserve valid raw game actions exactly as described in the task context.
3. Use only JAX-safe traced policy logic: no Python `if`, `int()`, or `float()` inside policy().
4. Prefer small helper functions over deeply nested `jnp.where`.
5. Preserve clearly working sub-logic, but do not preserve passive or low-scoring fallback behavior.
6. Return only valid Python module code, with no explanation.
7. The policy function signature must be exactly `policy(obs_flat, params)`.
8. Do not add `state`, `init_state()`, or any recurrent-memory API.

## Game-Specific Improvement Guidelines
{improvement_guidelines}

Generate the improved policy now.
""".strip()


SKIING_ENVIRONMENT_DESCRIPTION = """
## Skiing Environment Description

You are generating a policy for the Atari Skiing game implemented in JAX.

### Objective
The skier must pass through 20 gates as quickly as possible while avoiding trees and flag poles.
The optimization loop uses the intended ALE-style Skiing objective: every frame gives a small negative time penalty, and terminal scoring penalizes gates that were not passed. Therefore the optimizer still maximizes return, but better play means a less negative total return.

### Action Space
The wrapped action ids are compact indices:

```python
NOOP = 0
RIGHT = 1   # steer toward larger x
LEFT = 2    # steer toward smaller x
FIRE = 3    # jump; mostly irrelevant unless moguls are collidable
DOWN = 4    # tuck / accelerate action
```

Do not use raw ALE action ids. Return only values 0..4.

### Observation Space
The policy receives 146 flattened features because Skiing uses `frame_stack=2`.

```python
FRAME_SIZE = 73
prev = obs_flat[0:73]
curr = obs_flat[73:146]
```

Each single frame is:

```python
# skier object
skier.x           = frame[0]
skier.y           = frame[1]   # fixed near 46
skier.width       = frame[2]
skier.height      = frame[3]
skier.active      = frame[4]
skier.orientation = frame[7]   # degrees: 270,292.5,315,337.5,22.5,45,67.5,90

# two gate objects; each gate stores the left flag pole.
# right flag pole is at flag_x + 32.
flag_x[0:2]       = frame[8:10]
flag_y[0:2]       = frame[10:12]
flag_active[0:2]  = frame[16:18]

# four trees
tree_x[0:4]       = frame[24:28]
tree_y[0:4]       = frame[28:32]
tree_active[0:4]  = frame[40:44]

# two moguls
mogul_x[0:2]      = frame[56:58]
mogul_y[0:2]      = frame[58:60]
mogul_active[0:2] = frame[64:66]

# remaining missed-gate counter; starts at 20 and decreases when gates are passed
remaining_gates   = frame[72]
```

### Useful Derived Quantities
Use stacked frames to estimate object motion:

```python
flag_dy = curr_flag_y - prev_flag_y
tree_dy = curr_tree_y - prev_tree_y
```

Objects move upward on the screen as the skier moves downhill. The skier should position `skier.x` between `flag_x` and `flag_x + 32` before the active gate reaches `skier.y`.

### Steering Mechanics
LEFT and RIGHT are not direct lateral velocity actions. They change the skier's heading one notch at a time. Holding LEFT or RIGHT for too long turns the skier nearly horizontal, which makes downhill speed collapse and prevents the episode from finishing.

The heading sequence is:

```python
# skier_pos implied by orientation
# 0: 270.0  -> hard left, almost no downhill speed
# 1: 292.5  -> strong left
# 2: 315.0  -> moderate left
# 3: 337.5  -> straight/slight left, high downhill speed
# 4: 22.5   -> straight/slight right, high downhill speed
# 5: 45.0   -> moderate right
# 6: 67.5   -> strong right
# 7: 90.0   -> hard right, almost no downhill speed
```

Good policies should use short LEFT/RIGHT corrections to set a useful heading, then release with NOOP or DOWN to keep moving downhill. A policy that returns RIGHT or LEFT almost every frame will time out with very low speed.

### Strategy Notes
1. The core problem is target selection plus smooth steering.
2. Pick the next active gate that is below/near the skier and aim for the gate center, approximately `flag_x + 16`.
3. Steer left/right before the gate arrives; do not wait until the poles are at the skier.
4. Avoid steering through trees when their y position is near the skier.
5. Avoid the flag poles; passing through the center is safer than grazing the sides.
6. Keep movement smooth because turns change orientation gradually, not instantly. Correct heading, then release the turn.
7. FIRE/jump is usually unnecessary in the default game because moguls are not collidable.
8. A passive straight-down policy is weak: it misses gates and accumulates terminal penalties.
""".strip()


GAME_PROMPT_SPECS = {
    "pong": GamePromptSpec(
        environment_description=PONG_ENVIRONMENT_DESCRIPTION,
        design_principles="""
1. Use the two stacked frames to compute `ball_dx` and `ball_dy`.
2. Predict the intercept point when the ball is moving toward the player paddle.
3. Use a dead zone to avoid jitter and overreaction.
4. Favor off-center returns that create hard-to-catch trajectories.
5. FIRE actions are valid and may help, but they should be used deliberately.
""".strip(),
        failure_modes="""
1. Too many parameters or layered control logic.
2. Wrong error direction for paddle motion.
3. Hyperactive movement with almost no NOOPs.
4. Confusing raw action ids or observation indices.
""".strip(),
        improvement_guidelines="""
1. Increase predictive interception quality rather than adding extra controller layers.
2. Preserve any strong winning behavior once it appears.
3. Improve scoring, not just rally survival.
4. If FIRE is used, tie it to meaningful contact situations instead of constant use.
""".strip(),
        benchmark_context="""
## Benchmark Context
Pong ranges roughly from -21 (lose all points) to +21 (win all points).
A strong policy should approach +21.
""".strip(),
    ),
    "freeway": GamePromptSpec(
        environment_description=FREEWAY_ENVIRONMENT_DESCRIPTION_V2,
        design_principles="""
1. Use the two stacked frames to estimate lane-wise car motion.
2. Optimize timing of safe lane entry, not static one-frame overlap.
3. Prefer UP whenever the projected next-lane window is safe enough.
4. Use NOOP sparingly and DOWN only as a rare emergency escape action.
5. Do not stop just before the top border; finish the crossing.
6. Prefer current-lane plus immediate-next-lane safety checks over checking many future lanes.
7. Optimize time efficiency: repeated fast crossings per 1000 frames.
""".strip(),
        failure_modes="""
1. Static distance checks with no temporal reasoning.
2. Excessive hesitation and high NOOP rates.
3. Conservative stalling one step before scoring.
4. Symmetric UP/DOWN behavior that destroys upward progress.
5. Requiring multiple future lanes to be safe before moving.
6. Using DOWN as normal route planning instead of emergency recovery.
""".strip(),
        improvement_guidelines="""
1. Improve lane timing rather than adding many parameters.
2. Default to continued upward progress once a lane is judged safe.
3. Remove rules that wait too long in locally safe states.
4. Keep the controller lane-centric and simple.
5. If time_efficiency is low, reduce waiting thresholds or check fewer lanes.
6. If collision_rate is high but crossings are low, prefer shorter waits over retreating downward.
7. Preserve an aggressive UP/NOOP structure when it scores consistently.
""".strip(),
        benchmark_context="""
## Benchmark Context
Freeway score is successful crossings.
Random is near 0, DQN is about 30.3, and human is about 29.6.
""".strip(),
    ),
    "asterix": GamePromptSpec(
        environment_description=ASTERIX_ENVIRONMENT_DESCRIPTION,
        design_principles="""
1. Estimate lane danger from enemy positions, orientation, and frame-to-frame x motion.
2. Compute a lane score as collectible value minus danger, then choose one target lane.
3. Prefer adjacent safe collectible lanes first, then fall back to safer repositioning lanes.
4. Once in a safe target lane, commit horizontally to intercept the nearest collectible instead of drifting back toward center.
5. If the current lane is dangerous, dodge horizontally away from the closest threat before trying to recenter.
6. Use diagonal actions to reach selected safe collectible lanes; do not assume a pure UP action exists.
7. Keep the code modular: small helpers such as `_lane_danger`, `_lane_value`, and `_target_collectible_x` are preferred.
8. Keep NOOP usage low once a lane is judged safe.
9. Use both `prev` and `curr` frames explicitly; do not ignore the stacked frame information.
10. Weight collectible targets by `collect.visual_id` value rather than item count alone.
11. Use `collect_dx = curr_collect_x - prev_collect_x` to chase where a target item is moving, not only where it is now.
12. Prefer the current safe lane if it already contains a good target; only switch lanes when another lane has a clearly better value-danger tradeoff.
13. If the chosen lane has a target item and danger is low, lateral movement should be the default action every frame until the item is collected or danger changes.
14. Exploit the lane-aligned array structure directly: lane `i` should read from `enemy_x[i]`, `collect_x[i]`, `collect_visual_id[i]`, etc.
15. If the selected lane has no visible positive-value item but another safe lane does, switch or patrol toward that item; do not set `target_x = player_x` as a default safe-lane behavior.
16. Treat NOOP as an emergency or alignment action only. A high-NOOP Asterix policy is usually under-exploring and under-harvesting.
17. Average return is the primary objective. Action-rate metrics are diagnostics only; do not optimize them at the expense of score.
18. Avoid hard value thresholds above 50. Prefer scoring all active positive-value collectibles and let CMA-ES tune value and danger weights.
19. Lane switching must be easy enough to happen frequently. Keep lane-switch hysteresis modest, or the policy will remain stuck in the starting lane.
20. If a lane is dangerous, prefer `UPRIGHT`/`UPLEFT`, `DOWN`, or a diagonal escape toward a safer lane; do not make pure horizontal dodging the dominant fallback.
""".strip(),
        failure_modes="""
1. Safe-but-passive behavior with high NOOP rate.
2. Returning to center instead of harvesting available safe collectibles.
3. Overcomplicated global heuristics instead of simple lane logic.
4. Deeply nested `jnp.where` expressions that become syntactically fragile.
5. Treating low-hundreds score as good when the benchmark gap is still huge.
6. Changing lanes too often without ever committing laterally to a collectible.
7. Using the wrong action aliases, especially assuming action 1 means UP.
8. Ignoring the previous frame and making decisions from a static snapshot only.
9. Treating all collectibles as equal value and wasting time on low-value or zero-value items.
10. Switching lanes for tiny score differences instead of committing to the current safe harvest opportunity.
11. Falling back to NOOP whenever target x is nearly aligned, even though continued lateral patrol or target commitment would score more.
12. Recomputing lane identity from object y-values instead of using the already lane-aligned array index.
13. Classifying a low-thousands score as solved and making only minimal edits while the policy still shows high NOOP and low item-pickup metrics.
14. Choosing a target lane, then doing no lateral patrol because the target item is absent, inactive, or below a hard value threshold.
15. Overcorrecting high NOOP into constant horizontal motion, lane thrashing, or zero-NOOP behavior that lowers score.
16. Overly hard danger gates that reject most collectible lanes and reduce harvesting even though survival metrics look better.
17. A danger branch that overrides all collection logic with pure `RIGHT`/`LEFT`, trapping the agent in a dangerous lane.
18. Comparing weighted danger against a threshold on a different scale.
19. Value thresholds or lane-switch bonuses so high that 50-point items and nearby lane changes are suppressed.
""".strip(),
        improvement_guidelines="""
1. Push score upward by increasing continuous collectible harvesting, not just survival.
2. Reduce passive waiting inside safe lanes.
3. First pick a target lane, then pick a target x inside that lane.
4. Reward sustained harvesting in adjacent lanes more than frequent recentering.
5. Improve lane prioritization without making the parameterization large.
6. Keep helper functions short and explicit so the module remains syntactically robust.
7. Balance safety with continued item collection across multiple lives.
8. Build vertical progress from the actual diagonal actions exposed by the wrapped environment.
9. Use prev/curr deltas to make danger prediction one-step predictive instead of purely reactive.
10. Prefer value-weighted harvesting: 300-point items should dominate 50-point items when danger is comparable.
11. When a lane is safe and a target collectible exists, move laterally toward its predicted x every frame instead of waiting.
12. Add lane-switch hysteresis or commit logic so the agent does not abandon a good current lane for a tiny advantage elsewhere.
13. Avoid any fallback that returns NOOP in a safe lane with a visible positive-value collectible.
14. Use the direct lane-index structure to simplify the code; lane scoring should usually iterate over indices rather than inferring lane membership from y.
15. If logger feedback shows high NOOP rate or low horizontal rate, rewrite the action-selection fallback so the agent keeps harvesting or patrolling instead of waiting.
16. If item pickups remain modest, lower brittle value thresholds and let CMA-ES tune collection aggressiveness rather than hard-blocking low-value items.
17. Do not make zero NOOP a goal. A small amount of alignment waiting is acceptable if it improves average return.
18. Prefer targeted fixes to the current best policy over wholesale rewrites when the current score already exceeds random by a large margin.
19. If item pickups are near zero, first remove hard collection gates and make vertical lane selection aggressive before adding more danger logic.
20. If horizontal_rate is very high but item_pickups are low, the policy is likely stuck dodging in one lane; add vertical escape and lower lane-switch barriers.
21. Prefer nearest safe collectible lanes with a lane-distance/reachability penalty. A blanket upper-lane bias can reduce score by pulling the agent away from reachable items.
22. Empty lanes should be fallback targets only. In lane scoring, mask inactive collectible lanes out of the collectible objective and handle them in a separate safest-lane fallback.
23. If the current policy scores above random but below DQN, first replace the lane objective with active-safe-collectible scoring before adding new strategic concepts.
""".strip(),
        benchmark_context="""
## Benchmark Context
Asterix benchmark context:
- Random ~= 210
- DQN ~= 6012
- Human ~= 8503
Scores in the low hundreds are weak; strong play should move well beyond 1000.
""".strip(),
    ),
    "breakout": GamePromptSpec(
        environment_description=BREAKOUT_ENVIRONMENT_DESCRIPTION,
        design_principles="""
1. Solve interception first: keep the paddle under the ball reliably.
2. Use the correct raw action mapping for horizontal movement only.
3. Keep the controller small and centered on paddle_x, ball_x, and ball_y.
4. Add a dead zone so the paddle does not oscillate around the target x.
5. Prefer robust rally continuation over complicated brick-targeting heuristics at first.
""".strip(),
        failure_modes="""
1. Wrong action ids for LEFT and RIGHT.
2. Excessive oscillation caused by no dead zone or reversed error sign.
3. Overcomplicated logic that hurts interception reliability.
4. Losing lives because the paddle centers on the wrong x reference.
""".strip(),
        improvement_guidelines="""
1. Improve ball interception quality before adding any higher-level strategy.
2. Preserve working horizontal tracking if it already keeps the ball alive.
3. Tune dead zone, paddle centering, and anticipation before adding extra rules.
4. Keep the parameterization compact so CMA-ES can optimize it efficiently.
""".strip(),
        benchmark_context="""
## Benchmark Context
Breakout random is about 1.7, human is about 31.8, and DQN is about 401.2.
Scores above human are already meaningful even if still far below DQN.
""".strip(),
    ),
    "skiing": GamePromptSpec(
        environment_description=SKIING_ENVIRONMENT_DESCRIPTION,
        design_principles="""
1. Maximize return by making it less negative: fast gate completion and few missed gates.
2. Use the active gate center as the primary target, not the current screen center.
3. Steer early because the skier heading changes gradually.
4. Avoid tree and flag-pole collisions near the skier y coordinate.
5. Keep the controller compact: target gate center, steering dead zone, tree avoidance, and optional speed action are enough.
6. Prefer short LEFT/RIGHT correction taps followed by NOOP/DOWN; continuous turning kills downhill speed and causes timeouts.
7. Use `skier.orientation` as heading feedback. Convert angles greater than 180 to negative signed angles if that is useful.
""".strip(),
        failure_modes="""
1. Treating negative reward as something to minimize inside the policy instead of letting the optimizer maximize return.
2. Driving straight down and missing most gates.
3. Waiting too long before steering toward the next gate.
4. Aiming at the left flag pole instead of the center between poles.
5. Ignoring trees and triggering recovery freezes.
6. Using FIRE frequently even though default moguls are not collidable.
7. Using raw ALE ids instead of compact wrapped actions 0..4.
8. Returning LEFT or RIGHT almost every frame: this rotates the skier toward horizontal and stops downhill progress.
9. Treating `skier.orientation` as a simple positive/negative direction without accounting for the 270..337.5 left-facing angles.
""".strip(),
        improvement_guidelines="""
1. If missed_gate_count is high, increase early gate-center targeting and reduce the steering dead zone.
2. If collision_count is high, widen obstacle clearance and steer around trees earlier.
3. If average_steps is high or average_speed_y is low, reduce continuous LEFT/RIGHT usage and release to NOOP/DOWN after setting the heading.
4. If left/right oscillation is high, use the skier orientation to damp turns and add a heading dead zone.
5. Preserve any policy that reliably passes gates; tune speed and obstacle clearance before rewriting everything.
6. For lateral control, choose a desired signed heading from the gate-center error, then press LEFT/RIGHT only when the current signed heading is far from that target.
""".strip(),
        benchmark_context="""
## Benchmark Context
Skiing uses negative ALE-style returns: less negative is better. The optimization loop applies the intended missed-gate terminal penalty because the local environment field is a remaining-gates counter.
Do not compare raw sign with positive-score games. A run improving from -10000 to -5000 is a major improvement.
""".strip(),
    ),
}


# ============================================================================
# LLM Client
# ============================================================================

class LLMClient:
    """Client for interacting with LLM APIs."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup the appropriate LLM client."""
        if self.config.llm_provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        elif self.config.llm_provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("Please install anthropic: pip install anthropic")
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.llm_provider}")
    
    def ask(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a prompt to the LLM and get a response."""
        system_prompt = system_prompt or "You are an expert AI programmer specializing in JAX and reinforcement learning."
        if self.config.llm_provider == "openai":
            kwargs = {}
            if self.config.temperature is not None:
                kwargs["temperature"] = self.config.temperature
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content
        elif self.config.llm_provider == "anthropic":
            kwargs = {}
            if self.config.temperature is not None:
                kwargs["temperature"] = self.config.temperature
            response = self.client.messages.create(
                model=self.config.llm_model,
                max_tokens=self.config.max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **kwargs,
            )
            return response.content[0].text
        else:
            raise ValueError(f"Unknown provider: {self.config.llm_provider}")


# ============================================================================
# Code Extraction and Validation
# ============================================================================

def extract_python_code(response: str) -> str:
    """Extract Python code from LLM response."""
    import re
    
    # Find all code blocks (```python ... ``` or ``` ... ```)
    code_blocks = []
    
    # Pattern for ```python blocks
    python_pattern = r'```python\s*(.*?)```'
    matches = re.findall(python_pattern, response, re.DOTALL)
    code_blocks.extend(matches)
    
    # If no python-specific blocks, try generic code blocks
    if not code_blocks:
        generic_pattern = r'```\s*(.*?)```'
        matches = re.findall(generic_pattern, response, re.DOTALL)
        for match in matches:
            # Skip if it starts with a language identifier that's not python
            lines = match.strip().split('\n')
            if lines and lines[0].lower() in ['python', 'py']:
                code_blocks.append('\n'.join(lines[1:]))
            elif lines and lines[0].lower() not in ['json', 'bash', 'shell', 'markdown', 'md']:
                code_blocks.append(match)
    
    if code_blocks:
        # Find the block that looks most like a complete policy module
        # Prefer blocks with 'def policy' and 'def init_params'
        for block in code_blocks:
            if 'def policy' in block and 'def init_params' in block:
                return block.strip()
        
        # Otherwise return the longest block
        longest = max(code_blocks, key=len)
        return longest.strip()
    
    # If no code blocks at all, return the whole response
    return response.strip()


def save_policy_module(code: str, version: int, output_dir: str) -> str:
    """Save the generated policy code to a file."""
    filepath = Path(output_dir) / f"policy_v{version}.py"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Add header
    header = f'''"""
Auto-generated policy v{version}
Generated at: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""

'''
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header + code)
    
    return str(filepath)


def to_jsonable(value: Any) -> Any:
    """Recursively convert JAX / NumPy / dataclass-friendly values to JSON-safe objects."""
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def get_git_commit() -> Optional[str]:
    """Best-effort git commit lookup for reproducibility metadata."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_runtime_metadata() -> Dict[str, Any]:
    """Capture lightweight runtime metadata for reproducibility."""
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "jax_version": getattr(jax, "__version__", None),
        "git_commit": get_git_commit(),
    }


def load_policy_module(filepath: str) -> Tuple[Callable, Callable, Callable]:
    """Dynamically load a policy module and return its functions."""
    spec = importlib.util.spec_from_file_location("policy_module", filepath)
    module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Failed to load module: {e}")
    
    # Validate required functions exist
    if not hasattr(module, 'init_params'):
        raise ValueError("Module missing 'init_params' function")
    if not hasattr(module, 'policy'):
        raise ValueError("Module missing 'policy' function")
    if not hasattr(module, 'measure_main'):
        raise ValueError("Module missing 'measure_main' function")
    
    return module.init_params, module.policy, module.measure_main


def get_latest_policy_version(output_dir: str) -> int:
    """Return the highest policy_vN.py version already present in output_dir."""
    latest = 0
    for path in Path(output_dir).glob("policy_v*.py"):
        suffix = path.stem.replace("policy_v", "", 1)
        if suffix.isdigit():
            latest = max(latest, int(suffix))
    return latest


# ============================================================================
# Parallel Environment Evaluation
# ============================================================================

class ParallelEvaluator:
    """Evaluator for running policies in parallel using JAX."""
    
    def __init__(self, config: OptimizationConfig, game: str = "pong"):
        self.config = config
        self.game = game
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup the game environment with wrappers."""
        base_env = jaxatari.make(self.game)
        # Current thesis scope uses raw frame progression for all active games.
        frame_skip = 1
        frame_stack_size = self.config.frame_stack_size if self.config.frame_stack_size > 0 else (2 if self.game in ("pong", "freeway", "asterix", "skiing") else 1)
        self.frame_stack_size = frame_stack_size
        # Disable episodic_life for games where full-episode scoring is the objective.
        episodic_life = False if self.game in ("freeway", "asterix", "skiing") else True
        # Use ObjectCentricWrapper for flat observations
        self.env = FlattenObservationWrapper(
            ObjectCentricWrapper(
                AtariWrapper(base_env, episodic_life=episodic_life),
                frame_stack_size=frame_stack_size,
                frame_skip=frame_skip,
                clip_reward=False,
            )
        )
        self.action_space_n = self.env.action_space().n
        self.action_clip_max = self.action_space_n - 1
        
        # Get observation size by sampling
        key = jrandom.PRNGKey(0)
        obs, _ = self.env.reset(key)
        self.obs_size = obs.shape[-1]
        self.frame_obs_size = self.obs_size // self.frame_stack_size

    def validate_policy_output(self, policy_fn: Callable, params: Dict) -> None:
        """Fail fast when an LLM policy violates the optimizer-facing API."""
        signature = inspect.signature(policy_fn)
        positional = [
            p
            for p in signature.parameters.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        required_positional = [p for p in positional if p.default is inspect.Parameter.empty]
        has_varargs = any(
            p.kind is inspect.Parameter.VAR_POSITIONAL
            for p in signature.parameters.values()
        )
        if len(positional) != 2 or len(required_positional) != 2 or has_varargs:
            raise ValueError(
                "policy() must have exactly two positional arguments: "
                f"(obs_flat, params). Got signature {signature}."
            )

        dummy_obs = jnp.zeros((self.obs_size,), dtype=jnp.float32)
        action = jnp.asarray(policy_fn(dummy_obs, params))
        if action.shape != ():
            raise ValueError(
                f"policy() must return one scalar action, got shape {tuple(action.shape)}"
            )
    
    def _run_single_episode(
        self,
        policy_fn: Callable,
        params: Dict,
        key: jrandom.PRNGKey,
        max_steps: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Run a single episode and collect metrics."""
        
        obs, state = self.env.reset(key)
        
        # Get the last frame's observations (flatten removes frame stack dimension)
        # Use the dynamic observation size for the game
        obs_flat = obs[-self.obs_size:] if obs.shape[0] > self.obs_size else obs
        obs_flat = jnp.asarray(obs_flat, dtype=jnp.float32)
        
        def step_fn(carry, _):
            obs_flat, state, total_reward, done = carry
            
            # Get action from policy
            action = policy_fn(obs_flat, params)
            action = jnp.clip(action, 0, self.action_clip_max).astype(jnp.int32)
            
            # Step environment
            next_obs, next_state, reward, terminated, truncated, info = self.env.step(state, action)
            
            # Get last frame observation using dynamic observation size
            next_obs_flat = next_obs[-self.obs_size:] if next_obs.shape[0] > self.obs_size else next_obs
            next_obs_flat = jnp.asarray(next_obs_flat, dtype=jnp.float32)
            
            # Once an episode is done, later scan slots are padding and must not
            # add rewards again. This matters for terminal-penalty games such as Skiing.
            effective_reward = jnp.where(done, jnp.array(0.0, dtype=jnp.float32), reward)
            next_total_reward = total_reward + effective_reward
            
            # Continue or terminate
            next_done = jnp.logical_or(terminated, truncated)
            new_done = jnp.logical_or(done, next_done)
            episode_finished = jnp.logical_and(jnp.logical_not(done), next_done)
            
            # Keep old state if already done
            next_obs_flat = jax.lax.cond(
                done,
                lambda: obs_flat,
                lambda: next_obs_flat
            )
            next_state = jax.lax.cond(
                done,
                lambda: state,
                lambda: next_state
            )
            
            return (next_obs_flat, next_state, next_total_reward, new_done), (effective_reward, obs_flat, episode_finished)
        
        init_carry = (obs_flat, state, jnp.array(0.0), jnp.array(False))
        (final_obs, final_state, total_reward, _), (rewards, obs_history, terminal_flags) = lax.scan(
            step_fn, init_carry, None, length=max_steps
        )
        
        # Extract final scores (game-specific)
        # Pong reward is +1 for player points and -1 for enemy points, while the
        # flattened score fields stay at zero in this wrapper, so reconstruct the
        # actual match score from the per-step rewards instead.
        final_frame = final_obs[-self.frame_obs_size:]
        if self.game == "pong":
            player_score = jnp.sum(rewards > 0)
            enemy_score = jnp.sum(rewards < 0)
        elif self.game == "skiing":
            # The local Skiing env stores a remaining-gates counter in
            # successful_gates, but its terminal reward subtracts passed gates.
            # Keep the game implementation untouched and correct only the
            # optimizer-facing objective to the intended ALE-style score:
            # time penalties + penalty for gates missed.
            terminal_reward = jnp.sum(jnp.where(terminal_flags, rewards, 0.0))
            passed_gates = jnp.clip(-terminal_reward / 500.0, 0.0, 20.0)
            missed_gates = 20.0 - passed_gates
            intended_terminal_penalty = -missed_gates * 500.0
            total_reward = jnp.where(
                jnp.any(terminal_flags),
                total_reward - terminal_reward + intended_terminal_penalty,
                total_reward,
            )
            player_score = total_reward
            enemy_score = jnp.array(0.0)
        else:
            # For non-Pong games, use total reward as score
            player_score = total_reward
            enemy_score = jnp.array(0.0)
        
        return total_reward, rewards, player_score, enemy_score
    
    def evaluate_policy(
        self,
        policy_fn: Callable,
        params: Dict,
        key: jrandom.PRNGKey,
        num_episodes: int = None,
        max_steps: int = None
    ) -> Dict[str, Any]:
        """Evaluate a policy over multiple episodes in parallel."""
        
        num_episodes = num_episodes or self.config.num_eval_episodes
        max_steps = max_steps or self.config.max_steps_per_episode
        
        # Generate keys for each episode
        keys = jrandom.split(key, num_episodes)
        
        # Vectorize episode runner
        vmapped_run = vmap(
            lambda k: self._run_single_episode(policy_fn, params, k, max_steps)
        )
        
        # Run all episodes in parallel
        total_rewards, all_rewards, player_scores, enemy_scores = vmapped_run(keys)
        
        # Compute metrics
        avg_return = jnp.mean(total_rewards)
        avg_player_score = jnp.mean(player_scores)
        avg_enemy_score = jnp.mean(enemy_scores)
        win_rate = jnp.mean(player_scores > enemy_scores)
        
        return {
            'avg_return': float(avg_return),
            'avg_player_score': float(avg_player_score),
            'avg_enemy_score': float(avg_enemy_score),
            'win_rate': float(win_rate),
            'total_rewards': total_rewards,
            'player_scores': player_scores,
            'enemy_scores': enemy_scores,
        }
    
    def run_episode_with_logging(
        self,
        policy_fn: Callable,
        params: Dict,
        key: jrandom.PRNGKey,
        logger,
        max_steps: int = None
    ) -> Dict[str, Any]:
        """
        Run a single episode with per-step logging for detailed metrics.
        
        This is a Python-mode (non-JIT) evaluation that allows the Logger
        to access full game state at each step.
        
        Args:
            policy_fn: Policy function mapping observations to actions
            params: Policy parameters
            key: Random key for episode
            logger: Logger instance (PongLogger, FreewayLogger, BreakoutLogger)
            max_steps: Maximum steps per episode
            
        Returns:
            Dict with episode results and logger metrics
        """
        max_steps = max_steps or self.config.max_steps_per_episode
        
        # Reset environment and logger
        obs, state = self.env.reset(key)
        logger.reset()
        
        # Get flat observation
        obs_flat = obs[-self.obs_size:] if obs.shape[0] > self.obs_size else obs
        obs_flat = jnp.asarray(obs_flat, dtype=jnp.float32)
        
        total_reward = 0.0
        done = False
        step = 0
        terminal_reward = None
        
        while not done and step < max_steps:
            # Get action from policy
            action = policy_fn(obs_flat, params)
            action = int(jnp.clip(action, 0, self.action_clip_max))
            
            # Step environment
            next_obs, next_state, reward, terminated, truncated, info = self.env.step(state, action)
            
            # Log full transition. Some wrappers autoreset on env_done, so
            # game-specific loggers may need both pre-step and post-step state.
            try:
                logger.log_transition(
                    state,
                    next_state,
                    action=action,
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=info,
                )
            except Exception as e:
                # If logging fails, continue without logging
                if step == 0:
                    print(f"  Logger warning: {e}")
            
            # Update for next step
            total_reward += float(reward)
            if bool(terminated):
                terminal_reward = float(reward)
            obs_flat = next_obs[-self.obs_size:] if next_obs.shape[0] > self.obs_size else next_obs
            obs_flat = jnp.asarray(obs_flat, dtype=jnp.float32)
            state = next_state
            done = bool(jnp.logical_or(terminated, truncated))
            step += 1

        if self.game == "skiing" and terminal_reward is not None:
            passed_gates = max(0.0, min(20.0, -terminal_reward / 500.0))
            missed_gates = 20.0 - passed_gates
            intended_terminal_penalty = -missed_gates * 500.0
            total_reward = total_reward - terminal_reward + intended_terminal_penalty
        
        # Get computed metrics from logger
        logger_metrics = logger.return_metrics()
        
        return {
            'total_reward': total_reward,
            'steps': step,
            'done': done,
            'logger_metrics': logger_metrics
        }
    
    def evaluate_with_logging(
        self,
        policy_fn: Callable,
        params: Dict,
        key: jrandom.PRNGKey,
        logger,
        num_episodes: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluate policy over multiple episodes with logging.
        
        Runs a small number of episodes in Python mode to collect
        detailed metrics via the Logger.
        
        Args:
            policy_fn: Policy function
            params: Policy parameters
            key: Random key
            logger: Logger instance
            num_episodes: Number of episodes to run (default 3 for speed)
            
        Returns:
            Aggregated metrics from all episodes
        """
        all_metrics = []
        total_rewards = []
        
        keys = jrandom.split(key, num_episodes)
        
        for i in range(num_episodes):
            episode_result = self.run_episode_with_logging(
                policy_fn, params, keys[i], logger
            )
            all_metrics.append(episode_result['logger_metrics'])
            total_rewards.append(episode_result['total_reward'])
        
        # Aggregate metrics across episodes
        aggregated = {}
        if all_metrics:
            metric_keys = all_metrics[0].keys()
            for mk in metric_keys:
                values = [m[mk] for m in all_metrics if mk in m]
                if values:
                    aggregated[mk] = sum(values) / len(values)
        
        return {
            'avg_reward': sum(total_rewards) / len(total_rewards) if total_rewards else 0.0,
            'total_rewards': total_rewards,
            'aggregated_metrics': aggregated,
            'per_episode_metrics': all_metrics
        }


# ============================================================================
# Parameter Optimization (CMA-ES, Evolution Strategies, Random Search)
# ============================================================================

def params_to_vector(params: Dict) -> Tuple[jnp.ndarray, Any]:
    """Flatten a parameter dict to a 1D vector."""
    flat_params, tree_def = jax.tree_util.tree_flatten(params)
    # Convert all to float arrays
    flat_arrays = [jnp.atleast_1d(jnp.asarray(p, dtype=jnp.float32)) for p in flat_params]
    vector = jnp.concatenate([a.ravel() for a in flat_arrays])
    # Store shapes for reconstruction
    shapes = [a.shape for a in flat_arrays]
    return vector, (tree_def, shapes)


def vector_to_params(vector: jnp.ndarray, structure: Tuple) -> Dict:
    """Reconstruct parameter dict from 1D vector."""
    tree_def, shapes = structure
    # Split vector back into arrays
    flat_arrays = []
    idx = 0
    for shape in shapes:
        size = int(jnp.prod(jnp.array(shape)))
        arr = vector[idx:idx + size].reshape(shape)
        # Convert single-element arrays back to scalars
        if shape == (1,):
            arr = arr[0]
        flat_arrays.append(arr)
        idx += size
    return jax.tree_util.tree_unflatten(tree_def, flat_arrays)


class CMAESOptimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
    
    This is a powerful derivative-free optimizer that:
    1. Maintains a multivariate Gaussian distribution over parameters
    2. Samples a population of candidates
    3. Evaluates fitness of each candidate
    4. Updates the mean and covariance based on best candidates
    """
    
    def __init__(self, config: OptimizationConfig, evaluator: 'ParallelEvaluator'):
        self.config = config
        self.evaluator = evaluator
    
    def optimize(
        self,
        policy_fn: Callable,
        init_params_fn: Callable,
        key: jrandom.PRNGKey
    ) -> Tuple[Dict, Dict[str, Any]]:
        """Run CMA-ES optimization."""
        
        # Get initial parameters and convert to vector
        base_params = init_params_fn()
        x0, structure = params_to_vector(base_params)
        n = len(x0)
        
        # CMA-ES hyperparameters
        sigma = self.config.param_perturbation_scale
        lambda_ = self.config.num_param_samples  # Population size
        mu = lambda_ // 2  # Number of parents
        
        # Weights for recombination
        weights = jnp.log(mu + 0.5) - jnp.log(jnp.arange(1, mu + 1))
        weights = weights / jnp.sum(weights)
        mu_eff = 1.0 / jnp.sum(weights ** 2)
        
        # Adaptation parameters
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, jnp.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
        
        # Initialize state
        mean = x0
        C = jnp.eye(n)  # Covariance matrix
        ps = jnp.zeros(n)  # Evolution path for sigma
        pc = jnp.zeros(n)  # Evolution path for C
        
        chi_n = jnp.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        best_params = base_params
        best_score = float('-inf')
        best_metrics = None
        search_max_steps = self.config.search_max_steps or self.config.max_steps_per_episode
        
        if self.config.verbose:
            print(f"  CMA-ES: n={n} params, Î»={lambda_} population, {self.config.cma_es_generations} generations")
        
        for gen in range(self.config.cma_es_generations):
            # Sample population
            key, sample_key = jrandom.split(key)
            
            # Eigendecomposition for sampling
            eigenvalues, eigenvectors = jnp.linalg.eigh(C)
            eigenvalues = jnp.maximum(eigenvalues, 1e-10)  # Ensure positive
            sqrt_C = eigenvectors @ jnp.diag(jnp.sqrt(eigenvalues)) @ eigenvectors.T
            
            # Generate offspring
            z = jrandom.normal(sample_key, shape=(lambda_, n))
            offspring = mean + sigma * (z @ sqrt_C.T)
            
            # Evaluate all offspring in parallel
            fitness_scores = []
            all_metrics = []
            
            for i in range(lambda_):
                candidate_params = vector_to_params(offspring[i], structure)
                key, eval_key = jrandom.split(key)
                
                try:
                    metrics = self.evaluator.evaluate_policy(
                        policy_fn, candidate_params, eval_key,
                        num_episodes=self.config.num_parallel_envs,
                        max_steps=search_max_steps,
                    )
                    score = metrics['avg_return']
                    fitness_scores.append(score)
                    all_metrics.append((score, candidate_params, metrics))
                    
                    # Track best
                    if score > best_score:
                        best_score = score
                        best_params = candidate_params
                        best_metrics = metrics
                except Exception as e:
                    if i == 0 and gen == 0:  # Only print first error
                        print(f"    WARNING: Evaluation failed: {e}")
                    fitness_scores.append(float('-inf'))
                    all_metrics.append((float('-inf'), None, None))
            
            # Sort by fitness (descending)
            sorted_indices = jnp.argsort(jnp.array(fitness_scores))[::-1]
            
            # Select top mu individuals
            selected_z = z[sorted_indices[:mu]]
            selected_offspring = offspring[sorted_indices[:mu]]
            
            # Update mean
            old_mean = mean
            mean = jnp.sum(weights[:, None] * selected_offspring, axis=0)
            
            # Update evolution paths
            ps = (1 - cs) * ps + jnp.sqrt(cs * (2 - cs) * mu_eff) * (mean - old_mean) / sigma
            
            hsig = (jnp.linalg.norm(ps) / jnp.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chi_n) < (1.4 + 2 / (n + 1))
            
            pc = (1 - cc) * pc + hsig * jnp.sqrt(cc * (2 - cc) * mu_eff) * (mean - old_mean) / sigma
            
            # Update covariance matrix
            artmp = (selected_offspring - old_mean) / sigma
            C = (1 - c1 - cmu) * C + \
                c1 * (jnp.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu * (artmp.T @ jnp.diag(weights) @ artmp)
            
            # Update sigma
            sigma = sigma * jnp.exp((cs / damps) * (jnp.linalg.norm(ps) / chi_n - 1))
            sigma = float(jnp.clip(sigma, 0.01, 2.0))  # Prevent explosion
            
            # Report progress
            gen_best = max(fitness_scores)
            gen_mean = jnp.mean(jnp.array([s for s in fitness_scores if s > float('-inf')]))
            
            if self.config.verbose:
                print(f"    Gen {gen+1}: best={gen_best:.2f}, mean={gen_mean:.2f}, Ïƒ={sigma:.3f}, overall_best={best_score:.2f}")
        
        # If no valid evaluations, return base params with default metrics
        if best_metrics is None:
            best_params = base_params
            best_metrics = {
                'avg_return': float('-inf'),
                'avg_player_score': 0.0,
                'avg_enemy_score': 0.0,
                'win_rate': 0.0,
            }
        
        return best_params, best_metrics


class RandomSearchOptimizer:
    """Simple random search optimizer (baseline)."""
    
    def __init__(self, config: OptimizationConfig, evaluator: 'ParallelEvaluator'):
        self.config = config
        self.evaluator = evaluator
    
    def _perturb_params(self, params: Dict, key: jrandom.PRNGKey, scale: float) -> Dict:
        """Add random perturbations to parameters."""
        def perturb_leaf(leaf, k):
            if isinstance(leaf, jnp.ndarray):
                noise = jrandom.normal(k, shape=leaf.shape) * scale
                return leaf + noise
            elif isinstance(leaf, (int, float)):
                noise = float(jrandom.normal(k, shape=()) * scale)
                return leaf + noise
            return leaf
        
        flat_params, tree_def = jax.tree_util.tree_flatten(params)
        keys = jrandom.split(key, len(flat_params))
        perturbed_flat = [perturb_leaf(p, k) for p, k in zip(flat_params, keys)]
        return jax.tree_util.tree_unflatten(tree_def, perturbed_flat)
    
    def optimize(
        self,
        policy_fn: Callable,
        init_params_fn: Callable,
        key: jrandom.PRNGKey
    ) -> Tuple[Dict, Dict[str, Any]]:
        """Run random search optimization."""
        
        base_params = init_params_fn()
        best_params = base_params
        best_score = float('-inf')
        best_metrics = None
        search_max_steps = self.config.search_max_steps or self.config.max_steps_per_episode
        
        # Evaluate base parameters
        eval_key, key = jrandom.split(key)
        base_metrics = self.evaluator.evaluate_policy(
            policy_fn, base_params, eval_key,
            num_episodes=self.config.num_parallel_envs,
            max_steps=search_max_steps,
        )
        
        if base_metrics['avg_return'] > best_score:
            best_score = base_metrics['avg_return']
            best_params = base_params
            best_metrics = base_metrics
        
        if self.config.verbose:
            print(f"  Base params: return={base_metrics['avg_return']:.2f}, "
                  f"player={base_metrics['avg_player_score']:.1f}, "
                  f"enemy={base_metrics['avg_enemy_score']:.1f}")
        
        # Random search
        for i in range(self.config.num_param_samples - 1):
            perturb_key, eval_key, key = jrandom.split(key, 3)
            
            perturbed_params = self._perturb_params(
                base_params, perturb_key, self.config.param_perturbation_scale
            )
            
            metrics = self.evaluator.evaluate_policy(
                policy_fn, perturbed_params, eval_key,
                num_episodes=self.config.num_parallel_envs,
                max_steps=search_max_steps,
            )
            
            if metrics['avg_return'] > best_score:
                best_score = metrics['avg_return']
                best_params = perturbed_params
                best_metrics = metrics
                
                if self.config.verbose:
                    print(f"  Sample {i+1}: NEW BEST return={metrics['avg_return']:.2f}")
        
        return best_params, best_metrics


class GaussianProcess:
    """
    Gaussian Process regressor for Bayesian Optimization.
    
    Uses RBF (squared exponential) kernel with automatic relevance determination.
    """
    
    def __init__(self, length_scale: float = 1.0, noise: float = 1e-6):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None
        
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (squared exponential) kernel."""
        dists = cdist(X1 / self.length_scale, X2 / self.length_scale, metric='sqeuclidean')
        return np.exp(-0.5 * dists)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the GP to training data."""
        self.X_train = np.array(X)
        self.y_train = np.array(y).ravel()
        
        # Normalize y for numerical stability
        self.y_mean = np.mean(self.y_train)
        self.y_std = np.std(self.y_train) + 1e-8
        y_normalized = (self.y_train - self.y_mean) / self.y_std
        
        # Compute kernel matrix
        K = self._rbf_kernel(self.X_train, self.X_train)
        K += self.noise * np.eye(len(self.X_train))
        
        # Cholesky decomposition for stable inversion
        try:
            L = cholesky(K, lower=True)
            self.alpha = cho_solve((L, True), y_normalized)
            self.L = L
        except np.linalg.LinAlgError:
            # Fallback: add more noise for numerical stability
            K += 1e-4 * np.eye(len(self.X_train))
            L = cholesky(K, lower=True)
            self.alpha = cho_solve((L, True), y_normalized)
            self.L = L
    
    def predict(self, X: np.ndarray, return_std: bool = False):
        """Predict mean and optionally std at new points."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        K_star = self._rbf_kernel(X, self.X_train)
        
        # Mean prediction
        y_mean = K_star @ self.alpha
        y_mean = y_mean * self.y_std + self.y_mean  # Denormalize
        
        if return_std:
            # Variance prediction
            K_star_star = self._rbf_kernel(X, X)
            v = cho_solve((self.L, True), K_star.T)
            y_var = np.diag(K_star_star) - np.sum(K_star.T * v, axis=0)
            y_var = np.maximum(y_var, 0)  # Ensure non-negative
            y_std = np.sqrt(y_var) * self.y_std  # Denormalize
            return y_mean, y_std
        
        return y_mean


class BayesianOptimizer:
    """
    Gaussian Process-based Bayesian Optimization.
    
    This implementation uses:
    1. A Gaussian Process as surrogate model
    2. Upper Confidence Bound (UCB) acquisition function
    3. Proper exploration-exploitation trade-off via beta parameter
    
    The UCB acquisition: a(x) = Î¼(x) + Î² * Ïƒ(x)
    where Î² controls exploration (higher = more exploration)
    """

    def __init__(self, config: 'OptimizationConfig', evaluator: 'ParallelEvaluator'):
        self.config = config
        self.evaluator = evaluator
        self.history_X = []
        self.history_y = []

    def _ucb_acquisition(self, X: np.ndarray, gp: GaussianProcess, beta: float = 2.0) -> np.ndarray:
        """Upper Confidence Bound acquisition function."""
        mean, std = gp.predict(X, return_std=True)
        return mean + beta * std
    
    def _expected_improvement(self, X: np.ndarray, gp: GaussianProcess, y_best: float, xi: float = 0.01) -> np.ndarray:
        """Expected Improvement acquisition function."""
        from scipy.stats import norm
        
        mean, std = gp.predict(X, return_std=True)
        std = np.maximum(std, 1e-8)
        
        z = (mean - y_best - xi) / std
        ei = (mean - y_best - xi) * norm.cdf(z) + std * norm.pdf(z)
        return ei

    def optimize(
        self,
        policy_fn: Callable,
        init_params_fn: Callable,
        key: jrandom.PRNGKey
    ) -> Tuple[Dict, Dict[str, Any]]:
        """
        Run Gaussian Process Bayesian Optimization.
        
        Algorithm:
        1. Evaluate initial point and a few random samples
        2. Fit GP to observations
        3. Optimize acquisition function to find next point
        4. Evaluate and update GP
        5. Repeat for budget iterations
        """
        # Initial point
        base_params = init_params_fn()
        x0, structure = params_to_vector(base_params)
        x0 = np.array(x0)
        n = len(x0)

        best_params = base_params
        best_score = float('-inf')
        best_metrics = None
        
        # Track history for GP
        self.history_X = []
        self.history_y = []
        search_max_steps = self.config.search_max_steps or self.config.max_steps_per_episode

        # Evaluate the initial point
        key, eval_key = jrandom.split(key)
        base_metrics = self.evaluator.evaluate_policy(
            policy_fn,
            base_params,
            eval_key,
            num_episodes=self.config.num_parallel_envs,
            max_steps=search_max_steps,
        )
        score = float(base_metrics["avg_return"])
        self.history_X.append(x0.copy())
        self.history_y.append(score)
        
        if score > best_score:
            best_score = score
            best_params = base_params
            best_metrics = base_metrics

        if self.config.verbose:
            print(f"  Bayesian Opt: n={n} params, {self.config.cma_es_generations} iterations")
            print(f"    Initial: return={score:.2f}, win_rate={base_metrics['win_rate']:.2%}")

        # Initial random exploration (warm-start GP with diverse samples)
        n_initial = min(5, self.config.num_param_samples)
        sigma = self.config.param_perturbation_scale
        
        for i in range(n_initial):
            key, noise_key, eval_key = jrandom.split(key, 3)
            noise = np.array(jrandom.normal(noise_key, shape=(n,))) * sigma * 2.0
            x_candidate = x0 + noise
            candidate_params = vector_to_params(jnp.array(x_candidate), structure)
            
            try:
                metrics = self.evaluator.evaluate_policy(
                    policy_fn,
                    candidate_params,
                    eval_key,
                    num_episodes=self.config.num_parallel_envs,
                    max_steps=search_max_steps,
                )
                score = float(metrics["avg_return"])
            except Exception:
                score = float('-inf')
                metrics = None
            
            self.history_X.append(x_candidate.copy())
            self.history_y.append(score)
            
            if score > best_score:
                best_score = score
                best_params = candidate_params
                best_metrics = metrics
                if self.config.verbose:
                    print(f"    Random init {i+1}: NEW BEST return={score:.2f}")

        # Main Bayesian Optimization loop
        gp = GaussianProcess(length_scale=sigma, noise=1e-4)
        
        # Compute total budget: iterations * batch_size
        total_budget = self.config.cma_es_generations * self.config.num_param_samples
        remaining_budget = total_budget - len(self.history_X)
        
        # Beta schedule for UCB (start high for exploration, decay for exploitation)
        beta_start = 2.5
        beta_end = 0.5
        
        for iteration in range(remaining_budget):
            # Fit GP to current observations
            X_train = np.array(self.history_X)
            y_train = np.array(self.history_y)
            
            # Filter out -inf values for GP fitting
            valid_mask = y_train > float('-inf')
            if np.sum(valid_mask) < 2:
                # Not enough valid observations, do random sampling
                key, noise_key, eval_key = jrandom.split(key, 3)
                noise = np.array(jrandom.normal(noise_key, shape=(n,))) * sigma
                x_next = x0 + noise
            else:
                gp.fit(X_train[valid_mask], y_train[valid_mask])
                
                # Anneal beta for UCB
                progress = iteration / max(remaining_budget - 1, 1)
                beta = beta_start * (1 - progress) + beta_end * progress
                
                # Generate candidate points for acquisition optimization
                key, sample_key = jrandom.split(key)
                n_candidates = 1000
                
                # Sample around best point and globally
                best_idx = np.argmax(y_train)
                x_best_observed = X_train[best_idx]
                
                # 70% local, 30% global exploration
                n_local = int(0.7 * n_candidates)
                n_global = n_candidates - n_local
                
                local_candidates = x_best_observed + np.random.randn(n_local, n) * sigma * 0.5
                global_candidates = x0 + np.random.randn(n_global, n) * sigma * 2.0
                candidates = np.vstack([local_candidates, global_candidates])
                
                # Evaluate acquisition function
                acquisition_values = self._ucb_acquisition(candidates, gp, beta=beta)
                
                # Select best candidate
                best_acq_idx = np.argmax(acquisition_values)
                x_next = candidates[best_acq_idx]
            
            # Evaluate the selected point
            key, eval_key = jrandom.split(key)
            candidate_params = vector_to_params(jnp.array(x_next), structure)
            
            try:
                metrics = self.evaluator.evaluate_policy(
                    policy_fn,
                    candidate_params,
                    eval_key,
                    num_episodes=self.config.num_parallel_envs,
                    max_steps=search_max_steps,
                )
                score = float(metrics["avg_return"])
            except Exception:
                score = float('-inf')
                metrics = None
            
            self.history_X.append(x_next.copy())
            self.history_y.append(score)
            
            if score > best_score:
                best_score = score
                best_params = candidate_params
                best_metrics = metrics
            
            # Progress reporting (every 10% or when new best)
            if self.config.verbose and (iteration + 1) % max(1, remaining_budget // 10) == 0:
                print(f"    Iter {iteration + 1}/{remaining_budget}: "
                      f"best={best_score:.2f}, current={score:.2f}, "
                      f"Î²={beta:.2f}")
        
        # Final summary
        if self.config.verbose and best_metrics:
            print(f"    Final: return={best_score:.2f}, win_rate={best_metrics['win_rate']:.2%}")
        
        return best_params, best_metrics


class NoSearchOptimizer:
    """LLM-only ablation: evaluate init_params() directly without numeric search."""

    def __init__(self, config: 'OptimizationConfig', evaluator: 'ParallelEvaluator'):
        self.config = config
        self.evaluator = evaluator

    def optimize(
        self,
        policy_fn: Callable,
        init_params_fn: Callable,
        key: jrandom.PRNGKey
    ) -> Tuple[Dict, Dict[str, Any]]:
        params = init_params_fn()
        search_max_steps = self.config.search_max_steps or self.config.max_steps_per_episode
        metrics = self.evaluator.evaluate_policy(
            policy_fn,
            params,
            key,
            num_episodes=self.config.num_parallel_envs,
            max_steps=search_max_steps,
        )
        if self.config.verbose:
            print(
                "  No-search ablation: evaluated LLM init_params directly "
                f"return={metrics['avg_return']:.2f}, win_rate={metrics['win_rate']:.2%}"
            )
        return params, metrics


class ParameterSearcher:
    """Unified parameter searcher that selects optimizer based on config."""
    
    def __init__(self, config: OptimizationConfig, evaluator: 'ParallelEvaluator'):
        self.config = config
        self.evaluator = evaluator
        
        # Select optimizer
        if config.optimizer == "none":
            self.optimizer = NoSearchOptimizer(config, evaluator)
        elif config.optimizer == "cma-es":
            self.optimizer = CMAESOptimizer(config, evaluator)
        elif config.optimizer == "bayes":
            self.optimizer = BayesianOptimizer(config, evaluator)
        else:
            self.optimizer = RandomSearchOptimizer(config, evaluator)
    
    def search(
        self,
        policy_fn: Callable,
        init_params_fn: Callable,
        key: jrandom.PRNGKey
    ) -> Tuple[Dict, Dict[str, Any]]:
        """Search for optimal parameters using configured optimizer."""
        return self.optimizer.optimize(policy_fn, init_params_fn, key)
        
        return best_params, best_metrics


# ============================================================================
# Feedback Generator
# ============================================================================

def generate_feedback(metrics: Dict[str, Any], params: Dict, logger_metrics: Dict[str, float] = None, game: str = "pong") -> str:
    """Generate feedback for the LLM based on evaluation results.
    
    Args:
        metrics: Basic evaluation metrics (avg_return, win_rate, etc.)
        params: Best parameters found
        logger_metrics: Detailed metrics from Logger (if available)
        game: Game name for metric descriptions
    """
    feedback_parts = []
    
    avg_return = metrics['avg_return']
    player_score = metrics['avg_player_score']
    enemy_score = metrics['avg_enemy_score']
    win_rate = metrics['win_rate']
    
    # Overall performance assessment
    if avg_return < 0:
        feedback_parts.append(
            "The agent is LOSING on average. The enemy scores more points than the player."
        )
    elif avg_return < 5:
        feedback_parts.append(
            "The agent is slightly winning but performance is weak."
        )
    else:
        feedback_parts.append(
            "The agent is performing reasonably well."
        )
    
    # Score analysis
    if player_score < 5:
        feedback_parts.append(
            "The player scores very few points. The paddle may not be intercepting the ball effectively."
        )
    
    if enemy_score > 15:
        feedback_parts.append(
            "The enemy scores many points. The paddle may be positioned incorrectly or moving too slowly."
        )
      # Win rate analysis
    if win_rate < 0.3:
        feedback_parts.append(
            "Win rate is very low. Consider more aggressive ball tracking."
        )
    
    # Add detailed Logger metrics if available
    if logger_metrics:
        feedback_parts.append("\n## Detailed Performance Metrics (from Logger)")
        try:
            descriptions = get_all_metric_descriptions(game)
            metrics_text = format_metrics_for_llm(logger_metrics, descriptions)
            feedback_parts.append(metrics_text)
            
            # Game-specific insights based on logger metrics
            if game == "pong":
                if logger_metrics.get('avg_ball_tracking_error', 0) > 20:
                    feedback_parts.append("\nâš ï¸ High tracking error - paddle is not following the ball well.")
                if logger_metrics.get('paddle_jitter_count', 0) > 100:
                    feedback_parts.append("âš ï¸ High paddle jitter - consider increasing dead_zone parameter.")
                if logger_metrics.get('interception_rate', 0) < 0.5:
                    feedback_parts.append("âš ï¸ Low interception rate - paddle not positioned to intercept ball.")
            elif game == "freeway":
                if logger_metrics.get('hesitation_rate', 0) > 0.7:
                    feedback_parts.append("\nâš ï¸ Too much hesitation - chicken should move UP more aggressively.")
                if logger_metrics.get('collision_rate', 0) > 0.5:
                    feedback_parts.append("âš ï¸ High collision rate - improve car avoidance logic.")
                if logger_metrics.get('crossings', 0) == 0:
                    feedback_parts.append("âš ï¸ No crossings! Chicken must reach the top of the screen.")
        except Exception as e:
            feedback_parts.append(f"(Logger metrics formatting error: {e})")
    
      # Suggestions
    feedback_parts.append("\n## CRITICAL IMPLEMENTATION RULES:")
    feedback_parts.append("1. ERROR CALCULATION: error = ball_y - paddle_center_y (NOT the other way around!)")
    feedback_parts.append("   - When error > 0: ball is BELOW paddle, so move DOWN (action 3)")
    feedback_parts.append("   - When error < 0: ball is ABOVE paddle, so move UP (action 4)")
    feedback_parts.append("2. Paddle center: paddle_center_y = player_y + 8 (paddle height is 16)")
    feedback_parts.append("3. Action mapping: 3=DOWN (Y increases), 4=UP (Y decreases), 0=NOOP")
    feedback_parts.append("4. CRITICAL: Do NOT use Python's int() - it breaks JAX tracing!")
    feedback_parts.append("5. Use jax.lax.cond with integer literals (0, 3, 4)")
    
    feedback_parts.append("\n## Working Example:")
    feedback_parts.append("""
```python
def policy(obs_flat, params):
    player_y = obs_flat[1]
    ball_y = obs_flat[9]
    paddle_center = player_y + 8.0
    error = ball_y - paddle_center  # CORRECT: ball_y MINUS paddle
    
    action = jax.lax.cond(
        error > params['dead_zone'],
        lambda: 3,  # Ball below paddle -> move DOWN
        lambda: jax.lax.cond(
            error < -params['dead_zone'],
            lambda: 4,  # Ball above paddle -> move UP
            lambda: 0   # Within dead zone -> NOOP
        )
    )
    return action
```""")
    
    return "\n".join(feedback_parts)


# Override the legacy feedback helper above with a game-aware version.
def generate_feedback(metrics: Dict[str, Any], params: Dict, logger_metrics: Dict[str, float] = None, game: str = "pong") -> str:
    """Generate concise, game-aware feedback for the next LLM iteration."""
    feedback_parts = []

    avg_return = metrics['avg_return']
    player_score = metrics['avg_player_score']
    enemy_score = metrics['avg_enemy_score']
    win_rate = metrics['win_rate']

    if game == "pong":
        if avg_return < 0:
            feedback_parts.append("The agent is losing on average. The enemy scores more points than the player.")
        elif avg_return < 5:
            feedback_parts.append("The agent is slightly winning but performance is weak.")
        else:
            feedback_parts.append("The agent is performing reasonably well.")
        if player_score < 5:
            feedback_parts.append("The player scores very few points. The paddle may not be intercepting the ball effectively.")
        if enemy_score > 15:
            feedback_parts.append("The enemy scores many points. The paddle may be positioned incorrectly or moving too slowly.")
        if win_rate < 0.3:
            feedback_parts.append("Win rate is very low. Consider more aggressive ball tracking.")
    elif game == "freeway":
        if avg_return < 5:
            feedback_parts.append("The agent is collecting very few crossings. It likely hesitates too much or enters unsafe lanes at the wrong time.")
        elif avg_return < 20:
            feedback_parts.append("The agent is crossing sometimes, but timing and lane choice are still limiting score.")
        else:
            feedback_parts.append("The agent is crossing consistently. Further gains likely come from cleaner lane timing and less hesitation.")
    elif game == "asterix":
        if avg_return < 500:
            feedback_parts.append("The agent is still close to random-level Asterix performance. It likely misses too many collectibles, idles in safe lanes, or changes lanes too cautiously.")
        elif avg_return < 1500:
            feedback_parts.append("The agent is scoring, but it is still far below strong Asterix play. It should pursue safe collectibles more continuously and reduce passive movement.")
        elif avg_return < 3000:
            feedback_parts.append("The agent is improving, but still well below DQN-level Asterix. More sustained item collection and better lane prioritization are needed.")
        else:
            feedback_parts.append("The agent is scoring well. Further gains likely come from better lane prioritization and safer collectible pursuit.")
    elif game == "skiing":
        if avg_return < -12000:
            feedback_parts.append("The Skiing return is very poor. The policy likely misses many gates, moves too slowly, or collides often.")
        elif avg_return < -7000:
            feedback_parts.append("The Skiing policy is improving but still loses substantial time or gate penalties. Better gate-center targeting and obstacle avoidance are needed.")
        else:
            feedback_parts.append("The Skiing policy is reaching a more competitive negative return. Further gains should come from faster, smoother gate traversal.")
    else:
        feedback_parts.append("The agent is improving, but there is still room to raise return.")

    if logger_metrics:
        feedback_parts.append("\n## Detailed Performance Metrics (from Logger)")
        try:
            descriptions = get_all_metric_descriptions(game)
            metrics_text = format_metrics_for_llm(logger_metrics, descriptions)
            feedback_parts.append(metrics_text)

            if game == "pong":
                if logger_metrics.get('avg_ball_tracking_error', 0) > 20:
                    feedback_parts.append("\nHigh tracking error: the paddle is not following the ball well.")
                if logger_metrics.get('paddle_jitter_count', 0) > 100:
                    feedback_parts.append("High paddle jitter: consider increasing dead_zone.")
                if logger_metrics.get('interception_rate', 0) < 0.5:
                    feedback_parts.append("Low interception rate: the paddle is not positioned to intercept rallies reliably.")
            elif game == "freeway":
                if logger_metrics.get('hesitation_rate', 0) > 0.7:
                    feedback_parts.append("\nToo much hesitation: the chicken should move UP more aggressively.")
                if logger_metrics.get('time_efficiency', 0) < 3.0:
                    feedback_parts.append("Low time efficiency: reduce waiting and optimize faster crossings per 1000 frames.")
                if logger_metrics.get('up_action_rate', 0) < 0.5:
                    feedback_parts.append("UP action rate is too low for Freeway. Prefer an aggressive UP/NOOP controller with rare DOWN.")
                if logger_metrics.get('down_action_rate', 0) > 0.05:
                    feedback_parts.append("DOWN is used too often. It should be a rare emergency escape, not normal route planning.")
                if logger_metrics.get('collision_rate', 0) > 0.5:
                    feedback_parts.append("High collision rate: improve current-lane and next-lane projected-overlap checks, but do not become overly passive.")
                if logger_metrics.get('crossings', 0) == 0:
                    feedback_parts.append("No crossings recorded: the chicken must actually reach the top of the screen.")
            elif game == "asterix":
                if logger_metrics.get('hits_taken', 0) >= 2:
                    feedback_parts.append("\nToo many enemy hits: lane danger estimation is too weak or too late.")
                if logger_metrics.get('terminal_game_over', 0) > 0.5:
                    feedback_parts.append("The logged episodes usually end by game over. Treat survival as a primary bottleneck, not just item routing.")
                if logger_metrics.get('lives_remaining', 3) <= 0:
                    feedback_parts.append("The policy is exhausting all lives. Avoid terminal collisions even if this slightly reduces short-term pickup rate.")
                if logger_metrics.get('hit_rate_per_1000', 0) > 0.7:
                    feedback_parts.append("Hit rate per 1000 frames is high. Improve lane-entry clearance checks and abort unsafe transitions earlier.")
                if logger_metrics.get('hit_diagonal_fraction', 0) > 0.4:
                    feedback_parts.append("Many life losses happen on diagonal actions. The policy likely enters lanes that become unsafe during vertical transitions.")
                if logger_metrics.get('hit_horizontal_fraction', 0) > 0.4:
                    feedback_parts.append("Many life losses happen during horizontal chasing. The policy should add same-lane enemy clearance around the target x, not only lane-level safety.")
                if logger_metrics.get('avg_hit_enemy_gap', 999) < 20:
                    feedback_parts.append("Hits occur with very small enemy x-gaps. Use a larger collision buffer around projected enemy positions.")
                if logger_metrics.get('score_rate_per_1000', 0) < 600:
                    feedback_parts.append("Score density is still low. Improve survival without collapsing into passive safe-lane behavior.")
                if logger_metrics.get('item_pickups', 0) < 5:
                    feedback_parts.append("Very few item pickups: the policy is not committing to collectible lanes often enough.")
                if logger_metrics.get('item_pickups', 0) < 12:
                    feedback_parts.append("Item pickup count is still modest. The policy should harvest more lanes per life rather than settling into safe but low-scoring behavior.")
                if logger_metrics.get('item_pickups', 0) < 18:
                    feedback_parts.append("The agent still leaves too many collectibles unharvested. Prefer strong lane commitment and value-weighted target pursuit over cautious recentering.")
                if logger_metrics.get('respawn_rate', 0) > 0.15:
                    feedback_parts.append("High respawn rate: collisions are consuming too much time.")
                if logger_metrics.get('noop_rate', 0) > 0.2:
                    feedback_parts.append("Too much idle behavior: prefer deliberate movement toward safe items.")
                if logger_metrics.get('noop_rate', 0) > 0.4:
                    feedback_parts.append("NOOP rate is far too high. Remove any safe-lane fallback that returns NOOP when a visible target exists or when lateral patrol would maintain collection pressure.")
                topmost_stage = logger_metrics.get('topmost_stage_index', 7)
                avg_stage = logger_metrics.get('avg_stage_index', 0)
                if topmost_stage > 2:
                    feedback_parts.append("The agent is not reaching upper lanes often enough. Lower lane-switch barriers and use UPRIGHT/UPLEFT toward safer collectible lanes.")
                if avg_stage > 4 and topmost_stage > 2:
                    feedback_parts.append("The agent spends too much time in lower lanes and rarely reaches the top. It should traverse vertically toward reachable safe collectibles.")
                elif avg_stage > 4:
                    feedback_parts.append("The agent can reach upper lanes but does not collect densely enough. Do not add blanket upper-lane bias; prefer nearest safe collectible lanes with a reachability/lane-distance penalty.")
                if logger_metrics.get('horizontal_rate', 0) < 0.4:
                    feedback_parts.append("Horizontal commitment is too low. Once a lane is safe, move laterally to intercept items instead of waiting.")
                if logger_metrics.get('horizontal_rate', 0) < 0.2:
                    feedback_parts.append("Pure horizontal chasing is extremely weak. In a safe target lane, lateral interception should dominate over idling or unnecessary lane changes.")
                if logger_metrics.get('horizontal_rate', 0) > 0.75 and logger_metrics.get('item_pickups', 0) < 5:
                    feedback_parts.append("Horizontal movement is dominating but not producing pickups. The policy is likely dodging/camping in one lane; use vertical target-lane selection and escape instead.")
            elif game == "skiing":
                if logger_metrics.get('missed_gates', 20) > 8:
                    feedback_parts.append("\nToo many missed gates: target the active gate center earlier and reduce passive straight-line movement.")
                if logger_metrics.get('collision_count', 0) > 2:
                    feedback_parts.append("Too many collisions: widen tree/flag-pole clearance and steer around obstacles before they reach skier_y.")
                if logger_metrics.get('average_speed_y', 0) < 0.6:
                    feedback_parts.append("Downhill speed is low. The policy is probably holding LEFT/RIGHT too often; use short heading corrections, then NOOP/DOWN to keep descending.")
                if logger_metrics.get('turn_rate', 0) > 0.8:
                    feedback_parts.append("Turn rate is very high. LEFT/RIGHT are heading-change commands, not direct movement; reduce continuous turning with orientation-based damping.")
                if logger_metrics.get('down_action_rate', 0) + logger_metrics.get('noop_rate', 0) < 0.3:
                    feedback_parts.append("NOOP/DOWN usage is too low for Skiing. The skier needs release/coast actions after steering corrections.")
                if logger_metrics.get('gate_pass_rate', 0) < 0.5:
                    feedback_parts.append("Gate pass rate is low. Aim for flag_x + 16, not the left pole or screen center.")
        except Exception as e:
            feedback_parts.append(f"(Logger metrics formatting error: {e})")

    feedback_parts.append("\n## CRITICAL IMPLEMENTATION RULES:")
    feedback_parts.append("0. The optimizer calls exactly `policy(obs_flat, params)`. Do not add `state`, `init_state()`, or recurrent-memory APIs.")
    if game == "pong":
        feedback_parts.append("1. ERROR CALCULATION: error = target_y - paddle_center_y.")
        feedback_parts.append("2. Current wrapper uses 52 features: prev=obs_flat[0:26], curr=obs_flat[26:52].")
        feedback_parts.append("3. Use curr.player.y=curr[1], curr.ball.x=curr[16], curr.ball.y=curr[17].")
        feedback_parts.append("4. When error > 0 move DOWN with action 3 or 5; when error < 0 move UP with action 2 or 4.")
        feedback_parts.append("5. Do not use raw ALE ids 11 or 12; valid compact actions are 0..5.")
        feedback_parts.append("6. Do not use Python int() inside policy().")
    elif game == "freeway":
        feedback_parts.append("1. Current wrapper uses 176 features: prev=obs_flat[0:88], curr=obs_flat[88:176].")
        feedback_parts.append("2. Cars are stored as arrays: x=frame[8+i], y=frame[18+i], w=frame[28+i], h=frame[38+i].")
        feedback_parts.append("3. Use prev/curr frames to estimate car dx per lane.")
        feedback_parts.append("4. Optimize for safe lane-entry timing, not static nearest-car checks.")
        feedback_parts.append("5. Action mapping is 0=NOOP, 1=UP, 2=DOWN in this wrapper.")
        feedback_parts.append("6. Do not use Python int() inside policy().")
        feedback_parts.append("7. Strong Freeway policies usually use mostly UP and NOOP; DOWN should be rare.")
        feedback_parts.append("8. Do not require many future lanes to be safe simultaneously; check current and immediate next lane first.")
    elif game == "asterix":
        feedback_parts.append("1. Each frame is 136 features; stacked observations have 272 features.")
        feedback_parts.append("2. Use prev/curr enemy and collectible x positions to estimate lane-wise dx.")
        feedback_parts.append("3. The optimizer-facing wrapped action mapping is 0=NOOP, 1=RIGHT, 2=LEFT, 3=DOWN, 4=UPRIGHT, 5=UPLEFT, 6=DOWNRIGHT, 7=DOWNLEFT, 8=duplicate DOWNLEFT-like action.")
        feedback_parts.append("4. There is no pure UP action in this wrapped environment; upward progress must use actions 4 or 5.")
        feedback_parts.append("5. The observation does not contain score, lives, or timers directly.")
        feedback_parts.append("6. Asterix score comes from repeated item collection; safe-but-passive policies are not enough.")
        feedback_parts.append("7. Use collect.visual_id as a value signal instead of treating every item as equally rewarding.")
        feedback_parts.append("8. Do not use Python int() or Python if-statements inside policy().")
    elif game == "skiing":
        feedback_parts.append("1. Current wrapper uses 146 features: prev=obs_flat[0:73], curr=obs_flat[73:146].")
        feedback_parts.append("2. Action mapping is 0=NOOP, 1=RIGHT, 2=LEFT, 3=FIRE, 4=DOWN.")
        feedback_parts.append("3. Gate left-pole arrays are flag_x=frame[8:10], flag_y=frame[10:12]; aim at flag_x + 16.")
        feedback_parts.append("4. Tree arrays are tree_x=frame[24:28], tree_y=frame[28:32]; avoid trees near skier_y.")
        feedback_parts.append("5. Skiing returns are negative; maximize return by making it less negative.")
        feedback_parts.append("6. LEFT/RIGHT adjust heading one notch; holding them continuously turns horizontal and stops downhill movement.")
        feedback_parts.append("7. Use orientation as signed heading feedback: angles above 180 are left-facing if converted by angle - 360.")
        feedback_parts.append("8. Do not use Python int() or Python if-statements inside policy().")
    else:
        feedback_parts.append("1. Keep the policy simple and parameterized with 3-8 floats.")
        feedback_parts.append("2. Use jax.lax.cond or jnp.where for branching.")
        feedback_parts.append("3. Do not use Python int() inside policy().")

    return "\n".join(feedback_parts)


# ============================================================================
# Main Optimization Loop
# ============================================================================

class LLMOptimizationLoop:
    """Main class orchestrating the unified LeGPS optimization loop."""
    
    def __init__(self, config: OptimizationConfig, game: str = "pong"):
        self.config = config
        self.game = game
        self.llm_client = LLMClient(config)
        self.evaluator = ParallelEvaluator(config, game=game)
        self.searcher = ParameterSearcher(config, self.evaluator)
        
        self.iteration = 0
        self.history = []
        
        # Keep the logger in the provided-metrics/full-state regime for the
        # current unified thesis experiments.
        self.logger_config = AblationConfig(metrics_source="A1", info_access="B1")
        
        # Create game-specific logger
        try:
            self.logger = create_logger(game, self.logger_config)
            print(f"Logger initialized for {game}")
        except ValueError:
            self.logger = None
            print(f"No logger available for game: {game}")
        
        # Load game-specific prompts
        self._load_game_prompts()
    
    def _load_game_prompts(self):
        """Load the structured prompt spec for the specific game."""
        fallback_game = "pong"
        self.prompt_spec = GAME_PROMPT_SPECS.get(self.game, GAME_PROMPT_SPECS[fallback_game])
        self.environment_description = self.prompt_spec.environment_description
        self.system_prompt = LLM_ONLY_SYSTEM_PROMPT if self.config.optimizer == "none" else UNIFIED_SYSTEM_PROMPT

    def _method_prompt_context(self) -> Dict[str, str]:
        if self.config.optimizer == "none":
            return {
                "method_objective": (
                    "Produce one clean controller whose numeric constants are chosen by the LLM itself. "
                    "No CMA-ES or other optimizer will tune these values afterward. The structure should "
                    "generalize well and avoid game-specific prompt hacks."
                ),
                "method_design_rule": (
                    "Keep the policy self-calibrated: small parameter count, sensible default values, "
                    "shallow logic, and no brittle special cases."
                ),
                "improvement_objective": (
                    "Improve score while keeping the controller compact and self-calibrated. "
                    "Do not assume later CMA-ES tuning will repair weak constants."
                ),
                "rewrite_objective": (
                    "Keep the controller compact and self-calibrated because this ablation evaluates "
                    "the LLM-chosen constants directly."
                ),
            }
        return {
            "method_objective": (
                "Produce one clean parametric controller that can be optimized by CMA-ES. "
                "The structure should generalize well and avoid game-specific prompt hacks."
            ),
            "method_design_rule": (
                "Keep the policy optimizer-friendly: small parameter count, shallow logic, no brittle special cases."
            ),
            "improvement_objective": "Improve score while keeping the controller simple enough for CMA-ES.",
            "rewrite_objective": "Keep the controller compact and optimizer-friendly.",
        }

    def _build_initial_prompt(self) -> str:
        return UNIFIED_INITIAL_PROMPT.format(
            game_name=self.game.capitalize(),
            environment_description=self.prompt_spec.environment_description,
            **self._method_prompt_context(),
            design_principles=self.prompt_spec.design_principles,
            failure_modes=self.prompt_spec.failure_modes,
            benchmark_context=self.prompt_spec.benchmark_context,
        )

    def _build_improvement_prompt(
        self,
        previous_code: str,
        metrics: Dict[str, Any],
        params_str: str,
        feedback: str,
    ) -> str:
        avg_return = float(metrics.get('avg_return', 0.0))
        win_rate = float(metrics.get('win_rate', 0.0))

        if self.game == "pong":
            strong_policy = avg_return >= 18.0 or win_rate >= 0.95
        elif self.game == "freeway":
            strong_policy = avg_return >= 28.0
        elif self.game == "asterix":
            strong_policy = avg_return >= max(2500.0, self.config.target_score)
        else:
            strong_policy = avg_return >= self.config.target_score

        if strong_policy:
            revision_strategy = (
                "The current policy is already strong. Preserve the overall structure, helper layout, "
                "and main control logic. Only make the smallest set of changes needed to address the "
                "specific weaknesses mentioned in the feedback."
            )
            change_budget = (
                "Minimal edit budget: keep parameter names where possible, keep the same high-level "
                "policy shape, and change only one or two concrete behaviors."
            )
        elif self.game == "asterix" and avg_return >= 1000.0:
            revision_strategy = (
                "The current Asterix policy is meaningfully above random but still far below DQN. "
                "Do not perform a wholesale rewrite. Treat the current controller as a working baseline "
                "and make a targeted patch to the diagnosed bottleneck."
            )
            change_budget = (
                "Conservative Asterix edit budget: preserve the init_params key set exactly unless a "
                "parameter is impossible to reuse. Prefer changing formulas and fallback actions over "
                "adding, removing, or renaming tunable parameters, so CMA-ES can continue from the known "
                "best numeric vector."
            )
        else:
            revision_strategy = (
                "The current policy is not yet strong enough. Keep what is clearly working, but you may "
                "restructure the controller moderately if it directly fixes the failures described in the feedback."
            )
            change_budget = (
                "Moderate edit budget: preserve useful helper concepts when possible, but redesign lane "
                "selection, interception, or timing logic if that is the main bottleneck."
            )

        return UNIFIED_IMPROVEMENT_PROMPT.format(
            game_name=self.game.capitalize(),
            environment_description=self.prompt_spec.environment_description,
            avg_return=avg_return,
            avg_player_score=metrics.get('avg_player_score', 0),
            avg_enemy_score=metrics.get('avg_enemy_score', 0),
            win_rate=win_rate,
            best_params=params_str,
            previous_code=previous_code,
            feedback=feedback,
            **self._method_prompt_context(),
            revision_strategy=revision_strategy,
            change_budget=change_budget,
            benchmark_context=self.prompt_spec.benchmark_context,
            improvement_guidelines=self.prompt_spec.improvement_guidelines,
        )

    def _build_diagnostic_prompt(
        self,
        previous_code: str,
        metrics: Dict[str, Any],
        params_str: str,
        feedback: str,
        failed_candidate_context: str = "",
    ) -> str:
        return UNIFIED_DIAGNOSTIC_PROMPT.format(
            game_name=self.game.capitalize(),
            environment_description=self.prompt_spec.environment_description,
            avg_return=float(metrics.get('avg_return', 0.0)),
            avg_player_score=metrics.get('avg_player_score', 0),
            avg_enemy_score=metrics.get('avg_enemy_score', 0),
            win_rate=float(metrics.get('win_rate', 0.0)),
            best_params=params_str,
            previous_code=previous_code,
            feedback=feedback,
            failed_candidate_context=failed_candidate_context,
            benchmark_context=self.prompt_spec.benchmark_context,
        )

    def _build_diagnostic_rewrite_prompt(
        self,
        previous_code: str,
        metrics: Dict[str, Any],
        params_str: str,
        diagnosis: str,
        failed_candidate_context: str = "",
    ) -> str:
        avg_return = float(metrics.get('avg_return', 0.0))
        if self.game == "asterix" and avg_return >= 1000.0:
            rewrite_strategy = (
                "This is a conservative continuation from a working Asterix baseline. Preserve the "
                "`init_params()` key set exactly, preserve the general lane-value / lane-danger / "
                "target-x structure, and patch only the specific weak behavior identified by the diagnosis. "
                "Do not add `upper_bias`, `upward_bias`, or other new parameters unless the existing "
                "parameters cannot express the fix. Do not add a blanket structural upper-lane bias; prefer "
                "nearest safe collectible selection with a lane-distance/reachability penalty and better "
                "danger avoidance. Use the current best numeric parameters as the CMA-ES starting point."
            )
        else:
            rewrite_strategy = (
                "Use the diagnosis to improve the policy while keeping the parameterization compact. "
                "Preserve working helper concepts where possible, but redesign the controller if the "
                "current structure is clearly the bottleneck."
            )

        return UNIFIED_DIAGNOSTIC_REWRITE_PROMPT.format(
            game_name=self.game.capitalize(),
            environment_description=self.prompt_spec.environment_description,
            avg_return=avg_return,
            avg_player_score=metrics.get('avg_player_score', 0),
            avg_enemy_score=metrics.get('avg_enemy_score', 0),
            win_rate=float(metrics.get('win_rate', 0.0)),
            best_params=params_str,
            previous_code=previous_code,
            diagnosis=diagnosis,
            failed_candidate_context=failed_candidate_context,
            **self._method_prompt_context(),
            rewrite_strategy=rewrite_strategy,
            benchmark_context=self.prompt_spec.benchmark_context,
            improvement_guidelines=self.prompt_spec.improvement_guidelines,
        )
    
    def _get_initial_policy(self) -> str:
        """Get the initial policy from the LLM."""
        prompt = self._build_initial_prompt()
        
        if self.config.verbose:
            print("Requesting initial policy from LLM...")
        
        response = self.llm_client.ask(prompt, system_prompt=self.system_prompt)
        return extract_python_code(response)

    def _recent_failed_candidate_context(self, best_score: float) -> str:
        """Summarize the latest worse rewrite so the next prompt avoids repeating it."""
        if not self.history:
            return ""

        for entry in reversed(self.history):
            metrics = entry.get("metrics", {})
            avg_return = float(metrics.get("avg_return", float("-inf")))
            validation_metrics = entry.get("long_horizon_validation", {})
            validation_return = float(validation_metrics.get("avg_return", float("-inf"))) if validation_metrics else float("-inf")
            if validation_return >= best_score:
                continue
            if avg_return >= best_score:
                continue

            logger_metrics = metrics.get("logger_metrics", {}) or {}
            parts = [
                "## Recent Worse Candidate To Avoid",
                (
                    f"The previous generated candidate `{entry.get('filepath', 'unknown')}` "
                    f"scored {avg_return:.2f}, below the current best {best_score:.2f}."
                ),
                "Use this as negative evidence: avoid repeating the same broad rewrite direction.",
            ]
            if logger_metrics:
                metric_names = [
                    "final_score",
                    "item_pickups",
                    "hits_taken",
                    "respawn_rate",
                    "noop_rate",
                    "horizontal_rate",
                    "diagonal_rate",
                ]
                metric_lines = []
                for name in metric_names:
                    if name in logger_metrics:
                        metric_lines.append(f"- {name}: {float(logger_metrics[name]):.3f}")
                if metric_lines:
                    parts.append("Failed-candidate logger metrics:")
                    parts.extend(metric_lines)
            return "\n".join(parts)

        return ""

    def _get_improved_policy(
        self,
        previous_code: str,
        metrics: Dict[str, Any],
        params: Dict,
        feedback: str
    ) -> str:
        """Get an improved policy from the LLM."""
        
        # Convert params to string representation
        params_str = json.dumps(
            jax.tree_util.tree_map(
                lambda x: x.tolist() if hasattr(x, 'tolist') else x,
                params
            ),
            indent=2
        )
        failed_candidate_context = self._recent_failed_candidate_context(
            float(metrics.get('avg_return', float('-inf')))
        )
        
        if self.config.diagnostic_refinement:
            if self.config.verbose:
                print("Requesting policy diagnosis from LLM...")

            diagnosis_prompt = self._build_diagnostic_prompt(
                previous_code, metrics, params_str, feedback, failed_candidate_context
            )
            diagnosis = self.llm_client.ask(
                diagnosis_prompt,
                system_prompt=UNIFIED_DIAGNOSTIC_SYSTEM_PROMPT,
            ).strip()

            diagnosis_path = Path(self.config.output_dir) / f"diagnosis_v{self.iteration}.txt"
            diagnosis_path.parent.mkdir(parents=True, exist_ok=True)
            diagnosis_path.write_text(diagnosis, encoding="utf-8")

            prompt = self._build_diagnostic_rewrite_prompt(
                previous_code, metrics, params_str, diagnosis, failed_candidate_context
            )
            if self.config.verbose:
                print(f"Saved policy diagnosis to: {diagnosis_path}")
                print("Requesting improved policy from clean diagnostic rewrite prompt...")
        else:
            prompt = self._build_improvement_prompt(previous_code, metrics, params_str, feedback)
            if self.config.verbose:
                print("Requesting improved policy from LLM...")
        
        response = self.llm_client.ask(prompt, system_prompt=self.system_prompt)
        return extract_python_code(response)

    def _is_strong_result(self, metrics: Optional[Dict[str, Any]]) -> bool:
        """Return True when a game is already in a regime where preservation matters more than exploration."""
        if not metrics:
            return False

        avg_return = float(metrics.get("avg_return", float("-inf")))
        win_rate = float(metrics.get("win_rate", 0.0))

        if self.game == "pong":
            return avg_return >= 18.0 or win_rate >= 0.98
        if self.game == "freeway":
            return avg_return >= 28.0
        if self.game == "asterix":
            return avg_return >= max(2500.0, self.config.target_score)

        return avg_return >= self.config.target_score

    def _load_resume_state(self) -> Optional[Dict[str, Any]]:
        """Load the best policy and metrics from a previous optimization_results.json."""
        results_path = Path(self.config.output_dir) / "optimization_results.json"
        if not results_path.exists():
            print(f"Resume requested, but no results found at: {results_path}")
            return None

        try:
            # PowerShell 5's UTF8 writer can add a BOM; accept it so seeded
            # resume files do not silently fall back to a fresh policy.
            data = json.loads(results_path.read_text(encoding="utf-8-sig"))
        except Exception as e:
            print(f"Resume requested, but failed to read results: {e}")
            return None

        policy_candidates: List[Path] = []
        best_policy_path = data.get("best_policy_path")
        if best_policy_path:
            policy_candidates.append(Path(best_policy_path))
            policy_candidates.append(Path(str(best_policy_path).replace("\\", "/")))
        policy_candidates.extend(sorted(Path(self.config.output_dir).glob("policy_v*.py"), reverse=True))

        policy_path = next((path for path in policy_candidates if path.exists()), None)
        if policy_path is None:
            print("Resume requested, but no previous policy file could be found.")
            return None

        try:
            policy_code = policy_path.read_text(encoding="utf-8-sig")
        except Exception as e:
            print(f"Resume requested, but failed to read policy file: {e}")
            return None

        history = data.get("history", [])
        self.history = history if isinstance(history, list) else []

        best_metrics = data.get("best_metrics") or {}
        best_params = data.get("best_params") or {}
        best_score = float(data.get("best_score", best_metrics.get("avg_return", float("-inf"))))

        print(f"Resuming from previous best policy: {policy_path}")
        print(f"Previous best score: {best_score:.2f}")

        return {
            "current_code": policy_code,
            "best_code": policy_code,
            "best_metrics": best_metrics,
            "best_params": best_params,
            "best_score": best_score,
            "best_policy_path": str(policy_path),
            "best_overall_params": best_params,
        }

    def _seed_init_params_from_best(
        self,
        init_params_fn: Callable[[], Dict],
        best_params: Optional[Dict[str, Any]],
    ) -> Callable[[], Dict]:
        """Return an init_params function seeded from resumed best params when safe."""
        if not self.config.resume_best_params_for_cma or not best_params:
            return init_params_fn

        try:
            template_params = init_params_fn()
        except Exception as e:
            print(f"  Could not inspect init_params for CMA resume seeding: {e}")
            return init_params_fn

        if not isinstance(template_params, dict) or not isinstance(best_params, dict):
            return init_params_fn

        template_keys = set(template_params.keys())
        best_keys = set(best_params.keys())
        if template_keys != best_keys:
            missing = sorted(template_keys - best_keys)
            extra = sorted(best_keys - template_keys)
            print(
                "  CMA-ES will use policy defaults because resumed params are incompatible "
                f"(missing={missing}, extra={extra})."
            )
            return init_params_fn

        seeded_params = {}
        try:
            for name, template_value in template_params.items():
                value = best_params[name]
                template_array = jnp.asarray(template_value)
                value_array = jnp.asarray(value, dtype=jnp.float32)
                if template_array.shape:
                    value_array = jnp.reshape(value_array, template_array.shape)
                else:
                    value_array = jnp.reshape(value_array, ())
                seeded_params[name] = value_array
        except Exception as e:
            print(f"  CMA-ES will use policy defaults because best-param seeding failed: {e}")
            return init_params_fn

        def seeded_init_params() -> Dict:
            return dict(seeded_params)

        print("  CMA-ES initialized from resumed best parameters.")
        return seeded_init_params

    def _params_to_jax_scalars(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JSON-loaded scalar params back to JAX-friendly values."""
        converted = {}
        for name, value in params.items():
            try:
                converted[name] = jnp.asarray(value, dtype=jnp.float32)
            except Exception:
                converted[name] = value
        return converted

    def _refresh_resume_metrics(
        self,
        policy_path: Optional[str],
        best_params: Optional[Dict[str, Any]],
        key: jrandom.PRNGKey,
    ) -> Optional[Dict[str, Any]]:
        """Re-evaluate resumed best policy with current evaluator/logger code."""
        if not self.config.refresh_resume_metrics or not policy_path or not best_params:
            return None

        path = Path(policy_path)
        if not path.exists():
            print(f"Cannot refresh resumed metrics; policy path does not exist: {path}")
            return None

        try:
            _, policy_fn, _ = load_policy_module(str(path))
        except Exception as e:
            print(f"Cannot refresh resumed metrics; failed to load policy: {e}")
            return None

        params = self._params_to_jax_scalars(best_params)
        eval_key, log_key = jrandom.split(key)

        try:
            metrics = self.evaluator.evaluate_policy(
                policy_fn,
                params,
                eval_key,
                num_episodes=self.config.num_eval_episodes,
                max_steps=self.config.max_steps_per_episode,
            )
        except Exception as e:
            print(f"Cannot refresh resumed metrics; evaluation failed: {e}")
            return None

        if self.logger is not None:
            try:
                log_results = self.evaluator.evaluate_with_logging(
                    policy_fn,
                    params,
                    log_key,
                    self.logger,
                    num_episodes=3,
                )
                metrics["logger_metrics"] = log_results["aggregated_metrics"]
            except Exception as e:
                print(f"  Resume logger refresh failed: {e}")

        print("\nRefreshed resumed-best metrics with current logger:")
        print(f"  Average Return: {metrics['avg_return']:.2f}")
        logger_metrics = metrics.get("logger_metrics") or {}
        for name in [
            "final_score",
            "terminal_game_over",
            "hits_taken",
            "lives_remaining",
            "item_pickups",
            "respawn_rate",
            "noop_rate",
        ]:
            if name in logger_metrics:
                print(f"  {name}: {float(logger_metrics[name]):.3f}")

        return metrics

    def _write_results_snapshot(
        self,
        path: Path,
        seed: int,
        best_score: float,
        best_metrics: Optional[Dict[str, Any]],
        best_overall_params: Optional[Dict[str, Any]],
        best_policy_path: Optional[str],
    ) -> None:
        """Persist current loop state so long runs do not lose evaluated candidates."""
        snapshot = {
            'best_score': best_score,
            'seed': seed,
            'config': to_jsonable({
                key: ("<redacted>" if key == "api_key" and value else value)
                for key, value in self.config.__dict__.items()
            }),
            'runtime': get_runtime_metadata(),
            'best_params': best_overall_params,
            'best_metrics': best_metrics,
            'history': self.history,
            'best_policy_path': best_policy_path,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(
                snapshot,
                f,
                indent=2,
                default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x),
            )
    
    def run(self, seed: int = 42) -> Dict[str, Any]:
        """Run the full optimization loop."""
        
        key = jrandom.PRNGKey(seed)
        
        print("=" * 60)
        print(f"LLM Policy Optimization Loop for JAXAtari {self.game.upper()}")
        print("=" * 60)
        
        current_code = None
        best_metrics = None
        best_params = None
        best_code = None
        best_score = float('-inf')
        best_policy_path = None
        best_overall_params = None
        no_improvement_streak = 0
        version_offset = get_latest_policy_version(self.config.output_dir)

        if self.config.resume_from_results:
            resume_state = self._load_resume_state()
            if resume_state is not None:
                current_code = resume_state["current_code"]
                best_metrics = resume_state["best_metrics"]
                best_params = resume_state["best_params"]
                best_code = resume_state["best_code"]
                best_score = resume_state["best_score"]
                best_policy_path = resume_state["best_policy_path"]
                best_overall_params = resume_state["best_overall_params"]
                refresh_key, key = jrandom.split(key)
                refreshed_metrics = self._refresh_resume_metrics(
                    best_policy_path,
                    best_params,
                    refresh_key,
                )
                if refreshed_metrics is not None:
                    best_metrics = refreshed_metrics
                    # Use the refreshed score as the current-run baseline so the
                    # next rewrite is judged against the fixed diagnostic path.
                    best_score = refreshed_metrics["avg_return"]
        
        for iteration in range(self.config.max_iterations):
            self.iteration = version_offset + iteration + 1
            print(f"\n{'='*60}")
            print(f"Outer-loop step {iteration + 1}/{self.config.max_iterations} (policy v{self.iteration})")
            print("=" * 60)

            if (
                self.config.stop_on_strong_best
                and iteration > 0
                and self._is_strong_result(best_metrics)
                and no_improvement_streak >= 1
            ):
                print("Strong unified policy already found and the last revision did not improve it.")
                print("Stopping early to preserve the best-known structure instead of exploring weaker rewrites.")
                break
            
            # Step 1: Get policy from LLM
            if current_code is None:
                current_code = self._get_initial_policy()
            elif best_metrics is not None:
                # Only request improvement if we have metrics from a previous successful run
                # Extract logger metrics if available
                logger_metrics = best_metrics.get('logger_metrics', None)
                feedback = generate_feedback(
                    best_metrics, best_params, 
                    logger_metrics=logger_metrics, 
                    game=self.game
                )
                current_code = self._get_improved_policy(
                    best_code or current_code, best_metrics, best_params, feedback
                )
            # else: retry with same code structure but fresh LLM call
            else:
                print("Retrying initial policy generation...")
                current_code = self._get_initial_policy()
            
            # Step 2: Save and load policy
            filepath = save_policy_module(
                current_code, self.iteration, self.config.output_dir
            )
            print(f"Saved policy to: {filepath}")
            
            try:
                init_params_fn, policy_fn, measure_fn = load_policy_module(filepath)
                self.evaluator.validate_policy_output(policy_fn, init_params_fn())
                print("Successfully loaded policy module")
            except Exception as e:
                print(f"ERROR loading policy: {e}")
                print("Will retry with a new LLM request...")
                current_code = None  # Reset to trigger fresh generation
                continue
            
            # Step 3: Search for best parameters
            search_key, key = jrandom.split(key)
            if self.config.optimizer == "none":
                print("\nEvaluating LLM-chosen parameters directly (no inner optimizer)...")
            else:
                print(f"\nSearching for optimal parameters ({self.config.num_param_samples} samples)...")
            
            try:
                search_init_params_fn = (
                    init_params_fn
                    if self.config.optimizer == "none"
                    else self._seed_init_params_from_best(init_params_fn, best_params)
                )
                current_params, search_metrics = self.searcher.search(
                    policy_fn, search_init_params_fn, search_key
                )
            except Exception as e:
                print(f"ERROR during evaluation: {e}")
                continue

            eval_key, key = jrandom.split(key)
            try:
                metrics = self.evaluator.evaluate_policy(
                    policy_fn,
                    current_params,
                    eval_key,
                    num_episodes=self.config.num_eval_episodes,
                    max_steps=self.config.max_steps_per_episode,
                )
                metrics['search_avg_return'] = search_metrics.get('avg_return', float('-inf'))
                metrics['search_avg_player_score'] = search_metrics.get('avg_player_score', 0.0)
                metrics['search_avg_enemy_score'] = search_metrics.get('avg_enemy_score', 0.0)
                metrics['search_win_rate'] = search_metrics.get('win_rate', 0.0)
            except Exception as e:
                print(f"ERROR during full-match evaluation: {e}")
                metrics = search_metrics
            
            # Step 4: Report results
            print(f"\nResults for iteration {self.iteration}:")
            if 'search_avg_return' in metrics:
                print(f"  Search Return ({self.config.search_max_steps or self.config.max_steps_per_episode} steps): {metrics['search_avg_return']:.2f}")
            print(f"  Average Return: {metrics['avg_return']:.2f}")
            print(f"  Player Score: {metrics['avg_player_score']:.2f}")
            print(f"  Enemy Score: {metrics['avg_enemy_score']:.2f}")
            print(f"  Win Rate: {metrics['win_rate']:.2%}")
            
            # Step 5: Run detailed evaluation with Logger (for LLM feedback)
            logger_metrics = None
            if self.logger is not None:
                print("\nRunning detailed evaluation with the unified logger...")
                log_key, key = jrandom.split(key)
                try:
                    log_results = self.evaluator.evaluate_with_logging(
                        policy_fn, current_params, log_key, self.logger, num_episodes=3
                    )
                    logger_metrics = log_results['aggregated_metrics']
                    print(f"  Logger metrics collected: {len(logger_metrics)} metrics")
                    
                    # Print key metrics
                    for metric_name, value in list(logger_metrics.items())[:5]:
                        print(f"    {metric_name}: {value:.3f}")
                except Exception as e:
                    print(f"  Logger evaluation failed: {e}")
                    logger_metrics = None
            
            # Track best overall
            if metrics['avg_return'] > best_score:
                best_score = metrics['avg_return']
                best_metrics = metrics
                best_metrics['logger_metrics'] = logger_metrics  # Add logger metrics
                best_params = current_params
                best_overall_params = to_jsonable(current_params)
                best_code = current_code
                best_policy_path = filepath
                no_improvement_streak = 0
                print("  >>> New best overall!")
            else:
                no_improvement_streak += 1
                if self.config.stop_on_strong_best and self._is_strong_result(best_metrics):
                    print("  Current revision did not improve a strong best policy.")
                    print("  Stopping now to preserve the best policy instead of continuing weaker rewrites.")
                    self.history.append({
                        'iteration': self.iteration,
                        'metrics': {k: v for k, v in metrics.items() 
                                   if not isinstance(v, jnp.ndarray)},
                        'best_params': to_jsonable(current_params),
                        'filepath': filepath
                    })
                    break
            
            # Save history
            self.history.append({
                'iteration': self.iteration,
                'metrics': {k: v for k, v in metrics.items() 
                           if not isinstance(v, jnp.ndarray)},
                'best_params': to_jsonable(current_params),
                'filepath': filepath
            })

            checkpoint_path = Path(self.config.output_dir) / "optimization_checkpoint.json"
            self._write_results_snapshot(
                checkpoint_path,
                seed,
                best_score,
                best_metrics,
                best_overall_params,
                best_policy_path,
            )
            print(f"Checkpoint saved to: {checkpoint_path}")
            
            # Check if target reached
            if metrics['avg_return'] >= self.config.target_score:
                print(f"\n{'='*60}")
                print(f"TARGET REACHED! Score: {metrics['avg_return']:.2f}")
                print("=" * 60)
                break
        
        # Final report
        print(f"\n{'='*60}")
        print("Optimization Complete!")
        print("=" * 60)
        print(f"Best Score Achieved: {best_score:.2f}")
        print(f"Total Iterations: {self.iteration}")
        
        results = {
            'best_score': best_score,
            'best_params': best_overall_params,
            'best_metrics': best_metrics,
            'history': self.history,
            'best_policy_path': best_policy_path,
            'best_code': best_code,
        }

        results_path = Path(self.config.output_dir) / "optimization_results.json"
        self._write_results_snapshot(
            results_path,
            seed,
            best_score,
            best_metrics,
            best_overall_params,
            best_policy_path,
        )
        print(f"Results saved to: {results_path}")
        
        return results


# ============================================================================
# Demo Mode (without LLM)
# ============================================================================

PONG_DEMO_CODE = '''
import jax.numpy as jnp
import jax

def init_params() -> dict:
    """Initialize parameters for a simple tracking policy."""
    return {
        'dead_zone': jnp.array(2.0),
        'paddle_offset': jnp.array(8.0),
    }

def policy(obs_flat: jnp.ndarray, params: dict) -> int:
    paddle_y = obs_flat[1]
    ball_y = obs_flat[9]
    paddle_center = paddle_y + params['paddle_offset']
    error = ball_y - paddle_center
    should_move = jnp.abs(error) > params['dead_zone']
    move_down = error > 0
    action = jax.lax.cond(
        should_move,
        lambda: jax.lax.cond(move_down, lambda: 3, lambda: 4),
        lambda: 0
    )
    return action

def measure_main(episode_rewards: jnp.ndarray, episode_scores: dict) -> float:
    return jnp.sum(episode_rewards)
'''

KANGAROO_DEMO_CODE = '''
import jax.numpy as jnp
import jax

def init_params() -> dict:
    """Initialize parameters for Kangaroo."""
    return {
        'ladder_threshold': jnp.array(12.0),
        'move_speed': jnp.array(1.0),
    }

def policy(obs_flat: jnp.ndarray, params: dict) -> int:
    """Simple policy: find ladder and climb."""
    player_x = obs_flat[0]
    player_y = obs_flat[1]
    
    # Get first ladder position (indices 43-44)
    ladder_x = obs_flat[43]
    ladder_y = obs_flat[44]
    
    # Check if we're near a ladder
    dx = ladder_x - player_x
    at_ladder = jnp.abs(dx) < params['ladder_threshold']
    
    # If at ladder, climb up. Otherwise move towards ladder
    action = jax.lax.cond(
        at_ladder,
        lambda: 2,  # UP - climb
        lambda: jax.lax.cond(
            dx > 0,
            lambda: 3,  # RIGHT
            lambda: 4   # LEFT
        )
    )
    return action

def measure_main(episode_rewards: jnp.ndarray, episode_scores: dict) -> float:
    return jnp.sum(episode_rewards)
'''

FREEWAY_DEMO_CODE = '''
import jax.numpy as jnp
import jax

def init_params() -> dict:
    """Initialize parameters for Freeway."""
    return {
        'safety_margin': jnp.array(20.0),  # How close a car can be before we wait
        'lane_margin': jnp.array(12.0),    # How close to car y to consider in lane
    }

def policy(obs_flat: jnp.ndarray, params: dict) -> int:
    """Simple policy: move UP unless a car is too close."""
    chicken_x = obs_flat[0]  # Fixed at 40
    chicken_y = obs_flat[1]  # 187 = bottom, 15 = top/goal
    
    # Check all 10 cars for danger
    danger = jnp.bool_(False)
    
    # Car 0-9 check: each car at indices 4+i*4 (x) and 5+i*4 (y)
    for i in range(10):
        car_x = obs_flat[4 + i*4]
        car_y = obs_flat[5 + i*4]
        in_lane = jnp.abs(chicken_y - car_y) < params['lane_margin']
        car_close = jnp.abs(car_x - chicken_x) < params['safety_margin']
        danger = danger | (in_lane & car_close)
    
    # Action: NOOP if danger, else UP
    action = jax.lax.cond(
        danger,
        lambda: jnp.int32(0),  # NOOP - wait
        lambda: jnp.int32(2)   # UP - move towards goal
    )
    return action

def measure_main(episode_rewards: jnp.ndarray, episode_scores: dict) -> float:
    return jnp.sum(episode_rewards)
'''

def run_demo_mode(config: OptimizationConfig, seed: int = 42, game: str = "pong"):
    """Run a demo using a hand-crafted policy (no LLM required)."""
    
    print("=" * 60)
    print(f"DEMO MODE - Using hand-crafted policy for {game.upper()}")
    print("=" * 60)
    
    # Select demo code based on game
    if game == "freeway":
        demo_code = FREEWAY_DEMO_CODE
    else:
        demo_code = PONG_DEMO_CODE
    
    # Save and load
    filepath = save_policy_module(demo_code, 0, config.output_dir)
    print(f"Saved demo policy to: {filepath}")
    
    init_params_fn, policy_fn, measure_fn = load_policy_module(filepath)
    
    # Evaluate
    evaluator = ParallelEvaluator(config, game=game)
    searcher = ParameterSearcher(config, evaluator)
    
    key = jrandom.PRNGKey(seed)
    print(f"\nEvaluating policy ({config.num_param_samples} parameter samples)...")
    
    best_params, metrics = searcher.search(policy_fn, init_params_fn, key)
    
    print(f"\nDemo Results:")
    print(f"  Average Return: {metrics['avg_return']:.2f}")
    if 'avg_player_score' in metrics:
        print(f"  Player Score: {metrics['avg_player_score']:.2f}")
        print(f"  Enemy Score: {metrics['avg_enemy_score']:.2f}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    
    return metrics


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM-Optimized Policy Loop for JAXAtari Games"
    )
    # Game selection
    parser.add_argument('--game', type=str, default='pong',
                       choices=['pong', 'freeway', 'asterix', 'breakout', 'skiing'],
                       help='Game to optimize in the current unified thesis scope')
    
    # LLM settings
    parser.add_argument('--api-key', type=str, default=None,
                       help='API key for LLM (or set OPENAI_API_KEY/ANTHROPIC_API_KEY env var)')
    parser.add_argument('--provider', type=str, default='anthropic',
                       choices=['openai', 'anthropic'],
                       help='LLM provider to use')
    parser.add_argument('--model', type=str, default='claude-opus-4-7',
                       help='Model name to use (e.g., claude-opus-4-7, claude-sonnet-4-6, claude-sonnet-4-20250514, claude-opus-4-20250514)')
    parser.add_argument('--temperature', type=float, default=None,
                       help='Optional LLM sampling temperature. Leave unset for models that do not support it.')
    
    # Evaluation settings - MASSIVE PARALLELISM
    parser.add_argument('--num-envs', type=int, default=512,
                       help='Number of parallel environments (can be 1000+)')
    parser.add_argument('--num-episodes', type=int, default=512,
                       help='Number of evaluation episodes')
    parser.add_argument('--max-steps', type=int, default=2000,
                       help='Maximum steps per episode')
    parser.add_argument('--frame-stack', type=int, default=0,
                       help='Frame stack size (0 = game default, Pong uses 2)')
    parser.add_argument('--search-max-steps', type=int, default=0,
                       help='Inner-loop optimization horizon (0 = use --max-steps)')
    
    # Optimization settings
    parser.add_argument('--optimizer', type=str, default='cma-es',
                       choices=['none', 'random', 'cma-es', 'bayes'],
                       help='Optimizer to use: none for LLM-only ablation, random search, CMA-ES, or Bayesian search')
    parser.add_argument('--max-iters', type=int, default=10,
                       help='Maximum LLM optimization iterations')
    parser.add_argument('--target-score', type=float, default=10.0,
                       help='Target score to reach')
    parser.add_argument('--param-samples', type=int, default=32,
                       help='Population size for CMA-ES / samples for random search')
    parser.add_argument('--cma-generations', type=int, default=10,
                       help='Number of CMA-ES generations per LLM iteration')
    parser.add_argument('--no-diagnostic-refinement', action='store_true',
                       help='Disable the diagnosis-then-clean-rewrite improvement step')
    parser.add_argument('--resume-from-results', action='store_true',
                       help='Continue from output_dir/optimization_results.json instead of starting from a fresh LLM policy')
    parser.add_argument('--no-resume-best-params-for-cma', action='store_true',
                       help='When resuming, do not seed CMA-ES from compatible best parameters in optimization_results.json')
    parser.add_argument('--no-refresh-resume-metrics', action='store_true',
                       help='Do not re-evaluate resumed best policy before asking the LLM for a rewrite')
    parser.add_argument('--no-stop-on-strong-best', action='store_true',
                       help='Keep running all requested outer-loop iterations even after a strong best policy is found')
    
    # Other settings
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: scripts/llm_optimization/runs/single_game/<game>)')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo mode without LLM')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Set default output directory based on game
    if args.output_dir is None:
        args.output_dir = f'scripts/llm_optimization/runs/single_game/{args.game}'
    
    # Get API key: command line > environment variable > config default
    api_key = args.api_key
    if not api_key:
        if args.provider == 'openai':
            api_key = os.environ.get('OPENAI_API_KEY')
        elif args.provider == 'anthropic':
            api_key = os.environ.get('ANTHROPIC_API_KEY')
    
    if not api_key and not args.demo:
        raise SystemExit(
            f"No API key provided for {args.provider}. "
            f"Set the relevant environment variable or pass --demo explicitly."
        )
    
    # Create config
    config = OptimizationConfig(
        llm_provider=args.provider,
        llm_model=args.model,
        api_key=api_key,
        temperature=args.temperature,
        num_parallel_envs=args.num_envs,
        num_eval_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        frame_stack_size=args.frame_stack,
        search_max_steps=args.search_max_steps,
        optimizer=args.optimizer,
        max_iterations=args.max_iters,
        target_score=args.target_score,
        diagnostic_refinement=not args.no_diagnostic_refinement,
        resume_from_results=args.resume_from_results,
        resume_best_params_for_cma=not args.no_resume_best_params_for_cma,
        refresh_resume_metrics=not args.no_refresh_resume_metrics,
        stop_on_strong_best=not args.no_stop_on_strong_best,
        num_param_samples=args.param_samples,
        cma_es_generations=args.cma_generations,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        seed=args.seed,
    )
    
    # Print game info
    print(f"\n{'='*60}")
    print(f"Game: {args.game.upper()}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Search Horizon: {args.search_max_steps or args.max_steps} steps")
    print(f"Evaluation Horizon: {args.max_steps} steps")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")
    
    if args.demo:
        run_demo_mode(config, args.seed, game=args.game)
    else:
        loop = LLMOptimizationLoop(config, game=args.game)
        loop.run(args.seed)


if __name__ == "__main__":
    main()

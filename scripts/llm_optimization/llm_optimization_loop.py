"""
Language Guided Policy Search (LeGPS) loop for JAXAtari games.

This module implements the unified thesis pipeline:
1. Generate a parametric policy structure with an LLM
2. Optimize the policy's numeric parameters with CMA-ES
3. Evaluate the tuned controller on the base game task
4. Feed back measured behavior for conservative iterative revision
"""

import os
import sys
import json
import time
import argparse
import importlib.util
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
    llm_model: str = "claude-sonnet-4-6"
    api_key: Optional[str] = None
    max_tokens: int = 8192
    temperature: float = 0.7
    
    # Evaluation settings - MASSIVE PARALLELISM
    num_parallel_envs: int = 1024  # Run 1024 environments in parallel!
    max_steps_per_episode: int = 2000
    num_eval_episodes: int = 1024  # Evaluate over 1024 episodes
    frame_stack_size: int = 0  # 0 = use game default
    search_max_steps: int = 0  # 0 = use max_steps_per_episode
    
    # Parameter search settings - CMA-ES
    optimizer: str = "cma-es"  # "random", "cma-es", or "evolution"
    num_param_samples: int = 64  # Population size for CMA-ES
    param_perturbation_scale: float = 0.3  # Initial sigma for CMA-ES
    cma_es_generations: int = 20  # Number of CMA-ES generations per iteration
    
    # Optimization loop settings
    max_iterations: int = 10
    target_score: float = 15.0  # Win by 15 points on average
    
    # File paths
    output_dir: str = "scripts/llm_optimization/unified_prompt_main/pong"
    
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
The policy receives a flattened 1D array with 28 features because Pong uses `frame_stack=2`.
Think of it as two object-centric structs concatenated together:

```python
FRAME_SIZE = 14

prev = obs_flat[0:14]
curr = obs_flat[14:28]

prev.player.x = prev[0]
prev.player.y = prev[1]
prev.player.w = prev[2]
prev.player.h = prev[3]

prev.enemy.x = prev[4]
prev.enemy.y = prev[5]
prev.enemy.w = prev[6]
prev.enemy.h = prev[7]

prev.ball.x = prev[8]
prev.ball.y = prev[9]
prev.ball.w = prev[10]
prev.ball.h = prev[11]

prev.score_player = prev[12]
prev.score_enemy = prev[13]

curr.player.x = curr[0]
curr.player.y = curr[1]
curr.player.w = curr[2]
curr.player.h = curr[3]

curr.enemy.x = curr[4]
curr.enemy.y = curr[5]
curr.enemy.w = curr[6]
curr.enemy.h = curr[7]

curr.ball.x = curr[8]
curr.ball.y = curr[9]
curr.ball.w = curr[10]
curr.ball.h = curr[11]

curr.score_player = curr[12]
curr.score_enemy = curr[13]

ball_dx = curr.ball.x - prev.ball.x
ball_dy = curr.ball.y - prev.ball.y
```

### Useful Constants
You may define named constants in the generated module so you do not have to remember raw numbers:

```python
NOOP = 0
FIRE = 1
MOVE_DOWN = 3
MOVE_UP = 4
FIRE_DOWN = 11
FIRE_UP = 12

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
Relevant raw Atari actions for Pong:
- `0`: `NOOP`
- `3`: move paddle DOWN (implemented as `RIGHT`)
- `4`: move paddle UP (implemented as `LEFT`)
- `11`: move DOWN and FIRE (`RIGHTFIRE`)
- `12`: move UP and FIRE (`LEFTFIRE`)

Important:
- Y increases downward.
- The player controls the RIGHT paddle.
- `3` and `11` both move the paddle downward.
- `4` and `12` both move the paddle upward.
- `11` and `12` can accelerate the ball on player paddle hit.

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
6. FIRE actions are allowed and may help escape local optima.
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
The flattened observation has 88 features = previous frame (44) + current frame (44).

```python
FRAME_SIZE = 44
prev = obs_flat[0:44]
curr = obs_flat[44:88]

prev.chicken.x = prev[0]
prev.chicken.y = prev[1]
curr.chicken.x = curr[0]
curr.chicken.y = curr[1]

# Cars are 10 fixed lanes, each with [x, y, width, height]
# For lane i, the frame-local base index is 4 + 4*i
car_i_x = frame[4 + 4*i]
car_i_y = frame[5 + 4*i]
car_i_w = frame[6 + 4*i]
car_i_h = frame[7 + 4*i]

# Velocity estimate for lane i
car_i_dx = curr[4 + 4*i] - prev[4 + 4*i]
```

### Useful Constants
```python
NOOP = 0
UP = 2
DOWN = 5

SCREEN_WIDTH = 160
SCREEN_HEIGHT = 210
CHICKEN_X = 40
CHICKEN_WIDTH = 6
CHICKEN_HEIGHT = 8
TOP_BORDER = 15
BOTTOM_BORDER = 180
LANE_Y = [23, 39, 55, 71, 87, 103, 119, 135, 151, 167]
LANE_DIRECTION = [-1, -1, -1, -1, -1, +1, +1, +1, +1, +1]
```

### Key Insights
1. Chicken x is fixed at 40, only y changes.
2. Each car stays in a fixed lane, but lane traffic moves with different speeds.
3. A static one-frame distance check is too conservative.
4. A stronger policy should estimate lane-wise motion from the last two frames and time safe lane entries.
5. `UP` should still dominate, but only when the projected overlap for the next lane is safe enough.
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

### Observation Space (118 features)
The observation is a flattened 1D array:

Index    | Feature              | Description
---------|----------------------|------------------------------------------
0        | player.x             | Paddle left edge X position (0-144)
1        | player.y             | Paddle Y position (~189, fixed at bottom)
2        | player.w             | Paddle width (16)
3        | player.h             | Paddle height (4)
4        | ball.x               | Ball X position (0-160)
5        | ball.y               | Ball Y position (0-210, higher=closer to paddle)
6        | ball.w               | Ball width (2)
7        | ball.h               | Ball height (4)
8-115    | blocks               | 108 block states: 1=present, 0=broken
116      | score                | Current game score
117      | lives                | Remaining lives (0-5)

### Action Space (CRITICAL - Atari standard mapping!)
- Action 0: NOOP (do nothing)
- Action 1: FIRE (not needed, ball auto-launches)
- Action 3: RIGHT (move paddle right, +X)
- Action 4: LEFT (move paddle left, -X)

**WARNING**: Actions 2, 5+ do NOT move the paddle! Use 3 and 4 only!

### Key Insights
1. Paddle center = player.x + 8 (half of paddle width 16)
2. Ball auto-launches, no FIRE needed
3. Goal: Keep ball in play by positioning paddle under it
4. Track ball.x and move paddle to match
5. Y coordinates: 0 = top, ~210 = bottom (paddle area)
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
Valid raw actions for this environment:
- `0`: NOOP
- `1`: UP
- `2`: RIGHT
- `3`: LEFT
- `4`: DOWN
- `5`: UPRIGHT
- `6`: UPLEFT
- `7`: DOWNRIGHT
- `8`: DOWNLEFT

### Useful Facts
1. The player moves between horizontal lanes and can also move laterally within a lane.
2. Each lane may contain an enemy or a collectible moving left/right.
3. Enemy collisions cost lives and trigger respawn / hit timers internally, but those timers are not in the observation.
4. Collectibles increase score; `collect.visual_id` distinguishes item types.
5. Lower `player.y` means higher on the screen.
6. Enemy orientation is encoded as 0=stationary, 1=moving right, 2=moving left.
7. Lane centers are approximately at y = [27, 43, 59, 75, 91, 107, 123, 139].
8. Pure survival is not enough; high score comes from repeatedly entering safe collectible lanes and moving laterally to intercept items.
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

UNIFIED_INITIAL_PROMPT = """
You are generating an initial parametric policy for Atari {game_name}.

{environment_description}

## Shared Method Objective
Produce one clean parametric controller that can be optimized by CMA-ES.
The structure should generalize well and avoid game-specific prompt hacks.

## Common Design Rules
1. Prefer 3-8 tunable float parameters.
2. Use the object-centric observation exactly as documented.
3. Prefer simple score-seeking logic over complex hand-crafted heuristics.
4. Use named constants and helper aliases instead of unexplained magic indices.
5. Keep the policy optimizer-friendly: small parameter count, shallow logic, no brittle special cases.

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

### 3. measure_main(episode_rewards: jnp.ndarray, episode_scores: dict) -> float
Return `jnp.sum(episode_rewards)`.

## JAX Constraints
1. Use `jax.lax.cond` or `jnp.where` for branching; no Python `if` in the traced policy.
2. Do not use Python `int()` or `float()` inside `policy`.
3. Keep outputs as valid integer actions.
4. Keep the code concise and stable under tracing.
5. Prefer short helper functions if that reduces syntax risk or nested control flow.

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
Improve score while keeping the controller simple enough for CMA-ES.
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

Generate the improved policy now.
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
4. Use NOOP sparingly and DOWN mostly as an escape action.
5. Do not stop just before the top border; finish the crossing.
""".strip(),
        failure_modes="""
1. Static distance checks with no temporal reasoning.
2. Excessive hesitation and high NOOP rates.
3. Conservative stalling one step before scoring.
4. Symmetric UP/DOWN behavior that destroys upward progress.
""".strip(),
        improvement_guidelines="""
1. Improve lane timing rather than adding many parameters.
2. Default to continued upward progress once a lane is judged safe.
3. Remove rules that wait too long in locally safe states.
4. Keep the controller lane-centric and simple.
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
6. Use vertical movement to reach higher-value safe lanes, not just to wander.
7. Keep the code modular: small helpers such as `_lane_danger`, `_lane_value`, and `_target_collectible_x` are preferred.
8. Keep NOOP usage low once a lane is judged safe.
""".strip(),
        failure_modes="""
1. Safe-but-passive behavior with high NOOP rate.
2. Returning to center instead of harvesting available safe collectibles.
3. Overcomplicated global heuristics instead of simple lane logic.
4. Deeply nested `jnp.where` expressions that become syntactically fragile.
5. Treating low-hundreds score as good when the benchmark gap is still huge.
6. Changing lanes too often without ever committing laterally to a collectible.
""".strip(),
        improvement_guidelines="""
1. Push score upward by increasing continuous collectible harvesting, not just survival.
2. Reduce passive waiting inside safe lanes.
3. First pick a target lane, then pick a target x inside that lane.
4. Reward sustained harvesting in adjacent lanes more than frequent recentering.
5. Improve lane prioritization without making the parameterization large.
6. Keep helper functions short and explicit so the module remains syntactically robust.
7. Balance safety with continued item collection across multiple lives.
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
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            return response.choices[0].message.content
        elif self.config.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.config.llm_model,
                max_tokens=self.config.max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
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
        frame_stack_size = self.config.frame_stack_size if self.config.frame_stack_size > 0 else (2 if self.game in ("pong", "freeway", "asterix") else 1)
        self.frame_stack_size = frame_stack_size
        # Disable episodic_life for Freeway and Asterix so evaluation matches full-episode scoring.
        episodic_life = False if self.game in ("freeway", "asterix") else True
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
        self.action_clip_max = 12 if self.game == "pong" else self.action_space_n - 1
        
        # Get observation size by sampling
        key = jrandom.PRNGKey(0)
        obs, _ = self.env.reset(key)
        self.obs_size = obs.shape[-1]
        self.frame_obs_size = self.obs_size // self.frame_stack_size
    
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
            
            # Update cumulative reward
            next_total_reward = total_reward + reward
            
            # Continue or terminate
            next_done = jnp.logical_or(terminated, truncated)
            new_done = jnp.logical_or(done, next_done)
            
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
            
            return (next_obs_flat, next_state, next_total_reward, new_done), (reward, obs_flat)
        
        init_carry = (obs_flat, state, jnp.array(0.0), jnp.array(False))
        (final_obs, final_state, total_reward, _), (rewards, obs_history) = lax.scan(
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
        
        while not done and step < max_steps:
            # Get action from policy
            action = policy_fn(obs_flat, params)
            action = int(jnp.clip(action, 0, self.action_clip_max))
            
            # Step environment
            next_obs, next_state, reward, terminated, truncated, info = self.env.step(state, action)
            
            # Log state (Logger will auto-unwrap to get game state)
            try:
                logger.log_state(next_state, action=action, reward=float(reward))
            except Exception as e:
                # If logging fails, continue without logging
                if step == 0:
                    print(f"  Logger warning: {e}")
            
            # Update for next step
            total_reward += float(reward)
            obs_flat = next_obs[-self.obs_size:] if next_obs.shape[0] > self.obs_size else next_obs
            obs_flat = jnp.asarray(obs_flat, dtype=jnp.float32)
            state = next_state
            done = bool(jnp.logical_or(terminated, truncated))
            step += 1
        
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


class ParameterSearcher:
    """Unified parameter searcher that selects optimizer based on config."""
    
    def __init__(self, config: OptimizationConfig, evaluator: 'ParallelEvaluator'):
        self.config = config
        self.evaluator = evaluator
        
        # Select optimizer
        if config.optimizer == "cma-es":
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
                if logger_metrics.get('collision_rate', 0) > 0.5:
                    feedback_parts.append("High collision rate: improve car avoidance and lane-entry timing.")
                if logger_metrics.get('crossings', 0) == 0:
                    feedback_parts.append("No crossings recorded: the chicken must actually reach the top of the screen.")
            elif game == "asterix":
                if logger_metrics.get('hits_taken', 0) >= 2:
                    feedback_parts.append("\nToo many enemy hits: lane danger estimation is too weak or too late.")
                if logger_metrics.get('item_pickups', 0) < 5:
                    feedback_parts.append("Very few item pickups: the policy is not committing to collectible lanes often enough.")
                if logger_metrics.get('item_pickups', 0) < 12:
                    feedback_parts.append("Item pickup count is still modest. The policy should harvest more lanes per life rather than settling into safe but low-scoring behavior.")
                if logger_metrics.get('respawn_rate', 0) > 0.15:
                    feedback_parts.append("High respawn rate: collisions are consuming too much time.")
                if logger_metrics.get('noop_rate', 0) > 0.2:
                    feedback_parts.append("Too much idle behavior: prefer deliberate movement toward safe items.")
                if logger_metrics.get('max_stage_reached', 0) < 4:
                    feedback_parts.append("The agent is not exploring enough lanes: vertical movement is too conservative.")
                if logger_metrics.get('horizontal_rate', 0) < 0.4:
                    feedback_parts.append("Horizontal commitment is too low. Once a lane is safe, move laterally to intercept items instead of waiting.")
        except Exception as e:
            feedback_parts.append(f"(Logger metrics formatting error: {e})")

    feedback_parts.append("\n## CRITICAL IMPLEMENTATION RULES:")
    if game == "pong":
        feedback_parts.append("1. ERROR CALCULATION: error = target_y - paddle_center_y.")
        feedback_parts.append("2. When error > 0 move DOWN; when error < 0 move UP.")
        feedback_parts.append("3. Use prev/curr frames to compute ball_dx and ball_dy.")
        feedback_parts.append("4. Do not use Python int() inside policy().")
    elif game == "freeway":
        feedback_parts.append("1. Use prev/curr frames to estimate car dx per lane.")
        feedback_parts.append("2. Optimize for safe lane-entry timing, not static nearest-car checks.")
        feedback_parts.append("3. Action mapping is 0=NOOP, 2=UP, 5=DOWN in this wrapper.")
        feedback_parts.append("4. Do not use Python int() inside policy().")
    elif game == "asterix":
        feedback_parts.append("1. Each frame is 136 features; stacked observations have 272 features.")
        feedback_parts.append("2. Use prev/curr enemy and collectible x positions to estimate lane-wise dx.")
        feedback_parts.append("3. The action set is 0..8 only: NOOP, UP, RIGHT, LEFT, DOWN, UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT.")
        feedback_parts.append("4. The observation does not contain score, lives, or timers directly.")
        feedback_parts.append("5. Asterix score comes from repeated item collection; safe-but-passive policies are not enough.")
        feedback_parts.append("6. Do not use Python int() or Python if-statements inside policy().")
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
        self.system_prompt = UNIFIED_SYSTEM_PROMPT

    def _build_initial_prompt(self) -> str:
        return UNIFIED_INITIAL_PROMPT.format(
            game_name=self.game.capitalize(),
            environment_description=self.prompt_spec.environment_description,
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
            strong_policy = avg_return >= 1000.0
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
            revision_strategy=revision_strategy,
            change_budget=change_budget,
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
            return avg_return >= 1000.0

        return avg_return >= self.config.target_score
    
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
        
        for iteration in range(self.config.max_iterations):
            self.iteration = iteration + 1
            print(f"\n{'='*60}")
            print(f"Iteration {self.iteration}/{self.config.max_iterations}")
            print("=" * 60)

            if iteration > 0 and self._is_strong_result(best_metrics) and no_improvement_streak >= 1:
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
                print("Successfully loaded policy module")
            except Exception as e:
                print(f"ERROR loading policy: {e}")
                print("Will retry with a new LLM request...")
                current_code = None  # Reset to trigger fresh generation
                continue
            
            # Step 3: Search for best parameters
            search_key, key = jrandom.split(key)
            print(f"\nSearching for optimal parameters ({self.config.num_param_samples} samples)...")
            
            try:
                current_params, search_metrics = self.searcher.search(
                    policy_fn, init_params_fn, search_key
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
                if self._is_strong_result(best_metrics):
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
        
        # Save final results
        results = {
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
            'best_code': best_code
        }
        
        results_path = Path(self.config.output_dir) / "optimization_results.json"
        with open(results_path, 'w') as f:
            json.dump(
                {k: v for k, v in results.items() if k != 'best_code'},
                f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)
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
                       choices=['pong', 'freeway', 'asterix'],
                       help='Game to optimize in the current unified thesis scope')
    
    # LLM settings
    parser.add_argument('--api-key', type=str, default=None,
                       help='API key for LLM (or set OPENAI_API_KEY/ANTHROPIC_API_KEY env var)')
    parser.add_argument('--provider', type=str, default='anthropic',
                       choices=['openai', 'anthropic'],
                       help='LLM provider to use')
    parser.add_argument('--model', type=str, default='claude-sonnet-4-6',
                       help='Model name to use (e.g., claude-sonnet-4-6, claude-sonnet-4-20250514, claude-opus-4-20250514)')
    
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
                       choices=['random', 'cma-es', 'bayes'],
                       help='Optimizer to use: random search, CMA-ES, or Bayesian search')
    parser.add_argument('--max-iters', type=int, default=10,
                       help='Maximum LLM optimization iterations')
    parser.add_argument('--target-score', type=float, default=10.0,
                       help='Target score to reach')
    parser.add_argument('--param-samples', type=int, default=32,
                       help='Population size for CMA-ES / samples for random search')
    parser.add_argument('--cma-generations', type=int, default=10,
                       help='Number of CMA-ES generations per LLM iteration')
    
    # Other settings
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: scripts/llm_optimization/unified_prompt_main/<game>)')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo mode without LLM')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Set default output directory based on game
    if args.output_dir is None:
        args.output_dir = f'scripts/llm_optimization/unified_prompt_main/{args.game}'
    
    # Get API key: command line > environment variable > config default
    api_key = args.api_key
    if not api_key:
        if args.provider == 'openai':
            api_key = os.environ.get('OPENAI_API_KEY')
        elif args.provider == 'anthropic':
            api_key = os.environ.get('ANTHROPIC_API_KEY')
    
    if not api_key and not args.demo:
        print(f"Warning: No API key provided for {args.provider}. Running in demo mode.")
        args.demo = True
    
    # Create config
    config = OptimizationConfig(
        llm_provider=args.provider,
        llm_model=args.model,
        api_key=api_key,
        num_parallel_envs=args.num_envs,
        num_eval_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        frame_stack_size=args.frame_stack,
        search_max_steps=args.search_max_steps,
        optimizer=args.optimizer,
        max_iterations=args.max_iters,
        target_score=args.target_score,
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

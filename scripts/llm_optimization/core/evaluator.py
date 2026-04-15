"""
Evaluator for parallel policy evaluation using JAX.

Uses vmap over episode keys (each key = independent env reset+rollout).
Matches the working pattern from llm_optimization_loop.py.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import lax, vmap
from typing import Callable, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import jaxatari
from jaxatari.wrappers import AtariWrapper, ObjectCentricWrapper, FlattenObservationWrapper


@dataclass
class EvaluationMetrics:
    """Metrics from policy evaluation."""
    avg_return: float
    win_rate: float
    avg_episode_length: float
    total_episodes: int


class ParallelEvaluator:
    """
    Parallel policy evaluator using JAX vmap over episodes.
    
    Each episode gets its own key -> independent reset -> independent rollout.
    vmap parallelizes across episodes automatically on GPU/CPU.
    """
    
    def __init__(
        self,
        game_name: str,
        num_envs: int = 1024,
        max_steps: int = 2000,
        verbose: bool = False,
        action_clip_max: Optional[int] = None,
        frame_skip: Optional[int] = None,
        episodic_life: Optional[bool] = None,
    ):
        self.game_name = game_name
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.verbose = verbose
        self.action_clip_max = action_clip_max
        self.frame_skip = frame_skip
        self.episodic_life = episodic_life
        
        # Initialize environment
        self._init_environment()
    
    def _init_environment(self):
        """Initialize the environment with wrappers."""
        base_env = jaxatari.make(self.game_name)
        frame_skip = self.frame_skip if self.frame_skip is not None else 1
        episodic_life = self.episodic_life if self.episodic_life is not None else (False if self.game_name == "freeway" else True)
        self.env = FlattenObservationWrapper(
            ObjectCentricWrapper(
                AtariWrapper(base_env, episodic_life=episodic_life),
                frame_stack_size=1,
                frame_skip=frame_skip,
                clip_reward=False,
            )
        )
        self.action_space_n = self.env.action_space().n
        self.effective_action_clip_max = (
            self.action_space_n - 1
            if self.action_clip_max is None
            else self.action_clip_max
        )
        
        # Get observation size by sampling
        key = jrandom.PRNGKey(0)
        obs, _ = self.env.reset(key)
        self.obs_size = obs.shape[-1]
        
        if self.verbose:
            print(f"  Environment: {self.game_name}")
            print(f"  Parallel envs: {self.num_envs}")
            print(f"  Observation size: {self.obs_size}")
            print(f"  Action space: {self.action_space_n}")
            print(f"  Action clip max: {self.effective_action_clip_max}")
            print(f"  Frame skip: {frame_skip}")
            print(f"  Episodic life: {episodic_life}")
    
    def _run_single_episode(
        self,
        policy_fn: Callable,
        params: Dict,
        key: jrandom.PRNGKey,
        max_steps: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Run a single episode and return reward, scores, length, and completion."""
        
        obs, state = self.env.reset(key)
        obs_flat = obs[-self.obs_size:] if obs.shape[0] > self.obs_size else obs
        
        def step_fn(carry, _):
            obs_flat, state, total_reward, done, step_count = carry
            
            # Get action from policy
            action = policy_fn(obs_flat, params)
            action = jnp.clip(action, 0, self.effective_action_clip_max).astype(jnp.int32)
            
            # Step environment (state first, then action)
            next_obs, next_state, reward, terminated, truncated, info = self.env.step(state, action)
            next_obs_flat = next_obs[-self.obs_size:] if next_obs.shape[0] > self.obs_size else next_obs
            
            next_total_reward = total_reward + reward
            
            # Track done state
            next_done = jnp.logical_or(terminated, truncated)
            new_done = jnp.logical_or(done, next_done)
            
            # Track step count for episode length
            new_step_count = step_count + jnp.where(done, 0.0, 1.0)
            
            # Keep old state if already done
            next_obs_flat = jax.lax.cond(done, lambda: obs_flat, lambda: next_obs_flat)
            next_state = jax.lax.cond(done, lambda: state, lambda: next_state)
            
            return (next_obs_flat, next_state, next_total_reward, new_done, new_step_count), reward
        
        init_carry = (obs_flat, state, jnp.float32(0.0), jnp.bool_(False), jnp.float32(0.0))
        (final_obs, _, total_reward, done, ep_length), _ = lax.scan(
            step_fn, init_carry, None, length=max_steps
        )

        if self.obs_size == 14:
            player_score = final_obs[12]
            enemy_score = final_obs[13]
        else:
            player_score = total_reward
            enemy_score = jnp.float32(0.0)
        
        return total_reward, player_score, enemy_score, ep_length, done
    
    def evaluate(
        self,
        policy_fn: Callable,
        params: Dict,
        key: jrandom.PRNGKey
    ) -> Dict[str, Any]:
        """
        Evaluate a policy over num_envs parallel episodes.
        
        Args:
            policy_fn: Policy function (obs_flat, params) -> action
            params: Policy parameters
            key: Random key
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Generate keys for each episode
        keys = jrandom.split(key, self.num_envs)
        
        # Vectorize episode runner
        vmapped_run = vmap(
            lambda k: self._run_single_episode(policy_fn, params, k, self.max_steps)
        )
        
        # Run all episodes in parallel
        total_rewards, player_scores, enemy_scores, ep_lengths, dones = vmapped_run(keys)
        
        # Calculate metrics
        avg_return = float(jnp.mean(total_rewards))
        avg_player_score = float(jnp.mean(player_scores))
        avg_enemy_score = float(jnp.mean(enemy_scores))
        
        # Win rate: positive return is a win
        win_rate = float(jnp.mean(player_scores > enemy_scores))
        
        avg_length = float(jnp.mean(ep_lengths))
        completion_rate = float(jnp.mean(dones))
        
        return {
            'avg_return': avg_return,
            'avg_player_score': avg_player_score,
            'avg_enemy_score': avg_enemy_score,
            'win_rate': win_rate,
            'avg_episode_length': avg_length,
            'completion_rate': completion_rate,
            'total_episodes': self.num_envs,
        }
    
    def get_observation_size(self) -> int:
        """Return the flattened observation size."""
        return self.obs_size


def create_evaluator(
    game_name: str,
    num_envs: int = 1024,
    max_steps: int = 2000,
    verbose: bool = False,
    action_clip_max: Optional[int] = None,
    frame_skip: Optional[int] = None,
    episodic_life: Optional[bool] = None,
) -> ParallelEvaluator:
    """Factory function to create an evaluator."""
    return ParallelEvaluator(
        game_name=game_name,
        num_envs=num_envs,
        max_steps=max_steps,
        verbose=verbose,
        action_clip_max=action_clip_max,
        frame_skip=frame_skip,
        episodic_life=episodic_life,
    )

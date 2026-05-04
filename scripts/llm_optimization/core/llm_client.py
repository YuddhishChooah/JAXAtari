"""
LLM Client for policy generation using Anthropic Claude.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    temperature: float = 0.7
    api_key: Optional[str] = None


class LLMClient:
    """
    Client for interacting with Anthropic Claude API.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the LLM client."""
        self.config = config or LLMConfig()
        
        # Get API key from config, environment, or raise error
        api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key in config."
            )

        try:
            import anthropic
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The anthropic package is required to construct LLMClient. "
                "Install the LLM provider dependencies before running generation."
            ) from exc
        
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            
        Returns:
            Generated text response
        """
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = self.client.messages.create(**kwargs)
        return response.content[0].text
    
    def generate_policy(
        self,
        game_context: str,
        observation_description: str,
        action_description: str,
        feedback: Optional[str] = None,
        previous_policy: Optional[str] = None,
        design_principles: Optional[str] = None,
        version: int = 1
    ) -> str:
        """
        Generate a policy for a game.
        
        Args:
            game_context: Description of the game
            observation_description: Description of observations
            action_description: Description of actions
            feedback: Optional feedback from previous evaluations
            previous_policy: Optional previous policy code for improvement
            design_principles: Optional design principles to follow
            version: Policy version number
            
        Returns:
            Generated policy code as string
        """
        system_prompt = self._build_system_prompt(design_principles)
        user_prompt = self._build_user_prompt(
            game_context=game_context,
            observation_description=observation_description,
            action_description=action_description,
            feedback=feedback,
            previous_policy=previous_policy,
            version=version
        )
        
        return self.generate(user_prompt, system_prompt=system_prompt)
    
    def _build_system_prompt(self, design_principles: Optional[str] = None) -> str:
        """Build the system prompt for policy generation."""
        base = """You are an expert reinforcement learning policy designer.
Your task is to create JAX-compatible policy functions for Atari games.

Key requirements:
1. Use only JAX/NumPy operations (jax.numpy, jax.lax)
2. Policies must be JIT-compilable (no Python control flow on traced values)
3. Use jax.lax.cond/select instead of Python if/else
4. Keep policies simple and interpretable
5. Return integer action indices

Output format:
- Return ONLY valid Python code
- Include 'policy' function and 'init_params' function
- Use type hints
- Include brief docstrings"""

        if design_principles:
            base += f"\n\nDesign Principles:\n{design_principles}"
        
        return base
    
    def _build_user_prompt(
        self,
        game_context: str,
        observation_description: str,
        action_description: str,
        feedback: Optional[str] = None,
        previous_policy: Optional[str] = None,
        version: int = 1
    ) -> str:
        """Build the user prompt for policy generation."""
        
        prompt = f"""Create a policy for the following game:

## Game Context
{game_context}

## Observation Space
{observation_description}

## Action Space
{action_description}

"""
        
        if previous_policy and feedback:
            prompt += f"""## Previous Policy (v{version - 1})
```python
{previous_policy}
```

## Feedback from Evaluation
{feedback}

Please improve the policy based on the feedback above.
"""
        else:
            prompt += "Create an initial policy using your game knowledge.\n"
        
        prompt += """
## Required Output Format
```python
import jax.numpy as jnp
import jax.lax

def init_params():
    \"\"\"Initialize tunable parameters.\"\"\"
    return {
        # Add tunable parameters here (3-8 total recommended)
    }

def policy(observation: jnp.ndarray, params: dict) -> int:
    \"\"\"
    Policy function.
    
    Args:
        observation: Flattened observation array
        params: Tunable parameters
        
    Returns:
        Action index (integer)
    \"\"\"
    # Your policy logic here
    return action
```

Return ONLY the Python code, no explanations."""

        return prompt


def create_llm_client(
    model: str = "claude-sonnet-4-6",
    api_key: Optional[str] = None,
    temperature: float = 0.7
) -> LLMClient:
    """Factory function to create an LLM client."""
    config = LLMConfig(
        model=model,
        api_key=api_key,
        temperature=temperature
    )
    return LLMClient(config)

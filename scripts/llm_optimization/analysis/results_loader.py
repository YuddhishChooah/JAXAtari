"""
Unified results loader for the active LeGPS thesis runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


ACTIVE_GAMES = ["pong", "freeway", "asterix", "breakout", "skiing"]


@dataclass
class IterationResult:
    iteration: int
    metrics: Dict[str, Any]
    best_params: Dict[str, float] = field(default_factory=dict)
    policy_path: str = ""

    @property
    def avg_return(self) -> float:
        return float(self.metrics.get("avg_return", 0.0))

    @property
    def win_rate(self) -> float:
        return float(self.metrics.get("win_rate", 0.0))


@dataclass
class ExperimentResult:
    game: str
    best_score: float
    best_metrics: Dict[str, Any]
    best_params: Dict[str, float]
    best_policy_path: str
    config: Dict[str, Any]
    runtime: Dict[str, Any]
    iterations: List[IterationResult]

    @property
    def total_rewards(self) -> List[float]:
        return list(self.best_metrics.get("total_rewards", []))

    @property
    def reward_mean(self) -> float:
        rewards = self.total_rewards
        return float(np.mean(rewards)) if rewards else 0.0

    @property
    def reward_std(self) -> float:
        rewards = self.total_rewards
        return float(np.std(rewards)) if rewards else 0.0


class ResultsLoader:
    """Load active LeGPS experiments from canonical best-policy or run folders."""

    def __init__(self, base_path: Optional[str] = None, unified_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent
        self.project_root = self.base_path.parents[1]
        self.unified_path = Path(unified_path) if unified_path else self.base_path / "runs" / "best_10000_steps"
        if not self.unified_path.is_absolute():
            self.unified_path = self.project_root / self.unified_path

    def discover_experiments(self) -> List[str]:
        available: List[str] = []
        for game in ACTIVE_GAMES:
            game_dir = self.unified_path / game
            if (game_dir / "best_policy.json").exists() or (game_dir / "optimization_results.json").exists():
                available.append(game)
        return available

    def load_experiment(self, game: str) -> Optional[ExperimentResult]:
        canonical_file = self.unified_path / game / "best_policy.json"
        if canonical_file.exists():
            data = json.loads(canonical_file.read_text(encoding="utf-8-sig"))
            rewards = list(data.get("total_rewards_10000", []))
            best_metrics = {
                "avg_return": float(data.get("avg_return_10000", 0.0)),
                "avg_player_score": float(data.get("avg_player_score_10000", data.get("avg_return_10000", 0.0))),
                "avg_enemy_score": float(data.get("avg_enemy_score_10000", 0.0)),
                "win_rate": float(data.get("win_rate_10000", 0.0)),
                "total_rewards": rewards,
            }
            return ExperimentResult(
                game=game,
                best_score=float(data.get("avg_return_10000", 0.0)),
                best_metrics=best_metrics,
                best_params={k: float(v) for k, v in data.get("best_params", {}).items()},
                best_policy_path=data.get("canonical_policy_path", ""),
                config={
                    "max_steps_per_episode": int(data.get("max_steps", 0)),
                    "num_eval_episodes": int(data.get("num_episodes", 0)),
                    "source": data.get("source", ""),
                },
                runtime={},
                iterations=[],
            )

        results_file = self.unified_path / game / "optimization_results.json"
        if not results_file.exists():
            return None

        data = json.loads(results_file.read_text(encoding="utf-8-sig"))
        iterations = [
            IterationResult(
                iteration=int(entry["iteration"]),
                metrics=dict(entry.get("metrics", {})),
                best_params={k: float(v) for k, v in entry.get("best_params", {}).items()},
                policy_path=entry.get("filepath", ""),
            )
            for entry in data.get("history", [])
        ]

        return ExperimentResult(
            game=game,
            best_score=float(data.get("best_score", 0.0)),
            best_metrics=dict(data.get("best_metrics", {})),
            best_params={k: float(v) for k, v in data.get("best_params", {}).items()},
            best_policy_path=data.get("best_policy_path", ""),
            config=dict(data.get("config", {})),
            runtime=dict(data.get("runtime", {})),
            iterations=iterations,
        )

    def load_all_experiments(self) -> Dict[str, ExperimentResult]:
        results: Dict[str, ExperimentResult] = {}
        for game in self.discover_experiments():
            exp = self.load_experiment(game)
            if exp:
                results[game] = exp
        return results

    def get_iteration_progression(self, game: str) -> Dict[str, List[float]]:
        exp = self.load_experiment(game)
        if not exp:
            return {}

        progression = {
            "iteration": [],
            "avg_return": [],
            "win_rate": [],
            "avg_player_score": [],
            "avg_enemy_score": [],
        }

        logger_keys = set()
        for it in exp.iterations:
            logger_keys.update(it.metrics.get("logger_metrics", {}).keys())
        for key in sorted(logger_keys):
            progression[f"logger_{key}"] = []

        for it in exp.iterations:
            metrics = it.metrics
            progression["iteration"].append(it.iteration)
            progression["avg_return"].append(float(metrics.get("avg_return", 0.0)))
            progression["win_rate"].append(float(metrics.get("win_rate", 0.0)))
            progression["avg_player_score"].append(float(metrics.get("avg_player_score", 0.0)))
            progression["avg_enemy_score"].append(float(metrics.get("avg_enemy_score", 0.0)))
            logger_metrics = metrics.get("logger_metrics", {})
            for key in logger_keys:
                progression[f"logger_{key}"].append(float(logger_metrics.get(key, np.nan)))

        return progression

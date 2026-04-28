"""
Figure generation for the active unified LeGPS experiments.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .benchmarks import compute_human_normalized_score, get_benchmark_comparison
from .results_loader import ACTIVE_GAMES, ExperimentResult, ResultsLoader


class UnifiedVisualizer:
    GAME_COLORS = {
        "pong": "#2ecc71",
        "freeway": "#3498db",
        "asterix": "#f39c12",
        "breakout": "#e74c3c",
        "skiing": "#1abc9c",
    }

    def __init__(self, loader: Optional[ResultsLoader] = None, output_dir: Optional[str] = None):
        self.loader = loader or ResultsLoader()
        self.output_dir = Path(output_dir) if output_dir else self.loader.base_path / "analysis" / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["savefig.dpi"] = 300

    def _get_experiment(self, game: str) -> ExperimentResult:
        exp = self.loader.load_experiment(game)
        if not exp:
            raise ValueError(f"No unified results found for {game}")
        return exp

    def _save(self, fig: plt.Figure, filename: str, save: bool) -> None:
        if save:
            fig.savefig(self.output_dir / filename, bbox_inches="tight")

    def _resolve_policy_path(self, policy_path: str) -> Path:
        path = Path(policy_path.replace("\\", "/"))
        if path.is_absolute():
            return path
        candidates = [
            self.loader.project_root / path,
            self.loader.unified_path / path,
            self.loader.base_path / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def plot_reward_distribution(self, game: str, save: bool = True, show: bool = False) -> plt.Figure:
        exp = self._get_experiment(game)
        rewards = np.asarray(exp.total_rewards, dtype=float)
        color = self.GAME_COLORS[game]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(rewards, bins=min(20, max(8, len(rewards) // 2)), color=color, alpha=0.8, edgecolor="white")
        ax.axvline(rewards.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean {rewards.mean():.2f}")
        ax.axvline(np.median(rewards), color="black", linestyle=":", linewidth=1.5, label=f"Median {np.median(rewards):.2f}")
        ax.set_title(f"{game.capitalize()} Unified Reward Distribution")
        ax.set_xlabel("Episode reward")
        ax.set_ylabel("Count")
        ax.legend()
        self._save(fig, f"unified_{game}_reward_distribution.png", save)
        if show:
            plt.show()
        return fig

    def plot_iteration_progression(self, game: str, save: bool = True, show: bool = False) -> plt.Figure:
        progression = self.loader.get_iteration_progression(game)
        if not progression:
            raise ValueError(f"No iteration progression found for {game}")
        color = self.GAME_COLORS[game]
        iterations = progression["iteration"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        metrics = [("avg_return", "Average return"), ("win_rate", "Win rate")]
        for ax, (key, label) in zip(axes, metrics):
            values = progression[key]
            ax.plot(iterations, values, marker="o", color=color, linewidth=2)
            best_idx = int(np.nanargmax(values))
            ax.scatter([iterations[best_idx]], [values[best_idx]], s=140, marker="*", color="gold", edgecolor="black", zorder=5)
            ax.set_title(label)
            ax.set_xlabel("Iteration")
            ax.set_xticks(iterations)
        fig.suptitle(f"{game.capitalize()} Unified Iteration Progression")
        plt.tight_layout()
        self._save(fig, f"unified_{game}_iteration_progression.png", save)
        if show:
            plt.show()
        return fig

    def plot_parameter_landscape(self, game: str, save: bool = True, show: bool = False) -> plt.Figure:
        exp = self._get_experiment(game)
        policy_path = self._resolve_policy_path(exp.best_policy_path)
        spec = importlib.util.spec_from_file_location(policy_path.stem, policy_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load policy module from {policy_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        initial_params = {k: float(v) for k, v in module.init_params().items()}
        tuned_params = exp.best_params
        names = list(tuned_params.keys())

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
        x = np.arange(len(names))
        width = 0.38
        ax.bar(x - width / 2, [initial_params.get(name, 0.0) for name in names], width, label="LLM init")
        ax.bar(x + width / 2, [tuned_params[name] for name in names], width, label="CMA-ES best")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_title(f"{game.capitalize()} Unified Parameter Landscape")
        ax.set_ylabel("Parameter value")
        ax.legend()
        plt.tight_layout()
        self._save(fig, f"unified_{game}_parameter_landscape.png", save)
        if show:
            plt.show()
        return fig

    def plot_single_game_benchmark_comparison(self, game: str, save: bool = True, show: bool = False) -> plt.Figure:
        exp = self._get_experiment(game)
        comp = get_benchmark_comparison(game, exp.best_score)
        labels = ["Random", "LeGPS", "DQN", "Human"]
        values = [comp["random"], comp["our_score"], comp.get("dqn", np.nan), comp["human"]]
        colors = ["#95a5a6", self.GAME_COLORS[game], "#34495e", "#8e44ad"]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(labels, values, color=colors)
        ax.set_title(f"{game.capitalize()} Unified Benchmark Comparison")
        ax.set_ylabel("Score")
        self._save(fig, f"unified_{game}_benchmark_comparison.png", save)
        if show:
            plt.show()
        return fig

    def plot_unified_overview(self, save: bool = True, show: bool = False) -> plt.Figure:
        experiments = self.loader.load_all_experiments()
        games = [game for game in ACTIVE_GAMES if game in experiments]
        scores = [experiments[game].best_score for game in games]
        colors = [self.GAME_COLORS[game] for game in games]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([g.capitalize() for g in games], scores, color=colors)
        ax.set_title("Unified LeGPS Scores Across Current Games")
        ax.set_ylabel("Best score")
        self._save(fig, "unified_overview_scores.png", save)
        if show:
            plt.show()
        return fig

    def plot_human_normalized_overview(self, save: bool = True, show: bool = False) -> plt.Figure:
        experiments = self.loader.load_all_experiments()
        games = [game for game in ACTIVE_GAMES if game in experiments]
        normalized_scores = [
            compute_human_normalized_score(game, experiments[game].best_score)
            for game in games
        ]
        colors = [self.GAME_COLORS[game] for game in games]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([g.capitalize() for g in games], normalized_scores, color=colors)
        ax.axhline(0.0, color="black", linewidth=1, linestyle=":")
        ax.axhline(100.0, color="#8e44ad", linewidth=1.5, linestyle="--", label="Human = 100")
        ax.set_title("Unified LeGPS Human-Normalized Scores")
        ax.set_ylabel("Human-normalized score (%)")
        ax.legend()
        self._save(fig, "unified_human_normalized_scores.png", save)
        if show:
            plt.show()
        return fig

    def generate_curated_figures(self, show: bool = False, include_benchmarks: bool = False) -> None:
        self.plot_unified_overview(show=show)
        if include_benchmarks:
            self.plot_human_normalized_overview(show=show)
        for game in self.loader.discover_experiments():
            exp = self._get_experiment(game)
            if exp.total_rewards:
                self.plot_reward_distribution(game, show=show)
            if exp.iterations:
                self.plot_iteration_progression(game, show=show)
            self.plot_parameter_landscape(game, show=show)
            if include_benchmarks:
                self.plot_single_game_benchmark_comparison(game, show=show)


# Backward-compatible export name inside the local package.
AblationVisualizer = UnifiedVisualizer

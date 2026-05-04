#!/usr/bin/env python
"""Generate RQ4 optimizer-ablation figures.

RQ4 asks how much performance comes from the numeric optimizer. This script
compares canonical full LeGPS results against a suite run with optimizer=none.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .benchmarks import BENCHMARK_SCORES, compute_human_normalized_score
from .results_loader import ACTIVE_GAMES, ResultsLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CANONICAL_DIR = PROJECT_ROOT / "scripts" / "llm_optimization" / "runs" / "best_10000_steps"
DEFAULT_LLM_ONLY_DIR = (
    PROJECT_ROOT
    / "scripts"
    / "llm_optimization"
    / "runs"
    / "unified_suite"
    / "rq4_llm_only_20260427"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "scripts" / "llm_optimization" / "analysis" / "figures"
DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "llm_optimization"
    / "analysis"
    / "evaluations"
    / "rq4_optimizer_ablation_summary.json"
)

GAME_COLORS = {
    "pong": "#2ecc71",
    "freeway": "#3498db",
    "asterix": "#f39c12",
    "breakout": "#e74c3c",
    "skiing": "#1abc9c",
}


def repo_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def load_scores(results_dir: Path) -> dict[str, float]:
    loader = ResultsLoader(unified_path=str(results_dir))
    scores: dict[str, float] = {}
    for game, result in loader.load_all_experiments().items():
        scores[game] = float(result.best_score)
    return scores


def build_summary(
    *,
    canonical_dir: Path,
    llm_only_dir: Path,
    games: list[str],
) -> dict[str, Any]:
    canonical_scores = load_scores(canonical_dir)
    llm_only_scores = load_scores(llm_only_dir)

    rows: list[dict[str, Any]] = []
    for game in games:
        if game not in canonical_scores or game not in llm_only_scores:
            continue

        full_score = canonical_scores[game]
        llm_score = llm_only_scores[game]
        full_hns = compute_human_normalized_score(game, full_score)
        llm_hns = compute_human_normalized_score(game, llm_score)
        dqn_score = BENCHMARK_SCORES.get(game, {}).get("dqn")

        rows.append(
            {
                "game": game,
                "full_legps_score": full_score,
                "llm_only_score": llm_score,
                "score_delta_full_minus_llm_only": full_score - llm_score,
                "full_legps_human_normalized": full_hns,
                "llm_only_human_normalized": llm_hns,
                "human_normalized_delta_full_minus_llm_only": None
                if full_hns is None or llm_hns is None
                else full_hns - llm_hns,
                "dqn_score": dqn_score,
            }
        )

    return {
        "canonical_dir": repo_relative_path(canonical_dir),
        "llm_only_dir": repo_relative_path(llm_only_dir),
        "interpretation": (
            "Full LeGPS uses the LLM to generate policy code and CMA-ES to tune numeric "
            "parameters. LLM-only uses the same unified outer loop but optimizer=none, "
            "so init_params() values are evaluated directly."
        ),
        "games": rows,
    }


def plot_raw_scores(summary: dict[str, Any], output_dir: Path) -> None:
    rows = summary["games"]
    games = [row["game"] for row in rows]
    x = np.arange(len(games))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    full = [row["full_legps_score"] for row in rows]
    llm = [row["llm_only_score"] for row in rows]
    colors = [GAME_COLORS.get(game, "#7f8c8d") for game in games]

    ax.bar(x - width / 2, full, width, label="LeGPS + CMA-ES", color=colors)
    ax.bar(x + width / 2, llm, width, label="LLM-only constants", color="#95a5a6")
    ax.set_xticks(x)
    ax.set_xticklabels([game.capitalize() for game in games])
    ax.set_ylabel("Best score")
    ax.set_title("RQ4: Effect of CMA-ES on Raw Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "rq4_optimizer_ablation_raw_scores.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_human_normalized(summary: dict[str, Any], output_dir: Path) -> None:
    rows = summary["games"]
    games = [row["game"] for row in rows]
    x = np.arange(len(games))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    full = [row["full_legps_human_normalized"] for row in rows]
    llm = [row["llm_only_human_normalized"] for row in rows]
    colors = [GAME_COLORS.get(game, "#7f8c8d") for game in games]

    ax.bar(x - width / 2, full, width, label="LeGPS + CMA-ES", color=colors)
    ax.bar(x + width / 2, llm, width, label="LLM-only constants", color="#95a5a6")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
    ax.axhline(100.0, color="#8e44ad", linestyle="--", linewidth=1.5, label="Human = 100")
    ax.set_xticks(x)
    ax.set_xticklabels([game.capitalize() for game in games])
    ax.set_ylabel("Human-normalized score (%)")
    ax.set_title("RQ4: Effect of CMA-ES on Human-Normalized Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "rq4_optimizer_ablation_human_normalized.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_hns_delta(summary: dict[str, Any], output_dir: Path) -> None:
    rows = summary["games"]
    games = [row["game"] for row in rows]
    deltas = [row["human_normalized_delta_full_minus_llm_only"] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [GAME_COLORS.get(game, "#7f8c8d") for game in games]
    ax.bar([game.capitalize() for game in games], deltas, color=colors)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
    ax.set_ylabel("HNS gain from CMA-ES (percentage points)")
    ax.set_title("RQ4: Human-Normalized Gain from Numeric Optimization")
    fig.tight_layout()
    fig.savefig(output_dir / "rq4_optimizer_ablation_hns_gain.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RQ4 optimizer-ablation figures")
    parser.add_argument("--canonical-dir", type=Path, default=DEFAULT_CANONICAL_DIR)
    parser.add_argument("--llm-only-dir", type=Path, default=DEFAULT_LLM_ONLY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--games", nargs="+", default=ACTIVE_GAMES, choices=ACTIVE_GAMES)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    summary = build_summary(
        canonical_dir=args.canonical_dir.resolve(),
        llm_only_dir=args.llm_only_dir.resolve(),
        games=args.games,
    )
    args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_raw_scores(summary, args.output_dir)
    plot_human_normalized(summary, args.output_dir)
    plot_hns_delta(summary, args.output_dir)

    print(f"Wrote summary: {args.summary_output}")
    print(f"Wrote figures to: {args.output_dir}")


if __name__ == "__main__":
    main()

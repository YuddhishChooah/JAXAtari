#!/usr/bin/env python
"""
Generate figures for the active unified LeGPS experiments.
"""

from __future__ import annotations

import argparse
import sys

from .results_loader import ResultsLoader
from .visualizer import UnifiedVisualizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unified analysis figures for LeGPS")
    parser.add_argument("--game", type=str, default=None, help="Specific game to analyze")
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom figure output directory")
    args = parser.parse_args()

    loader = ResultsLoader()
    available = loader.discover_experiments()
    if not available:
        print("No unified experiments found.")
        print(f"Looking in: {loader.unified_path}")
        sys.exit(1)

    viz = UnifiedVisualizer(loader=loader, output_dir=args.output_dir)
    if args.game:
        viz.plot_reward_distribution(args.game, show=args.show)
        viz.plot_iteration_progression(args.game, show=args.show)
        viz.plot_parameter_landscape(args.game, show=args.show)
        viz.plot_single_game_benchmark_comparison(args.game, show=args.show)
    else:
        viz.generate_curated_figures(show=args.show)


if __name__ == "__main__":
    main()

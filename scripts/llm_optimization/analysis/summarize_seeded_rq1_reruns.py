#!/usr/bin/env python
"""Summarize seeded RQ1 LeGPS suite reruns.

RQ1 needs more than single best-policy scores. This script turns one or more
unified-suite run directories into per-game stability evidence: completion
counts, score distributions, deltas from canonical policies, and
human-normalized summaries when benchmark scores are available.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.llm_optimization.analysis.benchmarks import compute_human_normalized_score


ACTIVE_GAMES = ["pong", "freeway", "asterix", "breakout", "skiing"]
DEFAULT_SUITE_ROOT = PROJECT_ROOT / "scripts" / "llm_optimization" / "runs" / "unified_suite"
DEFAULT_CANONICAL_MANIFEST = (
    PROJECT_ROOT / "scripts" / "llm_optimization" / "runs" / "best_10000_steps" / "manifest.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "scripts"
    / "llm_optimization"
    / "analysis"
    / "evaluations"
    / "rq1_seeded_reruns_summary.json"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def resolve_repo_path(path_text: str | Path) -> Path:
    path = Path(str(path_text).replace("\\", "/"))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def discover_suite_dirs(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.glob(pattern) if (path / "manifest.json").exists())


def load_canonical_scores(manifest_path: Path, games: list[str]) -> dict[str, float | None]:
    manifest = read_json(manifest_path)
    canonical: dict[str, float | None] = {}
    for game in games:
        entry = manifest.get("games", {}).get(game, {})
        canonical[game] = finite_float(entry.get("avg_return_10000"))
    return canonical


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "sample_std": None,
            "min": None,
            "max": None,
            "median": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) >= 2 else 0.0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }


def result_row(
    *,
    suite: dict[str, Any],
    suite_dir: Path,
    game: str,
    result: dict[str, Any] | None,
    canonical_score: float | None,
) -> dict[str, Any]:
    status = "missing" if result is None else str(result.get("status", "unknown"))
    score = finite_float(result.get("best_score")) if result is not None else None
    return {
        "run_id": suite.get("run_id", suite_dir.name),
        "suite_dir": str(suite_dir),
        "created_at": suite.get("created_at"),
        "game": game,
        "status": status,
        "returncode": None if result is None else result.get("returncode"),
        "seed": None if result is None else result.get("seed"),
        "best_score": score,
        "canonical_score": canonical_score,
        "delta_vs_canonical": None if score is None or canonical_score is None else score - canonical_score,
        "elapsed_seconds": None if result is None else result.get("elapsed_seconds"),
        "output_dir": None if result is None else result.get("output_dir"),
        "best_policy_path": None if result is None else result.get("best_policy_path"),
    }


def summarize_game(rows: list[dict[str, Any]], canonical_score: float | None) -> dict[str, Any]:
    attempted_rows = [row for row in rows if row["status"] != "missing"]
    completed_rows = [
        row for row in attempted_rows if row["status"] == "completed" and row["best_score"] is not None
    ]
    scores = [float(row["best_score"]) for row in completed_rows]
    deltas = [
        float(row["delta_vs_canonical"])
        for row in completed_rows
        if row["delta_vs_canonical"] is not None
    ]
    hns_values = [
        compute_human_normalized_score(str(row["game"]), float(row["best_score"]))
        for row in completed_rows
    ]
    hns_scores = [float(value) for value in hns_values if value is not None]

    best_score = max(scores) if scores else None
    return {
        "game": rows[0]["game"] if rows else None,
        "canonical_score": canonical_score,
        "canonical_human_normalized": None
        if canonical_score is None
        else compute_human_normalized_score(str(rows[0]["game"]), canonical_score),
        "attempted_runs": len(attempted_rows),
        "completed_runs": len(completed_rows),
        "failed_runs": len([row for row in attempted_rows if row["status"] != "completed"]),
        "missing_runs": len([row for row in rows if row["status"] == "missing"]),
        "score_summary": numeric_summary(scores),
        "delta_vs_canonical_summary": numeric_summary(deltas),
        "human_normalized_summary": numeric_summary(hns_scores),
        "best_score": best_score,
        "best_delta_vs_canonical": None
        if best_score is None or canonical_score is None
        else best_score - canonical_score,
        "completed_seeds": [row["seed"] for row in completed_rows],
        "failed_or_missing_runs": [
            {
                "run_id": row["run_id"],
                "seed": row["seed"],
                "status": row["status"],
                "returncode": row["returncode"],
            }
            for row in rows
            if row["status"] != "completed" or row["best_score"] is None
        ],
    }


def build_summary(
    *,
    suite_dirs: list[Path],
    canonical_manifest: Path,
    games: list[str],
) -> dict[str, Any]:
    canonical_scores = load_canonical_scores(canonical_manifest, games)
    suites: list[dict[str, Any]] = []
    rows_by_game = {game: [] for game in games}

    for suite_dir in suite_dirs:
        manifest_path = suite_dir / "manifest.json"
        manifest = read_json(manifest_path)
        suite_entry = {
            "run_id": manifest.get("run_id", suite_dir.name),
            "suite_dir": str(suite_dir),
            "manifest_path": str(manifest_path),
            "created_at": manifest.get("created_at"),
            "protocol": manifest.get("protocol", {}),
            "games": {},
        }
        results = manifest.get("results", {})

        for game in games:
            row = result_row(
                suite=manifest,
                suite_dir=suite_dir,
                game=game,
                result=results.get(game),
                canonical_score=canonical_scores.get(game),
            )
            rows_by_game[game].append(row)
            suite_entry["games"][game] = row

        suites.append(suite_entry)

    per_game = [
        summarize_game(rows_by_game[game], canonical_scores.get(game))
        for game in games
    ]

    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "RQ1 seeded-rerun stability summary for LeGPS unified-suite outputs.",
        "canonical_manifest": str(canonical_manifest),
        "suite_dirs": [str(path) for path in suite_dirs],
        "games": games,
        "interpretation_note": (
            "Scores are best_score values reported by each unified-suite manifest. "
            "Deltas compare those rerun scores against canonical 10000-step saved-policy scores."
        ),
        "per_game": per_game,
        "suites": suites,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize seeded RQ1 LeGPS rerun suites")
    parser.add_argument(
        "--suite-dirs",
        nargs="*",
        default=None,
        help="Explicit unified-suite run directories to summarize. Defaults to discovering --pattern under --suite-root.",
    )
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE_ROOT)
    parser.add_argument("--pattern", type=str, default="rq1_seeded_*")
    parser.add_argument("--canonical-manifest", type=Path, default=DEFAULT_CANONICAL_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--games", nargs="+", default=ACTIVE_GAMES, choices=ACTIVE_GAMES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.suite_dirs:
        suite_dirs = [resolve_repo_path(path) for path in args.suite_dirs]
    else:
        suite_dirs = discover_suite_dirs(resolve_repo_path(args.suite_root), args.pattern)

    if not suite_dirs:
        raise SystemExit(
            "No suite manifests found. Pass --suite-dirs explicitly or adjust --suite-root/--pattern."
        )

    missing = [path for path in suite_dirs if not (path / "manifest.json").exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise SystemExit(f"Missing manifest.json in suite dirs: {missing_text}")

    summary = build_summary(
        suite_dirs=suite_dirs,
        canonical_manifest=resolve_repo_path(args.canonical_manifest),
        games=list(args.games),
    )
    output_path = resolve_repo_path(args.output)
    write_json(output_path, summary)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

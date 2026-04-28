"""
Reset generated LLM-optimization artifacts for a fresh run.

Deletes only derived outputs and caches. Source code and neutral docs stay intact.
"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent

DIRS_TO_REMOVE = [
    ROOT / "outputs",
    ROOT / "analysis" / "figures",
    ROOT / "runs" / "single_game",
    PROJECT_ROOT / ".pytest_cache",
]

GLOB_DIRS_TO_REMOVE = [
    ROOT.glob("experiments/*/outputs"),
    ROOT.rglob("__pycache__"),
]


def remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        print(f"removed: {path}")


def main() -> None:
    for path in DIRS_TO_REMOVE:
        remove_dir(path)

    for pattern in GLOB_DIRS_TO_REMOVE:
        for path in pattern:
            remove_dir(path)


if __name__ == "__main__":
    main()

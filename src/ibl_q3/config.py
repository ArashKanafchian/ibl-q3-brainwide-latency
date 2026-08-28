"""Portable paths and execution-mode configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ExecutionMode(str, Enum):
    QUICK = "quick"
    FULL = "full"


def find_repo_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not find pyproject.toml above the working directory")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_root: Path
    artifact_root: Path

    def namespace(self, question: str) -> Path:
        path = self.artifact_root / question.lower()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure(self) -> "ProjectPaths":
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(start: str | Path | None = None) -> ProjectPaths:
    root = Path(os.environ.get("OCTAGRAM_PROJECT_ROOT", find_repo_root(start))).resolve()
    data_root = Path(os.environ.get("OCTAGRAM_DATA_ROOT", root / "data")).resolve()
    artifact_root = Path(
        os.environ.get("OCTAGRAM_ARTIFACT_ROOT", root / "results")
    ).resolve()
    return ProjectPaths(root, data_root, artifact_root).ensure()


def get_execution_mode() -> ExecutionMode:
    raw = os.environ.get("OCTAGRAM_EXECUTION_MODE", ExecutionMode.QUICK.value).lower()
    try:
        return ExecutionMode(raw)
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in ExecutionMode)
        raise ValueError(f"OCTAGRAM_EXECUTION_MODE must be one of: {allowed}") from exc

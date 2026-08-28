"""Small artifact store for local, mounted, or CI-backed results."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ArtifactStore:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_frame(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.path(name)
        frame.to_csv(path, index=False)
        return path

    def load_frame(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.path(name))

    def save_pickle(self, name: str, value: Any) -> Path:
        path = self.path(name)
        with path.open("wb") as stream:
            pickle.dump(value, stream)
        return path

    def load_pickle(self, name: str) -> Any:
        with self.path(name).open("rb") as stream:
            return pickle.load(stream)

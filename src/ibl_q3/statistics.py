"""Statistical utilities used by Q3."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return Benjamini–Hochberg FDR-adjusted q-values, preserving NaNs."""
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return result
    observed = values[finite]
    order = np.argsort(observed)
    ranked = observed[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    result[finite] = restored
    return result

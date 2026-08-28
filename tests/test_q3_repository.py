from pathlib import Path

import nbformat
import numpy as np

from ibl_q3.statistics import benjamini_hochberg


ROOT = Path(__file__).resolve().parents[1]


def test_benjamini_hochberg_is_monotone_in_rank():
    p_values = np.array([0.01, 0.04, 0.03, np.nan])
    q_values = benjamini_hochberg(p_values)
    assert np.isnan(q_values[-1])
    order = np.argsort(p_values[:-1])
    assert np.all(np.diff(q_values[:-1][order]) >= 0)


def test_single_notebook_is_clean_and_owned_by_arash():
    paths = list((ROOT / "notebooks").glob("*.ipynb"))
    assert [path.name for path in paths] == ["01_q3_arash_brainwide_recruitment.ipynb"]
    notebook = nbformat.read(paths[0], as_version=4)
    assert notebook.metadata["question"] == "Q3"
    assert notebook.metadata["contributors"] == ["Arash Kanafchian"]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []

# IBL Q3 — Brain-Wide Visual Response Latency

An end-to-end, reproducible analysis of visual-response latency and the temporal
recruitment of brain regions in the International Brain Laboratory (IBL)
Brain-Wide Map dataset.

**Author:** Arash Kanafchian

## Research question

For responsive units, compute response latency after stimulus onset, group the
regions anatomically, and assess whether the ordering is consistent with sensory
signal propagation through the brain.

The repository reports two complementary quantities:

- **Population latency:** first 70% crossing of the left-versus-right population
  trajectory distance, aligned with the reference paper.
- **Unit latency:** first sustained 70% crossing of each unit's absolute response
  relative to its pre-stimulus baseline.

## Repository structure

```text
notebooks/                 # final Q3 notebook
results/q3/figures/        # publication-ready figures
results/q3/tables/         # final result tables
src/ibl_q3/config.py       # portable data/output paths
src/ibl_q3/artifacts.py    # mounted/local checkpoint handling
src/ibl_q3/statistics.py   # FDR utilities
src/ibl_q3/plotting.py     # publication plotting code
docs/                      # report and code explanations
references/                # literature and provenance notes
tests/                     # fast tests that require no data download
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the tests:

```bash
pytest
```

Start the notebook:

```bash
jupyter lab notebooks/01_q3_arash_brainwide_recruitment.ipynb
```

The notebook downloads the public IBL/Neuromatch aligned dataset when it is not
already present. To reuse a local or mounted dataset and output directory:

```bash
export OCTAGRAM_DATA_ROOT=/path/to/ibl-data
export OCTAGRAM_ARTIFACT_ROOT=/path/to/persistent-results
```

## Main result

The estimated sequence progresses from visual thalamus, through primary and
higher visual cortex, to later midbrain/hindbrain and association/action systems.
All committed notebook outputs are cleared; the final numerical tables and
publication figures are retained under `results/q3/`.

## Provenance

The Q3-specific analysis, validation, reporting, and plotting are attributed to
Arash Kanafchian. Data access and tutorial foundations are credited to the
International Brain Laboratory and Neuromatch. See `references/README.md` for the
literature map.

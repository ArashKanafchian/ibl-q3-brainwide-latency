# Q3 Paper-Aligned Analysis — Exact Code Walkthrough

> This document explains **only notebook cell 163**, the paper-aligned population-trajectory validation. It does not explain the primary unit-wise analysis in cells 158–162.

## Open this notebook and this cell

Open `IBL_BWM_Neuromatch_tutorial_raw_Final.ipynb` and find the cell titled:

`FINAL Q3 — 6. Paper-aligned population-trajectory validation`

That is notebook cell **163**. Read this document beside that cell from top to bottom. The order below follows the executable order of that cell.

The notebook is the executable source of truth. This document expands the logic of the code; it does not replace the notebook.

## What this cell is actually doing

This cell is not training a machine-learning model. It has no learned weights, training set, test set, or prediction accuracy.

It performs a controlled population-level statistical test:

1. Keep all eligible units in predefined anatomical regions.
2. Compute the real left-versus-right population difference through time.
3. Create fake left/right labels while preserving prior block, choice, and label counts.
4. Compute the same population difference for every fake labeling.
5. Ask whether the real trajectory changes more than the fake trajectories.
6. Correct the regional probability values.
7. Compute latency only for regions whose trajectory is statistically credible.

The scientific question is:

> Does a region's recorded population show a time-varying stimulus-side separation stronger than expected after stimulus labels are disconnected from neural activity while prior block and choice are preserved?

---

# Part 1 — Settings

## Lines 9–12

```python
Q3_PAPER_SHUFFLES = 1000
Q3_PAPER_MIN_UNITS = 20
Q3_PAPER_FDR_ALPHA = 0.01
Q3_PAPER_SEED = 20260820
```

`Q3_PAPER_SHUFFLES` is the number of fake labelings per insertion. With the later plus-one correction, 1,000 fake experiments give a smallest possible probability of \(1/1001\approx0.001\).

`Q3_PAPER_MIN_UNITS` requires at least 20 accumulated units in a region. This is a coverage rule, not a test of individual responsiveness.

`Q3_PAPER_FDR_ALPHA` is the corrected regional significance threshold. `Q3_PAPER_SEED` makes the random label generation reproducible; it does not make the result biologically true.

---

# Part 2 — Fake labels within controlled strata

## Line 15 — Define the label generator

```python
def q3_paper_shuffled_labels(labels, strata, n_shuffles, rng):
```

This defines a function. It does not execute yet.

| Input | Meaning | Shape |
|---|---|---|
| `labels` | True left/right assignment for each usable trial | trials |
| `strata` | Block × choice group for each trial | trials |
| `n_shuffles` | Number of fake experiments | one number |
| `rng` | Reproducible random generator | one object |

The returned matrix will have shape `fake experiments × trials`. One means fake-left and zero means fake-right.

## Line 18 — Allocate the fake-label matrix

```python
pseudo = np.zeros((n_shuffles, len(labels)), dtype=np.float32)
```

`np.zeros` creates a matrix filled with zero. Its first dimension is one row per fake experiment; its second is one column per retained real trial. Zero initially means fake-right.

`float32` reduces memory use and works conveniently in later matrix multiplication. It does not change the conceptual binary label.

## Line 19 — Create row indices

```python
rows = np.arange(n_shuffles)[:, None]
```

`np.arange(n_shuffles)` creates `0, 1, ..., n_shuffles - 1`. `[:, None]` changes its shape from `(shuffles,)` to `(shuffles, 1)`.

That extra dimension allows one later assignment to address every fake-experiment row while selecting several trial columns in each row. This is broadcasting.

## Line 21 — Visit each stratum

```python
for stratum in np.unique(strata):
```

`np.unique` returns each distinct stratum code once. The loop is essential: labels are never exchanged between different prior-block/choice states.

## Line 22 — Find positions in this stratum

```python
trial_index = np.flatnonzero(strata == stratum)
```

The comparison creates a Boolean vector. `np.flatnonzero` returns the integer positions where it is true. “Flat” means it returns one simple index array; here the input is already one-dimensional.

## Line 23 — Count left labels

```python
n_left = int(labels[trial_index].sum())
```

The current stratum's labels are selected. Boolean true values count as one, so `.sum()` gives the number of observed left trials. This exact count will be preserved in every fake experiment.

## Lines 25–26 — Handle an all-right stratum

```python
if n_left == 0:
    continue
```

The matrix already contains zeros, so an all-right stratum needs no assignment. No legal relabeling can change it while preserving its side count.

## Lines 27–29 — Handle an all-left stratum

```python
if n_left == len(trial_index):
    pseudo[:, trial_index] = 1
    continue
```

All trials in this stratum are assigned fake-left in every row. The first colon selects every fake experiment and `trial_index` selects the current stratum's trial columns.

## Line 31 — Generate random scores

```python
scores = rng.random((n_shuffles, len(trial_index)))
```

One random number is assigned to every fake experiment and every trial in this stratum. The scores are temporary rankings, not neural values.

## Lines 32–34 — Select exactly the required count

```python
selected = np.argpartition(
    scores, n_left - 1, axis=1
)[:, :n_left]
```

`np.argpartition` places the smallest `n_left` scores at the beginning of each row without fully sorting all scores. `n_left - 1` is the zero-based boundary. `axis=1` performs the operation independently across trial columns for each fake experiment.

`[:, :n_left]` keeps the selected positions. The result has shape `fake experiments × n_left`.

Because the scores are independent and continuous, every subset of the required size is equally likely. The function changes the label assignment while preserving the count.

## Line 35 — Write fake-left labels

```python
pseudo[rows, trial_index[selected]] = 1
```

`selected` contains positions local to the current stratum. `trial_index[selected]` converts them back to positions in the full retained-trial sequence. `rows` identifies fake-experiment rows and broadcasts across the selected columns.

Each row receives exactly `n_left` ones in this stratum; all other positions remain zero.

## Line 37 — Return fake labels

```python
return pseudo
```

The function returns only alternative labels. It never generates or modifies neural activity.

---

# Part 3 — Population latency helper

## Line 40 — Define the latency function

```python
def q3_paper_latency(times, distance, fraction=0.70):
```

This receives a time axis and one regional population-distance curve. It returns the first threshold crossing, interpolated between samples when possible.

## Line 43 — Define the amplitude-relative threshold

```python
threshold = distance.min() + fraction * np.ptp(distance)
```

`distance.min()` is the curve minimum. `np.ptp(distance)` is maximum minus minimum. Therefore:

\[
T=D_{\min}+0.70(D_{\max}-D_{\min}).
\]

This is 70% of the curve's full rise. It is not always `0.70 * maximum`; those are equal only when the minimum is zero.

## Line 44 — Find sampled crossings

```python
crossing = np.flatnonzero(distance >= threshold)
```

The comparison produces a Boolean value per time sample. `flatnonzero` returns the positions that meet or exceed the threshold, in increasing order.

## Lines 45–46 — Handle no crossing

```python
if len(crossing) == 0:
    return np.nan
```

If no sample crosses, latency remains missing rather than being invented.

## Line 48 — Select the first crossing

```python
index = int(crossing[0])
```

The first position is the earliest sampled crossing.

## Lines 49–50 — Handle non-interpolatable cases

```python
if index == 0 or distance[index] == distance[index - 1]:
    return times[index] * 1000
```

If the crossing is at the first sample, there is no previous sample. If neighboring distances are equal, interpolation would divide by zero. The sampled time is returned directly and converted from seconds to milliseconds.

## Lines 52–54 — Calculate fractional progress

```python
weight = (threshold - distance[index - 1]) / (
    distance[index] - distance[index - 1]
)
```

The numerator is the distance still needed to reach the threshold. The denominator is the total increase between neighboring samples. Their ratio is the fraction of the time interval at which a straight-line curve would cross.

## Lines 55–57 — Interpolate time

```python
return (
    times[index - 1] + weight * (times[index] - times[index - 1])
) * 1000
```

The crossing is estimated between the neighboring time samples and converted to milliseconds. A value such as 49.7 ms is an interpolation on a 10-ms grid, not true 0.1-ms temporal resolution.

---

# Part 4 — Region and time setup

## Lines 60–64 — Reverse anatomical mapping

```python
q3_paper_region_to_group = {
    region: group
    for group, regions in q3_anatomical_regions.items()
    for region in regions
}
```

The source mapping stores one group with a list of regions. This comprehension reverses it, so a region acronym can directly return its broad anatomical group.

The dictionary keys also define the regional testing scope.

## Lines 65–68 — Select the population time window

```python
q3_paper_window = (
    (q3_meta.times >= 0) & (q3_meta.times <= 0.15)
)
q3_paper_times = q3_meta.times[q3_paper_window]
```

The Boolean mask selects 0 through 150 ms. Applying the same mask to the shared time axis creates the physical time values used later for latency.

## Line 69 — Create reproducible random state

```python
q3_paper_rng = np.random.default_rng(Q3_PAPER_SEED)
```

All controlled fake-label draws use this generator. Reusing the seed reproduces the same sequence of fake experiments.

---

# Part 5 — Streaming accumulators

## Lines 74–77 — True squared sums

```python
q3_true_sumsq = {
    region: np.zeros(q3_paper_window.sum())
    for region in q3_paper_region_to_group
}
```

Each target region receives a zero vector with one value per selected time sample. It will store the accumulated sum of squared true unit differences.

## Lines 78–81 — Null squared sums

```python
q3_null_sumsq = {
    region: np.zeros((Q3_PAPER_SHUFFLES, q3_paper_window.sum()))
    for region in q3_paper_region_to_group
}
```

Each region receives a matrix with shape `fake experiments × time`. Row `b` will accumulate the squared contributions for fake experiment `b` across insertions.

## Line 82 — Initialize unit counts

```python
q3_paper_n_units = dict.fromkeys(q3_paper_region_to_group, 0)
```

Each target region starts with zero accumulated units. The count later supplies the RMS normalization and coverage check.

## Lines 83–85 — Initialize PID sets

```python
q3_paper_pids = {
    region: set() for region in q3_paper_region_to_group
}
```

Each region receives its own set of insertion identifiers. A set prevents multiple units from one PID being counted as multiple insertions.

## Lines 87–89 — Find target insertions

```python
q3_paper_target_pids = q3_meta.clusters.loc[
    q3_meta.clusters['acronym'].isin(q3_paper_region_to_group), 'pid'
].dropna().unique()
```

Clusters in the predefined pathway regions are identified. Their PIDs are selected, missing PIDs are removed, and duplicate PIDs are collapsed. Only insertions that can contribute to the paper-aligned pathway are processed.

---

# Part 6 — Process one insertion

## Lines 91–93 — Start the insertion loop

```python
for pid in tqdm.tqdm(
    q3_paper_target_pids, desc='Paper-aligned Q3 trajectory'
):
```

The loop processes one target insertion at a time. `tqdm` only displays progress and does not change the statistic.

## Lines 94–96 — Load the current insertion

```python
psth_pid, clusters_pid, trials_pid = get_psth_for_insertion(
    pid, q3_meta
)
```

The helper returns aligned activity, cluster metadata, and trial metadata. This cell uses all eligible target units; it does not use the responsive-unit selection from cells 158–162.

## Lines 97–99 — Mark target units

```python
target_units = clusters_pid['acronym'].isin(
    q3_paper_region_to_group
).to_numpy()
```

`.isin` creates one Boolean per cluster. True means the acronym belongs to the paper-aligned region mapping. This mask will be applied to the unit axis of the activity tensor.

## Lines 100–101 — Skip an empty insertion

```python
if not target_units.any():
    continue
```

`.any()` checks whether at least one target unit exists. This is a defensive guard because target PIDs were already selected from target clusters.

## Lines 103–106 — Extract trial variables

```python
left_contrast = trials_pid['contrastLeft'].to_numpy(dtype=float)
right_contrast = trials_pid['contrastRight'].to_numpy(dtype=float)
choice = trials_pid['choice'].to_numpy(dtype=float)
block = trials_pid['probabilityLeft'].to_numpy(dtype=float)
```

Each array has one entry per trial. Contrast magnitude is not thresholded here. A finite zero still counts as an assigned side, matching the paper-oriented trial definition. Choice and block define the conditional label-exchange groups.

## Lines 110–111 — Define assigned-side labels

```python
left_label = np.isfinite(left_contrast)
right_label = np.isfinite(right_contrast)
```

Finite values, including zero, become true. Missing values become false. The side is determined by which contrast column is assigned and finite.

## Lines 112–116 — Keep usable trials

```python
keep = (
    (left_label | right_label) &
    np.isfinite(choice) &
    np.isfinite(block)
)
```

The trial must have an assigned side, a finite choice, and a finite prior-block value. The same mask is later applied to activity and labels so their rows remain aligned. `|` is elementwise OR and `&` is elementwise AND.

## Line 117 — Retain true labels

```python
labels = left_label[keep]
```

The retained Boolean labels contain true for assigned-left trials and false for assigned-right trials. Because `keep` requires an assigned side, false means right rather than missing.

## Lines 118–119 — Require both sides

```python
if not labels.any() or labels.all():
    continue
```

`not labels.any()` means no left trials. `labels.all()` means every retained trial is left, so no right trials exist. Either case makes a left-versus-right average impossible.

---

# Part 7 — Build strata and fake experiments

## Lines 122–125 — Encode block × choice groups

```python
strata = pd.factorize(pd.MultiIndex.from_arrays([
    block[keep] == 0.8,
    choice[keep] == 1,
]))[0]
```

The two Boolean arrays describe prior-block and choice conditions. `MultiIndex.from_arrays` pairs them trial by trial. `factorize` replaces each distinct pair with an integer code; `[0]` keeps only the codes.

There can be four combinations: block not 0.8/choice not 1, block not 0.8/choice 1, block 0.8/choice not 1, and block 0.8/choice 1. The integer codes have no meaning; equality of codes is what matters.

## Lines 126–128 — Generate controlled fake labels

```python
pseudo = q3_paper_shuffled_labels(
    labels, strata, Q3_PAPER_SHUFFLES, q3_paper_rng
)
```

The output shape is `1000 fake experiments × retained trials`. Each row preserves left/right counts inside every block × choice stratum while breaking the original trial-specific association between side and activity.

## Lines 129–130 — Count fake groups

```python
pseudo_left_n = pseudo.sum(axis=1)
pseudo_right_n = len(labels) - pseudo_left_n
```

Summing over trial columns gives one fake-left count per experiment. Subtracting from the total gives the fake-right count. These arrays later become row-wise divisors after `[:, None]` adds a singleton dimension.

## Lines 131–132 — Protect against empty fake groups

```python
if np.any(pseudo_left_n == 0) or np.any(pseudo_right_n == 0):
    continue
```

`np.any` checks whether any fake experiment has zero trials on one side. Such an average would divide by zero, so the insertion is skipped defensively.

---

# Part 8 — Select activity and align metadata

## Line 134 — Select trials, units, and time

```python
values = psth_pid[keep][:, target_units, :][:, :, q3_paper_window]
```

The operation selects usable trials, target units, and 0–150 ms. The resulting shape is `retained trials × target units × selected time`.

The same masks are used for metadata and activity, keeping all axes aligned.

## Line 135 — Use compact floating-point storage

```python
values = values.astype(np.float32)
```

This reduces memory during repeated fake-label calculations. It does not smooth or otherwise redefine the neural response.

## Line 136 — Align filtered metadata

```python
target_clusters = clusters_pid.loc[target_units].reset_index(drop=True)
```

The same unit mask is applied to metadata. Resetting the index makes metadata row positions match the unit positions in `values`.

---

# Part 9 — Compute true and fake regional trajectories

## Lines 138–140 — Select one region

```python
for region, region_units in target_clusters.groupby('acronym'):
    region_index = region_units.index.to_numpy()
    region_values = values[:, region_index, :]
```

The metadata is divided by region. The reset indices identify the corresponding activity columns. `region_values` has shape `trials × regional units × time`.

## Lines 142–146 — Compute and accumulate the true difference

```python
true_difference = (
    region_values[labels].mean(axis=0) -
    region_values[~labels].mean(axis=0)
)
q3_true_sumsq[region] += np.square(true_difference).sum(axis=0)
```

True-left and true-right trial averages are subtracted. The result has shape `regional units × time`. Squaring prevents opposite unit preferences from cancelling; summing over units leaves one value per time sample. `+=` streams the current insertion into the regional accumulator.

## Line 148 — Flatten unit and time

```python
flat_values = region_values.reshape(len(labels), -1)
```

The trial dimension remains first. `-1` infers `regional units × time`, turning for example `200 × 5 × 16` into `200 × 80`. This is only a temporary form for matrix multiplication.

## Lines 149–154 — Calculate fake left/right averages

```python
pseudo_left = (
    pseudo @ flat_values
) / pseudo_left_n[:, None]
pseudo_right = (
    (1 - pseudo) @ flat_values
) / pseudo_right_n[:, None]
```

The multiplication has shapes `shuffles × trials` by `trials × (units × time)`, producing `shuffles × (units × time)`. Ones in `pseudo` select fake-left trials; ones in `1 - pseudo` select fake-right trials.

The `[:, None]` operation changes counts from one-dimensional vectors into row-wise divisors. Each fake experiment is divided by its own number of trials.

## Lines 155–162 — Restore dimensions and accumulate null curves

```python
pseudo_difference = (pseudo_left - pseudo_right).reshape(
    Q3_PAPER_SHUFFLES,
    region_values.shape[1],
    region_values.shape[2]
)
q3_null_sumsq[region] += np.square(
    pseudo_difference
).sum(axis=1)
```

The flattened differences are restored to `shuffles × regional units × time`. Squaring removes sign, and `sum(axis=1)` sums over units, leaving `shuffles × time`. Each row remains one complete null trajectory across insertions.

## Lines 164–165 — Track units and insertions

```python
q3_paper_n_units[region] += region_values.shape[1]
q3_paper_pids[region].add(pid)
```

The first line adds this insertion's regional unit count. The second adds the PID to a set, so the same insertion is counted once even when it contains many units.

---

# Part 10 — Convert accumulators into regional tests

## Lines 167–171 — Iterate regions and enforce coverage

```python
q3_paper_rows = []
for region in q3_paper_region_to_group:
    n_units = q3_paper_n_units[region]
    if n_units < Q3_PAPER_MIN_UNITS:
        continue
```

The list will hold one row per eligible region. Regions with fewer than 20 units are not tested or reported.

## Lines 173–176 — Calculate true/null distances and amplitudes

```python
distance = np.sqrt(q3_true_sumsq[region] / n_units)
null_distance = np.sqrt(q3_null_sumsq[region] / n_units)
amplitude = np.ptp(distance)
null_amplitude = np.ptp(null_distance, axis=1)
```

The true distance is:

\[
D_r(t)=\sqrt{\frac{1}{N_r}\sum_{j\in r}\Delta_j(t)^2}.
\]

The same normalization is applied to each null trajectory. `np.ptp` means maximum minus minimum. One true amplitude and one fake amplitude per shuffle are produced.

## Lines 177–179 — Calculate the empirical p-value

```python
p_value = (
    1 + np.sum(null_amplitude >= amplitude)
) / (Q3_PAPER_SHUFFLES + 1)
```

The code counts fake amplitudes at least as large as the true amplitude and applies the plus-one correction:

\[
p=\frac{1+\#\{A_b\ge A_{true}\}}{1001}.
\]

This is a one-sided empirical test of unusually large trajectory amplitude. It is not a test of latency yet.

## Lines 181–191 — Store the regional row

```python
q3_paper_rows.append({
    'acronym': region,
    'anatomical_group': q3_paper_region_to_group[region],
    'n_pids': len(q3_paper_pids[region]),
    'n_units': n_units,
    'amplitude_hz': amplitude,
    'latency_ms': q3_paper_latency(
        q3_paper_times, distance, Q3_ONSET_FRACTION
    ),
    'p_value': p_value
})
```

One dictionary stores location, group, unique insertion count, unit count, true amplitude, interpolated true-curve latency, and rearrangement p-value.

Latency is calculated for storage but is interpreted only after significance filtering.

## Lines 193–202 — Create, correct, and filter the regional table

```python
q3_paper_regions = pd.DataFrame(q3_paper_rows)
q3_paper_regions['q_value'] = q3_bh_fdr(
    q3_paper_regions['p_value'].to_numpy()
)
q3_paper_regions['significant'] = (
    q3_paper_regions['q_value'] < Q3_PAPER_FDR_ALPHA
)
q3_paper_significant_regions = q3_paper_regions[
    q3_paper_regions['significant']
].copy()
```

The dictionaries become rows. All eligible regional p-values are corrected as one BH-FDR family. A Boolean column marks q-values below 0.01, and Boolean indexing keeps only significant regions.

The order matters: every random curve has a crossing, so latency is only biologically meaningful after the population trajectory has passed the amplitude test.

---

# Part 11 — Grouped output and plot

## Lines 204–215 — Summarize significant regions

```python
q3_paper_group_summary = (
    q3_paper_significant_regions
    .groupby('anatomical_group')
    .agg(
        n_regions=('acronym', 'nunique'),
        n_units=('n_units', 'sum'),
        latency_ms=('latency_ms', 'median'),
        q25_ms=('latency_ms', lambda x: x.quantile(0.25)),
        q75_ms=('latency_ms', lambda x: x.quantile(0.75))
    )
    .sort_values('latency_ms')
)
```

Only significant regional population latencies enter this summary. The median is across significant regions, not across individual units. Quartiles describe the middle half of regional population latencies.

## Lines 217–222 — Print and display results

```python
print('Paper-aligned validation: all units, block x choice pseudo-trials')
print(f'Pseudo-trials: {Q3_PAPER_SHUFFLES}')
print(f'Regions with >= {Q3_PAPER_MIN_UNITS} units: {len(q3_paper_regions)}')
print(f'FDR-significant regions: {len(q3_paper_significant_regions)}')
display(q3_paper_regions.sort_values('latency_ms').round(3))
display(q3_paper_group_summary.round(1))
```

The prints create an audit trail: the method, number of fake experiments, number of eligible regions, and number surviving FDR.

The first table displays every eligible region, including non-significant ones. The second displays the grouped significant-region summary. `round` affects only presentation; `display` affects notebook rendering, not the calculation.

## Lines 225–230 — Prepare plot coordinates

```python
plot_table = q3_paper_group_summary.copy()
y = np.arange(len(plot_table))
xerr = np.vstack([
    plot_table['latency_ms'] - plot_table['q25_ms'],
    plot_table['q75_ms'] - plot_table['latency_ms']
])
```

`np.arange` creates vertical positions. The plot requires distances from the median to the lower and upper quartile, so the code subtracts:

\[
\text{lower distance}=\text{median}-Q_{25},
\]

\[
\text{upper distance}=Q_{75}-\text{median}.
\]

`np.vstack` places the two distance arrays into the asymmetric-error format.

## Lines 232–237 — Draw the figure

```python
fig, ax = plt.subplots(figsize=(10, 5))
ax.errorbar(
    plot_table['latency_ms'], y, xerr=xerr,
    fmt='o', color='darkblue', ecolor='cornflowerblue',
    capsize=4, markersize=7
)
```

The x-position is group median latency and the y-position is group index. `xerr` is the descriptive IQR. The function is called `errorbar`, but the interval is not a confidence interval or standard error.

## Lines 238–246 — Label and render

```python
ax.set_yticks(y)
ax.set_yticklabels(plot_table.index)
ax.invert_yaxis()
ax.set_xlim(0, 150)
ax.set_xlabel('Population-trajectory latency after stimulus onset (ms)')
ax.set_title('Q3 paper-aligned validation: median and IQR across regions')
ax.grid(axis='x', linestyle=':', alpha=0.35)
plt.tight_layout()
plt.show()
```

The y positions receive anatomical names. The y-axis is inverted so the earliest sorted group appears at the top. The x-axis matches the 0–150 ms analysis window.

The title and x-label explicitly identify this as a population-trajectory result. The grid and layout calls only improve reading, and `show` renders the figure.

---

# Shape ledger for cell 163

| Variable | Shape | Meaning |
|---|---|---|
| `labels` | trials | True left/right assignment |
| `strata` | trials | Block × choice code |
| `pseudo` | shuffles × trials | Fake side assignments |
| `values` | trials × target units × time | Selected activity |
| `region_values` | trials × regional units × time | One region in one insertion |
| `true_difference` | regional units × time | Real left-minus-right averages |
| `flat_values` | trials × (regional units × time) | Temporary matrix form |
| `pseudo_left` | shuffles × (regional units × time) | Fake-left averages |
| `pseudo_right` | shuffles × (regional units × time) | Fake-right averages |
| `pseudo_difference` | shuffles × regional units × time | Fake left-minus-right averages |
| `distance` | time | True regional RMS distance |
| `null_distance` | shuffles × time | Fake regional RMS distances |
| `null_amplitude` | shuffles | Fake trajectory amplitudes |

# Final interpretation

Cell 163 supports this limited statement:

> In significant regions, the time-varying RMS difference between true left- and right-assigned population averages is larger than expected from controlled label rearrangements that preserve prior block, choice, and side counts within their combinations.

Its latency is the first interpolated 70%-of-amplitude crossing of that significant regional distance curve.

This is a population-level validation of the visual-signal timing pattern. It is not a trained decoder, not a causal connectivity analysis, and not proof that the brain uses one strictly serial route from the earliest region to the latest region.

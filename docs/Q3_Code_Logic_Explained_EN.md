# Q3 Paper-Aligned Logic — Conceptual Guide

> This file explains only notebook cell 163: `FINAL Q3 — 6. Paper-aligned population-trajectory validation`.

Read it beside [Q3 Annotated Code EN](Q3_Annotated_Code_EN.md). Use [Q3 Math Companion EN](Q3_Math_Companion_EN.md) when an equation or statistical assumption needs deeper treatment.

## Which notebook to open

Open:

`IBL_BWM_Neuromatch_tutorial_raw_Final.ipynb`

Go directly to cell **163**. Do not begin with the earlier Q3 cells if your goal is to understand the paper-aligned method.

## The question answered by cell 163

Cell 163 asks whether a region's recorded population contains a time-varying left-versus-right stimulus distinction that is stronger than expected when stimulus-side labels are rearranged under controlled conditions.

It is a region-level population validation. It is not the direct unit-wise implementation of the sentence “for responsive units compute latency.” It should therefore be read as a paper-aligned companion result, not as a replacement for the literal unit-wise answer.

## The entire logic in six sentences

The code keeps all usable trials with an assigned side, known choice, and known prior block. It divides trials into block × choice groups and creates fake side labels only inside those groups, preserving the observed number of left and right labels. For the real labels and for every fake labeling, it calculates left and right population averages. It converts their unit-by-unit difference into a non-negative RMS population distance through time. It tests whether the real distance curve has a larger amplitude than the fake curves and corrects those regional tests with BH-FDR. Only after a region passes that test does the code report its first interpolated 70%-of-amplitude crossing as latency.

## This is not machine-learning training

There is no fitted classifier in this cell. No parameter is learned from a training set, and no prediction is scored on held-out trials.

The word “model” applies only to the null model: a rule saying that, after block and choice are fixed, stimulus-side labels may be exchanged within their group. The code compares the real statistic with statistics generated under that rule.

## Why all units are used

The paper-aligned method does not first select units using a left-versus-right test. This avoids defining the population with the same signal later used to measure its trajectory.

Weak units can contribute small squared differences; strong units can contribute larger ones. The output describes the recorded population in a region, not the subset of individually strong units.

This choice changes the meaning of latency. It is the onset of a regional population trajectory, not the typical onset of selected cells.

## Why zero contrast is retained

The code defines side by which contrast column is finite. A finite value of zero is still an assigned side.

This follows the paper-oriented trial definition. It differs from a visible-stimulus filter because zero contrast contains no visible side evidence but still belongs to the assigned-side structure of the task.

Including such trials can reduce the size of the observed side separation. That is acceptable here because the purpose is closer methodological alignment with the paper, not maximizing the primary sensory effect.

## Why the null preserves block and choice

Prior block changes expectation. Choice is related to stimulus side and can also be related to neural activity. A global left/right shuffle would destroy both structures and create an unrealistic comparison.

Cell 163 instead creates four possible groups from:

- whether the prior block equals 0.8;
- whether choice equals 1.

Labels move only between trials with the same pair of conditions. The number of left and right labels inside each group stays fixed.

The null claim is:

> Once prior block and choice are held fixed, the particular connection between a trial's neural activity and its stimulus-side label is no stronger than a random allowed rearrangement.

This is more precise than saying “choice and prior are removed.” They are preserved in the fake data, not erased. The test asks whether stimulus side contributes additional structure beyond them.

## What the pseudo-trials really are

They are not invented neural traces. The activity of every trial remains unchanged.

Only the side labels are changed. Therefore, each fake experiment is a relabeling of real observations. This gives an empirical null distribution: the expected statistic is obtained by repeatedly applying the chosen null rule to the actual dataset.

## Why the population distance is squared

At each time, each unit has a left-minus-right difference. Some units prefer left and have positive values; others prefer right and have negative values.

If signed differences were averaged directly, those units could cancel. Squaring makes both preferences contribute positive separation:

\[
D(t)=\sqrt{\frac{1}{N}\sum_j\Delta_j(t)^2}.
\]

The result is a root-mean-square distance across units. It measures the magnitude of stimulus-side separation, not a common firing direction.

## Why the code streams insertions

The total squared regional contribution is additive:

\[
\sum_{s}\sum_{j\in s}\Delta_{sj}(t)^2.
\]

It can be accumulated insertion by insertion without storing every unit at once. This is mathematically exact, not an approximation.

There is one weighting consequence: regions with unit-rich insertions contribute more terms to the population distance. The method is population-weighted, not insertion-balanced.

## Why matrix multiplication appears

The fake-label matrix has shape `shuffles × trials`. The activity matrix is temporarily reshaped to `trials × (units × time)`.

Multiplying them gives all fake trial sums simultaneously:

\[
P X.
\]

Dividing each row by its fake-left or fake-right trial count converts sums to means. This is exactly equivalent to looping over fake experiments one at a time, but it is faster and uses a compact representation.

The difficult-looking `[:, None]` only adds a singleton dimension so each fake experiment receives its own divisor.

## Why amplitude is tested before latency

A random curve can have a first threshold crossing. Therefore a latency exists mathematically even when the curve contains no real stimulus-side information.

Cell 163 first measures trajectory amplitude:

\[
A=\max_tD(t)-\min_tD(t).
\]

The true amplitude is compared with fake amplitudes. Only significant regions are retained. Latency is then interpreted as the timing of a trajectory that has already passed an existence test.

## Why the p-value is empirical

The code counts how many fake amplitudes are at least as large as the real amplitude:

\[
p=\frac{1+\#(A_{null}\ge A_{real})}{1001}.
\]

It does not assume that amplitudes follow a normal distribution. The fake-label distribution is generated directly from the observed trials under the controlled null rule.

The smallest possible p-value is about 0.001 because only 1,000 fake experiments are used.

## Why BH-FDR is applied to regions

The cell produces one p-value for every region with at least 20 units. Several regions are tested, so some small p-values can occur by chance.

Benjamini–Hochberg converts the regional p-values into q-values. A q-value below 0.01 selects regions while controlling the expected false-discovery proportion under the method's assumptions.

The tested family is the eligible Q3 pathway regions in this notebook. It is not automatically identical to the full region family used in the paper, so exact significant-region counts are not an exact replication test.

## What the latency means here

For a significant region, the threshold is:

\[
D_{min}+0.70(D_{max}-D_{min}).
\]

The first crossing is found on the 10-ms grid. If the threshold lies between two samples, linear interpolation estimates the crossing time.

The decimal result is a more precise location estimate between bins, not a claim that the original data have sub-millisecond temporal resolution.

## What the anatomical summary means

The code maps significant region acronyms into broad groups and takes the median regional latency inside each group.

The group is a summary label, not a single processing station. A broad group can contain early and late regions. The region-level table should therefore be examined before making a broad anatomical statement.

The horizontal interval in the figure is an interquartile range across regional population latencies. It is not a confidence interval.

## What this method can support

Cell 163 can support the statement that significant regions show stimulus-side population trajectories whose amplitude is unusual under the controlled block × choice label null, with regional onset times that form an anatomically interpretable early-to-late pattern.

It can support close qualitative comparison with the paper's major waves:

- early visual relay and visual cortical activity;
- later brainstem and action-related activity.

## What this method cannot prove

It cannot prove that one region directly sends the measured signal to the next. It cannot establish a unique serial route, causal connectivity, or a physical propagation speed. It also does not show that every unit in a significant region is individually responsive.

The correct wording is “consistent with the propagation of stimulus-side information through the measured network,” not “proves the signal propagates from region A to region B.”

## Reading sequence

Read cell 163 and this guide in this order:

1. Settings and the two helper functions.
2. Region/time accumulators.
3. Trial labels and block × choice strata.
4. Fake labels and count preservation.
5. True and fake population differences.
6. RMS distance and amplitude comparison.
7. Regional p-values and BH-FDR.
8. Latency and anatomical summary.

For exact explanations of each executable line, use [Q3 Annotated Code EN](Q3_Annotated_Code_EN.md). For derivations, use [Q3 Math Companion EN](Q3_Math_Companion_EN.md).

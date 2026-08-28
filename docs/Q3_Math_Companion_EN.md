# Q3 Paper-Aligned Mathematics Companion

> This companion explains only the mathematics and statistics used in notebook cell 163, `FINAL Q3 — 6. Paper-aligned population-trajectory validation`.

Use it beside [Q3 Annotated Code EN](Q3_Annotated_Code_EN.md). The conceptual overview is in [Q3 Code Logic Explained EN](Q3_Code_Logic_Explained_EN.md).

## The statistical object

Cell 163 does not train a predictive model. It calculates a fixed population statistic and compares it with an empirical null distribution created by controlled relabeling.

For region \(r\) and time \(t\), let \(\Delta_{rj}(t)\) be unit \(j\)'s true left-average minus right-average. The central statistic is:

\[
D_r(t)=
\sqrt{\frac{1}{N_r}\sum_{j=1}^{N_r}\Delta_{rj}(t)^2}.
\]

The rest of the cell answers three questions:

1. Is the real trajectory's change larger than expected under controlled false labels?
2. After correction across regions, which trajectories are credible?
3. When does each credible trajectory cross its onset threshold?

---

# 1. The conditional null model

For trial \(i\), define:

- \(S_i\): assigned stimulus side;
- \(B_i\): prior-block state;
- \(C_i\): animal's choice;
- \(Y_i\): observed neural activity.

The null claim is:

\[
Y_i\perp S_i\mid(B_i,C_i).
\]

In words:

> Once prior block and choice are held fixed, the particular connection between neural activity and stimulus side contains no additional structure.

The code does not estimate this relationship with regression. It creates data arrangements that would be allowed under this null and compares the observed statistic with them.

## What is conditioned on

The code creates up to four groups from two Boolean conditions:

\[
(B_i=0.8,\ C_i=1).
\]

Trials can exchange side labels only within the same pair.

## What is preserved

- every observed neural trace;
- every prior-block value;
- every choice value;
- the number of left/right labels in each stratum;
- the number of units and time samples.

## What is broken

Only the original trial-specific association between side label and neural activity is broken inside each allowed stratum.

## What is not guaranteed

The null does not remove every possible confound. An unmeasured variable can still differ between left and right trials inside the same block × choice group. The method controls known task structure, not every source of dependence.

---

# 2. Why the fake-label subsets are fair

Suppose one stratum has \(n\) trials and exactly \(k\) observed left labels. The code gives each trial an independent continuous random score and selects the \(k\) smallest scores.

The number of possible subsets is:

\[
\binom{n}{k}=\frac{n!}{k!(n-k)!}.
\]

Because the scores are identically distributed and ties have probability zero, each size-\(k\) subset has probability:

\[
\frac{1}{\binom{n}{k}}.
\]

`argpartition` is only a computational method for finding the smallest `k` scores. It does not change this probability model.

## Why pure strata are not shuffled

If \(k=0\), every trial is right. If \(k=n\), every trial is left. There is no alternative subset with the same count, so the code leaves that stratum unchanged.

---

# 3. Matrix calculation of fake averages

Let \(B\) be the number of fake experiments and \(T\) the number of usable trials. Let \(P\) be the fake-label matrix:

\[
P\in\{0,1\}^{B\times T},
\]

where \(P_{bi}=1\) if trial \(i\) is fake-left in experiment \(b\).

Flatten one region's activity into:

\[
X\in\mathbb{R}^{T\times K},
\]

where \(K=N\times H\) combines units and time samples.

The product:

\[
PX
\]

has shape \(B\times K\). Its entry is:

\[
(PX)_{bk}=\sum_{i=1}^{T}P_{bi}X_{ik}.
\]

Since \(P_{bi}\) is one only for fake-left trials, this is the fake-left sum for experiment \(b\). Dividing by:

\[
n_{Lb}=\sum_iP_{bi}
\]

gives:

\[
\bar X^{(b)}_L=\frac{PX}{n_{Lb}}.
\]

The fake-right indicator is \(1-P\), so:

\[
\bar X^{(b)}_R=
\frac{(1-P)X}{T-n_{Lb}}.
\]

The code uses `[:, None]` to turn one count per fake experiment into a row-wise divisor across all unit-time columns.

This is exactly equivalent to looping through fake experiments and averaging selected trials one by one. Matrix multiplication performs the same sums together.

---

# 4. Population distance

For true left/right means, define:

\[
\Delta_j(t)=\bar X_{Lj}(t)-\bar X_{Rj}(t).
\]

The code uses:

\[
D(t)=\sqrt{\frac{1}{N}\sum_{j=1}^{N}\Delta_j(t)^2}.
\]

## Why square the differences

If one unit has +5 Hz and another has −5 Hz, signed averaging gives zero, even though both units distinguish side strongly:

\[
\frac{5+(-5)}{2}=0.
\]

The RMS distance gives:

\[
\sqrt{\frac{5^2+(-5)^2}{2}}=5.
\]

Thus left-preferring and right-preferring units both contribute information.

## Why divide by unit count

Without division, a region with more recorded units tends to have a larger Euclidean length simply because it has more terms. Dividing by \(N\) produces a root-mean-square scale.

This does not make sampling identical across regions. It only removes the most direct dependence on unit count.

## Why streaming is exact

If units are divided across insertions \(s\), then:

\[
\sum_s\sum_{j\in s}\Delta_{sj}(t)^2
\]

is exactly the same sum obtained after concatenating every unit. The code can therefore add squared contributions one insertion at a time and take the square root only at the end.

## Weighting consequence

Although the distance is normalized by total unit count, an insertion with more units still contributes more terms to the total. The paper-aligned population result is therefore unit-weighted rather than insertion-balanced.

## What this statistic is not

It is not a decoder, not classifier accuracy, and not a covariance-aware Mahalanobis distance. It is a transparent RMS separation measure chosen for the population trajectory.

---

# 5. Trajectory amplitude

For a distance curve \(D(t)\), the code defines:

\[
A=\max_tD(t)-\min_tD(t).
\]

This measures how much the curve changes over the 0–150 ms window.

A curve that is constantly high but flat has a small amplitude. That is useful because the test is meant to detect an event-related change, not simply a high offset.

The true curve supplies one \(A_{true}\). Each fake curve supplies one \(A_b\).

---

# 6. Rearrangement-test p-value

The code computes:

\[
p=\frac{1+\sum_{b=1}^{B}
\mathbf{1}(A_b\ge A_{true})}{B+1}.
\]

The indicator is one when a fake amplitude is at least as large as the true amplitude.

## Why the test is one-sided

The alternative is that the true trajectory is unusually large. A smaller fake amplitude supports the result; a larger or equal fake amplitude counts against it.

## Why add one

With finite random samples, no fake amplitude may exceed the true amplitude. Reporting zero would claim an impossible probability from finite simulation.

The plus-one correction also treats the observed labeling as one member of the comparison set.

For \(B=1000\),:

\[
p_{min}=\frac{1}{1001}\approx0.000999.
\]

## Empirical null distribution

The null distribution is empirical because its values come directly from the controlled relabelings. No normal or chi-square approximation is imposed on the trajectory amplitude.

---

# 7. Why significance comes before latency

Any curve, including a random curve, has a minimum, maximum, and first crossing of a relative threshold.

Therefore the logical order is:

\[
\text{test trajectory existence}
\rightarrow
\text{correct regional tests}
\rightarrow
\text{measure latency}.
\]

Cell 163 tests amplitude first. Only significant regions are placed into the grouped latency summary.

This prevents a random crossing from being interpreted as a biological onset.

---

# 8. Benjamini–Hochberg for regional p-values

Suppose the eligible regions produce \(m\) p-values. Sort them:

\[
p_{(1)}\le p_{(2)}\le\cdots\le p_{(m)}.
\]

For rank \(i\), calculate:

\[
r_i=p_{(i)}\frac{m}{i}.
\]

Then enforce monotonicity from right to left:

\[
q_{(i)}=\min_{j\ge i}r_j.
\]

Finally, return corrected values to the original regional order.

## What a q-value means here

It is a multiple-testing-adjusted value for the regional testing family. It is not the probability that a particular region is null, and it does not automatically account for every dependency between correlated neural regions.

The code uses `q < 0.01`. The exact testing family is the eligible Q3 pathway regions present in this notebook, not necessarily every region in the full reference analysis.

---

# 9. Latency threshold and interpolation

For a significant regional curve:

\[
D_{min}=\min_tD(t),\qquad
D_{max}=\max_tD(t).
\]

The threshold is:

\[
T=D_{min}+0.70(D_{max}-D_{min}).
\]

If the crossing lies between \((t_0,D_0)\) and \((t_1,D_1)\), the code uses:

\[
t_{cross}=t_0+
\frac{T-D_0}{D_1-D_0}(t_1-t_0).
\]

This assumes approximate linearity between neighboring samples.

## What interpolation does

It avoids forcing every latency to an exact multiple of 10 ms.

## What interpolation does not do

It does not create additional observations or true sub-millisecond temporal resolution. A result of 49.7 ms should be interpreted as approximately 50 ms on a 10-ms grid.

---

# 10. Why the paper-aligned result is not a trained model

A trained predictive model would have learned parameters, a loss function, a training split, and a test metric. Cell 163 has none of these.

Its “model” is the null rule:

> exchange side labels only among trials with the same block × choice state while preserving side counts.

The population distance is a fixed formula. The fake trajectories are a simulation of allowed label assignments, not synthetic neural recordings.

This method therefore asks whether the observed population geometry is unusual under the null, not whether a trained algorithm can predict stimulus side.

---

# 11. What the final group statistics represent

The code groups significant regional population latencies by anatomical group and reports:

- number of significant regions;
- sum of their unit counts;
- median regional population latency;
- 25th and 75th percentiles across those regional latencies.

The interval shown in the plot is an interquartile range. It is descriptive spread, not a confidence interval.

A broad group can contain regions with different functions and timing. Region-level results should therefore be inspected before treating a group median as one processing stage.

---

# 12. Assumptions and limitations

The conditional relabeling assumes side labels are exchangeable within block × choice strata under the null. If another unmeasured variable remains associated with side inside those strata, the null comparison may not remove it.

The population distance treats squared unit contributions as additive and does not model covariance between units.

The streaming calculation gives unit-rich insertions more influence than unit-poor insertions.

The 1,000 fake experiments limit p-value resolution.

The 10-ms time grid limits meaningful temporal precision.

The significant-region filter is based on trajectory amplitude, while latency is interpreted afterward.

These limitations do not invalidate the method. They define exactly what claim the result can support.

## Defensible final claim

Cell 163 supports:

> Significant regions show stimulus-side population trajectories whose amplitude is unusual under controlled block × choice label rearrangements, and their interpolated onset times form a biologically interpretable early-to-late pattern.

It does not prove direct causal propagation, a unique serial anatomical route, or machine-learning decoding performance.

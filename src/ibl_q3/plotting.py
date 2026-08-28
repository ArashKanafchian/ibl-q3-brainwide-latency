"""Publication-quality figures for the Q3 latency notebook.

The functions in this module are deliberately plot-only: they read tables
exported by the Q3 submission notebook and never recompute or modify an estimate.
"""

from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# Okabe-Ito-derived, colour-vision-safe palette with neutral inferential marks.
INK = "#17212B"
MUTED = "#66717E"
LIGHT = "#D7DCE2"
PALE = "#F4F7FA"
BLUE = "#0072B2"
SKY = "#56B4E9"
TEAL = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#9C6ADE"

GROUP_COLOURS = {
    "Visual thalamus": BLUE,
    "Parallel visual midbrain": TEAL,
    "Primary visual cortex": "#2A6F97",
    "Higher / sensory cortex": SKY,
    "Later midbrain / hindbrain": ORANGE,
    "Association / action": PURPLE,
}
GROUP_ORDER = list(GROUP_COLOURS)

PAPER_LATENCY = {
    "LGd": 34.0,
    "VISp": 42.0,
    "LP": 42.0,
    "VISpm": 57.0,
    "VISam": 78.0,
}

BENCHMARK_REGIONS = list(PAPER_LATENCY)


def _set_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "axes.edgecolor": "#9AA3AD",
        "axes.linewidth": 0.8,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 450,
    })


def _save(fig, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def plot_chronological_latency(final_regions, output_dir):
    """Chronological forest plot with anatomy separated from inference."""
    data = final_regions.dropna(subset=["latency_ms"]).copy()
    data = data.sort_values(["latency_ms", "acronym"]).reset_index(drop=True)
    y = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(10.8, 9.6))

    # A subtle band identifies the five direct paper benchmarks without using
    # colour as the only signal.
    for pos, row in data.iterrows():
        if row["acronym"] in BENCHMARK_REGIONS:
            ax.axhspan(pos - 0.44, pos + 0.44, color="#EAF4FA", zorder=0)

    # Full 95% insertion intervals remain visible but deliberately recede.
    for pos, row in data.iterrows():
        low = row.get("latency_ci_low_ms", np.nan)
        high = row.get("latency_ci_high_ms", np.nan)
        if np.isfinite(low) and np.isfinite(high):
            ax.plot([low, high], [pos, pos], color="#B7C0C9", lw=1.0,
                    solid_capstyle="round", zorder=1)
            ax.plot([low, high], [pos, pos], marker="|", color="#8D98A4",
                    markersize=4.5, lw=0, zorder=1)

    sig = data["significant"].fillna(False).to_numpy(dtype=bool)
    ax.scatter(data.loc[~sig, "latency_ms"], y[~sig], s=38,
               facecolors="white", edgecolors="#7A8793", linewidth=1.1,
               zorder=3, label="Not global-FDR significant")
    ax.scatter(data.loc[sig, "latency_ms"], y[sig], s=46,
               color=BLUE, edgecolor="white", linewidth=0.6,
               zorder=4, label="Global BH-FDR q < 0.01")

    seen_groups = set()
    for pos, row in data.iterrows():
        group_colour = GROUP_COLOURS.get(row["pathway_group"], MUTED)
        # Keep the anatomical key inside the plotting field so it never clips
        # or obscures the region acronym.
        ax.scatter(1.7, pos, s=28, marker="s", color=group_colour,
                   edgecolor="white", linewidth=0.4, zorder=4)
        if row["pathway_group"] not in seen_groups:
            ax.text(3.1, pos, row["pathway_group"], va="center", ha="left",
                    fontsize=7.2, color=MUTED)
            seen_groups.add(row["pathway_group"])
        if row["acronym"] in PAPER_LATENCY:
            paper = PAPER_LATENCY[row["acronym"]]
            ax.scatter(paper, pos, marker="D", s=32, facecolor="white",
                       edgecolor=PURPLE, linewidth=1.2, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(data["acronym"])
    for tick, acronym in zip(ax.get_yticklabels(), data["acronym"]):
        if acronym in BENCHMARK_REGIONS:
            tick.set_fontweight("bold")

    ax.invert_yaxis()
    ax.set_xlim(0, 150)
    ax.set_xticks(np.arange(0, 151, 25))
    ax.set_xlabel("Population-trajectory latency after stimulus onset (ms)")
    ax.set_ylabel("")
    ax.set_title("Regional recruitment unfolds from early visual to later action systems",
                 loc="left", pad=16)
    ax.text(0, 1.008,
            "Filled blue: global FDR   ·   hollow grey: not FDR   ·   purple diamond: paper   ·   thin line: 95% insertion CI",
            transform=ax.transAxes, color=MUTED, fontsize=8.5)

    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(False)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.tick_params(axis="x", length=3, color="#9AA3AD")

    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.08)
    return _save(fig, output_dir, "Q3_temporal_recruitment_chronological")


def plot_grouped_latency(final_regions, output_dir):
    """Keep anatomical families together; order regions by latency within family."""
    data = final_regions.dropna(subset=["latency_ms"]).copy()
    data["pathway_group"] = pd.Categorical(
        data["pathway_group"], categories=GROUP_ORDER, ordered=True
    )
    data = data.sort_values(
        ["pathway_group", "latency_ms", "acronym"]
    ).reset_index(drop=True)

    positions = []
    group_spans = {}
    cursor = 0.0
    for group in GROUP_ORDER:
        indexes = data.index[data["pathway_group"] == group].tolist()
        if not indexes:
            continue
        start = cursor
        for index in indexes:
            positions.append((index, cursor))
            cursor += 1.0
        group_spans[group] = (start, cursor - 1.0)
        cursor += 0.75

    y_lookup = dict(positions)
    fig, ax = plt.subplots(figsize=(12.8, 10.2))

    # Shaded anatomical lanes make grouping visible at presentation distance.
    for group, (start, stop) in group_spans.items():
        colour = GROUP_COLOURS[group]
        ax.axhspan(start - 0.48, stop + 0.48, color=colour, alpha=0.085,
                   zorder=0)
        ax.text(2.0, (start + stop) / 2, group, va="center", ha="left",
                fontsize=9.5, fontweight="bold", color=INK, zorder=1)

    for index, ypos in positions:
        row = data.loc[index]
        low = row.get("latency_ci_low_ms", np.nan)
        high = row.get("latency_ci_high_ms", np.nan)
        if np.isfinite(low) and np.isfinite(high):
            ax.plot([low, high], [ypos, ypos], color="#AEB8C2", lw=1.15,
                    solid_capstyle="round", zorder=2)
            ax.plot([low, high], [ypos, ypos], marker="|", color="#87939F",
                    markersize=5.2, lw=0, zorder=2)

        significant = bool(row["significant"])
        if significant:
            ax.scatter(row["latency_ms"], ypos, s=58, color=BLUE,
                       edgecolor="white", linewidth=0.7, zorder=4)
        else:
            ax.scatter(row["latency_ms"], ypos, s=48, facecolor="white",
                       edgecolor="#75828F", linewidth=1.25, zorder=4)

        if row["acronym"] in PAPER_LATENCY:
            ax.scatter(PAPER_LATENCY[row["acronym"]], ypos, marker="D", s=40,
                       facecolor="white", edgecolor=PURPLE, linewidth=1.3,
                       zorder=5)

    ordered_positions = [ypos for _, ypos in positions]
    ordered_labels = [data.loc[index, "acronym"] for index, _ in positions]
    ax.set_yticks(ordered_positions)
    ax.set_yticklabels(ordered_labels)
    for tick, acronym in zip(ax.get_yticklabels(), ordered_labels):
        if acronym in BENCHMARK_REGIONS:
            tick.set_fontweight("bold")

    ax.invert_yaxis()
    ax.set_xlim(0, 150)
    ax.set_xticks(np.arange(0, 151, 25))
    ax.set_xlabel("Population-trajectory latency after stimulus onset (ms)")
    ax.set_title(
        "Temporal recruitment by anatomical system",
        loc="left", pad=18,
    )
    ax.text(
        0, 1.008,
        "Regions are ordered by latency within each shaded anatomical group; estimates and results are unchanged.",
        transform=ax.transAxes, color=MUTED, fontsize=9.5,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.grid(False)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.92, bottom=0.09)
    return _save(fig, output_dir, "Q3_temporal_recruitment_grouped")


def plot_distance_small_multiples(curve_table, final_regions, output_dir):
    """Five aligned panels remove the spaghetti-plot failure mode."""
    curves = curve_table.copy()
    if "time_ms" in curves.columns:
        curves = curves.set_index("time_ms")
    curves.index = curves.index.astype(float)
    lookup = final_regions.set_index("acronym")
    ordered = sorted(BENCHMARK_REGIONS,
                     key=lambda r: float(lookup.loc[r, "latency_ms"]))

    fig, axes = plt.subplots(len(ordered), 1, figsize=(9.6, 7.6), sharex=True,
                             sharey=True)
    colour_map = {
        "LGd": BLUE, "VISp": TEAL, "LP": ORANGE,
        "VISpm": PURPLE, "VISam": VERMILLION,
    }

    for ax, region in zip(axes, ordered):
        curve = curves[region].to_numpy(dtype=float)
        normalised = curve / np.ptp(curve)
        times = curves.index.to_numpy(dtype=float)
        ours = float(lookup.loc[region, "latency_ms"])
        paper = PAPER_LATENCY[region]
        colour = colour_map[region]

        ax.plot(times, normalised, color=colour, lw=2.1,
                solid_capstyle="round")
        ax.fill_between(times, 0, normalised, color=colour, alpha=0.13)
        ax.axhline(0.70, color="#7A8793", lw=0.9, ls=(0, (3, 3)))
        ax.scatter(ours, 0.70, s=48, color=colour, edgecolor="white",
                   linewidth=0.7, zorder=4)
        ax.scatter(paper, 0.70, marker="D", s=40, facecolor="white",
                   edgecolor=PURPLE, linewidth=1.2, zorder=4)
        ax.text(2, 0.91, region, fontweight="bold", fontsize=10,
                color=INK, va="top")
        ax.text(148, 0.72, "70%", ha="right", va="bottom",
                fontsize=7.5, color=MUTED)
        ax.text(ours + 2, 0.63, f"{ours:.0f} ms", fontsize=7.8,
                color=colour, va="top")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks([0, 0.7, 1.0])
        ax.set_yticklabels(["0", "0.7", "1"])
        ax.grid(False)

    axes[-1].set_xlim(0, 150)
    axes[-1].set_xticks(np.arange(0, 151, 25))
    axes[-1].set_xlabel("Time after stimulus onset (ms)")
    fig.supylabel("Left–right population distance (normalised within region)",
                  x=0.025, fontsize=10)
    axes[0].set_title("Each region crosses the same relative threshold at a different time",
                      loc="left", pad=24)
    axes[0].text(0, 1.08,
                 "Filled circle: this analysis   ·   open diamond: paper   ·   curves contain 10 ms samples",
                 transform=axes[0].transAxes, color=MUTED, fontsize=8.5)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.09, hspace=0.15)
    return _save(fig, output_dir, "Q3_distance_curve_small_multiples")


def plot_bootstrap_comparison(final_regions, output_dir):
    """Explain long intervals rather than hiding them."""
    data = final_regions[final_regions["acronym"].isin(BENCHMARK_REGIONS)].copy()
    data = data.sort_values("latency_ms").reset_index(drop=True)
    y = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(9.3, 4.4))
    for pos, row in data.iterrows():
        if pos % 2 == 0:
            ax.axhspan(pos - 0.38, pos + 0.38, color=PALE, zorder=0)
        ax.plot([row["latency_ci_low_ms"], row["latency_ci_high_ms"]],
                [pos, pos], color=SKY, lw=5.2, alpha=0.35,
                solid_capstyle="round", zorder=1)
        ax.plot([row["trial_ci_low_ms"], row["trial_ci_high_ms"]],
                [pos, pos], color=BLUE, lw=2.0,
                solid_capstyle="round", zorder=2)
        ax.scatter(row["latency_ms"], pos, s=48, color=INK,
                   edgecolor="white", linewidth=0.7, zorder=3)
        ax.scatter(PAPER_LATENCY[row["acronym"]], pos, marker="D", s=40,
                   facecolor="white", edgecolor=PURPLE, linewidth=1.2, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(data["acronym"], fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 150)
    ax.set_xticks(np.arange(0, 151, 25))
    ax.set_xlabel("Latency after stimulus onset (ms)")
    ax.set_title("Recording-to-recording variability dominates trial resampling",
                 loc="left", pad=18)
    ax.text(0, 1.02,
            "Outer band: 95% insertion bootstrap   ·   inner line: trial bootstrap   ·   diamond: paper",
            transform=ax.transAxes, color=MUTED, fontsize=8.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    fig.tight_layout()
    return _save(fig, output_dir, "Q3_bootstrap_level_comparison")


def plot_responsive_lollipop(final_regions, output_dir):
    """Low-ink replacement for the default orange bar chart."""
    data = final_regions.dropna(subset=["percent_responsive"]).copy()
    data = data.sort_values("percent_responsive", ascending=False).reset_index(drop=True)
    y = np.arange(len(data))
    colours = [GROUP_COLOURS.get(group, MUTED) for group in data["pathway_group"]]

    fig, ax = plt.subplots(figsize=(9.8, 8.6))
    ax.hlines(y, 0, data["percent_responsive"], color="#D9DEE4", lw=1.2)
    ax.scatter(data["percent_responsive"], y, c=colours, s=58,
               edgecolor="white", linewidth=0.7, zorder=3)
    for pos, row in data.iterrows():
        ax.text(row["percent_responsive"] + 0.8, pos,
                f"{int(row['n_responsive_units']):,}/{int(row['n_tested_units']):,}",
                va="center", fontsize=7.5, color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(data["acronym"])
    for tick, acronym in zip(ax.get_yticklabels(), data["acronym"]):
        if acronym in BENCHMARK_REGIONS:
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, max(62, data["percent_responsive"].max() + 7))
    ax.set_xlabel("Responsive units (% of units tested in each region)")
    ax.set_ylabel("Region (ordered by responsive fraction)")
    ax.set_title("Stimulus-onset responsiveness is distributed beyond visual cortex",
                 loc="left", pad=18)
    ax.text(0, 1.008,
            "Dots show regional fractions; labels show responsive / tested units. Bold labels mark paper benchmarks.",
            transform=ax.transAxes, color=MUTED, fontsize=8.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=colour, label=group)
        for group, colour in GROUP_COLOURS.items()
    ]
    ax.legend(handles=handles, loc="lower right", ncol=2, fontsize=7.3,
              columnspacing=0.8, handletextpad=0.35)
    fig.tight_layout()
    return _save(fig, output_dir, "Q3_responsive_units_lollipop")


def plot_sensitivity_decision(sensitivity, output_dir):
    """Turn the three textual sensitivity checks into one slide-ready argument."""
    order = ["LGd", "VISp", "LP", "VISpm", "VISam"]
    data = sensitivity.set_index("acronym").loc[order].copy()
    data["gate_delta"] = data["A_gated_ms"] - data["A_paper_rule_ms"]
    data["raw_delta"] = data["B_no_interpolation_ms"] - data["A_paper_rule_ms"]
    current_error = np.abs(
        data["A_paper_rule_ms"] - data["paper_latency_ms"]
    ).sum()
    raw_error = np.abs(
        data["B_no_interpolation_ms"] - data["paper_latency_ms"]
    ).sum()

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(
        1, 2, left=0.065, right=0.97, top=0.79, bottom=0.22,
        wspace=0.28, width_ratios=[1.08, 1.0]
    )
    ax_change = fig.add_subplot(grid[0, 0])
    ax_threshold = fig.add_subplot(grid[0, 1])

    fig.suptitle(
        "Sensitivity checks support keeping one common 70% latency rule",
        x=0.065, y=0.94, ha="left", fontsize=24, fontweight="bold",
    )
    fig.text(
        0.065, 0.875,
        "Five paper-benchmark regions were recomputed under stricter gating, raw-bin timing, and alternative thresholds.",
        ha="left", fontsize=13, color=MUTED,
    )

    # Panel A: every alternative is displayed as a change from the reported
    # 70%-crossing estimate, so “0-ms impact” becomes visually literal.
    y = np.arange(len(data))
    ax_change.axvspan(-0.18, 0.18, color=TEAL, alpha=0.10, zorder=0)
    ax_change.axvline(0, color="#86919C", lw=1.2, zorder=1)
    for pos, (_, row) in enumerate(data.iterrows()):
        ax_change.plot(
            [0, row["raw_delta"]], [pos, pos], color="#CBD2D9",
            lw=2.0, solid_capstyle="round", zorder=1,
        )
        ax_change.scatter(
            row["gate_delta"], pos, marker="s", s=90, color=TEAL,
            edgecolor="white", linewidth=0.9, zorder=4,
        )
        ax_change.scatter(
            row["raw_delta"], pos, marker="o", s=92, color=ORANGE,
            edgecolor="white", linewidth=0.9, zorder=5,
        )
        ax_change.text(
            row["raw_delta"] + 0.28, pos, f"+{row['raw_delta']:.0f} ms",
            va="center", fontsize=10.5, color=VERMILLION,
        )

    ax_change.set_yticks(y)
    ax_change.set_yticklabels(order, fontweight="bold")
    ax_change.invert_yaxis()
    ax_change.set_xlim(-0.8, 9.7)
    ax_change.set_xticks([0, 2, 4, 6, 8])
    ax_change.set_xlabel("Change from the reported estimate (ms)")
    ax_change.set_title(
        "A  Which analysis choices move the latency?", loc="left", pad=42,
    )
    ax_change.text(
        0.0, 1.015,
        "Green squares: gate ON—every original crossing was already significant, so all remain at 0 ms.\n"
        f"Orange circles: no interpolation—0–8 ms later; total paper discrepancy {raw_error:.0f} → {current_error:.0f} ms with interpolation.",
        transform=ax_change.transAxes, fontsize=9.3, color=MUTED,
    )
    ax_change.spines[["top", "right", "left"]].set_visible(False)
    ax_change.tick_params(axis="y", length=0, pad=8)
    ax_change.grid(axis="x", color="#E4E8EC", lw=0.8)

    # Panel B: show why tuning is not a single defensible global correction.
    required = data["required_fraction"].to_numpy(dtype=float)
    ax_threshold.axvline(0.70, color=ORANGE, lw=3.0, zorder=1)
    for pos, (region, row) in enumerate(data.iterrows()):
        colour = VERMILLION if region == "VISam" else PURPLE
        ax_threshold.plot(
            [0.70, row["required_fraction"]], [pos, pos],
            color="#C7CED6", lw=2.0, zorder=1,
        )
        ax_threshold.scatter(
            row["required_fraction"], pos, s=100, color=colour,
            edgecolor="white", linewidth=0.9, zorder=4,
        )
        alignment = "left" if row["required_fraction"] <= 0.70 else "right"
        offset = 0.018 if alignment == "left" else -0.018
        ax_threshold.text(
            row["required_fraction"] + offset, pos - 0.25,
            f"{row['required_fraction']:.2f}", ha=alignment,
            fontsize=10.5, color=colour, fontweight="bold",
        )

    ax_threshold.set_yticks(y)
    ax_threshold.set_yticklabels(order, fontweight="bold")
    ax_threshold.invert_yaxis()
    ax_threshold.set_xlim(0, 1.0)
    ax_threshold.set_xticks(np.arange(0, 1.01, 0.2))
    ax_threshold.set_xlabel("Threshold fraction needed to reproduce each paper latency")
    ax_threshold.set_title(
        "B  Can one new threshold match the paper?", loc="left", pad=34,
    )
    ax_threshold.text(
        0.72, 3.45, "common rule  0.70", ha="left", va="center",
        color=VERMILLION, fontsize=10.5, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
    )
    ax_threshold.annotate(
        "required settings span 0.16–0.95",
        xy=(required.min(), 4.62), xytext=(required.max(), 4.62),
        arrowprops=dict(arrowstyle="<->", color="#7B8793", lw=1.2),
        ha="center", va="center", color=MUTED, fontsize=10.5,
    )
    ax_threshold.text(
        0.0, 1.015,
        "Each dot is the threshold that best reproduces that region's paper latency; one global replacement should cluster.",
        transform=ax_threshold.transAxes, fontsize=9.3, color=MUTED,
    )
    ax_threshold.spines[["top", "right", "left"]].set_visible(False)
    ax_threshold.tick_params(axis="y", length=0, pad=8)
    ax_threshold.grid(axis="x", color="#E4E8EC", lw=0.8)

    # One conclusion strip replaces the slide's paragraph block.
    fig.patches.append(mpl.patches.Rectangle(
        (0.065, 0.055), 0.905, 0.085, transform=fig.transFigure,
        facecolor="#EEF4F8", edgecolor="none", zorder=-1,
    ))
    fig.text(
        0.085, 0.097,
        "Decision  →  keep the prespecified 70% rule",
        ha="left", va="center", fontsize=16, fontweight="bold", color=BLUE,
    )
    fig.text(
        0.405, 0.097,
        "The significance gate is neutral; interpolation reduces binning error; region-specific retuning would be cherry-picking.",
        ha="left", va="center", fontsize=12.2, color=INK,
    )
    return _save(fig, output_dir, "Q3_sensitivity_decision_two_panel")


def plot_sensitivity_decision_condensed(sensitivity, output_dir):
    """Plot-only sensitivity summary for a fast presentation slide."""
    order = ["LGd", "VISp", "LP", "VISpm", "VISam"]
    data = sensitivity.set_index("acronym").loc[order].copy()
    data["gate_delta"] = data["A_gated_ms"] - data["A_paper_rule_ms"]
    data["raw_delta"] = (
        data["B_no_interpolation_ms"] - data["A_paper_rule_ms"]
    )
    y = np.arange(len(data))

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(
        1, 3, left=0.075, right=0.97, top=0.76, bottom=0.22,
        wspace=0.28, width_ratios=[0.78, 1.08, 1.34],
    )
    ax_gate = fig.add_subplot(grid[0, 0])
    ax_raw = fig.add_subplot(grid[0, 1], sharey=ax_gate)
    ax_threshold = fig.add_subplot(grid[0, 2], sharey=ax_gate)

    fig.suptitle(
        "Sensitivity checks support one prespecified 70% latency rule",
        x=0.075, y=0.93, ha="left", fontsize=25, fontweight="bold",
    )

    # Highlight the region that remains discrepant without turning the row
    # into a separate textual callout.
    for ax in (ax_gate, ax_raw, ax_threshold):
        ax.axhspan(3.55, 4.45, color="#FCEDEB", zorder=0)

    # A — a stricter significance gate leaves every estimate unchanged.
    ax_gate.axvspan(-0.06, 0.06, color=TEAL, alpha=0.12, zorder=0)
    ax_gate.axvline(0, color="#AAB3BC", lw=1.2, zorder=1)
    ax_gate.scatter(
        data["gate_delta"], y, marker="s", s=120, color=TEAL,
        edgecolor="white", linewidth=1.0, zorder=3,
    )
    for pos in y:
        ax_gate.text(
            0.08, pos, "0", va="center", ha="left", color=TEAL,
            fontsize=11, fontweight="bold",
        )
    ax_gate.set_xlim(-0.55, 0.55)
    ax_gate.set_xticks([0])
    ax_gate.set_xlabel("Latency change (ms)")
    ax_gate.set_title(
        "A  Significance gate  |  0 ms", loc="left", pad=18, color=TEAL,
    )

    # B — raw 10-ms bins can only preserve or delay a crossing.
    for pos, delta in enumerate(data["raw_delta"].to_numpy(dtype=float)):
        ax_raw.plot(
            [0, delta], [pos, pos], color="#C9D0D7", lw=3.0,
            solid_capstyle="round", zorder=1,
        )
        ax_raw.scatter(
            delta, pos, s=105, color=ORANGE, edgecolor="white",
            linewidth=1.0, zorder=3,
        )
        ax_raw.text(
            delta + 0.23, pos, f"+{delta:.0f}", va="center", ha="left",
            color=VERMILLION, fontsize=11, fontweight="bold",
        )
    ax_raw.axvline(0, color="#AAB3BC", lw=1.0, zorder=1)
    ax_raw.set_xlim(-0.4, 9.2)
    ax_raw.set_xticks([0, 2, 4, 6, 8])
    ax_raw.set_xlabel("Later without interpolation (ms)")
    ax_raw.set_title(
        "B  Raw-bin timing  |  0–8 ms later",
        loc="left", pad=18, color=VERMILLION,
    )

    # C — the threshold needed to force an exact paper match differs strongly
    # by region, so there is no coherent global replacement for 0.70.
    ax_threshold.axvline(0.70, color=ORANGE, lw=3.2, zorder=1)
    for pos, (region, row) in enumerate(data.iterrows()):
        value = float(row["required_fraction"])
        colour = VERMILLION if region == "VISam" else PURPLE
        ax_threshold.plot(
            [min(value, 0.70), max(value, 0.70)], [pos, pos],
            color="#C9D0D7", lw=3.0, solid_capstyle="round", zorder=1,
        )
        ax_threshold.scatter(
            value, pos, s=120, color=colour, edgecolor="white",
            linewidth=1.0, zorder=3,
        )
        ha = "left" if value < 0.88 else "right"
        dx = 0.025 if ha == "left" else -0.025
        ax_threshold.text(
            value + dx, pos, f"{value:.2f}", va="center", ha=ha,
            color=colour, fontsize=11, fontweight="bold",
        )
    ax_threshold.text(
        0.70, -0.48, "common 0.70", ha="center", va="bottom",
        color=VERMILLION, fontsize=10.5, fontweight="bold",
    )
    ax_threshold.set_xlim(0, 1.0)
    ax_threshold.set_xticks(np.arange(0, 1.01, 0.2))
    ax_threshold.set_xlabel("Threshold required for an exact paper match")
    ax_threshold.set_title(
        "C  Retuned threshold  |  0.16–0.95 required",
        loc="left", pad=18, color=PURPLE,
    )

    for ax in (ax_gate, ax_raw, ax_threshold):
        ax.set_ylim(4.55, -0.62)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=10)
        ax.grid(axis="x", color="#E5E9ED", lw=0.8)

    ax_gate.set_yticks(y)
    ax_gate.set_yticklabels(order, fontweight="bold")
    for ax in (ax_raw, ax_threshold):
        ax.tick_params(axis="y", left=False, labelleft=False)

    # Two short statements retain the original slide's decision and its
    # explicitly qualified VISam explanation without reintroducing paragraphs.
    fig.patches.append(mpl.patches.Rectangle(
        (0.075, 0.055), 0.895, 0.09, transform=fig.transFigure,
        facecolor="#EEF4F8", edgecolor="none", zorder=-1,
    ))
    fig.text(
        0.095, 0.10, "KEEP  →  one common 70% rule",
        ha="left", va="center", fontsize=16, fontweight="bold", color=BLUE,
    )
    fig.text(
        0.57, 0.10,
        "VISam remains earlier; insertion coverage is plausible, not tested.",
        ha="left", va="center", fontsize=12.5, color=VERMILLION,
    )
    return _save(fig, output_dir, "Q3_sensitivity_decision_condensed")


def plot_interpolation_resolution(sensitivity, curve_table, output_dir):
    """Explain interpolation and the remaining VISam disagreement visually."""
    order = ["LGd", "VISp", "LP", "VISpm", "VISam"]
    data = sensitivity.set_index("acronym").loc[order].copy()
    curves = curve_table.set_index("time_ms") if "time_ms" in curve_table else curve_table.copy()
    times = curves.index.to_numpy(dtype=float)
    curve = curves["LGd"].to_numpy(dtype=float)
    normalised = (curve - curve.min()) / np.ptp(curve)

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(
        1, 2, left=0.065, right=0.97, top=0.79, bottom=0.22,
        wspace=0.28, width_ratios=[1.05, 1.0]
    )
    ax_curve = fig.add_subplot(grid[0, 0])
    ax_gap = fig.add_subplot(grid[0, 1])

    fig.suptitle(
        "Interpolation locates a crossing between samples—it does not create new resolution",
        x=0.065, y=0.94, ha="left", fontsize=23, fontweight="bold",
    )
    fig.text(
        0.065, 0.875,
        "The sampled 10-ms interval remains the uncertainty floor; the paper comparison asks whether both estimates occupy the same interval.",
        ha="left", fontsize=13, color=MUTED,
    )

    # Panel A uses the real LGd curve around its first threshold crossing.
    mask = (times >= 25) & (times <= 55)
    ax_curve.plot(times[mask], normalised[mask], color=BLUE, lw=3.0,
                  marker="o", markersize=8, zorder=3)
    ax_curve.fill_between(times[mask], 0, normalised[mask], color=BLUE,
                          alpha=0.10, zorder=1)
    ax_curve.axhline(0.70, color="#7A8793", lw=1.4, ls=(0, (4, 3)))
    interp = float(data.loc["LGd", "A_paper_rule_ms"])
    raw = float(data.loc["LGd", "B_no_interpolation_ms"])
    ax_curve.axvline(interp, color=TEAL, lw=2.2, zorder=2)
    ax_curve.axvline(raw, color=ORANGE, lw=2.2, ls=(0, (4, 3)), zorder=2)
    ax_curve.scatter(interp, 0.70, s=120, color=TEAL, edgecolor="white",
                     linewidth=1.0, zorder=5)
    ax_curve.annotate(
        "linear crossing\n36 ms",
        xy=(interp, 0.70), xytext=(29.5, 0.43),
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5),
        color=TEAL, fontsize=12, fontweight="bold", ha="center",
    )
    ax_curve.annotate(
        "first sampled crossing\n≈45 ms  (reported 44 ms)",
        xy=(45, normalised[np.where(times == 45)[0][0]]),
        xytext=(49.2, 0.93),
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5),
        color=VERMILLION, fontsize=11.5, fontweight="bold", ha="center",
    )
    ax_curve.set_xlim(25, 55)
    ax_curve.set_ylim(0, 1.08)
    ax_curve.set_xticks([25, 35, 45, 55])
    ax_curve.set_yticks([0, 0.7, 1.0])
    ax_curve.set_xlabel("Time after stimulus onset (ms)")
    ax_curve.set_ylabel("LGd population distance (normalised)")
    ax_curve.set_title("A  What linear interpolation actually does", loc="left", pad=16)
    ax_curve.text(
        0.0, 1.015, "Real LGd samples from the notebook", transform=ax_curve.transAxes,
        fontsize=10, color=MUTED,
    )
    ax_curve.spines[["top", "right"]].set_visible(False)
    ax_curve.grid(color="#E6EAEE", lw=0.8)

    # Panel B visualises the unresolved interval instead of describing it in a table.
    y = np.arange(len(data))
    for pos, (region, row) in enumerate(data.iterrows()):
        ax_gap.plot(
            [row["gap_low_ms"], row["gap_high_ms"]], [pos, pos],
            color=SKY, lw=13, alpha=0.28, solid_capstyle="round", zorder=1,
        )
        ax_gap.plot(
            [row["gap_low_ms"], row["gap_high_ms"]], [pos, pos],
            color=BLUE, lw=2.0, solid_capstyle="round", zorder=2,
        )
        ax_gap.scatter(
            row["A_paper_rule_ms"], pos, s=92, color=BLUE,
            edgecolor="white", linewidth=0.9, zorder=4,
        )
        paper_colour = VERMILLION if region == "VISam" else PURPLE
        ax_gap.scatter(
            row["paper_latency_ms"], pos, marker="D", s=80,
            facecolor="white", edgecolor=paper_colour, linewidth=1.8, zorder=5,
        )
        if region == "VISam":
            verdict = "+13 ms beyond bracket"
            verdict_colour = VERMILLION
        elif bool(row["paper_inside_gap"]):
            verdict = "inside bracket"
            verdict_colour = TEAL
        else:
            verdict = f"{abs(row['paper_offset_from_gap_ms']):.0f} ms from bracket"
            verdict_colour = MUTED
        ax_gap.text(89.5, pos, verdict, ha="right", va="center",
                    fontsize=10.5, color=verdict_colour,
                    fontweight="bold" if region == "VISam" else "normal")

    ax_gap.set_yticks(y)
    ax_gap.set_yticklabels(order, fontweight="bold")
    ax_gap.invert_yaxis()
    ax_gap.set_xlim(25, 92)
    ax_gap.set_xticks(np.arange(30, 91, 10))
    ax_gap.set_xlabel("Latency after stimulus onset (ms)")
    ax_gap.set_title("B  Does the paper value occupy the same 10-ms bracket?",
                     loc="left", pad=16)
    ax_gap.text(
        0.0, 1.015,
        "Blue band: unresolved sample interval   ·   circle: this analysis   ·   diamond: paper",
        transform=ax_gap.transAxes, fontsize=9.5, color=MUTED,
    )
    ax_gap.spines[["top", "right", "left"]].set_visible(False)
    ax_gap.tick_params(axis="y", length=0, pad=8)
    ax_gap.grid(axis="x", color="#E6EAEE", lw=0.8)

    fig.patches.append(mpl.patches.Rectangle(
        (0.065, 0.055), 0.905, 0.085, transform=fig.transFigure,
        facecolor="#F4F1F8", edgecolor="none", zorder=-1,
    ))
    fig.text(
        0.085, 0.097,
        "Interpretation",
        ha="left", va="center", fontsize=16, fontweight="bold", color=PURPLE,
    )
    fig.text(
        0.21, 0.097,
        "Four benchmarks are inside or within 3 ms of the unresolved bracket. VISam is 13 ms outside: a real dataset/coverage difference.",
        ha="left", va="center", fontsize=12.4, color=INK,
    )
    return _save(fig, output_dir, "Q3_interpolation_resolution_two_panel")


def plot_benchmark_agreement_visual(final_regions, sensitivity, output_dir):
    """Replace the benchmark table with two directly comparable evidence panels."""
    order = ["LGd", "VISp", "LP", "VISpm", "VISam"]
    estimates = final_regions.set_index("acronym").loc[order].copy()
    checks = sensitivity.set_index("acronym").loc[order].copy()
    data = estimates.join(
        checks[[
            "paper_latency_ms", "gap_low_ms", "gap_high_ms",
            "paper_inside_gap", "paper_offset_from_gap_ms",
        ]]
    )
    data["resolution_consistent"] = (
        data["paper_offset_from_gap_ms"].abs() <= 5
    )
    data["paper_inside_ci"] = data["paper_latency_ms"].between(
        data["latency_ci_low_ms"], data["latency_ci_high_ms"]
    )

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(
        1, 2, left=0.07, right=0.97, top=0.79, bottom=0.22,
        wspace=0.25, width_ratios=[1.0, 1.08]
    )
    ax_resolution = fig.add_subplot(grid[0, 0])
    ax_ci = fig.add_subplot(grid[0, 1], sharey=ax_resolution)
    y = np.arange(len(data))

    fig.suptitle(
        "Four benchmarks are resolution-consistent; VISam remains earlier",
        x=0.07, y=0.94, ha="left", fontsize=24, fontweight="bold",
        fontfamily="serif",
    )
    fig.text(
        0.07, 0.875,
        "Temporal sampling and bootstrap uncertainty are different tests of agreement.",
        ha="left", fontsize=13, color=MUTED,
    )

    # Panel A: what the raw 10-ms sampling can and cannot distinguish.
    for pos, (region, row) in enumerate(data.iterrows()):
        if pos % 2 == 0:
            ax_resolution.axhspan(pos - 0.37, pos + 0.37, color=PALE, zorder=0)
        ax_resolution.plot(
            [row["gap_low_ms"], row["gap_high_ms"]], [pos, pos],
            color=SKY, lw=14, alpha=0.27, solid_capstyle="round", zorder=1,
        )
        ax_resolution.plot(
            [row["gap_low_ms"], row["gap_high_ms"]], [pos, pos],
            color=BLUE, lw=2.2, solid_capstyle="round", zorder=2,
        )
        ax_resolution.scatter(
            row["latency_ms"], pos, s=92, color=BLUE,
            edgecolor="white", linewidth=0.9, zorder=4,
        )
        paper_colour = VERMILLION if region == "VISam" else PURPLE
        ax_resolution.scatter(
            row["paper_latency_ms"], pos, marker="D", s=78,
            facecolor="white", edgecolor=paper_colour, linewidth=1.8, zorder=5,
        )
        if region == "VISam":
            verdict = "+13 ms beyond bracket"
            verdict_colour = VERMILLION
        elif bool(row["paper_inside_gap"]):
            verdict = "inside bracket"
            verdict_colour = TEAL
        else:
            verdict = f"within {abs(row['paper_offset_from_gap_ms']):.0f} ms"
            verdict_colour = BLUE
        if region == "VISam":
            ax_resolution.text(
                69.0, pos + 0.20, "+13 ms beyond bracket", ha="center",
                va="top", fontsize=10.5, color=verdict_colour,
                fontweight="bold",
            )
        else:
            ax_resolution.text(
                84.5, pos, verdict, ha="right", va="center", fontsize=10.5,
                color=verdict_colour,
            )

    ax_resolution.set_yticks(y)
    ax_resolution.set_yticklabels(order, fontweight="bold")
    ax_resolution.invert_yaxis()
    ax_resolution.set_xlim(25, 86)
    ax_resolution.set_xticks(np.arange(30, 86, 10))
    ax_resolution.set_xlabel("Latency after stimulus onset (ms)")
    ax_resolution.set_title("A  Temporal sampling agreement", loc="left", pad=18)
    ax_resolution.text(
        0, 1.015,
        "Blue band: unresolved 10-ms crossing interval   ·   filled circle: notebook   ·   open diamond: paper",
        transform=ax_resolution.transAxes, fontsize=9.5, color=MUTED,
    )
    ax_resolution.spines[["top", "right", "left"]].set_visible(False)
    ax_resolution.tick_params(axis="y", length=0, pad=8)
    ax_resolution.grid(axis="x", color="#E5E9ED", lw=0.8)

    # Panel B: insertion-level uncertainty, deliberately kept on its full scale.
    for pos, (region, row) in enumerate(data.iterrows()):
        if pos % 2 == 0:
            ax_ci.axhspan(pos - 0.37, pos + 0.37, color=PALE, zorder=0)
        low = row["latency_ci_low_ms"]
        high = row["latency_ci_high_ms"]
        ax_ci.plot(
            [low, high], [pos, pos], color=PURPLE, lw=5.5, alpha=0.23,
            solid_capstyle="round", zorder=1,
        )
        ax_ci.plot(
            [low, high], [pos, pos], color=PURPLE, lw=1.8,
            solid_capstyle="round", zorder=2,
        )
        ax_ci.plot(
            [low, high], [pos, pos], marker="|", color=PURPLE,
            markersize=8, lw=0, zorder=2,
        )
        ax_ci.scatter(
            row["latency_ms"], pos, s=92, color=BLUE,
            edgecolor="white", linewidth=0.9, zorder=4,
        )
        inside = bool(row["paper_inside_ci"])
        paper_colour = TEAL if inside else (VERMILLION if region == "VISam" else PURPLE)
        ax_ci.scatter(
            row["paper_latency_ms"], pos, marker="D", s=78,
            facecolor="white", edgecolor=paper_colour, linewidth=1.8, zorder=5,
        )
        verdict = "inside CI" if inside else "outside CI"
        ax_ci.text(
            128.0, pos, verdict, ha="right", va="center", fontsize=10.5,
            color=TEAL if inside else (VERMILLION if region == "VISam" else MUTED),
            fontweight="bold" if region in {"LGd", "VISam"} else "normal",
        )

    ax_ci.set_xlim(25, 130)
    ax_ci.set_xticks(np.arange(30, 131, 20))
    ax_ci.set_xlabel("Latency after stimulus onset (ms)")
    ax_ci.set_title("B  Stability across recording insertions", loc="left", pad=18)
    ax_ci.text(
        0, 1.015,
        "Purple line: 95% insertion-level bootstrap CI   ·   filled circle: notebook   ·   open diamond: paper",
        transform=ax_ci.transAxes, fontsize=9.5, color=MUTED,
    )
    ax_ci.spines[["top", "right", "left"]].set_visible(False)
    ax_ci.tick_params(axis="y", left=False, labelleft=False)
    ax_ci.grid(axis="x", color="#E5E9ED", lw=0.8)

    # Flat evidence strip replaces the slide's paragraph-heavy right column.
    fig.patches.append(mpl.patches.Rectangle(
        (0.07, 0.055), 0.90, 0.095, transform=fig.transFigure,
        facecolor="#F2F5F8", edgecolor="none", zorder=-1,
    ))
    fig.text(
        0.092, 0.102, "4 / 5", fontsize=22, fontweight="bold",
        color=BLUE, va="center",
    )
    fig.text(
        0.155, 0.102, "within ≤5 ms of the 10-ms bracket",
        fontsize=12.2, color=INK, va="center",
    )
    fig.text(
        0.425, 0.102, "1 / 5", fontsize=22, fontweight="bold",
        color=PURPLE, va="center",
    )
    fig.text(
        0.488, 0.102, "paper values inside the 95% insertion CI",
        fontsize=12.2, color=INK, va="center",
    )
    fig.text(
        0.755, 0.102, "VISam", fontsize=17, fontweight="bold",
        color=VERMILLION, va="center",
    )
    fig.text(
        0.817, 0.102, "13 ms beyond bracket; ~2 ms above CI",
        fontsize=11.2, color=INK, va="center",
    )
    return _save(fig, output_dir, "Q3_benchmark_agreement_visual")


def plot_benchmark_agreement_condensed(final_regions, sensitivity, output_dir):
    """Plot-only version of the benchmark slide, with no narrative paragraphs."""
    order = ["LGd", "VISp", "LP", "VISpm", "VISam"]
    estimates = final_regions.set_index("acronym").loc[order].copy()
    checks = sensitivity.set_index("acronym").loc[order].copy()
    data = estimates.join(
        checks[["paper_latency_ms", "gap_low_ms", "gap_high_ms"]]
    )
    y = np.arange(len(data))

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(
        1, 2, left=0.075, right=0.97, top=0.76, bottom=0.13,
        wspace=0.22, width_ratios=[1.0, 1.08]
    )
    ax_resolution = fig.add_subplot(grid[0, 0])
    ax_ci = fig.add_subplot(grid[0, 1], sharey=ax_resolution)

    fig.suptitle(
        "Four benchmarks are resolution-consistent; VISam remains earlier",
        x=0.075, y=0.94, ha="left", fontsize=25, fontweight="bold",
        fontfamily="serif",
    )
    legend_handles = [
        Line2D([], [], color=SKY, lw=9, alpha=0.35,
               label="Unresolved 10-ms bracket"),
        Line2D([], [], color=PURPLE, lw=3, alpha=0.65,
               label="95% insertion CI"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=BLUE,
               markeredgecolor="white", markersize=9, label="Notebook"),
        Line2D([], [], marker="D", linestyle="", markerfacecolor="white",
               markeredgecolor=PURPLE, markersize=8, label="Paper"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper left", bbox_to_anchor=(0.075, 0.865),
        ncol=4, fontsize=11, handlelength=2.0, columnspacing=1.8,
    )

    for pos, (region, row) in enumerate(data.iterrows()):
        if region == "VISam":
            ax_resolution.axhspan(pos - 0.38, pos + 0.38,
                                  color="#FBECEC", zorder=0)
            ax_ci.axhspan(pos - 0.38, pos + 0.38,
                          color="#FBECEC", zorder=0)

        ax_resolution.plot(
            [row["gap_low_ms"], row["gap_high_ms"]], [pos, pos],
            color=SKY, lw=16, alpha=0.28, solid_capstyle="round", zorder=1,
        )
        ax_resolution.plot(
            [row["gap_low_ms"], row["gap_high_ms"]], [pos, pos],
            color=BLUE, lw=2.3, solid_capstyle="round", zorder=2,
        )
        ax_resolution.scatter(
            row["latency_ms"], pos, s=105, color=BLUE,
            edgecolor="white", linewidth=1.0, zorder=4,
        )
        paper_colour = VERMILLION if region == "VISam" else PURPLE
        ax_resolution.scatter(
            row["paper_latency_ms"], pos, marker="D", s=88,
            facecolor="white", edgecolor=paper_colour, linewidth=2.0, zorder=5,
        )
        ax_resolution.text(
            row["latency_ms"], pos - 0.23, f"{row['latency_ms']:.0f}",
            ha="center", va="bottom", fontsize=9.5, color=BLUE,
            fontweight="bold",
        )
        ax_resolution.text(
            row["paper_latency_ms"], pos + 0.23,
            f"{row['paper_latency_ms']:.0f}", ha="center", va="top",
            fontsize=9.5, color=paper_colour, fontweight="bold",
        )

        low = row["latency_ci_low_ms"]
        high = row["latency_ci_high_ms"]
        ax_ci.plot(
            [low, high], [pos, pos], color=PURPLE, lw=6.0, alpha=0.22,
            solid_capstyle="round", zorder=1,
        )
        ax_ci.plot(
            [low, high], [pos, pos], color=PURPLE, lw=1.9,
            solid_capstyle="round", zorder=2,
        )
        ax_ci.plot(
            [low, high], [pos, pos], marker="|", color=PURPLE,
            markersize=8, lw=0, zorder=2,
        )
        ax_ci.scatter(
            row["latency_ms"], pos, s=105, color=BLUE,
            edgecolor="white", linewidth=1.0, zorder=4,
        )
        ax_ci.scatter(
            row["paper_latency_ms"], pos, marker="D", s=88,
            facecolor="white", edgecolor=paper_colour, linewidth=2.0, zorder=5,
        )

    ax_resolution.set_yticks(y)
    ax_resolution.set_yticklabels(order, fontweight="bold")
    ax_resolution.invert_yaxis()
    ax_resolution.set_xlim(25, 85)
    ax_resolution.set_xticks(np.arange(30, 86, 10))
    ax_resolution.set_xlabel("Latency after stimulus onset (ms)")
    ax_resolution.set_title(
        "A  Temporal sampling   |   4 / 5", loc="left", pad=16,
        color=BLUE,
    )
    ax_resolution.spines[["top", "right", "left"]].set_visible(False)
    ax_resolution.tick_params(axis="y", length=0, pad=8)
    ax_resolution.grid(axis="x", color="#E5E9ED", lw=0.8)

    ax_ci.set_xlim(25, 130)
    ax_ci.set_xticks(np.arange(30, 131, 20))
    ax_ci.set_xlabel("Latency after stimulus onset (ms)")
    ax_ci.set_title(
        "B  Insertion bootstrap   |   1 / 5", loc="left", pad=16,
        color=PURPLE,
    )
    ax_ci.spines[["top", "right", "left"]].set_visible(False)
    ax_ci.tick_params(axis="y", left=False, labelleft=False)
    ax_ci.grid(axis="x", color="#E5E9ED", lw=0.8)
    return _save(fig, output_dir, "Q3_benchmark_agreement_condensed")


def build_publication_figures(output_dir):
    output_dir = Path(output_dir)
    _set_style()
    final_regions = pd.read_csv(output_dir / "Q3_final_regional_results.csv")
    curve_table = pd.read_csv(output_dir / "Q3_paper_region_distance_curves.csv")
    sensitivity = pd.read_csv(output_dir / "Q3_latency_sensitivity_checks.csv")

    outputs = []
    chronological = plot_chronological_latency(final_regions, output_dir)
    grouped = plot_grouped_latency(final_regions, output_dir)
    curves = plot_distance_small_multiples(curve_table, final_regions, output_dir)
    bootstrap = plot_bootstrap_comparison(final_regions, output_dir)
    responsive = plot_responsive_lollipop(final_regions, output_dir)
    sensitivity_figure = plot_sensitivity_decision(sensitivity, output_dir)
    sensitivity_condensed = plot_sensitivity_decision_condensed(
        sensitivity, output_dir
    )
    interpolation_figure = plot_interpolation_resolution(
        sensitivity, curve_table, output_dir
    )
    benchmark_figure = plot_benchmark_agreement_visual(
        final_regions, sensitivity, output_dir
    )
    benchmark_condensed = plot_benchmark_agreement_condensed(
        final_regions, sensitivity, output_dir
    )
    outputs.extend(chronological)
    outputs.extend(grouped)
    outputs.extend(curves)
    outputs.extend(bootstrap)
    outputs.extend(responsive)
    outputs.extend(sensitivity_figure)
    outputs.extend(sensitivity_condensed)
    outputs.extend(interpolation_figure)
    outputs.extend(benchmark_figure)
    outputs.extend(benchmark_condensed)

    # Preserve the notebook's historical filenames so existing slides and
    # downstream references automatically receive the redesigned figures.
    aliases = {
        chronological[0]: output_dir / "Q3_paper_focused_propagation.png",
        chronological[1]: output_dir / "Q3_paper_focused_propagation.pdf",
        curves[0]: output_dir / "Q3_distance_curve_comparison.png",
        curves[1]: output_dir / "Q3_distance_curve_comparison.pdf",
        responsive[0]: output_dir / "Q3_responsive_unit_summary_by_region.png",
        responsive[1]: output_dir / "Q3_responsive_unit_summary_by_region.pdf",
    }
    for source, destination in aliases.items():
        shutil.copyfile(source, destination)
        outputs.append(destination)
    return outputs


if __name__ == "__main__":
    default_output = Path(__file__).resolve().parent / "outputs"
    for path in build_publication_figures(default_output):
        print(path)

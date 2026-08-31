# -*- coding: utf-8 -*-
"""Create the publication-ready Figure 7 cross-view interpretation matrix.

The figure is generated from the completed Credal-PAG cross-view table. No
plotted value or cross-view classification is hard-coded. The script validates
the class-feature support, reconstructs every cross-view pattern from the
Credal maximal-set and PAG null-reference indicators, and then exports editable
SVG/PDF files together with a high-resolution PNG.

Recommended command
-------------------
python make_figure7_cross_view_interpretation.py

Optional controls
-----------------
python make_figure7_cross_view_interpretation.py --font-scale 1.15 --dpi 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


# Increase this value to enlarge every text element, for example 1.15.
DEFAULT_FONT_SCALE = 1.50
FONT_SCALE = DEFAULT_FONT_SCALE

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "results" / "pag"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "generated" / "figure7"
INPUT_FILE = "credal_pag_cross_view_matrix.csv"

CLASS_ORDER = ["No injury", "Minor injury", "Moderate-plus injury"]

# The order follows the result narrative: cross-class anchors first, followed
# by class-specific and structure-only signals, then features prominent in
# neither channel under the current design.
FEATURE_ORDER = [
    "PostedSpeedLimit_Group",
    "CPMovement_Group",
    "RoadwayType_Group",
    "SVMovement_Group",
    "VehicleAge_Group",
    "CrashWith_Group",
    "RoadSurface_Group",
    "Incident_Month",
    "Mileage_Group",
    "Incident_Hour_Group",
    "Lighting_Group",
    "RoadCondition_Group",
    "Weather_Group",
    "Is_Weekend",
]

FEATURE_LABELS = {
    "Incident_Month": "Month",
    "Is_Weekend": "Weekend",
    "RoadwayType_Group": "Roadway type",
    "RoadSurface_Group": "Road surface",
    "RoadCondition_Group": "Road condition",
    "Lighting_Group": "Lighting",
    "Weather_Group": "Weather",
    "CrashWith_Group": "Crash counterpart",
    "CPMovement_Group": "Crash-partner movement",
    "SVMovement_Group": "Subject-vehicle movement",
    "Mileage_Group": "Mileage",
    "VehicleAge_Group": "Vehicle age",
    "PostedSpeedLimit_Group": "Posted speed limit",
    "Incident_Hour_Group": "Incident hour",
}

# Palette shared with Figure 6. Color and shape both carry meaning.
TEXT = "#000000"
SECONDARY = "#000000"
NAVY = "#294F78"
TEAL = "#3F8C87"
TERRACOTTA = "#C97643"
SLATE = "#8196AA"
PALE = "#C8D0D8"
GRID = "#DDE3E8"
ROW_SHADE = "#F6F8F9"
WHITE = "#FFFFFF"

PATTERN_DIRECT = "direct_correspondence"
PATTERN_TWO_STEP = "two_step_correspondence"
PATTERN_PREDICTIVE_ONLY = "predictive_only"
PATTERN_DIRECT_ONLY = "direct_proximity_only"
PATTERN_TWO_STEP_ONLY = "two_step_proximity_only"
PATTERN_NEITHER = "neither"

PATTERN_STYLE = {
    PATTERN_DIRECT: dict(marker="o", facecolor=NAVY, edgecolor=NAVY, size=145, linewidth=1.3),
    PATTERN_TWO_STEP: dict(marker="s", facecolor=TEAL, edgecolor=TEAL, size=132, linewidth=1.3),
    PATTERN_PREDICTIVE_ONLY: dict(
        marker="D", facecolor=TERRACOTTA, edgecolor=TERRACOTTA, size=122, linewidth=1.2
    ),
    PATTERN_DIRECT_ONLY: dict(marker="o", facecolor=WHITE, edgecolor=NAVY, size=145, linewidth=2.2),
    PATTERN_TWO_STEP_ONLY: dict(marker="s", facecolor=WHITE, edgecolor=TEAL, size=132, linewidth=2.2),
    PATTERN_NEITHER: dict(marker="o", facecolor=PALE, edgecolor=PALE, size=18, linewidth=0.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing credal_pag_cross_view_matrix.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory receiving Figure 7 outputs.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=DEFAULT_FONT_SCALE,
        help="Multiplier applied to all text sizes, for example 1.15.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def scaled(value: float) -> float:
    return value * FONT_SCALE


def configure_style(font_scale: float) -> None:
    global FONT_SCALE
    if font_scale <= 0:
        raise ValueError("--font-scale must be positive.")
    FONT_SCALE = font_scale
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12.5 * font_scale,
            "axes.labelsize": 13.2 * font_scale,
            "xtick.labelsize": 11.7 * font_scale,
            "ytick.labelsize": 12.1 * font_scale,
            "legend.fontsize": 10.7 * font_scale,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def as_binary(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise").astype(int)
    if not numeric.isin([0, 1]).all():
        raise ValueError(f"{name} must contain only 0 and 1.")
    return numeric


def derive_pattern(row: pd.Series) -> str:
    maximal = bool(row["in_maximal_set"])
    direct = bool(row["observed_direct_above_null_q95_descriptive"])
    within_two = bool(row["observed_within_two_steps_above_null_q95_descriptive"])

    if direct and not within_two:
        raise ValueError(
            "Direct outcome adjacency must also satisfy within-two-step proximity: "
            f"class={row['class_name']}, feature={row['feature']}"
        )
    if maximal and direct:
        return PATTERN_DIRECT
    if maximal and within_two:
        return PATTERN_TWO_STEP
    if maximal:
        return PATTERN_PREDICTIVE_ONLY
    if direct:
        return PATTERN_DIRECT_ONLY
    if within_two:
        return PATTERN_TWO_STEP_ONLY
    return PATTERN_NEITHER


def validate_reported_pattern(frame: pd.DataFrame) -> None:
    expected = {
        PATTERN_DIRECT: "predictive_structural_direct_correspondence",
        PATTERN_TWO_STEP: "predictive_prominence_with_structural_proximity",
        PATTERN_PREDICTIVE_ONLY: "predictive_channel_prominence_only_in_current_design",
        PATTERN_DIRECT_ONLY: "structural_proximity_without_credal_maximality",
        PATTERN_TWO_STEP_ONLY: "structural_proximity_without_credal_maximality",
        PATTERN_NEITHER: "neither_channel_prominent_in_current_design",
    }
    reported = frame["descriptive_cross_view_pattern"].astype(str)
    reconstructed = frame["plot_pattern"].map(expected)
    mismatch = frame.loc[reported.ne(reconstructed), ["class_name", "feature"]].copy()
    if not mismatch.empty:
        mismatch["reported"] = reported.loc[mismatch.index]
        mismatch["reconstructed"] = reconstructed.loc[mismatch.index]
        raise ValueError(
            "Reconstructed cross-view patterns disagree with the saved results:\n"
            + mismatch.to_string(index=False)
        )


def load_and_validate(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = data_dir / INPUT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Required cross-view result table not found: {path}")

    frame = pd.read_csv(path)
    required = [
        "class_name",
        "feature",
        "in_maximal_set",
        "observed_direct_frequency",
        "observed_exact_two_step_frequency",
        "observed_within_two_steps_frequency",
        "observed_direct_above_null_q95_descriptive",
        "observed_exact_two_step_above_null_q95_descriptive",
        "observed_within_two_steps_above_null_q95_descriptive",
        "n_observed_designs",
        "n_outer_partitions",
        "n_settings_per_partition",
        "n_null_replicates",
        "descriptive_cross_view_pattern",
    ]
    require_columns(frame, required)

    binary_columns = [
        "in_maximal_set",
        "observed_direct_above_null_q95_descriptive",
        "observed_exact_two_step_above_null_q95_descriptive",
        "observed_within_two_steps_above_null_q95_descriptive",
    ]
    for column in binary_columns:
        frame[column] = as_binary(frame[column], column)

    numeric_columns = [
        "observed_direct_frequency",
        "observed_exact_two_step_frequency",
        "observed_within_two_steps_frequency",
        "n_observed_designs",
        "n_outer_partitions",
        "n_settings_per_partition",
        "n_null_replicates",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    if frame.duplicated(["class_name", "feature"]).any():
        duplicate = frame.loc[
            frame.duplicated(["class_name", "feature"], keep=False),
            ["class_name", "feature"],
        ]
        raise ValueError("Duplicate class-feature rows:\n" + duplicate.to_string(index=False))

    if set(frame["class_name"]) != set(CLASS_ORDER):
        raise ValueError(f"Unexpected class support: {sorted(frame['class_name'].unique())}")
    if set(frame["feature"]) != set(FEATURE_ORDER):
        missing = sorted(set(FEATURE_ORDER) - set(frame["feature"]))
        extra = sorted(set(frame["feature"]) - set(FEATURE_ORDER))
        raise ValueError(f"Feature support mismatch. Missing={missing}; extra={extra}")

    counts = frame.groupby("class_name")["feature"].nunique()
    if not counts.eq(len(FEATURE_ORDER)).all() or len(frame) != len(CLASS_ORDER) * len(FEATURE_ORDER):
        raise ValueError(f"Incomplete class-feature support: {counts.to_dict()}")

    design_counts = frame["n_observed_designs"].unique()
    partition_counts = frame["n_outer_partitions"].unique()
    setting_counts = frame["n_settings_per_partition"].unique()
    null_counts = frame["n_null_replicates"].unique()
    if len(design_counts) != 1 or len(partition_counts) != 1 or len(setting_counts) != 1:
        raise ValueError("Observed structural design counts are not constant across rows.")
    if not np.isclose(design_counts[0], partition_counts[0] * setting_counts[0]):
        raise ValueError("n_observed_designs does not equal outer partitions times FCI settings.")
    if len(null_counts) != 1:
        raise ValueError("The number of label-null replicates is not constant across rows.")

    structural_columns = [
        "observed_direct_frequency",
        "observed_exact_two_step_frequency",
        "observed_within_two_steps_frequency",
        "observed_direct_above_null_q95_descriptive",
        "observed_exact_two_step_above_null_q95_descriptive",
        "observed_within_two_steps_above_null_q95_descriptive",
        "n_observed_designs",
    ]
    for feature, group in frame.groupby("feature", sort=False):
        for column in structural_columns:
            values = group[column].to_numpy(dtype=float)
            if not np.allclose(values, values[0], atol=1e-12, rtol=0.0):
                raise ValueError(
                    f"Outcome-level PAG quantity varies by injury class: feature={feature}, column={column}"
                )

    frame["plot_pattern"] = frame.apply(derive_pattern, axis=1)
    validate_reported_pattern(frame)

    matrix = (
        frame.set_index(["feature", "class_name"])
        .loc[pd.MultiIndex.from_product([FEATURE_ORDER, CLASS_ORDER])]
        .reset_index()
    )
    structural = (
        frame.drop_duplicates("feature")
        .set_index("feature")
        .loc[FEATURE_ORDER]
        .reset_index()
    )

    summary_rows: list[dict[str, object]] = []
    for class_name in CLASS_ORDER:
        subset = frame.loc[frame["class_name"].eq(class_name)]
        maximal = subset["in_maximal_set"].eq(1)
        structural_prominence = subset["observed_within_two_steps_above_null_q95_descriptive"].eq(1)
        summary_rows.append(
            {
                "class_name": class_name,
                "n_maximal": int(maximal.sum()),
                "n_maximal_with_pag_signal": int((maximal & structural_prominence).sum()),
                "alignment_fraction": float((maximal & structural_prominence).sum() / maximal.sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return matrix, structural, summary


def add_row_guides(ax: plt.Axes, n_rows: int) -> None:
    for row in range(n_rows):
        if row % 2 == 1:
            ax.axhspan(row - 0.5, row + 0.5, color=ROW_SHADE, zorder=-3)
    for boundary in np.arange(-0.5, n_rows, 1.0):
        ax.axhline(boundary, color=GRID, linewidth=0.65, zorder=-2)


def plot_cross_view_matrix(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    n_rows = len(FEATURE_ORDER)
    x_positions = {name: index for index, name in enumerate(CLASS_ORDER)}
    y_positions = {feature: n_rows - 1 - index for index, feature in enumerate(FEATURE_ORDER)}
    add_row_guides(ax, n_rows)

    for row in matrix.itertuples(index=False):
        style = PATTERN_STYLE[row.plot_pattern]
        ax.scatter(
            x_positions[row.class_name],
            y_positions[row.feature],
            s=style["size"] * FONT_SCALE**1.25,
            marker=style["marker"],
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            linewidth=style["linewidth"],
            zorder=4,
        )

    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-0.65, n_rows - 0.35)
    ax.set_yticks(
        [y_positions[feature] for feature in FEATURE_ORDER],
        [FEATURE_LABELS[feature] for feature in FEATURE_ORDER],
    )
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0, pad=scaled(7))

    ax.text(
        0.5,
        1.205,
        "Class-specific cross-view\nclassification",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=scaled(13.2),
        fontweight="bold",
        color=TEXT,
        linespacing=0.92,
    )
    for class_name, x in x_positions.items():
        row = summary.loc[summary["class_name"].eq(class_name)].iloc[0]
        label = f"PAG signal: {int(row.n_maximal_with_pag_signal)}/{int(row.n_maximal)}"
        ax.text(
            x,
            1.120,
            class_name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=scaled(10.4),
            color=TEXT,
            clip_on=False,
        )
        ax.text(
            x,
            1.045,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=scaled(10.4),
            color=SECONDARY,
            clip_on=False,
        )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(-0.5, color=GRID, linewidth=0.8, zorder=-1)
    for boundary in [0.5, 1.5, 2.5]:
        ax.axvline(boundary, color=GRID, linewidth=0.8, zorder=-1)


def count_label(frequency: float, n_designs: int) -> str:
    count = int(round(frequency * n_designs))
    return f"{count}/{n_designs}"


def plot_structural_frequency(ax: plt.Axes, structural: pd.DataFrame) -> None:
    n_rows = len(FEATURE_ORDER)
    y_positions = {feature: n_rows - 1 - index for index, feature in enumerate(FEATURE_ORDER)}
    add_row_guides(ax, n_rows)
    max_frequency = max(
        float(structural["observed_direct_frequency"].max()),
        float(structural["observed_exact_two_step_frequency"].max()),
    )
    x_max = max(0.30, max_frequency + 0.035)

    for row in structural.itertuples(index=False):
        y = y_positions[row.feature]
        n_designs = int(round(row.n_observed_designs))
        direct = float(row.observed_direct_frequency)
        exact_two = float(row.observed_exact_two_step_frequency)

        if direct > 0:
            ax.hlines(
                y + 0.15,
                0,
                direct,
                color=NAVY,
                linewidth=1.25,
                alpha=0.28,
                zorder=1,
            )
            ax.scatter(
                direct,
                y + 0.15,
                s=82 * FONT_SCALE**1.15,
                marker="o",
                facecolor=NAVY,
                edgecolor=WHITE,
                linewidth=0.8,
                zorder=4,
            )
            ax.text(
                direct + 0.007,
                y + 0.15,
                count_label(direct, n_designs),
                ha="left",
                va="center",
                fontsize=scaled(9.6),
                color=TEXT,
            )
        if exact_two > 0:
            ax.hlines(
                y - 0.15,
                0,
                exact_two,
                color=TEAL,
                linewidth=1.25,
                alpha=0.28,
                zorder=1,
            )
            ax.scatter(
                exact_two,
                y - 0.15,
                s=76 * FONT_SCALE**1.15,
                marker="s",
                facecolor=TEAL,
                edgecolor=WHITE,
                linewidth=0.8,
                zorder=4,
            )
            ax.text(
                exact_two + 0.007,
                y - 0.15,
                count_label(exact_two, n_designs),
                ha="left",
                va="center",
                fontsize=scaled(9.6),
                color=TEXT,
            )

    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.65, n_rows - 0.35)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xticks(np.arange(0.0, x_max + 0.001, 0.05))
    ax.grid(axis="x", color=GRID, linewidth=0.75, zorder=-2)
    ax.set_axisbelow(True)
    ax.text(
        0.5,
        1.205,
        "PAG proximity frequency\nacross 45 specifications",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=scaled(13.2),
        fontweight="bold",
        color=TEXT,
        linespacing=0.92,
    )
    ax.text(
        0.5,
        1.120,
        "Outcome-level structural context",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=scaled(10.0),
        color=SECONDARY,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

def frequency_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=scaled(7.5),
            markerfacecolor=NAVY,
            markeredgecolor=WHITE,
            label="Direct adjacency",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=scaled(7.2),
            markerfacecolor=TEAL,
            markeredgecolor=WHITE,
            label="Exact two-step",
        ),
    ]


def cross_view_legend_handles() -> list[Line2D]:
    definitions = [
        (PATTERN_DIRECT, "Direct correspondence"),
        (PATTERN_TWO_STEP, "Two-step correspondence"),
        (PATTERN_PREDICTIVE_ONLY, "Predictive prominence only"),
        (PATTERN_DIRECT_ONLY, "Direct proximity only"),
        (PATTERN_TWO_STEP_ONLY, "Two-step proximity only"),
        (PATTERN_NEITHER, "Neither channel"),
    ]
    handles: list[Line2D] = []
    for pattern, label in definitions:
        style = PATTERN_STYLE[pattern]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                linestyle="None",
                markersize=scaled(7.8 if pattern != PATTERN_NEITHER else 4.0),
                markerfacecolor=style["facecolor"],
                markeredgecolor=style["edgecolor"],
                markeredgewidth=style["linewidth"],
                label=label,
            )
        )
    return handles


def save_figure(fig: plt.Figure, output_stem: Path, dpi: int) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    options = {"bbox_inches": "tight", "pad_inches": 0.08, "facecolor": WHITE}
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, **options)
    fig.savefig(output_stem.with_suffix(".pdf"), **options)
    fig.savefig(output_stem.with_suffix(".svg"), **options)


def create_figure(
    matrix: pd.DataFrame,
    structural: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    # Grow the canvas moderately at large font scales so labels and legends
    # remain separated without undoing the requested type enlargement.
    growth = max(0.0, min(FONT_SCALE, 1.6) - 1.0)
    fig = plt.figure(
        figsize=(
            11.2 * (1.0 + 0.48 * growth),
            7.9 * (1.0 + 0.30 * growth),
        )
    )
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.10, 1.08],
        left=0.235,
        right=0.985,
        top=0.775,
        bottom=0.235,
        wspace=0.085,
    )
    ax_matrix = fig.add_subplot(grid[0, 0])
    ax_structural = fig.add_subplot(grid[0, 1])

    plot_cross_view_matrix(ax_matrix, matrix, summary)
    plot_structural_frequency(ax_structural, structural)

    fig.legend(
        handles=cross_view_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.315, 0.045),
        ncol=2,
        frameon=False,
        handletextpad=0.55,
        columnspacing=1.25,
        labelspacing=0.75,
    )
    fig.legend(
        handles=frequency_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.79, 0.125),
        ncol=2,
        frameon=False,
        handletextpad=0.55,
        columnspacing=1.35,
    )

    save_figure(fig, output_dir / "figure7_cross_view_patterns", dpi)
    plt.close(fig)


def export_supporting_files(
    matrix: pd.DataFrame,
    structural: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_columns = [
        "class_name",
        "feature",
        "in_maximal_set",
        "observed_direct_frequency",
        "observed_exact_two_step_frequency",
        "observed_within_two_steps_frequency",
        "observed_direct_above_null_q95_descriptive",
        "observed_exact_two_step_above_null_q95_descriptive",
        "observed_within_two_steps_above_null_q95_descriptive",
        "plot_pattern",
    ]
    matrix[plot_columns].to_csv(output_dir / "figure7_plot_data.csv", index=False)
    summary.to_csv(output_dir / "figure7_class_summary.csv", index=False)

    n_designs = int(round(structural["n_observed_designs"].iloc[0]))
    n_null = int(round(structural["n_null_replicates"].iloc[0]))
    caption = (
        "Fig. 7. Cross-view patterns of predictive relevance and PAG proximity. "
        "Filled symbols identify Credal maximal-set members: circles denote direct outcome "
        "adjacency, squares denote exact-two-step proximity without direct adjacency, and "
        "diamonds denote predictive prominence without a PAG signal above its feature-specific "
        "label-permutation 95th percentile. Open symbols mark PAG proximity outside the "
        "class-specific maximal set; small gray points indicate prominence in neither channel. "
        f"The right-hand display reports observed frequencies across {n_designs} outer-partition and "
        f"FCI-specification combinations; structural null references used {n_null} label "
        "permutations per specification."
    )
    (output_dir / "figure7_caption.txt").write_text(caption + "\n", encoding="utf-8")

    readme = f"""Figure 7 output guide

Primary manuscript figure
  figure7_cross_view_patterns.png
  figure7_cross_view_patterns.pdf
  figure7_cross_view_patterns.svg

Supporting files
  figure7_plot_data.csv
  figure7_class_summary.csv
  figure7_caption.txt

Input
  {INPUT_FILE}

Font control
  Run with --font-scale 1.15, or edit DEFAULT_FONT_SCALE near the top of the script.

Export contract
  PNG is exported at the requested DPI (default 600).
  PDF and SVG preserve vector geometry; SVG text remains editable.
  No values or cross-view classifications are hard-coded in the plot.
"""
    (output_dir / "README_figure7.txt").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style(args.font_scale)
    matrix, structural, summary = load_and_validate(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    create_figure(matrix, structural, summary, args.output_dir, args.dpi)
    export_supporting_files(matrix, structural, summary, args.output_dir)

    print("Figure 7 generated successfully.")
    print(f"Input: {args.data_dir / INPUT_FILE}")
    print(f"Output: {args.output_dir}")
    print("Class-level alignment:")
    for row in summary.itertuples(index=False):
        print(
            f"  {row.class_name}: {row.n_maximal_with_pag_signal}/"
            f"{row.n_maximal} maximal features with a PAG signal"
        )


if __name__ == "__main__":
    main()

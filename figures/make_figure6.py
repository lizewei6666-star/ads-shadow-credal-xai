# -*- coding: utf-8 -*-
"""Create publication-ready Figure 6 panels and design alternatives.

The script reads the completed PAG result tables. It never hard-codes plotted
values. Separate manuscript panels contain no panel letters or in-panel titles.
All text remains editable in the SVG and PDF outputs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache")
)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# User controls
# ---------------------------------------------------------------------------

# Increase this value to enlarge every text element. For example, use 1.15.
DEFAULT_FONT_SCALE = 1.00
FONT_SCALE = DEFAULT_FONT_SCALE

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "results" / "pag"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated" / "figure6"

GLOBAL_FILE = "global_outcome_node_structural_signal.csv"
FEATURE_FILE = "pag_multiscale_structural_compatibility_by_feature.csv"
BRIDGE_FILE = "recurrent_two_step_skeleton_bridges.csv"

N_DESIGNS = 45
N_NULL_REPLICATES = 99


# ---------------------------------------------------------------------------
# Typography and restrained, colorblind-readable palettes
# ---------------------------------------------------------------------------

TEXT = "#20252A"
NAVY = "#244A73"
TEAL = "#4F8883"
TERRACOTTA = "#C77645"
SLATE = "#718397"
PALE_TEAL = "#ADC9C6"
PALE_BLUE = "#D8E2EA"
PALE_GRAY = "#D4D9DF"
LIGHTEST_GRAY = "#EEF1F4"
MID_GRAY = "#717980"
GRID = "#E0E5EA"

BRIDGE_COLORS = {
    "Crash counterpart": "#3F7D78",
    "Posted speed limit": "#C77645",
    "Crash-partner movement": "#6F86A0",
}

FEATURE_LABELS = {
    "CPMovement_Group": "Crash-partner movement",
    "CrashWith_Group": "Crash counterpart",
    "Incident_Hour_Group": "Incident hour",
    "Incident_Month": "Month",
    "Is_Weekend": "Weekend",
    "Lighting_Group": "Lighting",
    "Mileage_Group": "Mileage",
    "PostedSpeedLimit_Group": "Posted speed limit",
    "RoadCondition_Group": "Road condition",
    "RoadSurface_Group": "Road surface",
    "RoadwayType_Group": "Roadway type",
    "SVMovement_Group": "Subject-vehicle movement",
    "VehicleAge_Group": "Vehicle age",
    "Weather_Group": "Weather",
}

GLOBAL_LABELS = {
    "mean_target_degree": "Mean outcome-node degree",
    "fraction_designs_with_any_neighbor": "PAGs with outcome adjacency",
    "mean_fraction_features_within_two_steps": "Predictors within two steps",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--font-scale",
        type=float,
        default=DEFAULT_FONT_SCALE,
        help="Multiply all font sizes by this value, for example 1.15.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def configure_style(font_scale: float) -> None:
    """Apply one typography contract to every output."""
    global FONT_SCALE
    FONT_SCALE = font_scale
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 14.0 * font_scale,
            "axes.labelsize": 16.0 * font_scale,
            "xtick.labelsize": 14.0 * font_scale,
            "ytick.labelsize": 14.0 * font_scale,
            "legend.fontsize": 13.0 * font_scale,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.7,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def scaled(size: float) -> float:
    """Scale explicit annotation sizes with the global font control."""
    return size * FONT_SCALE


def require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def load_results(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "global": data_dir / GLOBAL_FILE,
        "feature": data_dir / FEATURE_FILE,
        "bridge": data_dir / BRIDGE_FILE,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required PAG result table not found: {path}")

    global_df = pd.read_csv(paths["global"])
    feature_df = pd.read_csv(paths["feature"])
    bridge_df = pd.read_csv(paths["bridge"])

    require_columns(
        global_df,
        {
            "metric",
            "observed_value",
            "null_median",
            "null_q95",
            "n_observed_designs",
            "n_null_replicates",
        },
        paths["global"],
    )
    require_columns(
        feature_df,
        {
            "feature",
            "observed_direct_frequency",
            "observed_exact_two_step_frequency",
            "observed_within_two_steps_frequency",
            "n_observed_designs",
        },
        paths["feature"],
    )
    require_columns(
        bridge_df,
        {
            "feature",
            "bridge_node",
            "observed_design_count",
            "observed_bridge_frequency",
            "null_bridge_q95",
            "n_observed_designs",
        },
        paths["bridge"],
    )

    design_values = set(pd.to_numeric(feature_df["n_observed_designs"]).astype(int))
    bridge_design_values = set(
        pd.to_numeric(bridge_df["n_observed_designs"]).astype(int)
    )
    null_values = set(pd.to_numeric(global_df["n_null_replicates"]).astype(int))
    if design_values != {N_DESIGNS} or bridge_design_values != {N_DESIGNS}:
        raise ValueError(
            f"Expected {N_DESIGNS} PAG specifications; "
            f"found feature={design_values}, bridge={bridge_design_values}."
        )
    if null_values != {N_NULL_REPLICATES}:
        raise ValueError(
            f"Expected {N_NULL_REPLICATES} label-null replicates; found {null_values}."
        )

    for column in ["observed_value", "null_median", "null_q95"]:
        global_df[column] = pd.to_numeric(global_df[column], errors="raise")
    for column in [
        "observed_direct_frequency",
        "observed_exact_two_step_frequency",
        "observed_within_two_steps_frequency",
    ]:
        feature_df[column] = pd.to_numeric(feature_df[column], errors="raise")
    for column in [
        "observed_design_count",
        "observed_bridge_frequency",
        "null_bridge_q95",
    ]:
        bridge_df[column] = pd.to_numeric(bridge_df[column], errors="raise")

    return global_df, feature_df, bridge_df


def save_figure(fig: plt.Figure, output_stem: Path, dpi: int) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    options = {"bbox_inches": "tight", "pad_inches": 0.06, "facecolor": "white"}
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, **options)
    fig.savefig(output_stem.with_suffix(".pdf"), **options)
    fig.savefig(output_stem.with_suffix(".svg"), **options)


def clean_axis(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def prepare_global_data(global_df: pd.DataFrame) -> pd.DataFrame:
    order = list(GLOBAL_LABELS)
    data = global_df.set_index("metric").loc[order].reset_index()
    data["label"] = data["metric"].map(GLOBAL_LABELS)
    return data


def prepare_feature_data(feature_df: pd.DataFrame) -> pd.DataFrame:
    data = feature_df.loc[
        feature_df["observed_within_two_steps_frequency"] > 0
    ].copy()
    data["label"] = data["feature"].map(FEATURE_LABELS)
    if data["label"].isna().any():
        unknown = data.loc[data["label"].isna(), "feature"].tolist()
        raise ValueError(f"Missing feature labels: {unknown}")
    return data.sort_values(
        ["observed_within_two_steps_frequency", "observed_direct_frequency", "label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def prepare_bridge_data(bridge_df: pd.DataFrame) -> pd.DataFrame:
    # The completed analysis used three occurrences as the display threshold.
    data = bridge_df.loc[bridge_df["observed_design_count"] >= 3].copy()
    data["feature_label"] = data["feature"].map(FEATURE_LABELS)
    data["bridge_label"] = data["bridge_node"].map(FEATURE_LABELS)
    if data[["feature_label", "bridge_label"]].isna().any().any():
        raise ValueError("One or more bridge labels are missing.")
    data["observed_design_count"] = data["observed_design_count"].astype(int)
    return data


# ---------------------------------------------------------------------------
# Figure 6a alternatives
# ---------------------------------------------------------------------------

def plot_6a_null_benchmark(ax: plt.Axes, global_df: pd.DataFrame) -> None:
    """Recommended design: null envelope, exceedance span, observed marker."""
    data = prepare_global_data(global_df)
    y = np.arange(len(data))[::-1]

    for yi, row in zip(y, data.itertuples(index=False)):
        ax.hlines(yi, 0, 0.40, color=LIGHTEST_GRAY, linewidth=1.1, zorder=0)
        ax.hlines(
            yi,
            row.null_median,
            row.null_q95,
            color=PALE_GRAY,
            linewidth=13.0,
            zorder=1,
        )
        ax.hlines(
            yi,
            row.null_q95,
            row.observed_value,
            color=TEAL,
            linewidth=2.0,
            zorder=2,
        )
        ax.plot(
            row.null_median,
            yi,
            marker="|",
            markersize=13,
            markeredgewidth=2.0,
            color=MID_GRAY,
            zorder=3,
        )
        ax.plot(
            row.null_q95,
            yi,
            marker="s",
            markersize=6.8,
            markeredgewidth=1.2,
            markerfacecolor="white",
            markeredgecolor=TERRACOTTA,
            zorder=4,
        )
        ax.scatter(
            row.observed_value,
            yi,
            s=105,
            marker="D",
            facecolor=NAVY,
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
        )
        ax.text(
            row.observed_value + 0.009,
            yi,
            f"{row.observed_value:.3f}",
            ha="left",
            va="center",
            fontsize=scaled(13.0),
            color=NAVY,
        )

    ax.set_yticks(y, data["label"])
    ax.set_xlim(-0.006, 0.415)
    ax.set_xticks(np.arange(0.0, 0.41, 0.10))
    ax.set_xlabel("Value across 45 PAG specifications")
    clean_axis(ax)

    handles = [
        Line2D(
            [0],
            [0],
            color=PALE_GRAY,
            linewidth=8,
            marker="s",
            markerfacecolor="white",
            markeredgecolor=TERRACOTTA,
            markersize=5.5,
            label="Label-null median to 95th percentile",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor=NAVY,
            markeredgecolor="white",
            markersize=8,
            label="Observed",
        ),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.52, -0.20),
        ncol=2,
        handlelength=2.3,
    )


def plot_6a_tripoint_profile(ax: plt.Axes, global_df: pd.DataFrame) -> None:
    """Alternative design: explicit median, 95th percentile, and observed profile."""
    data = prepare_global_data(global_df)
    y = np.arange(len(data))[::-1]

    for yi, row in zip(y, data.itertuples(index=False)):
        values = [row.null_median, row.null_q95, row.observed_value]
        ax.plot(values, [yi] * 3, color=PALE_BLUE, linewidth=2.2, zorder=1)
        ax.scatter(values[0], yi, s=70, marker="|", color=MID_GRAY, linewidth=2.0, zorder=3)
        ax.scatter(
            values[1], yi, s=78, marker="s", facecolor="white",
            edgecolor=TERRACOTTA, linewidth=1.9, zorder=4
        )
        ax.scatter(
            values[2], yi, s=105, marker="D", facecolor=NAVY,
            edgecolor="white", linewidth=0.9, zorder=5
        )
        ax.text(
            values[2] + 0.009, yi, f"{values[2]:.3f}",
            va="center", ha="left", fontsize=scaled(13.0), color=NAVY
        )

    ax.set_yticks(y, data["label"])
    ax.set_xlim(-0.006, 0.415)
    ax.set_xticks(np.arange(0.0, 0.41, 0.10))
    ax.set_xlabel("Value across 45 PAG specifications")
    clean_axis(ax)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="|", linestyle="none", color=MID_GRAY,
                   markeredgewidth=2.0, markersize=10, label="Null median"),
            Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="white",
                   markeredgecolor=TERRACOTTA, markeredgewidth=1.8, markersize=7,
                   label="Null 95th percentile"),
            Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=NAVY,
                   markeredgecolor="white", markersize=8, label="Observed"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.52, -0.20),
        ncol=3,
    )


def plot_6a_profile_matrix(ax: plt.Axes, global_df: pd.DataFrame) -> None:
    """Alternative design: compact annotated profile matrix."""
    data = prepare_global_data(global_df)
    values = data[["null_median", "null_q95", "observed_value"]].to_numpy()
    cmap = LinearSegmentedColormap.from_list(
        "aap_blue", ["#F5F7F8", "#BFD1DC", "#244A73"]
    )
    image = ax.imshow(values, cmap=cmap, vmin=0, vmax=0.40, aspect="auto")

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            color = "white" if value >= 0.24 else TEXT
            ax.text(
                col,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=color,
                fontsize=scaled(13.3),
                fontweight="bold" if col == 2 else "normal",
            )

    ax.set_yticks(np.arange(len(data)), data["label"])
    ax.set_xticks(
        np.arange(3),
        ["Label-null\nmedian", "Label-null\n95th percentile", "Observed"],
    )
    ax.tick_params(axis="both", which="both", length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Value")
    colorbar.outline.set_linewidth(0.7)


# ---------------------------------------------------------------------------
# Figure 6b: recommended neighborhood profile
# ---------------------------------------------------------------------------

def plot_6b_pag_neighborhood(ax: plt.Axes, feature_df: pd.DataFrame) -> None:
    data = prepare_feature_data(feature_df)
    y = np.arange(len(data))[::-1]
    direct = data["observed_direct_frequency"].to_numpy()
    two_step = data["observed_exact_two_step_frequency"].to_numpy()
    total = direct + two_step

    ax.barh(
        y,
        direct,
        height=0.56,
        color=NAVY,
        edgecolor="white",
        linewidth=0.5,
        label="Direct adjacency",
    )
    ax.barh(
        y,
        two_step,
        left=direct,
        height=0.56,
        color=PALE_TEAL,
        edgecolor="white",
        linewidth=0.5,
        label="Exact two-step",
    )

    for yi, value in zip(y, total):
        count = int(round(value * N_DESIGNS))
        ax.text(
            value + 0.006,
            yi,
            f"{count}/45",
            va="center",
            ha="left",
            fontsize=scaled(13.0),
            color=TEXT,
        )

    ax.set_yticks(y, data["label"])
    ax.set_xlim(0, 0.315)
    ax.set_xticks(np.arange(0, 0.31, 0.05))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Frequency across 45 PAG specifications")
    clean_axis(ax)
    ax.legend(frameon=False, loc="lower right", ncol=2)


# ---------------------------------------------------------------------------
# Figure 6c alternatives
# ---------------------------------------------------------------------------

def plot_6c_lollipop(ax: plt.Axes, bridge_df: pd.DataFrame) -> None:
    """Original Figure 6c design, refined for a separate journal panel."""
    data = prepare_bridge_data(bridge_df).copy()
    data["display_label"] = (
        data["feature_label"] + "  |  " + data["bridge_label"]
    )
    data = data.sort_values(
        ["observed_bridge_frequency", "bridge_label", "feature_label"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    y = np.arange(len(data))[::-1]

    for yi, row in zip(y, data.itertuples(index=False)):
        color = BRIDGE_COLORS[row.bridge_label]
        ax.hlines(
            yi,
            0,
            row.observed_bridge_frequency,
            color=GRID,
            linewidth=1.7,
            zorder=1,
        )
        ax.scatter(
            row.observed_bridge_frequency,
            yi,
            s=95,
            facecolor="white",
            edgecolor=color,
            linewidth=2.4,
            zorder=3,
        )
        ax.text(
            row.observed_bridge_frequency + 0.004,
            yi,
            f"{row.observed_design_count}/45",
            va="center",
            ha="left",
            fontsize=scaled(12.8),
            color=TEXT,
        )

    ax.set_yticks(y, data["display_label"])
    ax.set_xlim(0, 0.158)
    ax.set_xticks(np.arange(0, 0.151, 0.05))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_label_position("bottom")
    ax.tick_params(axis="x", top=False, labeltop=False, bottom=True, labelbottom=True)
    ax.set_xlabel("Frequency across 45 PAG specifications")
    clean_axis(ax)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=2.0,
            markersize=8,
            label=label,
        )
        for label, color in BRIDGE_COLORS.items()
    ]
    ax.legend(
        handles=handles,
        title="Intermediate node",
        title_fontsize=scaled(13.0),
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        borderaxespad=0.0,
        handletextpad=0.6,
        labelspacing=0.45,
        columnspacing=1.25,
    )

def bridge_matrix(bridge_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    data = prepare_bridge_data(bridge_df)
    totals = (
        data.groupby("feature_label", as_index=True)["observed_design_count"]
        .sum()
        .sort_values(ascending=False)
    )
    feature_order = totals.index.tolist()
    preferred = [
        "Crash counterpart",
        "Posted speed limit",
        "Crash-partner movement",
    ]
    present = set(data["bridge_label"])
    bridge_order = [item for item in preferred if item in present]
    bridge_order.extend(sorted(present.difference(bridge_order)))
    matrix = (
        data.pivot_table(
            index="feature_label",
            columns="bridge_label",
            values="observed_design_count",
            aggfunc="max",
            fill_value=0,
        )
        .reindex(index=feature_order, columns=bridge_order, fill_value=0)
        .astype(int)
    )
    return matrix, feature_order, bridge_order


def plot_6c_bridge_matrix(ax: plt.Axes, bridge_df: pd.DataFrame) -> None:
    """Recommended design: bridge identity by column, recurrence by glyph."""
    matrix, feature_order, bridge_order = bridge_matrix(bridge_df)
    values = matrix.to_numpy()
    cmap = LinearSegmentedColormap.from_list(
        "bridge_teal", ["#E6F0EF", "#82B3AE", "#245E67"]
    )
    norm = Normalize(vmin=3, vmax=max(6, int(values.max())))

    rows, cols = values.shape
    for row in range(rows):
        for col in range(cols):
            count = int(values[row, col])
            if count <= 0:
                continue
            size = 150 + 68 * count
            color = cmap(norm(count))
            ax.scatter(
                col,
                row,
                s=size,
                facecolor=color,
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )
            ax.text(
                col,
                row,
                f"{count}",
                ha="center",
                va="center",
                color="white" if count >= 4 else TEXT,
                fontsize=scaled(13.2),
                fontweight="bold",
                zorder=4,
            )

    ax.set_xlim(-0.55, cols - 0.45)
    ax.set_ylim(rows - 0.45, -0.55)
    ax.set_yticks(np.arange(rows), feature_order)
    xlabels = [
        item.replace("Crash-partner movement", "Crash-partner\nmovement")
        for item in bridge_order
    ]
    ax.set_xticks(np.arange(cols), xlabels)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Intermediate node", labelpad=12)
    ax.set_ylabel("Focal predictor")
    ax.tick_params(axis="both", length=0, pad=9)
    for x in np.arange(-0.5, cols, 1):
        ax.axvline(x, color=GRID, linewidth=0.8, zorder=0)
    for y in np.arange(-0.5, rows, 1):
        ax.axhline(y, color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        1.0,
        -0.10,
        "Cell values are occurrences among 45 PAG specifications",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=scaled(12.5),
        color=MID_GRAY,
    )


def plot_6c_bridge_heatmap(ax: plt.Axes, bridge_df: pd.DataFrame) -> None:
    """Alternative design: restrained annotated heatmap."""
    matrix, feature_order, bridge_order = bridge_matrix(bridge_df)
    values = matrix.to_numpy(dtype=float)
    masked = np.ma.masked_where(values == 0, values)
    cmap = LinearSegmentedColormap.from_list(
        "bridge_slate", ["#E9EFF2", "#9EB5C4", "#355C7D"]
    )
    cmap.set_bad("white")
    image = ax.imshow(masked, cmap=cmap, vmin=3, vmax=max(6, values.max()), aspect="auto")

    rows, cols = values.shape
    for row in range(rows):
        for col in range(cols):
            count = int(values[row, col])
            label = f"{count}/45" if count else ""
            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=scaled(13.0),
                color="white" if count >= 5 else TEXT,
                fontweight="bold" if count else "normal",
            )

    ax.set_yticks(np.arange(rows), feature_order)
    ax.set_xticks(
        np.arange(cols),
        [item.replace("Crash-partner movement", "Crash-partner\nmovement") for item in bridge_order],
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Intermediate node", labelpad=12)
    ax.set_ylabel("Focal predictor")
    ax.tick_params(axis="both", length=0, pad=9)
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Occurrences")
    colorbar.set_ticks([3, 4, 5, 6])
    colorbar.outline.set_linewidth(0.7)


def draw_bezier(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linewidth: float,
    alpha: float,
    zorder: int = 1,
) -> None:
    x0, y0 = start
    x1, y1 = end
    span = x1 - x0
    vertices = [
        (x0, y0),
        (x0 + 0.42 * span, y0),
        (x1 - 0.42 * span, y1),
        (x1, y1),
    ]
    path = MplPath(
        vertices,
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(
        PathPatch(
            path,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor=color,
            linewidth=linewidth,
            alpha=alpha,
            capstyle="round",
            zorder=zorder,
        )
    )


def plot_6c_bridge_network(ax: plt.Axes, bridge_df: pd.DataFrame) -> None:
    """Alternative design: undirected two-step skeleton bridge map."""
    data = prepare_bridge_data(bridge_df)
    totals = (
        data.groupby("feature_label")["observed_design_count"]
        .sum()
        .sort_values(ascending=False)
    )
    features = totals.index.tolist()
    bridges = [
        item
        for item in ["Crash counterpart", "Posted speed limit", "Crash-partner movement"]
        if item in set(data["bridge_label"])
    ]

    feature_y = dict(zip(features, np.linspace(0.82, 0.16, len(features))))
    bridge_y = dict(zip(bridges, np.linspace(0.76, 0.24, len(bridges))))
    x_feature, x_bridge, x_outcome = 0.22, 0.61, 0.91

    for bridge in bridges:
        draw_bezier(
            ax,
            (x_bridge + 0.015, bridge_y[bridge]),
            (x_outcome - 0.018, 0.50),
            color="#B7C0C8",
            linewidth=1.8,
            alpha=0.8,
            zorder=1,
        )

    for row in data.itertuples(index=False):
        start = (x_feature + 0.012, feature_y[row.feature_label])
        end = (x_bridge - 0.012, bridge_y[row.bridge_label])
        color = BRIDGE_COLORS[row.bridge_label]
        linewidth = 1.1 + 0.58 * row.observed_design_count
        draw_bezier(
            ax,
            start,
            end,
            color=color,
            linewidth=linewidth,
            alpha=0.62,
            zorder=2,
        )

    for feature in features:
        y = feature_y[feature]
        ax.scatter(
            [x_feature], [y], transform=ax.transAxes, s=75,
            facecolor="white", edgecolor=TEXT, linewidth=1.2, zorder=5
        )
        ax.text(
            x_feature - 0.025, y, feature,
            transform=ax.transAxes, ha="right", va="center", fontsize=scaled(13.2)
        )

    for bridge in bridges:
        y = bridge_y[bridge]
        color = BRIDGE_COLORS[bridge]
        ax.scatter(
            [x_bridge], [y], transform=ax.transAxes, s=150,
            facecolor="white", edgecolor=color, linewidth=2.6, zorder=5
        )
        ax.text(
            x_bridge + 0.025, y, bridge,
            transform=ax.transAxes, ha="left", va="center", fontsize=scaled(13.2), color=color
        )

    ax.scatter(
        [x_outcome], [0.50], transform=ax.transAxes, s=190,
        facecolor=NAVY, edgecolor="white", linewidth=1.2, zorder=6
    )
    ax.text(
        x_outcome, 0.50 - 0.085, "Injury severity",
        transform=ax.transAxes, ha="center", va="top", fontsize=scaled(13.5), fontweight="bold"
    )

    ax.text(x_feature, 0.96, "Focal predictor", transform=ax.transAxes,
            ha="center", va="top", fontsize=scaled(14.0), fontweight="bold")
    ax.text(x_bridge, 0.96, "Intermediate node", transform=ax.transAxes,
            ha="center", va="top", fontsize=scaled(14.0), fontweight="bold")
    ax.text(x_outcome, 0.96, "Outcome", transform=ax.transAxes,
            ha="center", va="top", fontsize=scaled(14.0), fontweight="bold")

    width_handles = [
        Line2D([0], [0], color=SLATE, linewidth=1.1 + 0.58 * value,
               alpha=0.62, label=f"{value}/45")
        for value in [3, 4, 6]
    ]
    ax.legend(
        handles=width_handles,
        title="Configuration frequency",
        title_fontsize=scaled(13.0),
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.55, -0.02),
        ncol=3,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def create_single_panel(
    plotter,
    frame: pd.DataFrame,
    *,
    size: tuple[float, float],
    stem: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    plotter(ax, frame)
    save_figure(fig, stem, dpi)
    plt.close(fig)


def add_panel_letter(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.055,
        1.035,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=scaled(16.0),
        fontweight="bold",
        clip_on=False,
    )


def create_recommended_combined(
    global_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    bridge_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(8.4, 12.0), layout="constrained")
    grid = fig.add_gridspec(3, 1, height_ratios=[0.72, 1.18, 1.05], hspace=0.12)
    axes = [fig.add_subplot(grid[index, 0]) for index in range(3)]
    plot_6a_null_benchmark(axes[0], global_df)
    plot_6b_pag_neighborhood(axes[1], feature_df)
    plot_6c_bridge_matrix(axes[2], bridge_df)
    for label, ax in zip(["a", "b", "c"], axes):
        add_panel_letter(ax, label)
    save_figure(fig, output_dir / "figure6_recommended_combined", dpi)
    plt.close(fig)


def create_variant_contact_sheet(output_dir: Path, dpi: int) -> None:
    files = [
        ("figure6a_primary_null_benchmark.png", "6a: null benchmark"),
        ("figure6a_alt_tripoint_profile.png", "6a: three-point profile"),
        ("figure6a_alt_profile_matrix.png", "6a: profile matrix"),
        ("figure6c_primary_bridge_matrix.png", "6c: bridge matrix"),
        ("figure6c_alt_bridge_network.png", "6c: bridge map"),
        ("figure6c_alt_bridge_heatmap.png", "6c: annotated heatmap"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.2), layout="constrained")
    for ax, (name, title) in zip(axes.flat, files):
        image = plt.imread(output_dir / name)
        ax.imshow(image)
        ax.set_title(title, fontsize=scaled(15), fontweight="bold", pad=8)
        ax.axis("off")
    fig.savefig(
        output_dir / "figure6_variant_contact_sheet.png",
        dpi=min(dpi, 300),
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor="white",
    )
    plt.close(fig)


def write_notes(output_dir: Path) -> None:
    caption = (
        "Fig. 6. PAG-based structural context of injury severity. "
        "(a) Observed global outcome-node statistics relative to the median and "
        "95th percentile of 99 label-permutation null replicates. "
        "(b) Direct adjacency and exact-two-step frequencies among predictors "
        "appearing within two steps of injury severity. "
        "(c) Recurrent exact-two-step PAG skeleton configurations. Rows denote "
        "focal predictors, columns denote intermediate nodes, and cell values "
        "give occurrences among the 45 PAG specifications."
    )
    (output_dir / "figure6_recommended_caption.txt").write_text(
        caption + "\n", encoding="utf-8"
    )

    guide = """Recommended manuscript set
figure6a_primary_null_benchmark
figure6b_primary_pag_neighborhood
figure6c_primary_bridge_matrix

Alternative 6a designs
figure6a_alt_tripoint_profile
figure6a_alt_profile_matrix

Alternative 6c designs
figure6c_alt_lollipop
figure6c_alt_bridge_network
figure6c_alt_bridge_heatmap

Font control
Run with --font-scale 1.10 or edit DEFAULT_FONT_SCALE near the top of this script.
Every separate panel is exported without a panel letter or in-panel title.
SVG and PDF text remains editable.
"""
    (output_dir / "README_figure6_variants.txt").write_text(guide, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style(args.font_scale)
    global_df, feature_df, bridge_df = load_results(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        (plot_6a_null_benchmark, global_df, (9.2, 4.4), "figure6a_primary_null_benchmark"),
        (plot_6a_tripoint_profile, global_df, (9.2, 4.4), "figure6a_alt_tripoint_profile"),
        (plot_6a_profile_matrix, global_df, (9.0, 3.8), "figure6a_alt_profile_matrix"),
        (plot_6b_pag_neighborhood, feature_df, (9.4, 6.8), "figure6b_primary_pag_neighborhood"),
        (plot_6c_bridge_matrix, bridge_df, (9.2, 5.7), "figure6c_primary_bridge_matrix"),
        (plot_6c_lollipop, bridge_df, (10.4, 6.8), "figure6c_alt_lollipop"),
        (plot_6c_bridge_network, bridge_df, (11.0, 6.8), "figure6c_alt_bridge_network"),
        (plot_6c_bridge_heatmap, bridge_df, (9.2, 5.7), "figure6c_alt_bridge_heatmap"),
    ]
    for plotter, frame, size, stem in outputs:
        create_single_panel(
            plotter,
            frame,
            size=size,
            stem=args.output_dir / stem,
            dpi=args.dpi,
        )

    create_recommended_combined(
        global_df, feature_df, bridge_df, args.output_dir, args.dpi
    )
    create_variant_contact_sheet(args.output_dir, args.dpi)
    write_notes(args.output_dir)

    print(f"Validated PAG specifications: {N_DESIGNS}")
    print(f"Validated label-null replicates: {N_NULL_REPLICATES}")
    print(f"Output directory: {args.output_dir}")
    for _, _, _, stem in outputs:
        print(f"  {stem}.png / .pdf / .svg")
    print("  figure6_recommended_combined.png / .pdf / .svg")
    print("  figure6_variant_contact_sheet.png")


if __name__ == "__main__":
    main()

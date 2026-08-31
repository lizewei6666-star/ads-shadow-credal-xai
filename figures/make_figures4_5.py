from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable


# Keep the Matplotlib cache beside this script so the program is portable and
# does not depend on write access to the user's default Matplotlib directory.
SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib_cache"))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


DEFAULT_ROOT = SCRIPT_DIR.parent
DEFAULT_TABLE_DIR = DEFAULT_ROOT / "results" / "predictive"
DEFAULT_OUT_DIR = SCRIPT_DIR / "generated"

CLASS_LABELS = {
    0: "No injury",
    1: "Minor injury",
    2: "Moderate-plus injury",
}

FEATURES = [
    "Incident_Month",
    "Is_Weekend",
    "Incident_Hour_Group",
    "RoadwayType_Group",
    "RoadSurface_Group",
    "RoadCondition_Group",
    "Lighting_Group",
    "Weather_Group",
    "PostedSpeedLimit_Group",
    "CrashWith_Group",
    "CPMovement_Group",
    "SVMovement_Group",
    "Mileage_Group",
    "VehicleAge_Group",
]

FEATURE_LABELS = {
    "Incident_Month": "Month",
    "Is_Weekend": "Day type",
    "Incident_Hour_Group": "Hour period",
    "RoadwayType_Group": "Roadway type",
    "RoadSurface_Group": "Road surface",
    "RoadCondition_Group": "Road condition",
    "Lighting_Group": "Lighting",
    "Weather_Group": "Weather",
    "PostedSpeedLimit_Group": "Posted speed limit",
    "CrashWith_Group": "Crash counterpart",
    "CPMovement_Group": "Crash-partner movement",
    "SVMovement_Group": "Subject-vehicle movement",
    "Mileage_Group": "Mileage",
    "VehicleAge_Group": "Vehicle age",
}

# Wrapped labels are used only inside the compact Hasse nodes.
HASSE_LABELS = {
    "Incident_Month": "Month",
    "Is_Weekend": "Day\ntype",
    "Incident_Hour_Group": "Hour\nperiod",
    "RoadwayType_Group": "Roadway\ntype",
    "RoadSurface_Group": "Road\nsurface",
    "RoadCondition_Group": "Road\ncondition",
    "Lighting_Group": "Lighting",
    "Weather_Group": "Weather",
    "PostedSpeedLimit_Group": "Posted speed\nlimit",
    "CrashWith_Group": "Crash\ncounterpart",
    "CPMovement_Group": "Crash-partner\nmovement",
    "SVMovement_Group": "Subject-vehicle\nmovement",
    "Mileage_Group": "Mileage",
    "VehicleAge_Group": "Vehicle\nage",
}

# Extra wrapping for the 12-node lower layer in the Moderate-plus panel.
# The wording is unchanged; only line breaks and the optional hyphens differ.
DENSE_HASSE_LABELS = {
    **HASSE_LABELS,
    "CrashWith_Group": "Crash\ncounterpart",
    "CPMovement_Group": "Crash\npartner\nmovement",
    "SVMovement_Group": "Subject\nvehicle\nmovement",
}

EXPECTED_EDGE_COUNTS = {0: 28, 1: 13, 2: 12}

# Restrained, grayscale-safe palette used throughout both figures.
NAVY = "#1F4E79"
NAVY_DARK = "#173A5A"
SLATE = "#8796A3"
LIGHT_SLATE = "#B4BEC6"
GRID = "#DCE2E7"
TEXT = "#20262B"
WHITE = "#FFFFFF"

# ---------------------------------------------------------------------------
# USER-EDITABLE FONT SETTINGS FOR THE THREE INDEPENDENT FIGURE 4 PANELS
# Increase these values if the figures will be displayed or printed larger.
# These settings do not alter the combined three-panel Figure 4.
# ---------------------------------------------------------------------------
FIG4_SINGLE_YTICK_FONTSIZE = 12.0   # Predictor names on the left
FIG4_SINGLE_XTICK_FONTSIZE = 11.0   # 0.0, 0.2, ..., 1.0
FIG4_SINGLE_XLABEL_FONTSIZE = 12.0  # Horizontal axis title
FIG4_SINGLE_LEGEND_FONTSIZE = 10.5  # Legend above the plot
FIG4_SINGLE_FIGSIZE = (8.20, 5.35)  # Width and height in inches

# ---------------------------------------------------------------------------
# USER-EDITABLE FONT SETTINGS FOR THE THREE INDEPENDENT FIGURE 5 PANELS
# Node fonts are scaled from density-aware base sizes so the 12-node panel
# remains readable. Increase the canvas together with the font scale.
# ---------------------------------------------------------------------------
FIG5_SINGLE_NODE_FONT_SCALE = 1.20   # All text inside Hasse nodes
FIG5_SINGLE_NODE_HEIGHT_SCALE = 1.14 # Extra box height for larger text
FIG5_SINGLE_LEGEND_FONTSIZE = 11.0   # Legend above each independent panel
FIG5_SINGLE_FIGSIZE = (8.80, 4.05)   # Width and height in inches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate AAP-ready Figure 4, Figure 5, and Table 5 from the "
            "final Credal result tables."
        )
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=DEFAULT_TABLE_DIR,
        help="Directory containing the final Credal CSV tables.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory in which figures and Table 5 are written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Resolution for PNG exports (default: 600).",
    )
    return parser.parse_args()


def configure_style() -> None:
    # Fail clearly if the requested manuscript font is unavailable rather than
    # silently mixing fonts across panels.
    mpl.font_manager.findfont("Times New Roman", fallback_to_default=False)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_tables(table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    interval_path = table_dir / "credal_feature_intervals_and_partial_order.csv"
    edge_path = table_dir / "credal_hasse_edges.csv"
    for path in (interval_path, edge_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required result table not found: {path}")
    return pd.read_csv(interval_path), pd.read_csv(edge_path)


def validate_tables(intervals: pd.DataFrame, edges: pd.DataFrame) -> None:
    interval_columns = {
        "class_id",
        "feature",
        "lower_expected_percentile",
        "upper_expected_percentile",
        "specification_imprecision_width",
        "partial_order_level",
        "in_maximal_set",
        "n_strictly_dominates",
        "n_strictly_dominated_by",
    }
    edge_columns = {
        "class_id",
        "source_feature",
        "target_feature",
        "source_level",
        "target_level",
        "lower_difference",
    }
    missing_intervals = interval_columns.difference(intervals.columns)
    missing_edges = edge_columns.difference(edges.columns)
    if missing_intervals:
        raise ValueError(
            f"Missing interval columns: {sorted(missing_intervals)}"
        )
    if missing_edges:
        raise ValueError(f"Missing Hasse-edge columns: {sorted(missing_edges)}")

    if len(intervals) != len(CLASS_LABELS) * len(FEATURES):
        raise ValueError(
            f"Expected 42 class-feature rows, found {len(intervals)}."
        )
    if intervals.duplicated(["class_id", "feature"]).any():
        raise ValueError("Duplicate class-feature interval rows were found.")
    if set(intervals["class_id"].astype(int)) != set(CLASS_LABELS):
        raise ValueError("The interval table does not contain classes 0, 1, and 2.")
    if set(intervals["feature"]) != set(FEATURES):
        unknown = sorted(set(intervals["feature"]).difference(FEATURES))
        missing = sorted(set(FEATURES).difference(intervals["feature"]))
        raise ValueError(f"Feature mismatch. Unknown={unknown}; missing={missing}")

    lower = intervals["lower_expected_percentile"].to_numpy(float)
    upper = intervals["upper_expected_percentile"].to_numpy(float)
    width = intervals["specification_imprecision_width"].to_numpy(float)
    if not np.isfinite(np.column_stack([lower, upper, width])).all():
        raise ValueError("Non-finite Credal endpoints or widths were found.")
    if np.any(lower < -1e-12) or np.any(upper > 1.0 + 1e-12):
        raise ValueError("A Credal endpoint lies outside [0, 1].")
    if np.any(lower > upper + 1e-12):
        raise ValueError("A lower endpoint exceeds its upper endpoint.")
    if not np.allclose(upper - lower, width, atol=1e-12, rtol=0.0):
        raise ValueError("Stored interval widths do not match upper minus lower.")

    maximal = intervals["in_maximal_set"].astype(int)
    dominated_by = intervals["n_strictly_dominated_by"].astype(int)
    levels = intervals["partial_order_level"].astype(int)
    if not maximal.isin([0, 1]).all():
        raise ValueError("The maximal-set indicator must be binary.")
    if not ((maximal == 1) == (dominated_by == 0)).all():
        raise ValueError(
            "Maximal-set membership is inconsistent with domination counts."
        )
    if not ((maximal == 1) == (levels == 1)).all():
        raise ValueError(
            "Maximal-set membership is inconsistent with partial-order level 1."
        )

    if edges.duplicated(["class_id", "source_feature", "target_feature"]).any():
        raise ValueError("Duplicate Hasse edges were found.")
    valid_nodes = set(FEATURES)
    if not set(edges["source_feature"]).issubset(valid_nodes):
        raise ValueError("An unknown Hasse source feature was found.")
    if not set(edges["target_feature"]).issubset(valid_nodes):
        raise ValueError("An unknown Hasse target feature was found.")
    if (edges["source_feature"] == edges["target_feature"]).any():
        raise ValueError("A self-loop was found in the Hasse edge table.")
    if not (
        edges["source_level"].astype(int) < edges["target_level"].astype(int)
    ).all():
        raise ValueError("A Hasse edge does not point from an upper to a lower level.")
    if not (edges["lower_difference"].astype(float) > 0.0).all():
        raise ValueError("A Hasse edge has a non-positive lower difference.")

    observed_counts = edges.groupby(edges["class_id"].astype(int)).size().to_dict()
    if observed_counts != EXPECTED_EDGE_COUNTS:
        raise ValueError(
            "Unexpected Hasse-edge counts. "
            f"Expected {EXPECTED_EDGE_COUNTS}, observed {observed_counts}."
        )

    # Confirm that every edge agrees with the levels stored in the interval table.
    level_lookup = {
        (int(row.class_id), str(row.feature)): int(row.partial_order_level)
        for row in intervals.itertuples(index=False)
    }
    for row in edges.itertuples(index=False):
        class_id = int(row.class_id)
        if level_lookup[(class_id, row.source_feature)] != int(row.source_level):
            raise ValueError("A source level disagrees with the interval table.")
        if level_lookup[(class_id, row.target_feature)] != int(row.target_level):
            raise ValueError("A target level disagrees with the interval table.")


def export_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {"bbox_inches": "tight", "facecolor": WHITE, "pad_inches": 0.04}
    fig.savefig(out_dir / f"{stem}.pdf", **common)
    fig.savefig(out_dir / f"{stem}.svg", **common)
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, **common)


def interval_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=NAVY, lw=2.2, label="Maximal-set feature"),
        Line2D([0], [0], color=LIGHT_SLATE, lw=1.35, label="Other feature"),
        Line2D(
            [0],
            [0],
            color=SLATE,
            marker="o",
            markerfacecolor=WHITE,
            markeredgecolor=SLATE,
            lw=0,
            markersize=5.0,
            label="Lower bound",
        ),
        Line2D(
            [0],
            [0],
            color=SLATE,
            marker="o",
            markerfacecolor=SLATE,
            markeredgecolor=SLATE,
            lw=0,
            markersize=5.0,
            label="Upper bound",
        ),
    ]


def draw_interval_panel(
    ax: plt.Axes,
    intervals: pd.DataFrame,
    class_id: int,
    panel_label: str,
    *,
    show_panel_title: bool = True,
    ytick_fontsize: float | None = None,
    xtick_fontsize: float | None = None,
) -> None:
    group = intervals[intervals["class_id"].astype(int) == class_id].set_index(
        "feature"
    )
    y = np.arange(len(FEATURES), dtype=float)

    for y_pos, feature in zip(y, FEATURES):
        row = group.loc[feature]
        lower = float(row["lower_expected_percentile"])
        upper = float(row["upper_expected_percentile"])
        is_maximal = int(row["in_maximal_set"]) == 1
        color = NAVY if is_maximal else LIGHT_SLATE
        linewidth = 2.2 if is_maximal else 1.35
        marker_size = 25 if is_maximal else 20

        ax.hlines(
            y_pos,
            lower,
            upper,
            color=color,
            linewidth=linewidth,
            zorder=2,
        )
        ax.scatter(
            lower,
            y_pos,
            s=marker_size,
            facecolor=WHITE,
            edgecolor=color,
            linewidth=1.05,
            zorder=3,
        )
        ax.scatter(
            upper,
            y_pos,
            s=marker_size,
            facecolor=color,
            edgecolor=color,
            linewidth=0.8,
            zorder=3,
        )

    if show_panel_title:
        ax.set_title(
            f"({panel_label}) {CLASS_LABELS[class_id]}",
            loc="left",
            pad=7,
            color=TEXT,
            fontweight="bold",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_LABELS[feature] for feature in FEATURES])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.65, zorder=0)
    ax.tick_params(
        axis="y",
        length=0,
        pad=4,
        labelsize=ytick_fontsize,
    )
    ax.tick_params(
        axis="x",
        length=3,
        color=SLATE,
        labelsize=xtick_fontsize,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SLATE)
    ax.spines["bottom"].set_color(SLATE)


def make_figure4(intervals: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    # Main manuscript version: one injury class per row for readable labels.
    fig, axes = plt.subplots(3, 1, figsize=(7.35, 10.25), sharex=True)
    for index, class_id in enumerate(CLASS_LABELS):
        draw_interval_panel(
            axes[index], intervals, class_id, chr(ord("a") + index)
        )
    axes[-1].set_xlabel("Shadow-calibrated relevance percentile")
    fig.legend(
        handles=interval_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.59, 0.996),
        ncol=4,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.25,
    )
    fig.subplots_adjust(
        left=0.285,
        right=0.985,
        bottom=0.060,
        top=0.945,
        hspace=0.30,
    )
    export_figure(fig, out_dir, "figure4_credal_bounds", dpi)
    plt.close(fig)

    # Independent panels are useful for Word layout and detailed inspection.
    for index, class_id in enumerate(CLASS_LABELS):
        fig, ax = plt.subplots(figsize=FIG4_SINGLE_FIGSIZE)
        draw_interval_panel(
            ax,
            intervals,
            class_id,
            chr(ord("a") + index),
            show_panel_title=False,
            ytick_fontsize=FIG4_SINGLE_YTICK_FONTSIZE,
            xtick_fontsize=FIG4_SINGLE_XTICK_FONTSIZE,
        )
        ax.set_xlabel(
            "Shadow-calibrated relevance percentile",
            fontsize=FIG4_SINGLE_XLABEL_FONTSIZE,
        )
        ax.legend(
            handles=interval_legend_handles(),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.035),
            ncol=4,
            frameon=False,
            handlelength=1.8,
            columnspacing=1.1,
            fontsize=FIG4_SINGLE_LEGEND_FONTSIZE,
        )
        fig.subplots_adjust(left=0.31, right=0.985, bottom=0.125, top=0.88)
        stem = f"figure4{chr(ord('a') + index)}_{slug(CLASS_LABELS[class_id])}"
        export_figure(fig, out_dir, stem, dpi)
        plt.close(fig)


def slug(label: str) -> str:
    return label.lower().replace("-", "_").replace(" ", "_")


def feature_rank(feature: str) -> int:
    return FEATURES.index(feature)


def compute_layer_orders(
    class_intervals: pd.DataFrame,
    class_edges: pd.DataFrame,
) -> dict[int, list[str]]:
    """Return deterministic barycentric layer orders to reduce crossings."""
    levels = {
        str(row.feature): int(row.partial_order_level)
        for row in class_intervals.itertuples(index=False)
    }
    orders: dict[int, list[str]] = {}
    for level in sorted(set(levels.values())):
        nodes = [feature for feature, value in levels.items() if value == level]
        if level == 1:
            # Place the most connected maximal nodes centrally in the initial order.
            dominates = {
                str(row.feature): int(row.n_strictly_dominates)
                for row in class_intervals.itertuples(index=False)
            }
            nodes.sort(key=lambda f: (-dominates[f], feature_rank(f)))
        else:
            nodes.sort(key=feature_rank)
        orders[level] = nodes

    predecessors: dict[str, list[str]] = {feature: [] for feature in FEATURES}
    successors: dict[str, list[str]] = {feature: [] for feature in FEATURES}
    for row in class_edges.itertuples(index=False):
        predecessors[row.target_feature].append(row.source_feature)
        successors[row.source_feature].append(row.target_feature)

    layer_values = sorted(orders)
    for _ in range(18):
        rank = {
            node: position
            for level in layer_values
            for position, node in enumerate(orders[level])
        }
        # Downward sweep.
        for level in layer_values[1:]:
            old_position = {node: i for i, node in enumerate(orders[level])}

            def down_key(node: str) -> tuple[float, int, int]:
                parents = predecessors[node]
                barycenter = (
                    float(np.mean([rank[parent] for parent in parents]))
                    if parents
                    else float(old_position[node])
                )
                return barycenter, old_position[node], feature_rank(node)

            orders[level] = sorted(orders[level], key=down_key)

        rank = {
            node: position
            for level in layer_values
            for position, node in enumerate(orders[level])
        }
        # Upward sweep.
        for level in reversed(layer_values[:-1]):
            old_position = {node: i for i, node in enumerate(orders[level])}

            def up_key(node: str) -> tuple[float, int, int]:
                children = successors[node]
                barycenter = (
                    float(np.mean([rank[child] for child in children]))
                    if children
                    else float(old_position[node])
                )
                return barycenter, old_position[node], feature_rank(node)

            orders[level] = sorted(orders[level], key=up_key)

    return orders


def layer_x_positions(
    nodes: Iterable[str],
    node_width: float,
) -> dict[str, float]:
    node_list = list(nodes)
    if len(node_list) == 1:
        return {node_list[0]: 0.5}
    if len(node_list) == 2:
        positions = np.array([0.32, 0.68], dtype=float)
    elif len(node_list) == 3:
        positions = np.array([0.20, 0.50, 0.80], dtype=float)
    else:
        margin = max(0.045, node_width / 2.0 + 0.012)
        positions = np.linspace(margin, 1.0 - margin, len(node_list))
    return {node: float(x) for node, x in zip(node_list, positions)}


def layer_y_positions(levels: list[int]) -> dict[int, float]:
    if len(levels) == 1:
        return {levels[0]: 0.50}
    if len(levels) == 2:
        return {levels[0]: 0.72, levels[1]: 0.25}
    if len(levels) == 3:
        return {levels[0]: 0.78, levels[1]: 0.49, levels[2]: 0.20}
    values = np.linspace(0.80, 0.17, len(levels))
    return {level: float(y) for level, y in zip(levels, values)}


def node_dimensions(n_in_layer: int, n_levels: int) -> tuple[float, float, float]:
    if n_in_layer <= 1:
        width = 0.18
    else:
        spacing = 0.89 / (n_in_layer - 1)
        width = min(0.145, spacing * 0.82)
    if n_in_layer >= 10:
        height = 0.135
        fontsize = 6.35
    elif n_in_layer >= 7:
        height = 0.105 if n_levels <= 2 else 0.087
        fontsize = 7.3
    else:
        height = 0.105 if n_levels <= 2 else 0.087
        fontsize = 8.0
    return width, height, fontsize


def draw_hasse_panel(
    ax: plt.Axes,
    intervals: pd.DataFrame,
    edges: pd.DataFrame,
    class_id: int,
    panel_label: str,
    *,
    show_panel_title: bool = True,
    node_font_scale: float = 1.0,
    node_height_scale: float = 1.0,
) -> None:
    class_intervals = intervals[
        intervals["class_id"].astype(int) == class_id
    ].copy()
    class_edges = edges[edges["class_id"].astype(int) == class_id].copy()
    orders = compute_layer_orders(class_intervals, class_edges)
    levels = sorted(orders)
    y_by_level = layer_y_positions(levels)

    positions: dict[str, tuple[float, float]] = {}
    dimensions: dict[str, tuple[float, float, float]] = {}
    for level in levels:
        width, height, fontsize = node_dimensions(len(orders[level]), len(levels))
        height *= node_height_scale
        fontsize *= node_font_scale
        x_by_node = layer_x_positions(orders[level], width)
        for node in orders[level]:
            positions[node] = (x_by_node[node], y_by_level[level])
            dimensions[node] = (width, height, fontsize)

    # Edges are drawn first so node boxes mask line endpoints cleanly.
    for row in class_edges.itertuples(index=False):
        source = row.source_feature
        target = row.target_feature
        source_x, source_y = positions[source]
        target_x, target_y = positions[target]
        _, source_height, _ = dimensions[source]
        _, target_height, _ = dimensions[target]
        horizontal_shift = target_x - source_x
        curvature = float(np.clip(horizontal_shift * 0.055, -0.035, 0.035))
        arrow = FancyArrowPatch(
            (source_x, source_y - source_height / 2.0),
            (target_x, target_y + target_height / 2.0),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=7.0,
            linewidth=0.75,
            color=SLATE,
            alpha=0.70,
            connectionstyle=f"arc3,rad={curvature}",
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=1,
        )
        ax.add_patch(arrow)

    maximal_lookup = {
        str(row.feature): int(row.in_maximal_set) == 1
        for row in class_intervals.itertuples(index=False)
    }
    for feature, (x, y) in positions.items():
        width, height, fontsize = dimensions[feature]
        n_in_layer = len(orders[int(
            class_intervals.loc[
                class_intervals["feature"] == feature,
                "partial_order_level",
            ].iloc[0]
        )])
        display_label = (
            DENSE_HASSE_LABELS[feature]
            if n_in_layer >= 10
            else HASSE_LABELS[feature]
        )
        is_maximal = maximal_lookup[feature]
        facecolor = NAVY if is_maximal else WHITE
        edgecolor = NAVY_DARK if is_maximal else NAVY
        text_color = WHITE if is_maximal else TEXT
        linewidth = 1.05 if is_maximal else 0.95
        node = FancyBboxPatch(
            (x - width / 2.0, y - height / 2.0),
            width,
            height,
            transform=ax.transAxes,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=3,
        )
        ax.add_patch(node)
        ax.text(
            x,
            y,
            display_label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=text_color,
            fontsize=fontsize,
            linespacing=0.92,
            zorder=4,
        )

    if show_panel_title:
        ax.text(
            0.0,
            0.985,
            f"({panel_label}) {CLASS_LABELS[class_id]}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.0,
            fontweight="bold",
            color=TEXT,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")


def hasse_legend_handles() -> list[object]:
    return [
        Patch(facecolor=NAVY, edgecolor=NAVY_DARK, label="Maximal-set feature"),
        Patch(facecolor=WHITE, edgecolor=NAVY, label="Other feature"),
        Line2D(
            [0],
            [0],
            color=SLATE,
            marker=r"$\rightarrow$",
            markersize=8,
            lw=0.8,
            label="Strict dominance",
        ),
    ]


def make_figure5(
    intervals: pd.DataFrame,
    edges: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.35, 9.15))
    for index, class_id in enumerate(CLASS_LABELS):
        draw_hasse_panel(
            axes[index], intervals, edges, class_id, chr(ord("a") + index)
        )
    fig.legend(
        handles=hasse_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.025, top=0.955, hspace=0.20)
    export_figure(fig, out_dir, "figure5_credal_hasse_orders", dpi)
    plt.close(fig)

    for index, class_id in enumerate(CLASS_LABELS):
        fig, ax = plt.subplots(figsize=FIG5_SINGLE_FIGSIZE)
        draw_hasse_panel(
            ax,
            intervals,
            edges,
            class_id,
            chr(ord("a") + index),
            show_panel_title=False,
            node_font_scale=FIG5_SINGLE_NODE_FONT_SCALE,
            node_height_scale=FIG5_SINGLE_NODE_HEIGHT_SCALE,
        )
        ax.legend(
            handles=hasse_legend_handles(),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=False,
            handlelength=1.6,
            columnspacing=1.4,
            fontsize=FIG5_SINGLE_LEGEND_FONTSIZE,
        )
        fig.subplots_adjust(left=0.025, right=0.985, bottom=0.025, top=0.84)
        stem = f"figure5{chr(ord('a') + index)}_{slug(CLASS_LABELS[class_id])}"
        export_figure(fig, out_dir, stem, dpi)
        plt.close(fig)


def format_feature_list(features: list[str]) -> str:
    return "; ".join(FEATURE_LABELS[feature] for feature in features)


def make_table5(intervals: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    edge_counts = edges.groupby(edges["class_id"].astype(int)).size().to_dict()
    rows: list[dict[str, object]] = []
    for class_id, class_label in CLASS_LABELS.items():
        group = intervals[intervals["class_id"].astype(int) == class_id].copy()
        maximal_features = [
            feature
            for feature in FEATURES
            if int(
                group.loc[group["feature"] == feature, "in_maximal_set"].iloc[0]
            )
            == 1
        ]
        largest_lower = float(group["lower_expected_percentile"].max())
        leaders = set(
            group.loc[
                np.isclose(
                    group["lower_expected_percentile"].astype(float),
                    largest_lower,
                    atol=1e-12,
                    rtol=0.0,
                ),
                "feature",
            ]
        )
        ordered_leaders = [feature for feature in FEATURES if feature in leaders]
        rows.append(
            {
                "Injury class": class_label,
                "Maximal feature set": format_feature_list(maximal_features),
                "Set size": len(maximal_features),
                "Median bound width": float(
                    group["specification_imprecision_width"].median()
                ),
                "Largest lower endpoint": (
                    f"{format_feature_list(ordered_leaders)} ({largest_lower:.3f})"
                ),
                "Hasse edges": int(edge_counts[class_id]),
            }
        )
    return pd.DataFrame(rows)


def save_table5(table: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "table5_class_specific_credal_orders.csv", index=False)
    lines = [
        "| Injury class | Maximal feature set | Set size | Median bound width | Largest lower endpoint | Hasse edges |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:.3f} | {row[4]} | {row[5]} |"
        )
    (out_dir / "table5_class_specific_credal_orders.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def save_captions(out_dir: Path) -> None:
    captions = (
        "Fig. 4. Class-specific Credal bounds of shadow-calibrated predictive "
        "relevance. Open and filled markers indicate the lower and upper "
        "bounds, respectively; dark blue identifies maximal-set features.\n\n"
        "Fig. 5. Strict Credal partial orders by injury class. Filled nodes "
        "identify maximal features; arrows run from the dominant to the "
        "dominated feature, with transitive edges omitted.\n"
    )
    (out_dir / "figure4_figure5_captions.txt").write_text(
        captions,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    table_dir = args.table_dir.resolve()
    out_dir = args.out_dir.resolve()
    if args.dpi < 300:
        raise ValueError("PNG export resolution must be at least 300 dpi.")

    configure_style()
    intervals, edges = load_tables(table_dir)
    validate_tables(intervals, edges)

    make_figure4(intervals, out_dir, args.dpi)
    make_figure5(intervals, edges, out_dir, args.dpi)
    table5 = make_table5(intervals, edges)
    save_table5(table5, out_dir)
    save_captions(out_dir)

    print("Validated Credal interval rows:", len(intervals))
    print("Validated Hasse edges:", len(edges))
    print("Output directory:", out_dir)
    print("\nTable 5:\n")
    print(table5.to_string(index=False))


if __name__ == "__main__":
    main()

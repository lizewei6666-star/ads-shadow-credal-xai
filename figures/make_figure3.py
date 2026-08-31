from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib"),
)

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
TABLE_DIR = SCRIPT_DIR.parent / "results" / "predictive"
OUT_DIR = SCRIPT_DIR / "generated"

RELEVANCE_PATH = TABLE_DIR / "outer_holdout_shadow_calibrated_relevance.csv"
PARTITION_PATH = TABLE_DIR / "outer_partition_manifest.csv"
COVERAGE_PATH = TABLE_DIR / "credal_common_support_coverage.csv"

SOURCES = ["SHAP", "PFI", "LOCO", "FIRM"]
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
COLORS = {
    "SHAP": "#1F4E79",
    "PFI": "#2A7F78",
    "LOCO": "#B07A2A",
    "FIRM": "#A64B4B",
}


def validate_inputs(
    relevance: pd.DataFrame,
    partitions: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    expected_columns = {
        "outer_partition",
        "model",
        "source",
        "class_id",
        "feature",
        "n_shadow_replicates",
        "relevance_percentile",
    }
    missing = expected_columns.difference(relevance.columns)
    if missing:
        raise ValueError(f"Missing relevance columns: {sorted(missing)}")

    if set(relevance["source"].unique()) != set(SOURCES):
        raise ValueError("Unexpected explanation-source set.")
    if set(relevance["class_id"].unique()) != set(CLASS_LABELS):
        raise ValueError("Unexpected injury-class set.")
    if set(relevance["feature"].unique()) != set(FEATURES):
        raise ValueError("Unexpected feature set.")
    if set(relevance["outer_partition"].unique()) != set(range(1, 6)):
        raise ValueError("Expected five outer partitions.")
    if not (relevance["n_shadow_replicates"] == 49).all():
        raise ValueError("Every calibrated statistic must use 49 shadows.")
    if not np.isfinite(relevance["relevance_percentile"]).all():
        raise ValueError("Non-finite relevance percentiles found.")
    if not relevance["relevance_percentile"].between(0.0, 1.0).all():
        raise ValueError("Relevance percentiles must lie in [0, 1].")

    key = ["outer_partition", "model", "source", "class_id", "feature"]
    if relevance.duplicated(key).any():
        raise ValueError("Duplicate outer relevance units found.")

    model_instances = relevance[["outer_partition", "model"]].drop_duplicates()
    expected_rows = len(model_instances) * len(SOURCES) * len(CLASS_LABELS) * len(FEATURES)
    if len(model_instances) != 12 or len(relevance) != expected_rows or expected_rows != 2016:
        raise ValueError(
            f"Expected 12 model-partition instances and 2016 rows; "
            f"found {len(model_instances)} and {len(relevance)}."
        )

    if len(partitions) != 5 or partitions["n_holdout"].sum() != 1469:
        raise ValueError("Outer partition manifest does not cover 1,469 holdout records once.")
    if len(coverage) != 15 or not (coverage["coverage"] == 1.0).all():
        raise ValueError("Common-support coverage must be complete for all fold-class cells.")


def compute_source_intervals(
    relevance: pd.DataFrame,
    partitions: pd.DataFrame,
) -> pd.DataFrame:
    weights = partitions.set_index("outer_partition")["n_holdout"].astype(float)
    weights = weights / weights.sum()

    fold_extrema = (
        relevance.groupby(
            ["outer_partition", "source", "class_id", "feature"],
            as_index=False,
        )["relevance_percentile"]
        .agg(lower="min", upper="max")
    )
    fold_extrema["weight"] = fold_extrema["outer_partition"].map(weights)
    fold_extrema["weighted_lower"] = fold_extrema["weight"] * fold_extrema["lower"]
    fold_extrema["weighted_upper"] = fold_extrema["weight"] * fold_extrema["upper"]

    intervals = (
        fold_extrema.groupby(["source", "class_id", "feature"], as_index=False)
        .agg(Q_lower=("weighted_lower", "sum"), Q_upper=("weighted_upper", "sum"))
    )
    intervals["width"] = intervals["Q_upper"] - intervals["Q_lower"]
    intervals["class_label"] = intervals["class_id"].map(CLASS_LABELS)
    intervals["feature_label"] = intervals["feature"].map(FEATURE_LABELS)
    return intervals


def make_table(intervals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source in SOURCES:
        row: dict[str, object] = {"Explanation source": source}
        for class_id, class_label in CLASS_LABELS.items():
            subset = intervals[
                (intervals["source"] == source)
                & (intervals["class_id"] == class_id)
            ].sort_values(
                ["Q_lower", "Q_upper", "feature_label"],
                ascending=[False, False, True],
            )
            max_lower = float(subset["Q_lower"].max())
            leaders = subset[np.isclose(subset["Q_lower"], max_lower, atol=1e-12, rtol=0.0)]
            entries = [
                f"{leader.feature_label} "
                f"[{leader.Q_lower:.3f}, {leader.Q_upper:.3f}]"
                for leader in leaders.itertuples(index=False)
            ]
            row[class_label] = "; ".join(entries)
        rows.append(row)
    return pd.DataFrame(rows)


def save_table(table: pd.DataFrame) -> None:
    table.to_csv(OUT_DIR / "table4_source_specific_leaders.csv", index=False)
    lines = [
        "| Explanation source | No injury | Minor injury | Moderate-plus injury |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['Explanation source']} | {row['No injury']} | "
            f"{row['Minor injury']} | {row['Moderate-plus injury']} |"
        )
    (OUT_DIR / "table4_source_specific_leaders.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def draw_source_panel(
    ax: mpl.axes.Axes,
    intervals: pd.DataFrame,
    class_id: int,
    class_label: str,
    panel_label: str,
) -> None:
    base_y = np.arange(len(FEATURES))[::-1]
    offsets = {"SHAP": 0.24, "PFI": 0.08, "LOCO": -0.08, "FIRM": -0.24}
    class_data = intervals[intervals["class_id"] == class_id].set_index(
        ["source", "feature"]
    )

    for source in SOURCES:
        ys = base_y + offsets[source]
        lower = np.array(
            [class_data.loc[(source, feature), "Q_lower"] for feature in FEATURES]
        )
        upper = np.array(
            [class_data.loc[(source, feature), "Q_upper"] for feature in FEATURES]
        )
        color = COLORS[source]
        ax.hlines(ys, lower, upper, color=color, linewidth=1.7, zorder=2)
        ax.scatter(
            lower,
            ys,
            s=17,
            facecolor="white",
            edgecolor=color,
            linewidth=0.9,
            zorder=3,
        )
        ax.scatter(
            upper,
            ys,
            s=17,
            facecolor=color,
            edgecolor=color,
            linewidth=0.7,
            zorder=3,
        )

    ax.set_title(f"{panel_label}) {class_label}", loc="left", fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_yticks(base_y)
    ax.set_yticklabels([FEATURE_LABELS[feature] for feature in FEATURES])
    ax.grid(axis="x", color="#D9DEE3", linewidth=0.55, zorder=0)
    ax.tick_params(axis="y", length=0, labelleft=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAB2B9")
    ax.spines["bottom"].set_color("#AAB2B9")


def source_legend_handles() -> list[mpl.lines.Line2D]:
    return [
        mpl.lines.Line2D(
            [0],
            [0],
            color=COLORS[source],
            marker="o",
            markerfacecolor=COLORS[source],
            markeredgecolor=COLORS[source],
            linewidth=1.7,
            markersize=4.5,
            label=source,
        )
        for source in SOURCES
    ]


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        path = OUT_DIR / f"{stem}.{suffix}"
        save_kwargs: dict[str, object] = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }
        if suffix == "png":
            save_kwargs["dpi"] = 600
        fig.savefig(path, **save_kwargs)


def make_figure(intervals: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.1,
            "legend.fontsize": 8.4,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 11.8), sharex=True, sharey=True)
    for panel_index, (class_id, class_label) in enumerate(CLASS_LABELS.items()):
        draw_source_panel(
            axes[panel_index],
            intervals,
            class_id,
            class_label,
            chr(97 + panel_index),
        )
        axes[panel_index].tick_params(axis="x", labelbottom=True)

    handles = source_legend_handles()
    fig.legend(
        handles=handles,
        labels=SOURCES,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.995),
        ncol=4,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.5,
    )
    fig.supxlabel("Shadow-calibrated relevance percentile", y=0.018, fontsize=8.8)
    fig.subplots_adjust(left=0.27, right=0.985, bottom=0.055, top=0.955, hspace=0.27)
    save_figure(fig, "figure3_source_specific_relevance")
    plt.close(fig)

    panel_stems = {
        0: "figure3a_no_injury",
        1: "figure3b_minor_injury",
        2: "figure3c_moderate_plus_injury",
    }
    for panel_index, (class_id, class_label) in enumerate(CLASS_LABELS.items()):
        panel_fig, panel_ax = plt.subplots(figsize=(7.5, 4.25))
        draw_source_panel(
            panel_ax,
            intervals,
            class_id,
            class_label,
            chr(97 + panel_index),
        )
        panel_ax.set_xlabel("Shadow-calibrated relevance percentile")
        panel_fig.legend(
            handles=handles,
            labels=SOURCES,
            loc="upper center",
            bbox_to_anchor=(0.60, 0.995),
            ncol=4,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.5,
        )
        panel_fig.subplots_adjust(left=0.27, right=0.985, bottom=0.13, top=0.90)
        save_figure(panel_fig, panel_stems[class_id])
        plt.close(panel_fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    relevance = pd.read_csv(RELEVANCE_PATH)
    partitions = pd.read_csv(PARTITION_PATH)
    coverage = pd.read_csv(COVERAGE_PATH)
    validate_inputs(relevance, partitions, coverage)

    intervals = compute_source_intervals(relevance, partitions)
    intervals.to_csv(OUT_DIR / "figure3_source_specific_relevance_data.csv", index=False)
    table = make_table(intervals)
    save_table(table)
    make_figure(intervals)

    print("Validated rows:", len(relevance))
    print("Model-partition instances:", len(relevance[["outer_partition", "model"]].drop_duplicates()))
    print("Common-support fold-class cells:", len(coverage))
    print("\nTable 4 values:")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

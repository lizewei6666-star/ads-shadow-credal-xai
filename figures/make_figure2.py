import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(SCRIPT_DIR / ".matplotlib_cache"),
)

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = SCRIPT_DIR.parent / "results" / "predictive"
OUT = SCRIPT_DIR / "generated"
OUT.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = ROOT / "model_selection_metrics_summary.csv"
HOLDOUT_PATH = ROOT / "admissible_model_holdout_metrics_raw.csv"

summary = pd.read_csv(SUMMARY_PATH).sort_values("rank_balanced_accuracy").reset_index(drop=True)
holdout = pd.read_csv(HOLDOUT_PATH)

model_labels = {
    "Logistic regression": "Logistic regression",
    "CatBoost": "CatBoost",
    "XGBoost": "XGBoost",
    "Extra trees": "Extra Trees",
    "LightGBM": "LightGBM",
    "Histogram gradient boosting": "Hist. gradient boosting",
    "Random forest": "Random forest",
}
summary["plot_label"] = summary["model_display"].map(model_labels)


def mean_sd_text(mean: float, sd: float, n: int | None = None) -> str:
    if pd.isna(mean):
        return "NA"
    if n == 1 or pd.isna(sd):
        return f"{mean:.3f}"
    return f"{mean:.3f} ({sd:.3f})"


outer = (
    holdout.groupby(["model", "model_display"], as_index=False)
    .agg(
        n_outer_holdouts=("outer_partition", "nunique"),
        outer_mean_balanced_accuracy=("balanced_accuracy", "mean"),
        outer_sd_balanced_accuracy=("balanced_accuracy", "std"),
        outer_mean_macro_f1=("macro_f1", "mean"),
        outer_sd_macro_f1=("macro_f1", "std"),
        outer_mean_accuracy=("accuracy", "mean"),
        outer_sd_accuracy=("accuracy", "std"),
    )
)

table = summary.merge(outer, on=["model", "model_display"], how="left")
table["admitted_outer_folds"] = (table["one_se_selection_frequency"] * 5).round().astype(int)
table["Inner CV BA"] = [
    mean_sd_text(m, s) for m, s in zip(table["mean_balanced_accuracy"], table["std_balanced_accuracy"])
]
table["Inner CV macro-F1"] = [
    mean_sd_text(m, s) for m, s in zip(table["mean_macro_f1"], table["std_macro_f1"])
]
table["Inner CV accuracy"] = [
    mean_sd_text(m, s) for m, s in zip(table["mean_accuracy"], table["std_accuracy"])
]
table["Admitted folds"] = table["admitted_outer_folds"].astype(str) + "/5"
table["Outer holdout BA"] = [
    mean_sd_text(m, s, int(n) if pd.notna(n) else None)
    for m, s, n in zip(
        table["outer_mean_balanced_accuracy"],
        table["outer_sd_balanced_accuracy"],
        table["n_outer_holdouts"],
    )
]
table["Outer holdout macro-F1"] = [
    mean_sd_text(m, s, int(n) if pd.notna(n) else None)
    for m, s, n in zip(
        table["outer_mean_macro_f1"],
        table["outer_sd_macro_f1"],
        table["n_outer_holdouts"],
    )
]
table["Outer holdout accuracy"] = [
    mean_sd_text(m, s, int(n) if pd.notna(n) else None)
    for m, s, n in zip(
        table["outer_mean_accuracy"],
        table["outer_sd_accuracy"],
        table["n_outer_holdouts"],
    )
]

display_columns = [
    "model_display",
    "Inner CV BA",
    "Inner CV macro-F1",
    "Inner CV accuracy",
    "Admitted folds",
    "Outer holdout BA",
    "Outer holdout macro-F1",
    "Outer holdout accuracy",
]
table_out = table[display_columns].rename(columns={"model_display": "Learner"})
table_out.to_csv(OUT / "table3_predictive_performance.csv", index=False, encoding="utf-8-sig")
headers = list(table_out.columns)
markdown_lines = [
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
]
for row in table_out.itertuples(index=False, name=None):
    markdown_lines.append("| " + " | ".join(str(value) for value in row) + " |")
(OUT / "table3_predictive_performance.md").write_text(
    "\n".join(markdown_lines) + "\n",
    encoding="utf-8",
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.8,
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

fig, ax = plt.subplots(figsize=(5.35, 3.45))

recall = summary[["mean_recall_class_0", "mean_recall_class_1", "mean_recall_class_2"]].to_numpy()
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "aap_blue",
    ["#F5F7F8", "#D8E7EA", "#8CC2C5", "#3A8F99", "#184F67"],
)
image = ax.imshow(recall, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
ax.set_xticks(np.arange(3))
ax.set_xticklabels(["No injury", "Minor injury", "Moderate-plus"])
ax.xaxis.tick_top()
ax.tick_params(axis="x", length=0, pad=7)
ax.set_yticks(np.arange(len(summary)))
ax.set_yticklabels(summary["plot_label"])
ax.tick_params(axis="y", length=0, pad=7)

for row in range(recall.shape[0]):
    for col in range(recall.shape[1]):
        value = recall[row, col]
        ax.text(
            col,
            row,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=8.2,
            color="white" if value >= 0.52 else "#26343D",
        )

for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(summary), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.3)
ax.tick_params(which="minor", bottom=False, left=False)

cbar = fig.colorbar(image, ax=ax, fraction=0.050, pad=0.035)
cbar.set_label("Mean recall", rotation=90, labelpad=6)
cbar.set_ticks([0, 0.5, 1.0])
cbar.ax.tick_params(labelsize=7.5)

fig.subplots_adjust(left=0.34, right=0.91, top=0.86, bottom=0.06)

for suffix in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"figure2_class_recall.{suffix}")
plt.close(fig)

print(table_out.to_string(index=False))
print(f"\nSaved outputs to {OUT}")

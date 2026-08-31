"""Create the corrected Figure 1 methodological framework.

Outputs are written beside this script as SVG, PDF, and 600-dpi PNG.
The SVG keeps text as editable text elements.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image


OUT_DIR = Path(__file__).resolve().parent / "generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAVY = "#0B2F6B"
BLUE = "#2457B2"
BLUE_2 = "#173F82"
GREEN = "#185C37"
GREEN_2 = "#2B7A4B"
PURPLE = "#5A2A82"
ORANGE = "#B85520"
TEAL = "#0F766E"
RED = "#A72B2B"
AMBER = "#A86408"
INK = "#172033"
MID = "#667085"
GRID = "#D6DCE6"
SOFT = "#F7F9FC"
WHITE = "#FFFFFF"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.2,
            "text.color": INK,
            "axes.edgecolor": INK,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def rounded_box(ax, x, y, w, h, edge, face=WHITE, lw=0.8, radius=0.9, z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.18,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def text(
    ax,
    x,
    y,
    value,
    size=6.2,
    weight="normal",
    color=INK,
    ha="center",
    va="center",
    z=5,
    linespacing=1.18,
    **kwargs,
):
    return ax.text(
        x,
        y,
        value,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        zorder=z,
        linespacing=linespacing,
        **kwargs,
    )


def arrow(ax, x1, y1, x2, y2, color=INK, lw=0.9, dashed=False, mutation=8, z=4):
    style = "--" if dashed else "-"
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        linestyle=style,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def section_header(ax, x, y, w, label, color, panel=None, size=7.2):
    rounded_box(ax, x, y, w, 2.25, color, face=color, lw=0, radius=1.1, z=8)
    if panel:
        ax.add_patch(Circle((x + 1.45, y + 1.12), 0.92, facecolor=WHITE, edgecolor=WHITE, zorder=9))
        text(ax, x + 1.45, y + 1.12, panel, size=8.5, weight="bold", color=color, z=10)
        text(ax, x + 2.75, y + 1.12, label, size=size, weight="bold", color=WHITE, ha="left", z=10)
    else:
        text(ax, x + w / 2, y + 1.12, label, size=size, weight="bold", color=WHITE, z=10)


def database_icon(ax, x, y, color=NAVY, scale=1.0):
    w, h = 2.25 * scale, 2.6 * scale
    ax.add_patch(Rectangle((x - w / 2, y - h / 2 + 0.35 * scale), w, h - 0.7 * scale,
                           facecolor=color, edgecolor=color, zorder=4))
    ax.add_patch(mpl.patches.Ellipse((x, y + h / 2 - 0.35 * scale), w, 0.7 * scale,
                                    facecolor=WHITE, edgecolor=color, lw=0.8, zorder=5))
    ax.add_patch(mpl.patches.Ellipse((x, y + h / 2 - 0.35 * scale), w - 0.42 * scale, 0.34 * scale,
                                    facecolor=color, edgecolor=color, zorder=6))
    for yy in [y + 0.35 * scale, y - 0.55 * scale]:
        ax.add_patch(Arc((x, yy), w, 0.7 * scale, theta1=180, theta2=360,
                         edgecolor=WHITE, lw=0.65, zorder=6))
    ax.add_patch(Arc((x, y - h / 2 + 0.35 * scale), w, 0.7 * scale, theta1=0, theta2=180,
                     edgecolor=color, lw=0.8, zorder=5))


def road_icon(ax, x, y, color=NAVY, scale=1.0):
    poly = Polygon(
        [(x - 1.65 * scale, y - 1.5 * scale), (x + 1.65 * scale, y - 1.5 * scale),
         (x + 0.72 * scale, y + 1.5 * scale), (x - 0.72 * scale, y + 1.5 * scale)],
        closed=True,
        facecolor=color,
        edgecolor=color,
        zorder=4,
    )
    ax.add_patch(poly)
    for yy in [-0.95, 0.0, 0.95]:
        ax.plot([x, x], [y + yy * scale - 0.28 * scale, y + yy * scale + 0.28 * scale],
                color=WHITE, lw=1.0, zorder=5)


def calendar_icon(ax, x, y, color=NAVY, scale=1.0):
    rounded_box(ax, x - 1.45 * scale, y - 1.25 * scale, 2.9 * scale, 2.5 * scale,
                color, face=WHITE, lw=0.8, radius=0.28 * scale, z=4)
    ax.add_patch(Rectangle((x - 1.45 * scale, y + 0.55 * scale), 2.9 * scale, 0.45 * scale,
                           facecolor=color, edgecolor=color, zorder=5))
    for dx in [-0.78, 0.0, 0.78]:
        for dy in [-0.58, 0.0]:
            ax.add_patch(Rectangle((x + dx * scale - 0.16 * scale, y + dy * scale - 0.14 * scale),
                                   0.32 * scale, 0.28 * scale, facecolor=color, edgecolor="none", zorder=5))


def car_icon(ax, x, y, color=NAVY, scale=1.0):
    body = FancyBboxPatch((x - 1.45 * scale, y - 0.65 * scale), 2.9 * scale, 1.15 * scale,
                          boxstyle=f"round,pad=0.06,rounding_size={0.28 * scale}",
                          facecolor=WHITE, edgecolor=color, linewidth=0.85, zorder=4)
    ax.add_patch(body)
    roof = Polygon([(x - 0.85 * scale, y + 0.5 * scale), (x - 0.45 * scale, y + 1.05 * scale),
                    (x + 0.55 * scale, y + 1.05 * scale), (x + 0.95 * scale, y + 0.5 * scale)],
                   closed=True, facecolor=WHITE, edgecolor=color, linewidth=0.85, zorder=4)
    ax.add_patch(roof)
    for xx in [x - 0.88 * scale, x + 0.88 * scale]:
        ax.add_patch(Circle((xx, y - 0.68 * scale), 0.25 * scale, facecolor=color, edgecolor=color, zorder=5))


def collision_icon(ax, x, y, color=NAVY, scale=1.0):
    car_icon(ax, x - 1.15 * scale, y + 0.2 * scale, color, 0.62 * scale)
    car_icon(ax, x + 1.15 * scale, y - 0.25 * scale, color, 0.62 * scale)
    for angle in [0, 45, 90, 135]:
        rad = mpl.transforms.Affine2D().rotate_deg_around(x, y, angle)
        line = mpl.lines.Line2D([x, x + 0.55 * scale], [y, y], color=ORANGE, lw=0.8, zorder=7)
        line.set_transform(rad + ax.transData)
        ax.add_line(line)


def split_icon(ax, x, y, color=NAVY):
    for i, frac in enumerate([0.30, 0.22, 0.18, 0.16, 0.14]):
        theta1 = 90 - 360 * sum([0.30, 0.22, 0.18, 0.16, 0.14][:i + 1])
        theta2 = 90 - 360 * sum([0.30, 0.22, 0.18, 0.16, 0.14][:i])
        ax.add_patch(mpl.patches.Wedge((x, y), 2.2, theta1, theta2,
                                      facecolor=mpl.colors.to_rgba(color, 0.34 + 0.12 * i),
                                      edgecolor=WHITE, lw=0.45, zorder=4))
    ax.add_patch(Circle((x, y), 2.2, facecolor="none", edgecolor=color, lw=0.8, zorder=5))


def admission_icon(ax, x, y, color=NAVY):
    ax.add_patch(Circle((x, y), 2.25, facecolor="none", edgecolor=GRID, lw=0.7, zorder=3))
    points = [(-1.25, -0.65), (-0.55, 0.95), (0.25, -0.15), (0.95, 0.72), (1.35, -0.82), (-1.1, 0.25), (0.55, -1.1)]
    for idx, (dx, dy) in enumerate(points):
        admitted = idx in {0, 1, 2, 3, 5}
        ax.add_patch(Circle((x + dx, y + dy), 0.26 if admitted else 0.20,
                            facecolor=color if admitted else WHITE,
                            edgecolor=color if admitted else MID,
                            lw=0.7, zorder=5))
    ax.plot([x - 1.65, x + 1.65], [y - 1.45, y + 1.25], color=ORANGE, lw=0.75, zorder=4)


def null_icon(ax, x, y):
    ax.plot([x - 2.7, x + 2.7], [y, y], color=GRID, lw=0.75, zorder=2)
    observed = [(-2.2, 0.7), (-1.6, -0.4), (-0.9, 0.35), (-0.3, -0.75), (0.45, 0.6), (1.0, -0.15)]
    shadows = [(-1.95, 0.15), (-1.35, 0.58), (-0.65, -0.15), (0.05, 0.2), (0.75, -0.55), (1.45, 0.25)]
    for dx, dy in shadows:
        ax.add_patch(Polygon([(x + dx, y + dy + 0.22), (x + dx + 0.22, y + dy),
                              (x + dx, y + dy - 0.22), (x + dx - 0.22, y + dy)],
                             closed=True, facecolor=ORANGE, edgecolor=ORANGE, zorder=4))
    for dx, dy in observed:
        ax.add_patch(Circle((x + dx, y + dy), 0.18, facecolor=NAVY, edgecolor=NAVY, zorder=5))
    arrow(ax, x - 2.0, y - 1.35, x + 2.05, y - 1.35, color=ORANGE, lw=0.8, dashed=True, mutation=6)


def bounds_icon(ax, x, y, color=NAVY):
    rows = [(-1.5, 1.0, 1.7), (-1.2, 0.15, 1.1), (-0.8, -0.7, 1.45)]
    for lo, yy, hi in rows:
        ax.plot([x + lo, x + hi], [y + yy, y + yy], color=color, lw=1.15, zorder=4)
        ax.add_patch(Circle((x + lo, y + yy), 0.19, facecolor=WHITE, edgecolor=color, lw=0.8, zorder=5))
        ax.add_patch(Circle((x + hi, y + yy), 0.19, facecolor=color, edgecolor=color, lw=0.8, zorder=5))
    ax.plot([x - 1.7, x + 1.9], [y - 1.35, y - 1.35], color=GRID, lw=0.7, zorder=2)


def partial_order_icon(ax, x, y, color=NAVY):
    nodes = [(x - 1.55, y - 0.9), (x - 0.1, y - 0.15), (x + 1.4, y + 0.9),
             (x + 1.55, y - 0.75), (x - 0.65, y + 1.15)]
    edges = [(0, 1), (1, 2), (1, 3), (4, 2)]
    for a, b in edges:
        arrow(ax, *nodes[a], *nodes[b], color=color, lw=0.7, mutation=5, z=3)
    for idx, (xx, yy) in enumerate(nodes):
        ax.add_patch(Circle((xx, yy), 0.32, facecolor=color if idx in {2, 3} else WHITE,
                            edgecolor=color, lw=0.8, zorder=5))


def grid_icon(ax, x, y, rows=5, cols=9, color=GREEN):
    cell_w, cell_h = 0.46, 0.42
    x0 = x - cols * cell_w / 2
    y0 = y - rows * cell_h / 2
    for r in range(rows):
        for c in range(cols):
            alpha = 0.24 + 0.65 * ((r * cols + c) % 7 == 0)
            ax.add_patch(Rectangle((x0 + c * cell_w, y0 + r * cell_h), cell_w - 0.06, cell_h - 0.06,
                                   facecolor=mpl.colors.to_rgba(color, alpha), edgecolor="none", zorder=4))


def pag_icon(ax, x, y, color=GREEN):
    nodes = [(x - 2.0, y + 0.8), (x - 0.9, y + 1.5), (x + 0.1, y + 0.55),
             (x + 1.4, y + 1.35), (x + 2.0, y - 0.3), (x + 0.5, y - 1.2),
             (x - 1.4, y - 0.8)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 4), (2, 5), (5, 6), (0, 6), (1, 6)]
    for a, b in edges:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color=MID, lw=0.65, zorder=2)
    for idx, (xx, yy) in enumerate(nodes):
        ax.add_patch(Circle((xx, yy), 0.28, facecolor=color if idx == 4 else WHITE,
                            edgecolor=color, lw=0.75, zorder=4))


def neighborhood_icon(ax, x, y, color=GREEN):
    ax.add_patch(Circle((x, y), 0.72, facecolor=color, edgecolor=color, zorder=6))
    text(ax, x, y, "Y", size=6.2, weight="bold", color=WHITE, z=7)
    direct = [(x - 2.1, y + 0.8), (x + 2.0, y + 0.9), (x - 1.7, y - 1.2)]
    two_step = [(x + 3.3, y + 1.55), (x + 3.4, y - 0.75), (x - 3.2, y - 1.7)]
    for xx, yy in direct:
        ax.plot([x, xx], [y, yy], color=color, lw=0.8, zorder=3)
        ax.add_patch(Circle((xx, yy), 0.38, facecolor=color, edgecolor=color, zorder=5))
    for idx, (xx, yy) in enumerate(two_step):
        dx, dy = direct[idx]
        ax.plot([dx, xx], [dy, yy], color=TEAL, lw=0.75, zorder=3)
        ax.add_patch(Circle((xx, yy), 0.38, facecolor=WHITE, edgecolor=TEAL, lw=0.8, zorder=5))


def pattern_glyph(ax, x, y, kind):
    if kind == "direct":
        ax.add_patch(Circle((x - 0.55, y), 0.42, facecolor=NAVY, edgecolor=NAVY, zorder=5))
        ax.plot([x - 0.05, x + 0.55], [y, y], color=PURPLE, lw=1.2, zorder=4)
        ax.add_patch(Circle((x + 0.9, y), 0.42, facecolor=GREEN, edgecolor=GREEN, zorder=5))
    elif kind == "two":
        ax.add_patch(Circle((x - 0.75, y), 0.42, facecolor=NAVY, edgecolor=NAVY, zorder=5))
        ax.plot([x - 0.3, x + 0.1], [y, y], color=PURPLE, lw=1.0, zorder=4)
        ax.add_patch(Circle((x + 0.45, y), 0.30, facecolor=WHITE, edgecolor=GREEN, lw=0.9, zorder=5))
        ax.plot([x + 0.78, x + 1.18], [y, y], color=GREEN, lw=1.0, zorder=4)
        ax.add_patch(Circle((x + 1.48, y), 0.36, facecolor=GREEN, edgecolor=GREEN, zorder=5))
    elif kind == "predictive":
        ax.add_patch(Circle((x - 0.45, y), 0.46, facecolor=NAVY, edgecolor=NAVY, zorder=5))
        ax.plot([x + 0.15, x + 1.2], [y, y], color=GRID, lw=1.0, linestyle="--", zorder=3)
        ax.add_patch(Circle((x + 1.45, y), 0.38, facecolor=WHITE, edgecolor=GRID, lw=0.8, zorder=4))
    else:
        ax.add_patch(Circle((x - 0.45, y), 0.40, facecolor=WHITE, edgecolor=GRID, lw=0.8, zorder=4))
        ax.plot([x + 0.05, x + 1.1], [y, y], color=GRID, lw=1.0, linestyle="--", zorder=3)
        ax.add_patch(Circle((x + 1.4, y), 0.46, facecolor=GREEN, edgecolor=GREEN, zorder=5))


def draw_data_band(ax):
    x, y, w, h = 2.4, 52.0, 95.2, 7.1
    rounded_box(ax, x, y, w, h, NAVY, face=WHITE, lw=0.9, radius=1.0)
    section_header(ax, 41.8, 58.0, 16.4, "DATA FOUNDATION", NAVY, size=7.0)

    cuts = [16.2, 35.5, 54.0, 70.5, 84.4]
    for xx in cuts:
        ax.plot([xx, xx], [52.65, 57.9], color=GRID, lw=0.65, zorder=2)

    database_icon(ax, 3.95, 54.8, NAVY, 0.56)
    text(ax, 10.3, 56.65, "Analytical sample", size=5.85, weight="bold", color=NAVY)
    text(ax, 10.3, 55.15, "N = 1,469 ADS crashes", size=5.2, weight="bold")
    text(ax, 10.3, 53.7, "14 predictors | 3 classes", size=4.95, color=MID)

    collision_icon(ax, 18.2, 54.85, NAVY, 0.58)
    text(ax, 20.5, 56.65, "Crash interaction", size=6.0, weight="bold", ha="left", color=NAVY)
    text(ax, 21.1, 55.15, "Crash counterpart", size=5.4, ha="left")
    text(ax, 21.1, 54.0, "Subject and partner movement", size=5.4, ha="left")

    road_icon(ax, 38.0, 54.85, NAVY, 0.72)
    text(ax, 40.6, 56.65, "Road context", size=6.35, weight="bold", ha="left", color=NAVY)
    text(ax, 40.6, 55.15, "Roadway | speed limit", size=5.4, ha="left")
    text(ax, 40.6, 54.0, "Surface | lighting | weather", size=5.4, ha="left")

    calendar_icon(ax, 56.5, 54.85, NAVY, 0.68)
    text(ax, 58.9, 56.65, "Temporal context", size=6.25, weight="bold", ha="left", color=NAVY)
    text(ax, 58.9, 55.15, "Month | day type", size=5.35, ha="left")
    text(ax, 58.9, 54.0, "Incident hour", size=5.35, ha="left")

    car_icon(ax, 72.6, 54.85, NAVY, 0.66)
    text(ax, 75.0, 56.65, "Vehicle history", size=5.85, weight="bold", ha="left", color=NAVY)
    text(ax, 75.0, 55.15, "Mileage", size=5.35, ha="left")
    text(ax, 75.0, 54.0, "Vehicle age", size=5.35, ha="left")

    text(ax, 91.0, 56.65, "Injury severity", size=5.9, weight="bold", color=NAVY)
    labels = [("No injury", GREEN), ("Minor injury", AMBER), ("Moderate-plus", RED)]
    for idx, (lab, col) in enumerate(labels):
        yy = 55.35 - idx * 1.13
        ax.add_patch(Rectangle((85.8, yy - 0.3), 0.6, 0.6, facecolor=col, edgecolor=col, zorder=4))
        text(ax, 86.85, yy, lab, size=5.35, ha="left")


def draw_predictive_panel(ax):
    x, y, w, h = 1.4, 27.4, 59.1, 22.1
    rounded_box(ax, x, y, w, h, NAVY, face=WHITE, lw=0.95, radius=1.45)
    section_header(ax, 7.0, 48.45, 29.0, "PREDICTIVE RELEVANCE", NAVY, "A", size=6.4)

    stage_x = [2.4, 11.6, 22.0, 30.5, 45.6]
    stage_w = [9.0, 10.2, 8.3, 14.9, 13.4]
    titles = ["1\nPARTITION", "2\nADMISSION", "3\nEXPLAINERS",
              "4\nSHADOW NULL", "5\nCREDAL ORDER"]
    for i, (sx, sw, title_value) in enumerate(zip(stage_x, stage_w, titles)):
        if i > 0:
            ax.plot([sx, sx], [31.2, 47.6], color=GRID, lw=0.65, zorder=2)
        text(ax, sx + sw / 2, 46.05, title_value, size=5.0, weight="bold", color=NAVY, linespacing=0.9)

    split_icon(ax, 6.8, 40.25, NAVY)
    text(ax, 6.8, 37.35, "Outer: 5-fold", size=5.1, weight="bold")
    text(ax, 6.8, 36.15, "Inner: 5-fold", size=5.1)

    admission_icon(ax, 16.7, 40.25, NAVY)
    text(ax, 16.7, 37.3, "7 learners", size=5.1, weight="bold")
    text(ax, 16.7, 36.05, "Class-balanced one-SE", size=4.8)

    tags = [("SHAP", BLUE), ("PFI", GREEN_2), ("LOCO", ORANGE), ("FIRM", PURPLE)]
    for idx, (lab, col) in enumerate(tags):
        yy = 42.8 - idx * 1.95
        rounded_box(ax, 23.65, yy - 0.6, 5.0, 1.25, col, face=WHITE, lw=0.85, radius=0.3, z=3)
        text(ax, 26.15, yy, lab, size=5.9, weight="bold", color=col)

    null_icon(ax, 38.05, 40.65)
    text(ax, 38.05, 43.75, "Observed | outer holdout", size=4.8, weight="bold", color=NAVY)
    text(ax, 38.05, 38.25, "49 feature-matched shadows", size=4.55, weight="bold", color=ORANGE)
    text(ax, 38.05, 37.15, "inner CV", size=4.45, color=ORANGE)
    text(ax, 38.05, 36.0, "same model | source | class | feature", size=3.95)
    rounded_box(ax, 32.3, 33.95, 11.5, 1.5, NAVY, face=NAVY, lw=0, radius=0.5, z=4)
    text(ax, 38.05, 34.7, "Relevance percentile  Z", size=4.5, weight="bold", color=WHITE)

    bounds_icon(ax, 49.7, 40.45, NAVY)
    partial_order_icon(ax, 56.4, 40.35, NAVY)
    text(ax, 52.4, 43.75, "Common support", size=4.9, weight="bold")
    text(ax, 52.4, 37.45, "Relevance bounds", size=4.95)
    text(ax, 52.4, 36.15, "Strict partial order", size=4.95)
    text(ax, 52.4, 34.8, "Maximal set", size=5.05, weight="bold", color=NAVY)

    rounded_box(ax, 3.2, 28.3, 55.2, 2.65, NAVY, face=SOFT, lw=0.75, radius=0.7, z=2)
    text(ax, 30.8, 29.62,
         "OUTPUT: RELEVANCE BOUNDS  |  PARTIAL ORDERS  |  MAXIMAL SETS",
         size=5.45, weight="bold", color=NAVY)


def draw_structural_panel(ax):
    x, y, w, h = 62.2, 27.4, 36.4, 22.1
    rounded_box(ax, x, y, w, h, GREEN, face=WHITE, lw=0.95, radius=1.45)
    section_header(ax, 68.8, 48.45, 23.5, "PAG STRUCTURAL CONTEXT", GREEN, "B", size=6.15)

    stage_x = [63.1, 74.5, 86.5]
    stage_w = [11.2, 11.8, 11.1]
    titles = ["1\nFCI GRID", "2\nPAG ENSEMBLE", "3\nLOCAL STRUCTURE"]
    for i, (sx, sw, title_value) in enumerate(zip(stage_x, stage_w, titles)):
        if i > 0:
            ax.plot([sx, sx], [31.2, 47.6], color=GRID, lw=0.65, zorder=2)
        text(ax, sx + sw / 2, 46.05, title_value, size=5.0, weight="bold", color=GREEN, linespacing=0.9)

    grid_icon(ax, 68.7, 40.9, 5, 9, GREEN)
    text(ax, 68.7, 38.3, "G-square CI tests", size=5.1, weight="bold")
    text(ax, 68.7, 36.9, "5 folds x 9 specs", size=4.65)

    pag_icon(ax, 80.4, 40.9, GREEN)
    text(ax, 80.4, 38.3, "45 observed PAGs", size=5.1, weight="bold")
    text(ax, 80.4, 36.85, "99 label permutations", size=4.35, weight="bold", color=ORANGE)
    text(ax, 80.4, 35.75, "per outer fold", size=4.25, weight="bold", color=ORANGE)
    text(ax, 80.4, 34.65, "same 9-specification grid", size=4.05, color=MID)

    neighborhood_icon(ax, 92.0, 41.0, GREEN)
    text(ax, 92.0, 38.2, "Direct adjacency", size=5.0, weight="bold")
    text(ax, 92.0, 36.95, "Within-two-step proximity", size=4.7)
    text(ax, 92.0, 35.75, "Exact-two-step bridges", size=4.7)
    text(ax, 92.0, 34.55, "Endpoint uncertainty", size=4.7)

    rounded_box(ax, 63.9, 28.3, 33.0, 2.65, GREEN, face=SOFT, lw=0.75, radius=0.7, z=2)
    text(ax, 80.4, 29.95,
         "OUTPUT: ADJACENCY  |  PROXIMITY  |  BRIDGES",
         size=4.45, weight="bold", color=GREEN)
    text(ax, 80.4, 29.05, "endpoint uncertainty", size=4.15, color=GREEN)


def draw_integration_panel(ax):
    x, y, w, h = 2.4, 5.4, 95.2, 18.3
    rounded_box(ax, x, y, w, h, PURPLE, face=WHITE, lw=0.95, radius=1.35)
    section_header(ax, 35.2, 22.6, 29.6, "CROSS-VIEW INTERPRETATION", PURPLE, "C", size=6.8)

    text(ax, 50.0, 20.6,
         "Separate inferential targets: class-specific predictive relevance and overall outcome-centered PAG structure",
         size=6.15, weight="bold", color=PURPLE)

    cards = [
        (4.2, "Predictive-structural\ncorrespondence", "Maximal feature + recurrent\ndirect adjacency", "direct", NAVY),
        (27.6, "Predictive prominence with\nstructural proximity", "Maximal feature + within-two-step\nPAG proximity", "two", TEAL),
        (51.0, "Predictive-only\nevidence", "Class-specific maximal feature\nwithout local PAG proximity", "predictive", ORANGE),
        (74.4, "Structural-only\nevidence", "PAG proximity outside the\nclass-specific predictive frontier", "structural", GREEN),
    ]
    for cx, title_value, desc, kind, col in cards:
        rounded_box(ax, cx, 9.0, 21.4, 9.4, col, face=WHITE, lw=0.9, radius=0.75, z=2)
        ax.add_patch(Rectangle((cx, 16.0), 21.4, 2.4, facecolor=col, edgecolor=col, zorder=3))
        text(ax, cx + 10.7, 17.2, title_value, size=5.75, weight="bold", color=WHITE)
        pattern_glyph(ax, cx + 9.9, 13.6, kind)
        text(ax, cx + 10.7, 10.65, desc, size=5.25, color=INK)

    rounded_box(ax, 13.5, 6.25, 73.0, 1.8, PURPLE, face=PURPLE, lw=0, radius=0.6, z=3)
    text(ax, 50.0, 7.15,
         "Correspondence and divergence are retained as distinct analytical findings.",
         size=6.1, weight="bold", color=WHITE)


def draw_flow(ax):
    # Data foundation to the two independent channels.
    ax.plot([31.0, 31.0], [52.0, 50.4], color=NAVY, lw=0.85, zorder=2)
    arrow(ax, 31.0, 50.4, 6.8, 49.55, color=NAVY, lw=0.8, mutation=7)
    ax.plot([76.0, 76.0], [52.0, 50.4], color=GREEN, lw=0.85, zorder=2)
    arrow(ax, 76.0, 50.4, 68.7, 49.55, color=GREEN, lw=0.8, mutation=7)

    # Independent outputs flow to the complementary interpretation panel.
    ax.plot([30.8, 30.8], [27.4, 25.35], color=NAVY, lw=0.85, zorder=2)
    ax.plot([30.8, 42.2], [25.35, 25.35], color=NAVY, lw=0.85, zorder=2)
    arrow(ax, 42.2, 25.35, 42.2, 23.72, color=NAVY, lw=0.8, mutation=7)
    ax.plot([80.4, 80.4], [27.4, 25.35], color=GREEN, lw=0.85, zorder=2)
    ax.plot([80.4, 57.8], [25.35, 25.35], color=GREEN, lw=0.85, zorder=2)
    arrow(ax, 57.8, 25.35, 57.8, 23.72, color=GREEN, lw=0.8, mutation=7)


def main() -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(7.7, 4.8))
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.set_aspect("auto")
    ax.axis("off")

    draw_data_band(ax)
    draw_predictive_panel(ax)
    draw_structural_panel(ax)
    draw_flow(ax)
    draw_integration_panel(ax)

    stem = OUT_DIR / "figure1_methodological_framework_corrected"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04, facecolor=WHITE)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04, facecolor=WHITE)
    png_path = stem.with_suffix(".png")
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.04, facecolor=WHITE)
    plt.close(fig)

    with Image.open(png_path) as raster:
        raster_array = np.asarray(raster.convert("RGB"))
        tifffile.imwrite(
            stem.with_suffix(".tiff"),
            raster_array,
            compression="deflate",
            resolution=(600, 600),
            resolutionunit="INCH",
        )


if __name__ == "__main__":
    main()

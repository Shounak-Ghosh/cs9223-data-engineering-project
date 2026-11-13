"""Optional plot of analysis.csv: selectivity error, CI coverage, and zero-pass rate vs. k.

matplotlib is not a project dependency; run with:

    uv run --with matplotlib python -m jev_milestone_3.plot results/milestone3/acting_quality-analysis
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
TARGET = "#8a8984"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument(
        "--output", type=Path, help="Default: <analysis_dir>/analysis.png"
    )
    args = parser.parse_args(argv)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        parser.error("matplotlib missing; run via `uv run --with matplotlib ...`")

    with (args.analysis_dir / "analysis.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    analysis = json.loads((args.analysis_dir / "analysis.json").read_text())
    k = [int(r["k"]) for r in rows]

    def col(name: str) -> list[float]:
        return [float(r[name]) for r in rows]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), facecolor=SURFACE)
    line = {"color": SERIES, "linewidth": 2, "marker": "o", "markersize": 5}

    def finish(ax, title: str, ylabel: str) -> None:
        ax.set_facecolor(SURFACE)
        ax.set_title(title, color=INK, loc="left", fontsize=11)
        ax.set_xlabel("sample size k")
        ax.set_ylabel(ylabel)
        ax.set_xticks(k)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    ax = axes[0]
    ax.plot(k, col("mean_abs_error_selectivity"), **line)
    ax.plot(k, col("p90_abs_error_selectivity"), linestyle="--", **line)
    for name, label in (
        ("mean_abs_error_selectivity", "mean"),
        ("p90_abs_error_selectivity", "p90"),
    ):
        ax.annotate(
            label,
            (k[-1], col(name)[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK_SECONDARY,
        )
    ax.set_ylim(bottom=0)
    finish(ax, "Selectivity absolute error", "|sample - full| pass rate")

    ax = axes[1]
    ax.axhline(0.95, color=TARGET, linewidth=1, linestyle=":")
    ax.annotate(
        "nominal 0.95",
        (k[-1], 0.95),
        xytext=(0, -12),
        textcoords="offset points",
        ha="right",
        color=INK_SECONDARY,
        fontsize=9,
    )
    ax.plot(k, col("ci_coverage"), **line)
    ax.set_ylim(min(0.8, min(col("ci_coverage")) - 0.03), 1.01)
    finish(ax, "95% Wilson CI coverage", "fraction of draws")

    ax = axes[2]
    ax.plot(k, col("zero_pass_fraction"), **line)
    ax.set_ylim(0, max(0.05, max(col("zero_pass_fraction")) * 1.15))
    finish(ax, "Samples with zero passes", "fraction of draws")

    ref = analysis["reference"]
    fig.suptitle(
        f"{analysis['filter']}: full-data pass rate {ref['selectivity']:.2f} "
        f"(N={analysis['population_size']}, {analysis['draws']} draws per k)",
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    output = args.output or args.analysis_dir / "analysis.png"
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

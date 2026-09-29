#!/usr/bin/env python3
"""Refresh report figures from completed benchmark results, without adding run dumps."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
FIGURES = REPORT / "figures"


def main():
    sys.path.insert(0, str(REPORT))
    import figures_mech as figures

    FIGURES.mkdir(parents=True, exist_ok=True)
    rendered = []
    for name, build in (("supremacy_map", figures.fig_supremacy_map),
                        ("e68_recall_training", figures.fig_e68_recall_training)):
        fig = build()
        if fig is None:
            continue
        fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        rendered.append(name)
    print("Refreshed report visualizations: " + ", ".join(rendered), flush=True)


if __name__ == "__main__":
    main()

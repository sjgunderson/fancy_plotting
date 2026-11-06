"""Offline count display, power weighting, and unweighted ratio diagnostics."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from fancy_plotting import plot_counts
from fit_plotting_demo import make_spectra


def main():
    spectra, models = make_spectra()
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig = plt.figure(figsize=(12, 6), layout="constrained")
        subfigures = fig.subfigures(1, 2)
        for subfigure, power, title in zip(subfigures, (1, 2),
                                          ("Counts per bin", "Counts per bin × energy")):
            axes = subfigure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            result = plot_counts([[1, 2], [3, 4]], datasets=spectra, models=models,
                                 ax=axes, res=6, power=power, rebin=2,
                                 color=["#205d91", "#b45b25"], marker=".", markersize=3,
                                 linestyle="none", elinewidth=.6,
                                 label=["Group A", "Group B"],
                                 model_styles={"linestyle": "solid", "linewidth": 1.4},
                                 legend=False)
            result.ax.set(yscale="log", title=title)
            result.ax.legend(frameon=False)
            result.residual_ax.set(xlim=(.6, 7), ylim=(.7, 1.3))
            result.residual_ax.grid(axis="y", alpha=.15)
        target = Path(__file__).with_name("counts_plotting_demo.png")
        fig.savefig(target, dpi=150)
        plt.close(fig)
        print(target)


if __name__ == "__main__":
    main()

"""Standalone visualization example; needs NumPy + Matplotlib, not Sherpa."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from fancy_plotting import SpectrumData, plot_data


def make_spectra():
    rng = np.random.default_rng(2026)
    edges = np.linspace(.6, 7, 161)
    e = (edges[:-1]+edges[1:])/2
    spectra = {}
    for key, exposure in enumerate([4000, 5000, 3500, 6500], 1):
        epoch = 1 if key < 3 else 1.35
        rate = epoch * (.8*e**(-1.3) + 2*np.exp(-.5*((e-1.85)/.055)**2)
                        + .6*np.exp(-.5*((e-6.4)/.07)**2))
        counts = rng.poisson(rate*np.diff(edges)*exposure)
        spectra[key] = SpectrumData(edges[:-1], edges[1:], counts, exposure)
    return spectra


def main():
    spectra = make_spectra()
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelsize": 10}):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
        plot_data([1, 2, 3, 4], datasets=spectra, ax=axes[0],
                  color=["#6482a3", "#adc0d1", "#d18a56", "#e5bea1"],
                  label=["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4"],
                  linewidth=.9, yerrorbars=False, ylog=True, rebin=2,
                  title="Four outer entries: four curves")
        plot_data([[1, 2], [3, 4]], datasets=spectra, ax=axes[1],
                  combine="exposure", color=["#205d91", "#b45b25"],
                  label=["1, 2: pooled rate", "3, 4: pooled rate"],
                  marker=["o", "s"], mfc="none", ms=3, capsize=1.5, elinewidth=.7,
                  yscale="log", rebin=2, title="Matplotlib markers and error bars")
        plot_data([[1, 2], [3, 4]], datasets=spectra, ax=axes[2],
                  combine=["sum", "mean"], color=["#205d91", "#b45b25"],
                  label=["1, 2: sum of rates", "3, 4: mean of rates"],
                  yscale="log", rebin=[2, 4],
                  styles=[{"linestyle": "solid", "ecolor": (.2, .4, .6)},
                          {"linestyle": (0, (5, 2)), "alpha": .85}],
                  title="Per-group styles and custom dashes")
        for ax in axes:
            ax.set_xlim(.6, 7)
            ax.set_ylim(.035, 7)
            ax.grid(axis="y", which="major", alpha=.15)
            ax.legend(frameon=False, fontsize=9, loc="upper right")
        target = Path(__file__).with_name("plotting_demo.png")
        fig.savefig(target, dpi=150)
        print(target)
        plt.close(fig)


if __name__ == "__main__":
    main()

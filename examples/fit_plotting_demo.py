"""Offline folded-count demonstration of ISIS residual modes 1, 4, and 6."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from fancy_plotting import SpectrumData, plot_data
from fancy_plotting.resample import register_data_match


def make_spectra():
    rng = np.random.default_rng(1408)
    spectra, models = {}, {}
    for key, (bins, exposure, normalization) in enumerate(
            [(120, 4000, .9), (180, 4000, 1.1), (120, 4500, 1.), (180, 5500, 1.)], 1):
        edges = np.linspace(.6, 7., bins+1)
        center = (edges[:-1]+edges[1:])/2
        template = .8*center**(-1.3)+1.8*np.exp(-.5*((center-1.85)/.07)**2)
        prediction = template*np.diff(edges)*exposure
        expected = prediction*normalization
        spectra[key] = SpectrumData(edges[:-1], edges[1:], rng.poisson(expected), exposure)
        models[key] = prediction
    register_data_match(spectra, reference=1)
    return spectra, models


def main():
    spectra, models = make_spectra()
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig = plt.figure(figsize=(15, 6.2), layout="constrained")
        subfigures = fig.subfigures(1, 3)
        for subfigure, mode, title in zip(subfigures, (1, 4, 6),
                                         ("res=1: individual diagnostics", "res=4: combined diagnostics",
                                          "res=6: combined data/model")):
            axes = subfigure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            residual_styles = [{"color": c, "alpha": .8} for c in
                               ("#184e77", "#66a3c5", "#aa4f1e", "#dca273")] if mode == 1 else None
            result = plot_data([[1, 2], [3, 4]], datasets=spectra, models=models, ax=axes,
                               res=mode, combine="exposure", rebin=2,
                               color=["#205d91", "#b45b25"], marker=".", linestyle="none",
                               markersize=3, elinewidth=.6, label=["Group A", "Group B"],
                               model_styles={"linestyle": "solid", "linewidth": 1.4},
                               residual_styles=residual_styles, legend=False)
            result.ax.set(yscale="log", ylim=(.035, 4), title=title)
            result.ax.legend(frameon=False, fontsize=8)
            result.residual_ax.set_xlim(.6, 7)
            result.residual_ax.set_ylim((.7, 1.3) if mode == 6 else (-4, 4))
            result.residual_ax.grid(axis="y", alpha=.15)
        target = Path(__file__).with_name("fit_plotting_demo.png")
        fig.savefig(target, dpi=150)
        plt.close(fig)
        print(target)


if __name__ == "__main__":
    main()

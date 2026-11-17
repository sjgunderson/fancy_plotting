"""Offline side-by-side residual plots with independently editable panels."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np

import fancy_plotting as fp


def main():
    rng = np.random.default_rng(17)
    edges = np.geomspace(.25, 6., 121)
    energy = (edges[:-1]+edges[1:])/2
    exposure = 4000.
    datasets, models = {}, {}
    for key, normalization in enumerate((1., .25, .8, .2), 1):
        rate = normalization*energy**(-.8)*np.exp(-energy/1.2)
        rate += normalization*.04*np.exp(-.5*((energy-2.4)/.06)**2)
        prediction = rate*np.diff(edges)*exposure
        datasets[key] = fp.SpectrumData(edges[:-1], edges[1:], rng.poisson(prediction), exposure)
        models[key] = prediction

    fig, slots = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    results = [fp.plot_data(ids, datasets=datasets, models=models, ax=slot,
                           res=4, residual_stat="cstat", xlog=True, ylog=True,
                           xlim=(.25, 6), ylim=(1.e-4, 4), color=["black", "m"],
                           label=["PN", "MOS"], yerrorbars=False,
                           model_styles=[{"color": "green"}, {"color": "darkorange"}],
                           title=title)
               for ids, slot, title in zip(([1, 2], [3, 4]), slots,
                                           ("Observation A", "Observation B"))]
    for result in results:
        result.residual_ax.set_ylim(-4, 4)
        result.residual_ax.xaxis.set_minor_locator(LogLocator(subs=[2, 3, 4, 6, 8]))
        result.residual_ax.xaxis.set_minor_formatter(ScalarFormatter())

    # Each result contains only its own two axes, regardless of figure order.
    right_data, right_residual = results[1].axes
    right_data.set_ylabel("")
    right_residual.set_ylabel("")
    for panel in results[1].axes:
        panel.tick_params(axis="y", which="both", labelleft=False)

    target = Path(__file__).with_suffix(".png")
    fig.savefig(target, dpi=150)
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()

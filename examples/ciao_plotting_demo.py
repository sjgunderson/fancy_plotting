"""Run inside CIAO with Matplotlib: no model or input files required."""
import numpy as np
from matplotlib import pyplot as plt
from sherpa.astro import ui
from sherpa.astro.data import DataPHA
from sherpa.astro.instrument import create_arf, create_delta_rmf

from fancy_plotting import match_dataset_grids, plot_data


def main():
    ui.clean()
    for key, bins in enumerate([64, 64, 128, 128], 1):
        edges = np.linspace(.5, 7, bins+1)
        exposure = 500 + 100*key
        expected = .08*(edges[1:]**(-.7)-edges[:-1]**(-.7))/(-.7)*30*exposure
        pha = DataPHA(f"spectrum_{key}", np.arange(1, bins+1),
                      np.random.default_rng(key).poisson(expected), exposure=exposure)
        pha.set_response(create_arf(edges[:-1], edges[1:]),
                         create_delta_rmf(edges[:-1], edges[1:],
                                          e_min=edges[:-1], e_max=edges[1:]))
        ui.set_data(key, pha)
    # The default PHA analysis can be channel, energy, or wavelength.
    # No unit setting is needed for matching and no model is required.
    matched = match_dataset_grids([1, 2, 3, 4])
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for quantity, ax in zip(("energy", "wavelength", "channel"), axes):
        ui.set_analysis(quantity)
        result = plot_data([[1, 2], [3, 4]], ax=ax,
                           combine=["sum", "mean"],
                           color=["tab:blue", "tab:orange"],
                           marker=["o", "s"], markerfacecolor="none",
                           capsize=2, elinewidth=.8,
                           label=["Datasets 1+2: sum", "Datasets 3+4: mean"],
                           rebin=[2, 4], ylog=True,
                           title=f"One matching plan; set_analysis('{quantity}')")
        print(quantity, [curve.members for curve in result.curves])
    assert matched.shared is None
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

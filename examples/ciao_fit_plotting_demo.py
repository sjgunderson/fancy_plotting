"""Inside CIAO: source assignment and all six ISIS-style residual modes."""
import numpy as np
from matplotlib import pyplot as plt
from sherpa.astro import ui
from sherpa.astro.data import DataPHA
from sherpa.astro.instrument import create_arf, create_delta_rmf
from sherpa.models.basic import PowLaw1D

from fancy_plotting import match_dataset_grids, plot_data


def main():
    ui.clean()
    for key, bins in enumerate((64, 128, 64, 128), 1):
        edges = np.linspace(.5, 7., bins+1)
        exposure = 500.+100*key
        flux = .08*(edges[1:]**(-.7)-edges[:-1]**(-.7))/(-.7)
        pha = DataPHA(f"spectrum_{key}", np.arange(1, bins+1),
                      np.random.default_rng(key).poisson(flux*30*exposure), exposure=exposure)
        pha.set_response(create_arf(edges[:-1], edges[1:], specresp=np.full(bins, 30.)),
                         create_delta_rmf(edges[:-1], edges[1:], e_min=edges[:-1], e_max=edges[1:]))
        ui.set_data(key, pha)
    # Prepare geometry before defining a model. Units require no arguments.
    matched = match_dataset_grids([1, 2, 3, 4])
    continuum = PowLaw1D("continuum")
    continuum.gamma = 1.7
    continuum.ampl = .08
    matched.set_source(continuum)
    ui.set_stat("cstat")
    ui.set_analysis("energy")
    fig = plt.figure(figsize=(15, 8), layout="constrained")
    for subfigure, mode in zip(fig.subfigures(2, 3).ravel(), range(1, 7)):
        axes = subfigure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        result = plot_data([[1, 2], [3, 4]], ax=axes, res=mode, rebin=2,
                           color=["tab:blue", "tab:orange"], marker=".", linestyle="none",
                           ylog=True, legend=False, title=f"res={mode}")
        print(mode, [residual.members for residual in result.residuals])
    print("Last plot source cache:", matched.cache_info())
    plt.show()


if __name__ == "__main__":
    main()

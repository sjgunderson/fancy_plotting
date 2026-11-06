"""Inside CIAO: matched counts, models/residuals, and scaled background views."""
import numpy as np
from matplotlib import pyplot as plt
from sherpa.astro import ui
from sherpa.astro.data import DataPHA
from sherpa.astro.instrument import create_arf, create_delta_rmf
from sherpa.models.basic import PowLaw1D

from fancy_plotting import match_dataset_grids, plot_counts


def main():
    ui.clean()
    for key, bins in enumerate((64, 128), 1):
        edges = np.linspace(.5, 7., bins+1)
        exposure = 500.+100*key
        flux = .08*(edges[1:]**(-.7)-edges[:-1]**(-.7))/(-.7)
        rng = np.random.default_rng(key)
        background = DataPHA(f"background_{key}", np.arange(1, bins+1),
                             rng.poisson(np.full(bins, 10.)), exposure=exposure, backscal=4.)
        pha = DataPHA(f"spectrum_{key}", np.arange(1, bins+1),
                      rng.poisson(flux*30*exposure+5.), exposure=exposure, backscal=2.)
        pha.set_response(create_arf(edges[:-1], edges[1:], specresp=np.full(bins, 30.)),
                         create_delta_rmf(edges[:-1], edges[1:], e_min=edges[:-1], e_max=edges[1:]))
        pha.set_background(background)
        ui.set_data(key, pha)
    matched = match_dataset_grids([1, 2])
    source = PowLaw1D("continuum")
    source.gamma = 1.7
    source.ampl = .08
    matched.set_source(source)
    ui.set_stat("chi2datavar")
    # Counts are displayed even if the Sherpa session selects rate + plot_fac.
    ui.set_analysis("energy", "rate", 2)
    fig = plt.figure(figsize=(12, 6), layout="constrained")
    left, right = fig.subfigures(1, 2)
    axes = left.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    plot_counts([[1, 2]], ax=axes, res=4, rebin=2,
                color="navy", marker=".", linestyle="none", title="Matched net counts",
                model_styles={"color": "orange", "linestyle": "solid"})
    plot_counts([[1, 2], [1, 2]], ax=right.subplots(), rebin=2,
                background=["include", "only"],
                label=["Source aperture", "Scaled background"],
                color=["black", "tab:green"], show_model=False,
                title="Measured source and background")
    plt.show()


if __name__ == "__main__":
    main()

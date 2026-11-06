"""Inside CIAO: general PHA unfolding with photon/erg and both model views."""
import numpy as np
from matplotlib import pyplot as plt
from sherpa.astro import ui
from sherpa.astro.data import DataPHA
from sherpa.astro.instrument import create_arf, create_delta_rmf
from sherpa.models.basic import PowLaw1D

from fancy_plotting import match_dataset_grids, plot_unfold


def main():
    ui.clean()
    for key, bins in enumerate((64, 128), 1):
        edges = np.linspace(.5, 7., bins+1)
        exposure = 1500.+500*key
        flux = .025*(edges[1:]**(-.6)-edges[:-1]**(-.6))/(-.6)
        area = (80+20*key)*((edges[1:]+edges[:-1])/4)**(-.15)
        pha = DataPHA(f"spectrum_{key}", np.arange(1, bins+1),
                      np.random.default_rng(key).poisson(flux*area*exposure), exposure=exposure)
        pha.set_response(create_arf(edges[:-1], edges[1:], specresp=area),
                         create_delta_rmf(edges[:-1], edges[1:], e_min=edges[:-1], e_max=edges[1:]))
        ui.set_data(key, pha)
    matched = match_dataset_grids([1, 2])
    source = PowLaw1D("continuum")
    source.gamma = 1.6
    source.ampl = .025
    matched.set_source(source)
    ui.set_stat("cstat")
    ui.set_analysis("energy", "counts", 3)  # Rate/factor do not select flux normalization.
    fig = plt.figure(figsize=(12, 6), layout="constrained")
    for subfigure, flux_unit, model_display in zip(fig.subfigures(1, 2),
                                                   ("photons", "ergs"), ("convolved", "source")):
        axes = subfigure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        plot_unfold([[1, 2]], ax=axes, flux_unit=flux_unit, model_display=model_display,
                    res=4, rebin=2, color="navy", marker=".", linestyle="none", ylog=True,
                    title=f"{flux_unit}: {model_display} model",
                    model_styles={"color": "orange", "linestyle": "solid"})
    # The existing match is valid after an analysis-coordinate change.
    ui.set_analysis("wavelength")
    plot_unfold([[1, 2]], flux_unit="ergs", power_scale="geometric", res=4, ylog=True)
    plt.show()


if __name__ == "__main__":
    main()

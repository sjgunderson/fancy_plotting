"""Run inside CIAO: python examples/ciao_demo.py. No data files required."""
import numpy as np

from sherpa.astro import ui
from sherpa.astro.data import DataPHA
from sherpa.astro.instrument import create_arf, create_delta_rmf
from sherpa.models.basic import PowLaw1D
from sherpa.optmethods import NelderMead

from fancy_plotting import combine_datasets, match_dataset_grids


def main():
    ui.clean()
    for i, bins in enumerate([64, 128, 64, 128], 1):
        edges = np.linspace(.5, 7, bins+1)
        pha = DataPHA(f"spectrum_{i}", np.arange(1, bins+1), np.zeros(bins),
                      exposure=500 + 100*i)
        pha.set_response(create_arf(edges[:-1], edges[1:],
                                    specresp=np.full(bins, 30.)),
                         create_delta_rmf(edges[:-1], edges[1:],
                                          e_min=edges[:-1], e_max=edges[1:]))
        pha.set_analysis("energy")
        # Analytic synthetic counts; no Sherpa source model exists yet.
        flux = .08 * (edges[1:]**(-.7) - edges[:-1]**(-.7)) / (-.7)
        pha.counts = np.random.default_rng(10+i).poisson(flux * 30 * pha.exposure)
        ui.set_data(i, pha)

    group = match_dataset_grids([1, 2, 3, 4])
    prepared = combine_datasets(group)  # joint mode also works before a model exists
    assert group.shared is None and prepared.shared is None
    print("Prepared common grid with", group.grid.size, "bins, without a model")

    source = PowLaw1D("continuum")
    source.gamma = 1.7
    source.gamma.freeze()
    source.ampl = .04
    group.set_source(source)
    for i in [1, 2, 3, 4]:
        ui.get_model(i)(ui.get_data(i).channel)
    print("One-vector cache counters:", group.cache_info())
    assert group.cache_info().misses == 1
    assert group.cache_info().hits == 3

    ui.set_stat("cstat")
    ui.set_method("neldermead")
    ui.fit(1, 2, 3, 4)
    print("Joint fit normalization:", source.ampl.val)
    print("Joint fit cache counters:", group.cache_info())

    source.ampl = .04
    summed = combine_datasets(group, mode="sum")
    summed.set_source(source)
    result = summed.make_fit(method=NelderMead()).fit()
    print("Summed-count fit:", result)
    # To use the UI instead: summed.register('sum'); ui.fit('sum').
    # Fitting 'sum' together with its original members would double-count data.


if __name__ == "__main__":
    main()

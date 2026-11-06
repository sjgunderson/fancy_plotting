"""Inside CIAO: match -> group_width -> combine -> define a model -> fit."""
import numpy as np
from sherpa.astro import ui
from sherpa.astro.data import DataPHA
from sherpa.astro.instrument import create_arf, create_delta_rmf
from sherpa.models.basic import PowLaw1D
from sherpa.optmethods import NelderMead

from fancy_plotting import match_dataset_grids, combine_datasets


def main():
    ui.clean()
    ids = [1, 2, 3, 4]
    edges = np.linspace(.5, 7, 129)
    for key in ids:
        exposure = 500 + 100*key
        expected = .08*(edges[1:]**(-.7)-edges[:-1]**(-.7))/(-.7)*30*exposure
        pha = DataPHA(f"spectrum_{key}", np.arange(1, 129),
                      np.random.default_rng(key).poisson(expected), exposure=exposure)
        pha.set_response(create_arf(edges[:-1], edges[1:], specresp=np.full(128, 30.)),
                         create_delta_rmf(edges[:-1], edges[1:]))
        pha.set_analysis("energy")
        ui.set_data(key, pha)
    group = match_dataset_grids(ids)  # No source model exists at matching.

    source = PowLaw1D("continuum")
    source.gamma = 1.7
    source.gamma.freeze()
    for width in (1, 2, 4, 8, 16, 32):
        for key in ids:
            pha = ui.get_data(key)
            pha.ungroup()
            pha.notice()
            ui.group_width(key, width)
        # Reuse the source-grid plan with the NEW grouping. The result is a
        # snapshot; earlier combinations keep their own grouping/data.
        combined = combine_datasets(group, mode="sum", name=f"width_{width}")
        source.ampl = .04
        combined.set_source(source)
        combined.model.startup(True)
        try:
            combined.data.eval_model_to_fit(combined.model)
            assert combined.cache_info().misses == 1
            calls = combined.cache_info().misses
        finally:
            combined.model.teardown()
        result = combined.make_fit(method=NelderMead()).fit()
        print(f"width={width:2d}, fit bins={len(combined.data.y):3d}, "
              f"source bins={combined.group.grid.size:3d}, "
              f"source calls/vector={calls}, norm={result.parvals[0]:.6f}")


if __name__ == "__main__":
    main()

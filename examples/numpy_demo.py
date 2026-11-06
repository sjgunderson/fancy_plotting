"""Independent numerical validation; runs with NumPy + SciPy, without CIAO.

Compare four separately folded synthetic spectra with a shared-grid fit.
No artificial delays or simulated Sherpa objects are used.
"""
import json

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

from fancy_plotting import EnergyGrid, SharedEvaluation, prepare_grids


class Spectrum:
    def __init__(self):
        self.calls = 0

    def __call__(self, p, lo, hi):
        self.calls += 1
        norm, gamma, line_norm = p
        continuum = norm * (hi ** (1-gamma) - lo ** (1-gamma)) / (1-gamma)
        width = .015 * np.sqrt(2)
        line = line_norm * (erf((hi-1.8)/width) - erf((lo-1.8)/width)) / 2
        return continuum + line


def main():
    rng = np.random.default_rng(17)
    grids = [EnergyGrid.from_edges(np.linspace(.5, 7, n+1)) for n in (64, 96, 128, 80)]
    # Matching depends only on bin geometry. No Spectrum instance exists yet.
    plan = prepare_grids(grids)
    master, projections = plan.grid, plan.projections
    channel_centres = np.linspace(.55, 6.95, 48)
    responses = []
    for i, grid in enumerate(grids):
        energy = (grid.lo + grid.hi) / 2
        redistribution = np.exp(-.5 * ((channel_centres[:, None] - energy) / .12)**2)
        redistribution /= redistribution.sum(axis=0)
        area = 45 * (energy / 2) ** (-.2) * (1 + .03*i)
        responses.append(redistribution * area * (600 + 100*i))
    native_source, shared_source = Spectrum(), Spectrum()
    shared = SharedEvaluation(master, shared_source)

    def native_predictions(p):
        return [r @ native_source(p, g.lo, g.hi) for r, g in zip(responses, grids)]

    def shared_predictions(p):
        return [r @ projection(shared.evaluate(p))
                for r, projection in zip(responses, projections)]

    truth = [.08, 1.85, .015]
    native = native_predictions(truth)
    accelerated = shared_predictions(truth)
    for a, b in zip(native, accelerated):
        np.testing.assert_allclose(a, b, rtol=2e-12, atol=1e-11)
    source_evaluations = {"ordinary": native_source.calls, "shared": shared_source.calls}
    observed = np.concatenate([rng.poisson(values) for values in native])
    weights = np.sqrt(np.maximum(observed, 1))

    def residual(prediction):
        return lambda p: (np.concatenate(prediction(p)) - observed) / weights

    options = dict(x0=[.06, 1.6, .01], bounds=([.001, 1.1, 0], [1, 3.5, 1]),
                   xtol=1e-11, ftol=1e-11, gtol=1e-9)
    ordinary = least_squares(residual(native_predictions), **options)
    accelerated_fit = least_squares(residual(shared_predictions), **options)
    np.testing.assert_allclose(ordinary.x, accelerated_fit.x, rtol=3e-7, atol=1e-9)
    np.testing.assert_allclose(ordinary.cost, accelerated_fit.cost, rtol=1e-11)
    print(json.dumps({
        "datasets": len(grids), "merged_bins": master.size,
        "grids_prepared_before_source_creation": True,
        "source_calls_for_one_parameter_vector": source_evaluations,
        "ordinary_fit_parameters": ordinary.x.tolist(),
        "shared_fit_parameters": accelerated_fit.x.tolist(),
        "ordinary_chisquare": float(2*ordinary.cost),
        "shared_chisquare": float(2*accelerated_fit.cost),
        "validation": "PASS"
    }, indent=2))


if __name__ == "__main__":
    main()

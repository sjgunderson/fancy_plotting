"""Validate and time ordinary versus shared-grid fits to the SAME count sum.

Requires real Sherpa. Run from a checkout with its package installed:
    python examples/benchmark_combined_fit.py --repeats 5 --output benchmark-results.json

This synthetic thermal continuum/line forest is not an AtomDB plasma model.
No artificial delays are used. A timing speedup is measured, never required.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import numpy as np
from numpy.testing import assert_allclose
import scipy
from scipy.special import erf
import sherpa
from sherpa.astro.data import DataARF, DataPHA, DataRMF
from sherpa.astro.instrument import Response1D
from sherpa.fit import Fit
from sherpa.models.model import ArithmeticModel
from sherpa.models.parameter import Parameter
from sherpa.optmethods import NelderMead
from sherpa.stats import CStat

import fancy_plotting
from fancy_plotting import match_pha_grids, combine_datasets
from fancy_plotting.adapter import _CombinedCountsModel


TRUTH = (.035, 1.5, 1.)
START = (.028, 1.1, .8)


class ThermalLineForest(ArithmeticModel):
    """Bin-integrated continuum plus temperature-dependent Gaussian lines."""
    ndim = 1

    def __init__(self, name, lines):
        self.norm = Parameter(name, "norm", START[0], 1.e-5, 1.)
        self.kT = Parameter(name, "kT", START[1], .2, 5.)
        self.metal = Parameter(name, "metal", START[2], .05, 3.5)
        self.centroids = np.geomspace(.56, 6.8, lines)
        self.sigma = .003 + self.centroids*.001
        self.excitation = .2 + self.centroids
        self.calls, self.evaluated_bins, self.kernel_seconds = 0, 0, 0.
        super().__init__(name, (self.norm, self.kT, self.metal))

    def reset(self):
        self.calls, self.evaluated_bins, self.kernel_seconds = 0, 0, 0.

    def calc(self, p, lo, hi=None, *args, **kwargs):
        start = perf_counter()
        self.calls += 1
        lo, hi = np.asarray(lo), np.asarray(hi)
        self.evaluated_bins += lo.size
        norm, temperature, metal = p
        continuum = temperature*(np.exp(-lo/temperature)-np.exp(-hi/temperature))
        width = np.sqrt(2)*self.sigma
        profiles = .5*(erf((hi[:, None]-self.centroids)/width) -
                        erf((lo[:, None]-self.centroids)/width))
        strengths = np.exp(-self.excitation/temperature)/np.sqrt(temperature)
        result = norm*(continuum + metal*np.sum(profiles*strengths, axis=1)/8.)
        self.kernel_seconds += perf_counter()-start
        return result


def make_pha(name, nenergy, index, nchan=96):
    edges = np.linspace(.5, 7., nenergy+1)
    channels = np.linspace(.5, 7., nchan+1)
    energy = (edges[:-1]+edges[1:])/2
    center = (channels[:-1]+channels[1:])/2
    matrix = np.exp(-.5*((energy[:, None]-center)/(.07+.01*index))**2)
    matrix /= matrix.sum(axis=1)[:, None]
    rmf = DataRMF(name+"_rmf", nchan, edges[:-1], edges[1:],
                  np.ones(nenergy, dtype=int), np.ones(nenergy, dtype=int),
                  np.full(nenergy, nchan, dtype=int), matrix.ravel(), offset=1,
                  e_min=channels[:-1], e_max=channels[1:])
    arf = DataARF(name+"_arf", edges[:-1], edges[1:],
                  30*(energy/2.)**(-.2)*(1+.06*index))
    pha = DataPHA(name, np.arange(1, nchan+1), np.zeros(nchan),
                  exposure=1500.+300*index, areascal=1.)
    pha.set_response(arf, rmf)
    pha.set_analysis("energy")
    return pha


def prediction(data, model):
    model.startup(True)
    try:
        return np.array(data.eval_model(model), copy=True)
    finally:
        model.teardown()


def set_parameters(source, values):
    for par, value in zip(source.pars, values):
        par.val = value


def timed_fit(data, model, source):
    set_parameters(source, START)
    source.reset()
    method = NelderMead()
    method.config.update(ftol=1.e-8, maxfev=4000)
    fit = Fit(data, model, stat=CStat(), method=method)
    start = perf_counter()
    result = fit.fit()
    elapsed = perf_counter()-start
    if not result.succeeded:
        raise AssertionError(f"Fit did not converge: {result.message}")
    return {"seconds": elapsed, "kernel_seconds": source.kernel_seconds,
            "source_calls": source.calls, "evaluated_source_bins": source.evaluated_bins,
            "nfev": int(result.nfev), "parameters": list(map(float, result.parvals)),
            "statistic": float(result.statval), "numpoints": int(result.numpoints),
            "dof": int(result.dof)}


def benchmark_case(name, sizes, *, repeats, lines, seed):
    rng = np.random.default_rng(seed)
    truth = ThermalLineForest("truth", lines)
    set_parameters(truth, TRUTH)
    phas = [make_pha(f"spectrum_{i}", size, i) for i, size in enumerate(sizes)]
    for pha in phas:
        pha.counts = rng.poisson(Response1D(pha)(truth)(pha.channel)).astype(float)

    setup_start = perf_counter()
    combined = combine_datasets(match_pha_grids(phas), mode="sum")
    ordinary_source = ThermalLineForest("ordinary", lines)
    shared_source = ThermalLineForest("shared", lines)
    combined.set_source(shared_source)
    native_models = [Response1D(pha)(ordinary_source) for pha in combined.group.datasets.values()]
    # Identical aggregation, response snapshots, selection, and projections.
    # Only the unconvolved source evaluation differs from the public model.
    ordinary_model = _CombinedCountsModel(native_models, tuple(combined.group.datasets.values()),
        combined.projections, combined.orders, combined.grid, "ordinary_combined_model")
    setup_seconds = perf_counter()-setup_start
    if combined.fractional:
        raise AssertionError("This benchmark requires a whole-bin Poisson count sum")

    ordinary_source.reset()
    ordinary_prediction = prediction(combined.data, ordinary_model)
    ordinary_calls = ordinary_source.calls
    shared_source.reset()
    shared_prediction = prediction(combined.data, combined.model)
    shared_calls = shared_source.calls
    assert_allclose(shared_prediction, ordinary_prediction, rtol=2.e-10, atol=1.e-10)
    if ordinary_calls != len(phas) or shared_calls != 1:
        raise AssertionError(f"Expected {len(phas)} ordinary source calls and one shared call; "
                             f"got {ordinary_calls} and {shared_calls}")

    models = {"ordinary": (ordinary_model, ordinary_source), "shared": (combined.model, shared_source)}
    # Warm both complete fitting paths once; exclude their times from results.
    for model, source in models.values():
        timed_fit(combined.data, model, source)
    trials = []
    for repeat in range(repeats):
        order = ("ordinary", "shared") if repeat % 2 == 0 else ("shared", "ordinary")
        trial = {key: timed_fit(combined.data, *models[key]) for key in order}
        a, b = trial["ordinary"], trial["shared"]
        assert_allclose(a["parameters"], b["parameters"], rtol=2.e-4, atol=2.e-7)
        assert_allclose(a["statistic"], b["statistic"], rtol=1.e-7, atol=5.e-6)
        if (a["numpoints"], a["dof"]) != (b["numpoints"], b["dof"]):
            raise AssertionError("Fits used different points or degrees of freedom")
        trials.append(trial)

    times = {key: [trial[key]["seconds"] for trial in trials] for key in models}
    medians = {key: median(values) for key, values in times.items()}
    return {"case": name, "native_source_bins": sizes, "shared_source_bins": combined.group.grid.size,
            "combined_detector_bins": combined.grid.size, "datasets": len(phas),
            "lines": lines, "seed": seed, "setup_seconds_excluded": setup_seconds,
            "source_calls_per_parameter_vector": {"ordinary": ordinary_calls, "shared": shared_calls},
            "maximum_prediction_absolute_difference": float(np.max(np.abs(
                shared_prediction-ordinary_prediction))),
            "fit_seconds": {key: {"median": medians[key], "minimum": min(values), "maximum": max(values)}
                            for key, values in times.items()},
            "speedup_ordinary_over_shared": medians["ordinary"]/medians["shared"],
            "trials": trials, "validation": "PASS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--lines", type=int, default=96)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    args = parser.parse_args()
    if args.repeats < 2 or args.lines < 1:
        parser.error("use at least two timed repeats and one spectral line")
    cases = [("aligned", [512]*4), ("nested", [256, 512, 1024, 2048]),
             ("staggered", [512, 519, 526, 533])]
    results = [benchmark_case(name, sizes, repeats=args.repeats, lines=args.lines, seed=710+i)
               for i, (name, sizes) in enumerate(cases)]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "machine": platform.machine(), "logical_cpus": os.cpu_count(),
                              "numpy": np.__version__, "scipy": scipy.__version__,
                              "sherpa": sherpa.__version__, "fancy_plotting": fancy_plotting.__version__,
                              "thread_environment": {name: os.environ.get(name) for name in
                                  ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
              "statistic": "cstat", "optimizer": "neldermead", "initial_parameters": START,
              "truth_parameters": TRUTH, "timed_repeats": args.repeats,
              "warmups_per_method": 1, "results": results,
              "timing_is_a_pass_criterion": False, "validation": "PASS"}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print("Combined-count fit benchmark (identical counts, responses, statistic, and start)")
    print("Times exclude preparation and one warm-up per method; median of", args.repeats, "fits.")
    for result in results:
        a, b = result["fit_seconds"]["ordinary"], result["fit_seconds"]["shared"]
        print(f'{result["case"]}: ordinary {a["median"]:.4f}s [{a["minimum"]:.4f}, {a["maximum"]:.4f}]; '
              f'shared {b["median"]:.4f}s [{b["minimum"]:.4f}, {b["maximum"]:.4f}]; '
              f'speedup {result["speedup_ordinary_over_shared"]:.3f}x; '
              f'source calls/vector {result["source_calls_per_parameter_vector"]}; '
              f'grid bins {result["native_source_bins"]} -> {result["shared_source_bins"]}; PASS')
    print("Timing gains depend on grid overlap, source cost, response folding, and hardware.")
    print("Full measurements and parameter/statistic comparisons:", args.output)


if __name__ == "__main__":
    main()

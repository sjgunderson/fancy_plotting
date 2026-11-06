"""Offline generalized-response example: corrected and intrinsic model views."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import ndtr

from fancy_plotting import SpectrumData, SourceSpectrum, plot_unfold
from fancy_plotting.resample import register_data_match
from fancy_plotting.units import HC_KEV_ANGSTROM as HC


def make_spectra():
    spectra, models, weights, sources = {}, {}, {}, {}
    for key, (nbin, nchan, exposure) in enumerate(((300, 100, 1800), (320, 120, 2400)), 1):
        edges = np.linspace(.5, 7., nbin+1)
        detector = np.linspace(.5, 7., nchan+1)
        lo, hi = edges[:-1], edges[1:]
        center = (lo+hi)/2
        flux = .025*(hi**(-.6)-lo**(-.6))/(-.6)
        flux += .006*(ndtr((hi-1.85)/.025)-ndtr((lo-1.85)/.025))
        sigma = .07+.025*center
        matrix = ndtr((detector[None, 1:]-center[:, None])/sigma[:, None])- \
            ndtr((detector[None, :-1]-center[:, None])/sigma[:, None])
        matrix /= matrix.sum(axis=1)[:, None]
        area = (80+20*key)*(center/2)**(-.15)*(1-.35*(center > 2.2))
        predicted = exposure*(matrix.T @ (area*flux))
        weights[key] = exposure*(matrix.T @ (area*HC*(1/lo-1/hi)))
        spectra[key] = SpectrumData(detector[:-1], detector[1:],
                                    np.random.default_rng(200+key).poisson(predicted), exposure)
        sources[key] = SourceSpectrum(lo, hi, flux)
        models[key] = predicted
    register_data_match(spectra, reference=1)
    return spectra, models, weights, sources


def main():
    spectra, models, weights, sources = make_spectra()
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig = plt.figure(figsize=(15, 6.2), layout="constrained")
        choices = (("photons", "convolved", "Photon density · corrected model"),
                   ("ergs", "convolved", "Energy flux · corrected model"),
                   ("ergs", "source", "Energy flux · intrinsic model"))
        for subfig, (flux_unit, display, title) in zip(fig.subfigures(1, 3), choices):
            axes = subfig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            result = plot_unfold([[1, 2]], datasets=spectra, models=models, response_weights=weights,
                                 source_spectra=sources if display == "source" else None,
                                 flux_unit=flux_unit, model_display=display, power_scale="geometric",
                                 ax=axes, res=4, rebin=2, color="#205d91", marker=".",
                                 markersize=4, linestyle="none", elinewidth=.6,
                                 label="Combined spectra", legend=False,
                                 model_styles={"color": "#b45b25", "linestyle": "solid", "linewidth": 1.5})
            result.ax.set(yscale="log", title=title)
            result.ax.legend(frameon=False, fontsize=8)
            result.residual_ax.set(xlim=(.5, 7), ylim=(-3.5, 3.5))
            result.residual_ax.grid(axis="y", alpha=.15)
        target = Path(__file__).with_name("unfold_plotting_demo.png")
        fig.savefig(target, dpi=150)
        plt.close(fig)
        print(target)


if __name__ == "__main__":
    main()

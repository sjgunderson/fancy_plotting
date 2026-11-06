"""ISIS-style linear-response flux correction, independent of instrument names.

Reference weights are the predicted counts for a unit photon flux density
per Angstrom, including exposure and AREASCAL. They do not use a fit model.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse

from ._counts_plot import counts_options, count_weights
from .grid import EnergyGrid, merge_grids
from .plotting import PlotCurve, SpectrumData, _aggregate, _convert, _array
from .resample import OverlapRebin
from .units import HC_KEV_ANGSTROM, convert_bins, normalize_unit, unit_family

KEV_TO_ERG = 1.602176634e-9


@dataclass(frozen=True)
class UnfoldOptions:
    power: object
    power_scale: str
    flux_unit: str
    model_display: str

    def resolved(self, unit):
        if self.power is not None:
            return self
        power = 3 if self.flux_unit == "ergs" and unit_family(unit) in ("energy", "frequency") else 1
        return replace(self, power=power)


def unfold_options(power, power_scale, flux_unit, model_display):
    _, mode = counts_options(1 if power is None else power, power_scale)
    aliases = {"photon": "photons", "photons": "photons", "erg": "ergs", "ergs": "ergs"}
    name = aliases.get(flux_unit.lower()) if isinstance(flux_unit, str) else None
    if name is None:
        raise ValueError("flux_unit must be photons or ergs")
    if model_display not in ("convolved", "source"):
        raise ValueError("model_display must be convolved or source")
    return UnfoldOptions(power, mode, name, model_display)


@dataclass(frozen=True, eq=False)
class SourceSpectrum:
    """Bin-integrated source photon flux, in photons/cm^2/s.

    Optional offline input to plot_unfold(source_spectra={id: ...}).
    Physical bin units describe the grid, never the integrated flux values.
    This is the complete unsmeared expression, including its absorption.
    """
    lo: np.ndarray
    hi: np.ndarray
    flux: np.ndarray
    unit: str = "keV"
    hc: float = HC_KEV_ANGSTROM

    def __post_init__(self):
        if not np.isfinite(self.hc) or self.hc <= 0:
            raise ValueError("hc must be finite and positive")
        data = SpectrumData(self.lo, self.hi, self.flux, unit=self.unit, hc=self.hc)
        if unit_family(data.unit) == "channel" or np.any(data.lo <= 0):
            raise ValueError("source spectra require positive physical spectral bin boundaries")
        for name, value in (("lo", data.lo), ("hi", data.hi), ("flux", data.counts), ("unit", data.unit)):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class UnfoldedCurve(PlotCurve):
    """Plotted flux plus pooled reference counts, count-to-flux factor, and mask.

    factor includes units and power, before presentation scale/offset.
    Invalid sensitivity rows of covariance are zero; valid identifies them.
    """
    weights: object = None
    factor: object = None
    valid: object = None


def _wave_grid(lo, hi, hc):
    """Increasing wavelength grid and indices into a response's native rows."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if lo.ndim != 1 or lo.shape != hi.shape or not lo.size or np.any(lo <= 0) or \
            not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi <= lo):
        raise ValueError("response energy bins must be finite and positive with high > low")
    order = np.argsort(hc/hi)
    return EnergyGrid(hc/hi[order], hc/lo[order]), order


def _positive_exposure(value):
    return value is not None and np.ndim(value) == 0 and np.isfinite(value) and value > 0


def response_weights(pha, exposure_source="arf", *, hc=HC_KEV_ANGSTROM):
    """Integrate t * R(h, lambda) * A(lambda) over incident wavelength.

    Effective area is piecewise constant in its tabulated bins. For distinct
    ARF/RMF grids, integrate ARF*d_lambda into each RMF incident bin before
    redistribution. Full incident and detector grids are used, so ignored
    detector channels cannot remove incident photons contributing elsewhere.
    RMF-only input is interpreted as a full RSP containing effective area.
    """
    if exposure_source not in ("arf", "pha"):
        raise ValueError("exposure_source must be arf or pha")
    if not np.isfinite(hc) or hc <= 0:
        raise ValueError("hc must be finite and positive")
    view = deepcopy(pha)
    result = np.zeros(len(view.channel))
    if not view.response_ids:
        raise ValueError("flux correction requires an ARF/RMF or full RSP response")
    for response_id in view.response_ids:
        arf, rmf = view.get_response(response_id)
        if arf is None and rmf is None:
            raise ValueError("empty response component cannot define flux correction")
        exposure = view.exposure
        arf_exposure = getattr(arf, "exposure", None)
        if exposure_source == "arf" and _positive_exposure(arf_exposure):
            exposure = arf_exposure
        if not _positive_exposure(exposure):
            raise ValueError("flux correction requires a positive exposure for each response")
        if arf is not None:
            area_grid, area_order = _wave_grid(arf.energ_lo, arf.energ_hi, hc)
            area = np.asarray(arf.specresp, float)
            if area.shape != area_order.shape or not np.isfinite(area).all() or np.any(area < 0):
                raise ValueError("effective area must be finite, nonnegative, and match its bins")
            integrated_area = area[area_order]*(area_grid.hi-area_grid.lo)
        if rmf is not None:
            response_grid, response_order = _wave_grid(rmf.energ_lo, rmf.energ_hi, hc)
            incident = response_grid.hi-response_grid.lo if arf is None else \
                OverlapRebin(area_grid, response_grid)(integrated_area)
            native = np.empty_like(incident)
            native[response_order] = incident
            rmf.notice()  # Only the deep-copied response is unfiltered.
            folded = np.asarray(rmf.apply_rmf(native), float)
            indices = np.asarray(view.channel)-getattr(rmf, "offset", 1)
            if not np.isfinite(indices).all() or np.any(indices < 0) or \
                    np.any(indices >= len(folded)) or np.any(indices != np.floor(indices)):
                raise ValueError("PHA channel labels must map into the RMF detector grid")
            folded = folded[indices.astype(int)]
        else:
            folded = np.empty_like(integrated_area)
            folded[area_order] = integrated_area
        if folded.shape != result.shape or not np.isfinite(folded).all() or np.any(folded < 0):
            raise ValueError("response correction weights must be finite, nonnegative, and match PHA channels")
        result += float(exposure)*folded
    area_scale = getattr(view, "areascal", None)
    area_scale = np.asarray(1. if area_scale is None else area_scale, float)
    if area_scale.shape not in ((), result.shape) or not np.isfinite(area_scale).all() or np.any(area_scale < 0):
        raise ValueError("AREASCAL must be finite, nonnegative, and scalar or one value per channel")
    result *= area_scale
    if not np.isfinite(result).all():
        raise ValueError("response correction weights overflowed")
    return result


def check_linear_response(session, keys):
    if session is not None and set(keys).intersection(getattr(session, "list_pileup_model_ids", lambda: [])()):
        raise NotImplementedError("model-independent flux correction requires a linear response; "
                                  "a source-independent normalization is not defined for pileup")


def weights_for(key, data, supplied, exposure_source):
    if supplied is not None:
        if not isinstance(supplied, Mapping):
            raise TypeError("response_weights must map dataset IDs to native reference-count arrays")
        if key not in supplied:
            raise ValueError(f"no response weights for dataset {key!r}")
        values = _array(supplied[key], len(data.counts), "response weights")
        if np.any(values < 0):
            raise ValueError("response weights must be nonnegative")
        return values
    if isinstance(data, SpectrumData):
        raise ValueError("offline unfolding requires response_weights in the input count bins")
    from sherpa.astro import hc
    return response_weights(data, exposure_source, hc=hc)


def weight_view(data, values, unit, options, *, raw=False):
    if isinstance(data, SpectrumData):
        return _convert(SpectrumData(data.lo, data.hi, values, staterror=np.zeros(len(values)),
                                     unit=data.unit, hc=data.hc), unit)
    from .plotting import _read_pha
    spectrum = _read_pha(data, unit, options, raw=raw, values=values)
    return replace(spectrum, staterror=np.zeros(len(spectrum.lo)), covariance=None)


def sensitivity_breaks(weights):
    """Keep plot-only rebinning within runs of pooled detector sensitivity."""
    pooled = np.zeros(len(weights[0].lo))
    first = weights[0]
    for spectrum in weights:
        if spectrum.lo.shape != first.lo.shape or not (
                np.allclose(spectrum.lo, first.lo, rtol=1e-12, atol=0) and
                np.allclose(spectrum.hi, first.hi, rtol=1e-12, atol=0)):
            raise ValueError("pooled response weights need matching selected detector bins")
        pooled += spectrum.counts
    live = pooled > 0
    return live[:-1] != live[1:]


def incident_grid(pha):
    """Preserve all ARF/RMF incident boundaries for unsmeared source curves."""
    grids = []
    for key in pha.response_ids:
        for response in pha.get_response(key):
            if response is None:
                continue
            lo, hi = np.asarray(response.energ_lo), np.asarray(response.energ_hi)
            order = np.argsort(lo)
            grid = EnergyGrid(lo[order], hi[order])
            if np.any(grid.lo <= 0):
                raise ValueError("source display requires positive incident energy boundaries")
            grids.append(grid)
    if not grids:
        raise ValueError("source display requires incident response bins")
    return merge_grids(grids)


def evaluate_sources(session, datasets, supplied=None):
    if supplied is not None:
        if not isinstance(supplied, Mapping) or any(not isinstance(v, SourceSpectrum) for v in supplied.values()):
            raise TypeError("source_spectra must map dataset IDs to SourceSpectrum objects")
        return {key: supplied[key] for key in datasets if key in supplied}
    if not datasets or session is None:
        return {}
    from sherpa.astro import hc
    from sherpa.utils.err import ModelErr, IdentifierErr
    jobs, started = {}, []
    for key, pha in datasets.items():
        try:
            source = session.get_source(key)
        except (ModelErr, IdentifierErr):
            continue
        if getattr(source, "integrate", True) is False:
            raise ValueError("source display requires bin-integrated source models")
        grid = incident_grid(pha)
        lo, hi = grid.lo, grid.hi
        # Match the response's calling convention, including _DatasetSource's
        # translation of a wavelength-analysis PHA back into its keV model.
        wave = pha.units == "wavelength"
        xlo, xhi = (hc/hi, hc/lo) if wave else (lo, hi)
        jobs[key] = (source, lo, hi, xlo, xhi)
    try:
        for source, *_ in jobs.values():
            source.startup(True)
            started.append(source)
        return {key: SourceSpectrum(lo, hi, source(xlo, xhi), hc=hc)
                for key, (source, lo, hi, xlo, xhi) in jobs.items()}
    finally:
        for source in reversed(started):
            source.teardown()


def _base_options(options):
    return {**options, "combine": "sum", "scale": 1., "offset": 0.}


def _display_multiplier(lo, hi, wave_lo, wave_hi, unit, config, hc):
    multiplier = count_weights(lo, hi, config.power, config.power_scale)
    if config.flux_unit == "photons":
        return multiplier
    family = unit_family(unit)
    if family == "channel":
        energy = count_weights(hc/wave_hi, hc/wave_lo, 2, config.power_scale)
        return multiplier*KEV_TO_ERG*energy
    if family == "wavelength":
        scale = convert_bins([1.], [2.], unit, "Angstrom", hc=hc)[0][0]
        return multiplier*KEV_TO_ERG*hc/scale
    scale = convert_bins([1.], [2.], unit, "keV", hc=hc)[0][0]
    return multiplier*KEV_TO_ERG*scale


def _reference_coordinates(first, options, unit, channel_bounds):
    """Carry wavelength boundaries through the exact same grouping/order."""
    from .plotting import _coarsen
    lo, hi, _, _ = _coarsen(first.lo, first.hi, np.zeros(len(first.lo)),
                            np.zeros(len(first.lo)), options["rebin"], options.get("_breaks"))
    if unit_family(unit) != "channel":
        _, _, order = convert_bins(lo, hi, "wavelength", unit, hc=first.hc)
    else:
        clo = np.asarray(channel_bounds[0])
        if options["rebin"] == 1:
            grouped = clo
        else:
            matrix = OverlapRebin(EnergyGrid(first.lo, first.hi), EnergyGrid(lo, hi)).matrix
            grouped = np.array([clo[matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]]].min()
                                for i in range(len(lo))])
        order = np.argsort(grouped)
    return lo[order], hi[order]


def unfold_curve(members, spectra, weights, options, config, unit, channel_bounds=None):
    if options["combine"] != "sum":
        raise ValueError("plot_unfold pools counts and response weights; combine must be sum")
    first = spectra[0]
    if normalize_unit(first.unit) != "wavelength":
        raise ValueError("internal unfolding spectra must use wavelength coordinates")
    base = _base_options(options)
    counts = _aggregate(members, spectra, base, False, False, unit, channel_bounds)
    response = _aggregate(members, weights, base, False, False, unit, channel_bounds)
    wlo, whi = _reference_coordinates(first, options, unit, channel_bounds)
    multiplier = _display_multiplier(counts.lo, counts.hi, wlo, whi, unit, config, first.hc)
    factor = np.zeros_like(response.y)
    valid = response.y > 0
    np.divide((whi-wlo)*multiplier/(counts.hi-counts.lo), response.y, out=factor, where=valid)
    valid &= np.isfinite(factor) & (factor > 0)
    if not np.any(valid):
        raise ValueError("no finite positive response correction weights in the plotted range")
    factor[~valid] = 0.
    scale, offset = float(options["scale"]), float(options["offset"])
    # Reuse validation of presentation controls before mutating axes.
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(offset):
        raise ValueError("scale must be finite and positive; offset must be finite")
    diagonal = sparse.diags(factor*scale)
    covariance = (diagonal @ counts.covariance @ diagonal).tocsr()
    y = np.where(valid, counts.y*factor*scale+offset, np.nan)
    error = np.where(valid, np.sqrt(covariance.diagonal()), np.nan)
    return UnfoldedCurve(members, counts.lo, counts.hi, y, error, "sum", counts.label,
                         covariance, response.y.copy(), factor, valid)


def source_curve(members, sources, spectra, options, config, unit, channel_bounds, data_curve):
    """ISIS's arithmetic mean of intrinsic spectra on the first model grid."""
    from ._fit_plot import ModelCurve
    from .plotting import _coarsen
    first = spectra[0]
    converted = [_convert(SpectrumData(s.lo, s.hi, s.flux, staterror=np.zeros(len(s.lo)),
                                        unit=s.unit, hc=s.hc), "wavelength") for s in sources]
    reference = converted[0]
    if unit_family(unit) == "channel":
        physical = _aggregate((0,), [first], _base_options(options), False, False, "wavelength")
        grid = EnergyGrid(physical.lo, physical.hi)
    else:
        grid = EnergyGrid(reference.lo, reference.hi)
    values = sum(OverlapRebin(EnergyGrid(s.lo, s.hi), grid)(s.counts) for s in converted)/len(converted)
    if unit_family(unit) == "channel":
        # Values already use the data's coarsened physical bins; obtain the
        # matching channel edges/order by aggregating the original reference.
        opts = _base_options(options)
        display = _aggregate((0,), [first], opts, False, False, unit, channel_bounds)
        wlo, whi = _reference_coordinates(first, options, unit, channel_bounds)
        index = np.searchsorted(grid.lo, wlo)
        lo, hi, values = display.lo, display.hi, values[index]
        valid = data_curve.valid.copy()
    else:
        lo, hi, order = convert_bins(grid.lo, grid.hi, "wavelength", unit, hc=first.hc)
        wlo, whi, values = grid.lo[order], grid.hi[order], values[order]
        valid = np.zeros(len(lo), bool)
        # Match ISIS's model-gap convention: complete source bins must fall
        # within one contiguous run of valid displayed data bins.
        mask = data_curve.valid
        dlo, dhi = data_curve.lo[mask], data_curve.hi[mask]
        if len(dlo):
            runs_lo, runs_hi, _, _ = _coarsen(dlo, dhi, np.zeros(len(dlo)), np.zeros(len(dlo)), len(dlo))
            for lower, upper in zip(runs_lo, runs_hi):
                valid |= (lo >= lower*(1-1e-12)) & (hi <= upper*(1+1e-12))
    factor = _display_multiplier(lo, hi, wlo, whi, unit, config, first.hc)/(hi-lo)
    y = np.where(valid, values*factor*float(options["scale"])+float(options["offset"]), np.nan)
    return ModelCurve(members, lo, hi, y, data_curve.label+" source")


def flux_label(unit, config):
    from .units import quantity_label, unit_mathtext
    family = unit_family(unit)
    if config.flux_unit == "photons":
        exponent = config.power-2
        base, energy = "Photons", 0
    else:
        exponent = config.power-(1 if family == "wavelength" else 2 if family == "channel" else 3)
        base, energy = "Energy flux", 1
    return quantity_label(base, (r"\mathrm{erg}", energy), (r"\mathrm{cm}", -2),
                          (r"\mathrm{s}", -1), (unit_mathtext(unit), exponent))


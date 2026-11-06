"""Grouped spectral plots using Sherpa settings and Matplotlib styles."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy import sparse

from .grid import EnergyGrid
from .resample import OverlapRebin, diagonal_covariance, get_data_match, _MATCH_COORDINATE
from .units import (HC_KEV_ANGSTROM, normalize_unit, unit_family,
                    axis_label, convert_bins, quantity_label, unit_mathtext)
from ._plot_compat import translate_legacy

_CURVE = {"combine": None, "color": None, "marker": None, "ecolor": None,
          "show_model": "auto",
          "linestyle": None, "linewidth": None, "markersize": None, "alpha": None,
          "elinewidth": None, "capsize": None, "capthick": None, "barsabove": False,
          "errorevery": 1, "markerfacecolor": None, "markeredgecolor": None,
          "markeredgewidth": None, "fillstyle": "full", "zorder": None,
          "rasterized": False, "drawstyle": "default", "dashes": None,
          "label": None, "scale": 1., "offset": 0., "rebin": 1,
          "xerrorbars": False, "yerrorbars": True,
          "subtract_background": True, "background": None,
          "error_method": "sqrt", "exposure_source": "arf"}
_GLOBAL = {"unit": None, "rate": None, "density": None,
           "xlog": None, "ylog": None, "xscale": None, "yscale": None,
           "xlim": None, "ylim": None,
           "xlabel": None, "ylabel": None, "title": None, "legend": True,
           "overplot": False, "clearwindow": True}
_GLOBAL.update({"align": "matched", "reference": None})
_LINE_KEYS = {"color", "marker", "linestyle", "linewidth", "markersize", "alpha",
              "markerfacecolor", "markeredgecolor", "markeredgewidth", "fillstyle",
              "zorder", "rasterized", "drawstyle", "dashes"}


def _unit(value):
    return normalize_unit(value)


def _data_unit(data):
    return _unit(getattr(data, "unit", getattr(data, "units", "energy")))


def _array(value, size, name):
    result = np.array(value, dtype=float, copy=True)
    if result.ndim != 1 or result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 1D array with {size} entries")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False)
class SpectrumData:
    """Already selected/grouped, integrated detector counts for one spectrum.

    Optional offline input to plot_data(datasets={id: SpectrumData(...)}).
    staterror is an absolute one-sigma count uncertainty, including any
    background/systematic contribution if the counts have been corrected.
    """
    lo: np.ndarray
    hi: np.ndarray
    counts: np.ndarray
    exposure: object = None
    staterror: object = None
    unit: str = "keV"
    covariance: object = None
    hc: float = HC_KEV_ANGSTROM

    def __post_init__(self):
        lo = np.array(self.lo, dtype=float, copy=True)
        if lo.ndim != 1 or not lo.size:
            raise ValueError("spectrum bins must be a nonempty 1D array")
        lo = _array(lo, lo.size, "lo")
        hi = _array(self.hi, lo.size, "hi")
        counts = _array(self.counts, lo.size, "counts")
        if np.any(hi <= lo) or np.any(lo[1:] < hi[:-1]):
            raise ValueError("spectrum bins must increase without overlap")
        error = None if self.staterror is None else _array(self.staterror, lo.size, "staterror")
        covariance = None if self.covariance is None else sparse.csr_matrix(self.covariance, copy=True)
        if covariance is not None:
            if covariance.shape != (lo.size, lo.size) or not np.isfinite(covariance.data).all():
                raise ValueError("covariance must be finite and match the spectrum bins")
            if np.any(covariance.diagonal() < 0):
                raise ValueError("covariance diagonal must be nonnegative")
            error = _array(np.sqrt(covariance.diagonal()), lo.size, "staterror")
        if error is not None and np.any(error < 0):
            raise ValueError("staterror must be nonnegative")
        exposure = None if self.exposure is None else float(self.exposure)
        if exposure is not None and (not np.isfinite(exposure) or exposure <= 0):
            raise ValueError("exposure must be finite and positive")
        for name, value in (("lo", lo), ("hi", hi), ("counts", counts),
                            ("staterror", error), ("exposure", exposure), ("unit", _unit(self.unit)),
                            ("covariance", covariance)):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PlotCurve:
    members: tuple
    lo: np.ndarray
    hi: np.ndarray
    y: np.ndarray
    yerr: np.ndarray
    combine: str
    label: str
    covariance: object = None


@dataclass
class PlotResult:
    ax: object
    curves: tuple
    artists: tuple
    model_curves: tuple = ()
    model_artists: tuple = ()
    residual_ax: object = None
    residuals: tuple = ()
    residual_artists: tuple = ()
    reference_line: object = None

    @property
    def figure(self):
        return self.ax.figure

    @property
    def lines(self):
        """The histogram Line2D artists, one per displayed curve."""
        return tuple(line for line, _ in self.artists)

    @property
    def errorbars(self):
        """The Matplotlib ErrorbarContainer objects, one per curve."""
        return tuple(errors for _, errors in self.artists)


def _is_id(value):
    return (isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))) or \
        (isinstance(value, str) and bool(value))


def normalize_groups(ids):
    """Outer entries are curves; an inner sequence combines dataset IDs."""
    if _is_id(ids):
        return ((ids,),)
    if not isinstance(ids, (list, tuple, np.ndarray)) or (isinstance(ids, np.ndarray) and ids.ndim == 0):
        raise ValueError("use an ID or an ordered list/tuple of IDs and ID groups")
    if not len(ids):
        raise ValueError("provide at least one dataset or group")
    groups = []
    for entry in ids:
        if _is_id(entry):
            members = (entry,)
        elif isinstance(entry, (list, tuple, np.ndarray)):
            members = tuple(entry)
            if not members or not all(_is_id(x) for x in members):
                raise ValueError("each group must be a nonempty flat sequence of dataset IDs")
        else:
            raise ValueError("dataset IDs must be integers or nonempty strings")
        if len(set(members)) != len(members):
            raise ValueError("a dataset cannot appear twice within one group")
        groups.append(members)
    return tuple(groups)


def _canonical_options(options):
    from matplotlib.cbook import normalize_kwargs
    from matplotlib.lines import Line2D
    result = normalize_kwargs(translate_legacy(options), Line2D)
    for key in result:
        if key not in _CURVE and key not in _GLOBAL and key != "sum_exp":
            raise TypeError(f"unsupported plotting option: {key}")
    return result


def _broadcast(value, count, name):
    from matplotlib.colors import is_color_like
    from matplotlib.markers import MarkerStyle
    if name in ("color", "ecolor", "markerfacecolor", "markeredgecolor") and is_color_like(value):
        return [value] * count
    if name == "marker":
        try:
            MarkerStyle(value if value is not None else "none")
            return [value] * count
        except (ValueError, TypeError):
            pass
    if name == "linestyle" and isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], Real):
        return [value] * count
    if name == "dashes" and value is not None and all(np.isscalar(v) for v in value):
        return [value] * count
    if name == "errorevery" and isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, Integral) for v in value):
        return [value] * count
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != count:
            raise ValueError(f"{name} requires one value per plotted curve ({count}), or a scalar")
        return list(value)
    return [value] * count


def _boolean(value, name):
    if not isinstance(value, (bool, np.bool_, Integral)) or value not in (0, 1):
        raise ValueError(f"{name} must be True/False or 1/0")
    return bool(value)


def _curve_options(count, options, styles, preferences=None):
    if "background" in options and "subtract_background" in options:
        raise ValueError("choose background or subtract_background, not both")
    preferences = [{} for _ in range(count)] if preferences is None else preferences
    global_options = {**_GLOBAL, **{k: v for k, v in preferences[0].items() if k in _GLOBAL},
                      **{k: v for k, v in options.items() if k in _GLOBAL}}
    curves = [{**_CURVE, **{k: v for k, v in prefs.items() if k in _CURVE}} for prefs in preferences]
    for key, value in options.items():
        if key in _CURVE or key == "sum_exp":
            for curve, item in zip(curves, _broadcast(value, count, key)):
                curve[key] = item
    if styles is not None:
        if not isinstance(styles, (list, tuple)) or len(styles) != count:
            raise ValueError("styles requires one dictionary per plotted curve")
        for curve, style in zip(curves, styles):
            if not isinstance(style, Mapping):
                raise TypeError("each styles entry must be a dictionary")
            style = _canonical_options(style)
            if set(style).intersection(_GLOBAL):
                raise ValueError("styles entries accept curve options, not panel options")
            # A local normalization explicitly replaces its global counterpart.
            if "combine" in style:
                curve.pop("sum_exp", None)
            if "sum_exp" in style:
                curve["combine"] = None
            if "background" in style and "subtract_background" in style:
                raise ValueError("choose background or subtract_background, not both")
            if "subtract_background" in style:
                curve["background"] = None
            curve.update(style)
    for curve in curves:
        background = curve["background"]
        if background is not None and background not in ("subtract", "include", "only"):
            raise ValueError("background must be subtract, include, or only")
        curve["background_only"] = background == "only"
        if background is not None:
            curve["subtract_background"] = background == "subtract"
        if curve["background_only"]:
            curve["show_model"] = False
        if curve["show_model"] != "auto":
            curve["show_model"] = _boolean(curve["show_model"], "show_model")
        if "sum_exp" in curve:
            if curve["combine"] is not None:
                raise ValueError("choose combine or sum_exp, not both")
            summed_exposure = _boolean(curve.pop("sum_exp"), "sum_exp")
            # ISISscripts make_data sums counts first, then divides by
            # the summed or mean exposure. It never sums C_i/T_i here.
            curve["combine"] = ("exposure" if summed_exposure else "mean_exposure") \
                if global_options["rate"] else "sum"
        if curve["combine"] is None:
            curve["combine"] = "exposure" if global_options["rate"] else "sum"
        if curve["combine"] == "average":
            curve["combine"] = "mean"
        if curve["combine"] not in ("sum", "mean", "exposure", "mean_exposure"):
            raise ValueError("combine must be sum, mean, exposure, or mean_exposure")
        if curve["combine"] in ("exposure", "mean_exposure") and not global_options["rate"]:
            raise ValueError("exposure normalization requires rate=True; counts use sum or mean")
        for key in ("subtract_background", "xerrorbars", "yerrorbars"):
            curve[key] = _boolean(curve[key], key)
        if not isinstance(curve["rebin"], Integral) or isinstance(curve["rebin"], bool) or curve["rebin"] < 1:
            raise ValueError("rebin must be a positive integer")
        if curve["error_method"] not in ("sqrt", "gehrels", "stored"):
            raise ValueError("error_method must be sqrt, gehrels, or stored")
        if curve["exposure_source"] not in ("arf", "pha"):
            raise ValueError("exposure_source must be arf or pha")
    return global_options, curves


def _error(counts, method):
    if np.any(counts < 0):
        raise ValueError("negative/net counts require supplied staterror values")
    if method == "stored":
        raise ValueError("error_method='stored' requires explicit statistical errors")
    return np.sqrt(counts) if method == "sqrt" else 1 + np.sqrt(counts + .75)


def _convert(data, unit):
    unit = _unit(unit)
    if data.unit == unit:
        return data
    lo, hi, order = convert_bins(data.lo, data.hi, data.unit, unit, hc=data.hc)
    return SpectrumData(lo, hi, data.counts[order], data.exposure,
                        None if data.staterror is None else data.staterror[order], unit,
                        None if data.covariance is None else data.covariance[order][:, order], data.hc)


def _background_arrays(pha, options):
    """Measured background counts/variance scaled to native source channels."""
    counts, variance = np.zeros(len(pha.channel)), np.zeros(len(pha.channel))
    for bid in pha.background_ids:
        background = deepcopy(pha.get_background(bid))
        background.ungroup()
        if not np.array_equal(background.channel, pha.channel):
            raise ValueError("associated background channels must match the source channels")
        values = np.asarray(background.counts, float)
        sigma = np.asarray(background.staterror, float) if background.staterror is not None else \
            _error(values, options["error_method"])
        systematic = background.get_syserror(filter=False)
        var = sigma**2 if systematic is None else sigma**2+np.asarray(systematic)**2
        scale = np.asarray(pha.get_background_scale(bid, units="counts", group=False, filter=False), float)
        if not np.isfinite(scale).all():
            raise ValueError("background count scaling must be finite")
        counts += scale*values
        variance += scale**2*var
    return counts, variance


def _read_pha(pha, unit, options, raw=False, values=None):
    """Sherpa's data methods perform grouping/filtering and background scaling."""
    from sherpa.astro import hc
    from sherpa.astro.data import DataPHA
    if not isinstance(pha, DataPHA):
        raise TypeError("plot_data currently accepts DataPHA or SpectrumData datasets")
    data = deepcopy(pha)
    if values is not None:
        values = _array(values, len(data.channel), "model counts")
        data.counts = values.copy()
        data.staterror = np.sqrt(np.maximum(values, 0.))
        data.syserror = None
        options = {**options, "subtract_background": False}
    elif options.get("background_only", False):
        counts, variance = _background_arrays(data, options)
        data.counts, data.staterror = counts, np.sqrt(variance)
        data.syserror = None
        options = {**options, "subtract_background": False}
    if raw:
        selected = np.array(data.get_noticed_channels(), copy=True)
        data.ungroup()
        data.quality_filter = None
        data.mask = np.isin(data.channel, selected)
    # Sherpa accepts coordinate families, with energy in keV and wavelength
    # in Angstrom. Scaled units/frequency belong to the conversion layer.
    family = unit_family(unit)
    analysis = "channel" if family == "channel" else "energy"
    data.set_analysis(analysis)
    if options["subtract_background"] and data.background_ids:
        data.subtract()
    elif data.subtracted:
        data.unsubtract()
    lo, hi = data._get_ebins(group=False)
    lo = data.apply_filter(lo, data._min)
    hi = data.apply_filter(hi, data._max)
    counts = data.get_dep(filter=True)
    method = options["error_method"]
    staterrfunc = None if method == "stored" else lambda x: _error(x, method)
    error = data.get_staterror(filter=True, staterrfunc=staterrfunc)
    if error is None:
        raise ValueError("statistical errors unavailable; choose sqrt or gehrels")
    systematic = data.get_syserror(filter=True)
    if systematic is not None:
        error = np.hypot(error, systematic)
    exposure = data.exposure
    if options["exposure_source"] == "arf":
        arf, _ = data.get_response()
        candidate = getattr(arf, "exposure", None)
        if candidate is not None and np.isfinite(candidate) and candidate > 0:
            exposure = candidate
    order = np.argsort(lo)
    return _convert(SpectrumData(lo[order], hi[order], counts[order], exposure,
                                np.asarray(error)[order], analysis, hc=hc), unit)


def _align_spectrum(data, target, unit, options):
    data = _convert(data, unit)
    mapping = OverlapRebin(EnergyGrid(data.lo, data.hi), EnergyGrid(target.lo, target.hi))
    error = data.staterror if data.staterror is not None else _error(data.counts, options["error_method"])
    covariance = data.covariance if data.covariance is not None else diagonal_covariance(error**2)
    covariance = mapping.covariance(covariance)
    return SpectrumData(target.lo, target.hi, mapping(data.counts), data.exposure,
                        unit=unit, covariance=covariance, hc=data.hc)


def _reference_channels(pha):
    """Channel edges in canonical matching order, using the reference calibration.

    Channel numbers are labels of the already matched reference bins. They
    are never used to align different instruments' channels numerically.
    """
    from sherpa.astro import hc
    from sherpa.astro.data import DataPHA
    if not isinstance(pha, DataPHA):
        raise ValueError("matched channel display requires a calibrated DataPHA reference")
    data = deepcopy(pha)
    data.set_analysis("energy")
    lo, hi = data._get_ebins(group=False)
    lo = np.asarray(data.apply_filter(lo, data._min))
    hi = np.asarray(data.apply_filter(hi, data._max))
    clo = np.asarray(data.apply_filter(data.channel, data._min))
    chi = np.asarray(data.apply_filter(np.asarray(data.channel)+1., data._max))
    order = np.argsort(lo)
    _, _, permutation = convert_bins(lo[order], hi[order], "energy", _MATCH_COORDINATE, hc=hc)
    order = order[permutation]
    return clo[order], chi[order]


def _coarsen(lo, hi, y, variance, factor, breaks=None):
    if factor == 1:
        return lo.copy(), hi.copy(), y, variance
    # Rebin within contiguous runs; never bridge ignored regions or gaps.
    discontinuity = ~np.isclose(hi[:-1], lo[1:], rtol=1e-12, atol=0)
    if breaks is not None:
        breaks = np.asarray(breaks, dtype=bool)
        if breaks.shape != discontinuity.shape:
            raise ValueError("plot grouping boundaries must match the selected bins")
        discontinuity |= breaks
    cuts = np.r_[0, np.flatnonzero(discontinuity) + 1, len(lo)]
    bins = [(i, min(i + factor, end)) for start, end in zip(cuts[:-1], cuts[1:])
            for i in range(start, end, factor)]
    return (np.array([lo[i] for i, _ in bins]), np.array([hi[j-1] for _, j in bins]),
            np.array([y[i:j].sum() for i, j in bins]),
            np.array([variance[i:j].sum() for i, j in bins]))


def _aggregate(members, spectra, options, rate, density, unit=None, channel_bounds=None, plot_factor=0):
    first = spectra[0]
    for data in spectra[1:]:
        if data.lo.shape != first.lo.shape or not (
                np.allclose(data.lo, first.lo, rtol=1e-12, atol=0) and
                np.allclose(data.hi, first.hi, rtol=1e-12, atol=0)):
            raise ValueError(f"group {members} needs matching noticed/grouped detector bins; "
                             "rerun match_dataset_grids with a reference or use align=True")
    counts = np.stack([data.counts for data in spectra])
    covariances = [data.covariance if data.covariance is not None else
                  diagonal_covariance((data.staterror if data.staterror is not None else
                                       _error(data.counts, options["error_method"]))**2)
                  for data in spectra]
    mode = options["combine"]
    if rate:
        if any(data.exposure is None for data in spectra):
            raise ValueError(f"group {members}: rate plotting requires exposure for every dataset")
        exposure = np.array([data.exposure for data in spectra])
        if mode in ("exposure", "mean_exposure"):
            normalization = exposure.sum() if mode == "exposure" else exposure.mean()
            coefficients = np.full(len(spectra), 1/normalization)
        else:
            coefficients = 1/exposure
    else:
        coefficients = np.ones(len(spectra))
    if mode == "mean":
        coefficients /= len(spectra)
    y = (coefficients[:, None]*counts).sum(axis=0)
    covariance = sum(coefficient**2*cov for coefficient, cov in zip(coefficients, covariances))
    lo, hi, y, _ = _coarsen(first.lo, first.hi, y, covariance.diagonal(), options["rebin"],
                            options.get("_breaks"))
    if options["rebin"] != 1:
        mapping = OverlapRebin(EnergyGrid(first.lo, first.hi), EnergyGrid(lo, hi))
        covariance = mapping.covariance(covariance)
    # Group in the fixed input/matching coordinate before display conversion.
    # Otherwise reversing an odd-length spectrum moves the partial group.
    if channel_bounds is None:
        lo, hi, order = convert_bins(lo, hi, first.unit, first.unit if unit is None else unit, hc=first.hc)
    else:
        clo, chi = (np.asarray(value) for value in channel_bounds)
        if clo.shape != first.lo.shape or chi.shape != first.hi.shape:
            raise ValueError("reference channel bounds must match its selected/grouped bins")
        if options["rebin"] == 1:
            lo, hi = clo.copy(), chi.copy()
        else:
            rows = [mapping.matrix.indices[mapping.matrix.indptr[i]:mapping.matrix.indptr[i+1]]
                    for i in range(len(lo))]
            lo = np.array([clo[row].min() for row in rows])
            hi = np.array([chi[row].max() for row in rows])
        order = np.argsort(lo)
        lo, hi = lo[order], hi[order]
    y, covariance = y[order], covariance[order][:, order]
    if density:
        y = y / (hi-lo)
        diagonal = sparse.diags(1/(hi-lo))
        covariance = (diagonal @ covariance @ diagonal).tocsr()
    if plot_factor > 0:
        multiplier = ((lo+hi)/2)**plot_factor
        y = y * multiplier
        diagonal = sparse.diags(multiplier)
        covariance = (diagonal @ covariance @ diagonal).tocsr()
    scale, offset = float(options["scale"]), float(options["offset"])
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(offset):
        raise ValueError("scale must be finite and positive; offset must be finite")
    label = options["label"]
    if label is None:
        label = str(members[0]) if len(members) == 1 else ", ".join(map(str, members))
    if not isinstance(label, str):
        raise ValueError("each label must be a string or None")
    covariance = covariance*scale**2
    return PlotCurve(members, lo, hi, y*scale + offset, np.sqrt(covariance.diagonal()), mode, label, covariance)


def _style(options):
    from matplotlib import rcParams
    from matplotlib.colors import is_color_like
    from matplotlib.lines import Line2D
    style = dict(options)
    for key in ("color", "ecolor", "markerfacecolor", "markeredgecolor"):
        value = style[key]
        automatic = key in ("markerfacecolor", "markeredgecolor") and isinstance(value, str) and value == "auto"
        if value is not None and not automatic and not is_color_like(value):
            raise ValueError(f"invalid {key}: {value!r}; use a Matplotlib color")
    for key in ("linewidth", "markersize", "markeredgewidth", "elinewidth", "capsize", "capthick"):
        if style[key] is None:
            continue
        style[key] = float(style[key])
        if not np.isfinite(style[key]) or style[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if style["alpha"] is not None and (not np.isfinite(style["alpha"]) or not 0 <= style["alpha"] <= 1):
        raise ValueError("alpha must lie between zero and one")
    if style["zorder"] is not None:
        style["zorder"] = float(style["zorder"])
        if not np.isfinite(style["zorder"]):
            raise ValueError("zorder must be finite")
    every = style["errorevery"]
    if isinstance(every, Integral):
        valid = every > 0
    else:
        valid = isinstance(every, tuple) and len(every) == 2 and all(isinstance(x, Integral) for x in every) \
            and every[0] >= 0 and every[1] > 0
    if not valid:
        raise ValueError("errorevery must be a positive integer or (start, positive stride)")
    for key in ("barsabove", "rasterized"):
        style[key] = _boolean(style[key], key)
    if style["marker"] is None:
        style["marker"] = rcParams["lines.marker"]
    # Let Matplotlib validate its own marker, line, fill, and dash syntax
    # before any existing axes can be cleared.
    line_options = {key: style[key] for key in _LINE_KEYS if style[key] is not None}
    Line2D([], [], **line_options)
    return style


def _plot_preferences(session, groups):
    """Copy recognized Sherpa preferences, never mutating the session dicts."""
    getter = getattr(session, "get_data_plot_prefs", None)
    keys = _LINE_KEYS | {"label", "ecolor", "capsize", "elinewidth", "capthick", "barsabove",
                         "errorevery", "xerrorbars", "yerrorbars", "xlog", "ylog"}
    return [{} for _ in groups] if getter is None else \
        [{k: v for k, v in dict(getter(group[0])).items() if k in keys} for group in groups]


def _histogram(lo, hi, values):
    x, y = np.empty(3*len(lo)), np.empty(3*len(lo))
    x[0::3], x[1::3], x[-1] = lo, hi, np.nan
    y[0::3], y[1::3], y[-1] = values, values, np.nan
    x[2:-1:3] = lo[1:]
    y[2:-1:3] = np.where(np.isclose(hi[:-1], lo[1:], rtol=1e-12, atol=0), values[1:], np.nan)
    return x, y


def _model_line(lo, hi, values):
    """Join bin centers, breaking at gaps and invalid values."""
    center = (np.asarray(lo)+np.asarray(hi))/2
    breaks = np.flatnonzero(~np.isclose(hi[:-1], lo[1:], rtol=1e-12, atol=0))+1
    return np.insert(center, breaks, np.nan), np.insert(values, breaks, np.nan)


def _place_errorbars(errors, line, barsabove):
    """Order the bars and caps relative to the separate histogram artist."""
    zorder = line.get_zorder() + (.1 if barsabove else -.1)
    for artist in (*errors.lines[1], *errors.lines[2]):
        artist.set_zorder(zorder)


def _plot_spectra(ids=None, *, session=None, datasets=None, ax=None,
              overplot=False, clearwindow=True, styles=None, res=0,
              models=None, model_styles=None, model_plot="histogram", residual_styles=None,
              residual_ax=None, residual_stat=None, count_options=None,
              unfold=None, response_weights=None, source_spectra=None, **kwargs):
    """Plot spectra, available folded models, and optional ISIS-style residuals.

    ``plot_data([[1, 2], [3, 4]], color=['red', 'blue'])`` plots two curves.
    ``plot_data([1, 2, 3, 4])`` plots four individual spectra.

    Curve options broadcast from a scalar or accept exactly one value per
    outer entry. ``styles=[{...}, {...}]`` overrides options per curve.
    Matplotlib styles include color, marker, linestyle, linewidth, ecolor,
    capsize, elinewidth, markerfacecolor, and markeredgecolor; short aliases
    such as lw and ms are supported. Use ax for Matplotlib subplots and
    customization. Panel options include xlim/ylim, xscale/yscale, Sherpa's
    xlog/ylog, legend, and clearwindow/overplot. Sherpa plot preferences are
    copied as defaults; explicit keywords and per-curve styles override them.
    combine='exposure' pools counts/exposure, while 'sum'/'mean' operate
    on individual rates. Old PGPLOT spellings issue DeprecationWarning.
    Analysis units, counts/rate, bin-width normalization, and positive
    plot_fac default to the first displayed PHA (or explicit reference).
    Matching itself never depends on these display settings.

    Existing session models are overlaid by default; show_model=False
    disables this. model_styles accepts a shared Matplotlib style dictionary
    or one per displayed curve. model_plot='histogram' (default) follows bin
    edges; 'line' joins bin centers without bridging gaps. Both can be
    customized independently of data; numerical model curves are unchanged.
    Error bars/caps lie behind histograms by default; barsabove=True reverses
    this order. Axis defaults use Mathtext for units and exponents.
    res=1/2/3 shows individual chi/signed-chi-square/ratio
    diagnostics; res=4/5/6 calculates them for each displayed group. Cash
    and CStat select signed deviance diagnostics for the first two kinds.
    Offline models maps IDs to folded count arrays in the input bins.
    background='subtract'/'include'/'only' selects the upper-panel measured
    counts; background-only display suppresses its model overlay.
    Returns data/model/residual arrays, Axes, and Matplotlib artists. No
    fit is run and loaded PHA arrays/analysis/grouping/filters are preserved.
    """
    from ._fit_plot import (evaluate_models, statistic_kind, component_styles, prepare_axes, preflight_axes,
                            ModelCurve, make_diagnostics, render_models, render_residuals)
    if not isinstance(res, Integral) or isinstance(res, (bool, np.bool_)) or res not in range(7):
        raise ValueError("res must be an integer from 0 through 6")
    if not res and (residual_styles is not None or residual_ax is not None):
        raise ValueError("residual styling/axes requires res=1 through 6")
    if "id" in kwargs:
        if ids is not None:
            raise TypeError("choose id or ids, not both")
        ids = kwargs.pop("id")
    kwargs = _canonical_options(kwargs)
    if "overplot" in kwargs:  # deprecated oplt spelling
        if overplot:
            raise ValueError("overplot was supplied twice through aliases")
        overplot = kwargs.pop("overplot")
    kwargs.update(overplot=overplot, clearwindow=clearwindow)
    if datasets is None:
        if session is None:
            from sherpa.astro import ui as session
        get_data = session.get_data
    elif isinstance(datasets, Mapping):
        if session is not None:
            raise ValueError("choose datasets or session, not both")
        get_data = datasets.__getitem__
    else:
        raise TypeError("datasets must be an ID-to-dataset mapping")
    if ids is None:
        if datasets is not None:
            if not datasets:
                raise ValueError("datasets must not be empty")
            ids = next(iter(datasets))
        else:
            ids = session.get_default_id()
    groups = normalize_groups(ids)
    if residual_stat is None and session is not None and hasattr(session, "get_stat"):
        residual_stat = session.get_stat()
    stat_kind = statistic_kind(residual_stat)
    if stat_kind == "gehrels" and "error_method" not in kwargs:
        kwargs["error_method"] = "gehrels"
    first_id = kwargs.get("reference") if kwargs.get("reference") is not None else groups[0][0]
    first_data = get_data(first_id)
    if kwargs.get("rate") is None:
        kwargs["rate"] = bool(getattr(first_data, "rate", True))
    if kwargs.get("density") is None:
        kwargs["density"] = bool(first_data.grouped or kwargs["rate"]) if hasattr(first_data, "grouped") else True
    plot_factor = getattr(first_data, "plot_fac", 0) if count_options is None and unfold is None else 0
    global_options, options = _curve_options(len(groups), kwargs, styles, _plot_preferences(session, groups))
    for key in ("rate", "density", "overplot", "clearwindow"):
        global_options[key] = _boolean(global_options[key], key)
    if not isinstance(global_options["legend"], Mapping):
        global_options["legend"] = _boolean(global_options["legend"], "legend")
    from matplotlib.scale import get_scale_names
    for coordinate in ("x", "y"):
        log, scale = coordinate+"log", coordinate+"scale"
        if scale in kwargs and log not in kwargs:
            global_options[log] = None  # explicit scale overrides inherited Sherpa xlog/ylog
        if global_options[log] is not None:
            desired = "log" if _boolean(global_options[log], log) else "linear"
            if global_options[scale] is not None and global_options[scale] != desired:
                raise ValueError(f"choose compatible {log} and {scale} values")
            global_options[scale] = desired
        if global_options[scale] is not None and global_options[scale] not in get_scale_names():
            raise ValueError(f"unknown {scale}: {global_options[scale]!r}")
    for key in ("xlim", "ylim"):
        value = global_options[key]
        if value is not None:
            if len(value) != 2 or any(v is not None and not np.isfinite(v) for v in value):
                raise ValueError(f"{key} must contain two finite bounds or None")
            if global_options[key[0]+"scale"] == "log" and any(v is not None and v <= 0 for v in value):
                raise ValueError(f"{key} must be positive for a logarithmic axis")
    options = [_style(item) for item in options]
    model_options = component_styles(model_styles, len(groups), "model_styles")
    model_plots = _broadcast(model_plot, len(groups), "model_plot")
    if any(not isinstance(value, str) or value not in ("histogram", "line") for value in model_plots):
        raise ValueError("model_plot must be 'histogram' or 'line', or one value per displayed curve")
    residual_count = len(groups) if res > 3 else sum(map(len, groups)) if res else 0
    residual_options = component_styles(residual_styles, residual_count, "residual_styles")
    if global_options["align"] not in (True, False, "matched"):
        raise ValueError("align must be True, False, or 'matched'")
    if global_options["align"] != "matched":
        global_options["align"] = bool(global_options["align"])
    requested_unit = global_options["unit"]
    if requested_unit is None or (isinstance(requested_unit, str) and requested_unit.lower() == "auto"):
        unit = _data_unit(first_data)
    else:
        unit = _unit(requested_unit)
    if unfold is not None:
        from ._unfold_plot import (check_linear_response, weights_for, weight_view, unfold_curve,
                                   evaluate_sources, source_curve, flux_label, sensitivity_breaks)
        unfold = unfold.resolved(unit)
        check_linear_response(session, (key for members in groups for key in members))
    needed = dict.fromkeys(key for members, item in zip(groups, options)
                           if res or item["show_model"] is not False for key in members)
    predictions = evaluate_models(session, {key: get_data(key) for key in needed}, models=models)
    intrinsic = {}
    if unfold is not None and unfold.model_display == "source":
        source_ids = dict.fromkeys(key for members, item in zip(groups, options)
                                  if item["show_model"] is not False for key in members)
        intrinsic = evaluate_sources(session, {key: get_data(key) for key in source_ids}, source_spectra)
    cache, curves, model_curves, residual_curves = {}, [], [], []
    for group_index, (members, item) in enumerate(zip(groups, options)):
        inputs = [get_data(key) for key in members]
        has_models = all(key in predictions for key in members)
        has_sources = unfold is not None and unfold.model_display == "source" and all(key in intrinsic for key in members)
        if (res and not has_models) or (item["show_model"] is True and not (has_sources or has_models)):
            missing = [key for key in members if key not in predictions]
            raise ValueError(f"no model for datasets {missing}; define models before requesting residuals/overlays")
        matches = [get_data_match(data) for data in inputs]
        match = matches[0] if matches[0] is not None and all(m is matches[0] for m in matches) else None
        alignment = global_options["align"]
        align = alignment is True or (alignment == "matched" and match is not None)
        if global_options["reference"] is not None and alignment is not False:
            align = True
        reference = get_data(global_options["reference"]) if global_options["reference"] is not None else \
            match.reference_data() if match is not None else inputs[0]
        match_unit = _MATCH_COORDINATE if align or unfold is not None else \
            _data_unit(reference) if isinstance(reference, SpectrumData) else "energy"
        target = None
        channel_bounds = None
        if align:
            target_key = ("target", id(reference), match_unit)
            if target_key not in cache:
                target_options = {**item, "error_method": "sqrt"}
                cache[target_key] = _convert(reference, match_unit) if isinstance(reference, SpectrumData) else \
                    _read_pha(reference, match_unit, target_options)
            target = cache[target_key]
            if unit == "channel":
                channel_key = ("channels", id(reference))
                if channel_key not in cache:
                    cache[channel_key] = _reference_channels(reference)
                channel_bounds = cache[channel_key]
        elif unfold is not None and unit == "channel":
            channel_bounds = _reference_channels(inputs[0])
        spectra, raw_spectra, net_spectra, predicted_spectra, display_models = [], [], [], [], []
        weight_spectra = []
        for key, data in zip(members, inputs):
            read_unit = match_unit if align or unit != "channel" or unfold is not None else unit
            cache_key = (key, item["subtract_background"], item["error_method"],
                         item["exposure_source"], read_unit, align, item["background_only"])
            if cache_key not in cache:
                if isinstance(data, SpectrumData):
                    view = SpectrumData(data.lo, data.hi, np.zeros(len(data.lo)), data.exposure,
                                        np.zeros(len(data.lo)), data.unit, hc=data.hc) if item["background_only"] else data
                    cache[cache_key] = _convert(view, read_unit)
                else:
                    cache[cache_key] = _read_pha(data, read_unit, item, raw=align)
            spectrum = cache[cache_key]
            if align:
                spectrum = _align_spectrum(spectrum, target, match_unit, item)
            spectra.append(spectrum)
            if unfold is not None:
                weight_key = ("unfold_weights", key, item["exposure_source"], read_unit, align)
                if weight_key not in cache:
                    values = weights_for(key, data, response_weights, item["exposure_source"])
                    cache[weight_key] = weight_view(data, values, read_unit, item, raw=align)
                weighted = cache[weight_key]
                if align:
                    weighted = _align_spectrum(weighted, target, match_unit, item)
                weight_spectra.append(weighted)
            if has_models:
                def selected_view(subtract):
                    opt = {**item, "subtract_background": subtract, "background_only": False}
                    native = _convert(data, read_unit) if isinstance(data, SpectrumData) else \
                        _read_pha(data, read_unit, opt, raw=align)
                    return _align_spectrum(native, target, match_unit, opt) if align else native
                raw_data, net_data = selected_view(False), selected_view(True)
                values = predictions[key]
                if isinstance(data, SpectrumData):
                    predicted = _convert(SpectrumData(data.lo, data.hi, values, data.exposure,
                                                      np.sqrt(np.maximum(values, 0.)), data.unit, hc=data.hc), read_unit)
                else:
                    predicted = _read_pha(data, read_unit, item, raw=align, values=values)
                if align:
                    predicted = _align_spectrum(predicted, target, match_unit, item)
                background = raw_data.counts-net_data.counts
                display = predicted.counts-background if item["subtract_background"] else predicted.counts
                display_models.append(SpectrumData(predicted.lo, predicted.hi, display, predicted.exposure,
                                                    np.zeros(len(display)), predicted.unit, hc=predicted.hc))
                raw_spectra.append(raw_data)
                net_spectra.append(net_data)
                predicted_spectra.append(predicted)
        if unfold is not None:
            item = {**item, "_breaks": sensitivity_breaks(weight_spectra)}
        curve = unfold_curve(members, spectra, weight_spectra, item, unfold, unit, channel_bounds) \
            if unfold is not None else \
            _aggregate(members, spectra, item, global_options["rate"], global_options["density"],
                       unit, channel_bounds, plot_factor)
        if count_options is not None:
            from ._counts_plot import weight_curve
            curve = weight_curve(curve, count_options, offset=float(item["offset"]))
        if global_options["xscale"] == "log" and np.any(curve.lo <= 0):
            raise ValueError("logarithmic x axes require positive plotted bin boundaries")
        curves.append(curve)
        if unfold is not None:
            if item["background_only"] or item["show_model"] is False:
                model_curves.append(None)
            elif unfold.model_display == "source":
                if not has_sources and (has_models or item["show_model"] is True):
                    raise ValueError("source display requires an unsmeared source for every group member; "
                                     "use model_display='convolved' for full/count-only models")
                model_curves.append(source_curve(members, [intrinsic[key] for key in members], spectra,
                                                 item, unfold, unit, channel_bounds, curve) if has_sources else None)
            elif has_models:
                folded = unfold_curve(members, display_models, weight_spectra, item, unfold, unit, channel_bounds)
                model_curves.append(ModelCurve(members, folded.lo, folded.hi, folded.y, curve.label+" model"))
            else:
                model_curves.append(None)
        elif has_models:
            folded = _aggregate(members, display_models, item, global_options["rate"], global_options["density"],
                                unit, channel_bounds, plot_factor)
            if count_options is not None:
                folded = weight_curve(folded, count_options, offset=float(item["offset"]))
            model_curves.append(None if item["background_only"] else
                                ModelCurve(members, folded.lo, folded.hi, folded.y, curve.label+" model"))
        else:
            model_curves.append(None)
        if res and has_models:
            residual_curves.extend(make_diagnostics(members, raw_spectra, net_spectra, predicted_spectra,
                                                    item, res, stat_kind, group_index, unit, channel_bounds))

    from matplotlib import pyplot as plt
    reuse = global_options["overplot"] or not global_options["clearwindow"]
    weight_signature = count_options if count_options is not None and count_options[0] != 1 else None
    if unfold is not None:
        weight_signature = ("unfold", unfold.flux_unit, unfold.power, unfold.power_scale)
    signature = (unit, global_options["rate"], global_options["density"], plot_factor, weight_signature)
    preflight_axes(ax, residual_ax, res, stat_kind, signature, reuse)
    ax, residual_ax = prepare_axes(ax, residual_ax, res=res, reuse=reuse)
    if global_options["overplot"] or not global_options["clearwindow"]:
        previous = getattr(ax, "_fancy_plotting_units", signature)
        if previous != signature:
            raise ValueError("overplotted spectra must have the same unit, rate, density, and power settings")
    else:
        ax.clear()
    ax._fancy_plotting_units = signature
    for coordinate in ("x", "y"):
        value = global_options[coordinate+"scale"]
        if value is not None:
            getattr(ax, "set_"+coordinate+"scale")(value)
    artists = []
    for curve, item in zip(curves, options):
        values = np.where(curve.y > 0, curve.y, np.nan) if ax.get_yscale() == "log" else curve.y
        x, y = _histogram(curve.lo, curve.hi, values)
        line_options = {k: item[k] for k in _LINE_KEYS - {"marker"} if item[k] is not None}
        line, = ax.plot(x, y, **line_options, marker="None",
                        label="_nolegend_")
        visible_line = line.get_linestyle() != "None" and line.get_linewidth() > 0
        line.set_label(curve.label if visible_line else "_nolegend_")
        color = line.get_color()  # One Matplotlib cycle entry per curve.
        center = (curve.lo+curve.hi)/2
        marker_options = {k: item[k] for k in _LINE_KEYS - {"color", "linestyle", "drawstyle", "dashes"}
                          if item[k] is not None}
        has_marker = item["marker"] is not None and str(item["marker"]).lower() not in ("none", "", " ")
        errors = ax.errorbar(center, values,
                            xerr=(curve.hi-curve.lo)/2 if item["xerrorbars"] else None,
                            yerr=curve.yerr if item["yerrorbars"] else None,
                            fmt="" if has_marker else "none", color=color,
                            ecolor=color if item["ecolor"] is None else item["ecolor"],
                            elinewidth=item["elinewidth"] if item["elinewidth"] is not None else item["linewidth"],
                            capsize=item["capsize"], capthick=item["capthick"],
                            barsabove=item["barsabove"], errorevery=item["errorevery"],
                            **marker_options, linestyle="none",
                            label="_nolegend_" if visible_line else curve.label)
        _place_errorbars(errors, line, item["barsabove"])
        artists.append((line, errors))
    ylabel = quantity_label("Counts", (r"\mathrm{s}", -int(global_options["rate"])),
                            (unit_mathtext(unit), plot_factor-int(global_options["density"])))
    if count_options is not None:
        from ._counts_plot import counts_label
        ylabel = counts_label(unit, count_options[0])
    if unfold is not None:
        ylabel = flux_label(unit, unfold)
    ax.set_xlabel(global_options["xlabel"] if global_options["xlabel"] is not None else axis_label(unit))
    ax.set_ylabel(global_options["ylabel"] if global_options["ylabel"] is not None else ylabel)
    if global_options["title"] is not None:
        ax.set_title(global_options["title"])
    for key, setter in (("xlim", ax.set_xlim), ("ylim", ax.set_ylim)):
        if global_options[key] is not None:
            setter(global_options[key])
    model_artists = render_models(ax, model_curves, artists, options, model_options, model_plots)
    if isinstance(global_options["legend"], Mapping):
        ax.legend(**global_options["legend"])
    elif global_options["legend"]:
        ax.legend()
    elif ax.get_legend() is not None:
        ax.get_legend().remove()
    residual_artists, reference_line = (), None
    if res:
        residual_artists, reference_line = render_residuals(
            ax, residual_ax, residual_curves, residual_options, artists, res, stat_kind,
            unit, reuse, global_options["xlabel"] if global_options["xlabel"] is not None else axis_label(unit))
    return PlotResult(ax, tuple(curves), tuple(artists), tuple(model_curves), tuple(model_artists),
                      residual_ax, tuple(residual_curves), tuple(residual_artists), reference_line)


def plot_data(ids=None, *, session=None, datasets=None, ax=None,
              overplot=False, clearwindow=True, styles=None, res=0,
              models=None, model_styles=None, model_plot="histogram", residual_styles=None,
              residual_ax=None, residual_stat=None, **kwargs):
    return _plot_spectra(ids, session=session, datasets=datasets, ax=ax, overplot=overplot,
                         clearwindow=clearwindow, styles=styles, res=res, models=models,
                         model_styles=model_styles, model_plot=model_plot, residual_styles=residual_styles,
                         residual_ax=residual_ax, residual_stat=residual_stat, **kwargs)


plot_data.__doc__ = _plot_spectra.__doc__


def plot_counts(ids=None, *, power=1, power_scale="arithmetic", session=None, datasets=None, ax=None,
                overplot=False, clearwindow=True, styles=None, res=0,
                models=None, model_styles=None, model_plot="histogram", residual_styles=None,
                residual_ax=None, residual_stat=None, **kwargs):
    """ISIS-style counts/bin, folded models, and optional res=0..6 panels.

    Nested ID groups combine counts exactly as plot_data combines curves.
    power=0/1/2/3 weights counts by 1/X, 1, X, or X**2, using arithmetic,
    geometric, or ISIS powerlaw bin weighting. The default ignores Sherpa's
    plot_fac and never divides by exposure or bin width. X units still come
    from set_analysis. Background can be subtract (default), include, or only.
    Matplotlib styles, model/residual controls, and PlotResult are shared
    with plot_data. Residual diagnostics are independent of display power.
    """
    from ._counts_plot import counts_options
    options = counts_options(power, power_scale)
    for key in ("rate", "density"):
        value = kwargs.pop(key, False)
        if value is not None and _boolean(value, key):
            raise ValueError(f"plot_counts requires {key}=False; use plot_data for rate/density normalization")
    return _plot_spectra(ids, session=session, datasets=datasets, ax=ax, overplot=overplot,
                         clearwindow=clearwindow, styles=styles, res=res, models=models,
                         model_styles=model_styles, model_plot=model_plot, residual_styles=residual_styles,
                         residual_ax=residual_ax, residual_stat=residual_stat,
                         count_options=options, rate=False, density=False, **kwargs)


def plot_unfold(ids=None, *, flux_unit="photons", power=None, power_scale="arithmetic",
                model_display="convolved", session=None, datasets=None, ax=None,
                overplot=False, clearwindow=True, styles=None, res=0,
                models=None, model_styles=None, model_plot="histogram", residual_styles=None,
                residual_ax=None, residual_stat=None, response_weights=None,
                source_spectra=None, **kwargs):
    """ISIS-style response-corrected photon/erg spectra and count residuals.

    Correction weights integrate exposure*RMF*ARF over incident wavelength,
    independently of the fitted source and current set_analysis coordinate.
    Counts and weights are matched/grouped/summed before division. All linear
    response components contribute; no instrument metadata selects a method.
    The default overlays a similarly corrected folded model; model_display=
    'source' overlays the unsmeared source expression (including absorption).

    Photon power defaults to 1 (density per displayed unit). Ergs default to
    power=3 in energy/frequency (nu F_nu), 1 in wavelength (lambda F_lambda).
    Calibrated channel displays use density per channel; ergs multiply by a
    representative physical photon energy, default power=1. Explicit power
    0..3 and arithmetic/geometric/powerlaw weighting follow ISISscripts.
    Sherpa rate/plot_fac settings do not change this flux normalization.

    Other styling, backgrounds, models, res=0..6, and returned artists follow
    plot_data. Returned UnfoldedCurve also has weights, factor, and valid.
    Residuals always use count-space diagnostics, independent of flux units.
    Zero sensitivity bins are masked; an entirely insensitive range errors.
    Nonlinear pileup responses have no model-independent normalization.

    Offline SpectrumData needs response_weights: native predicted counts for
    unit photon flux density per Angstrom, already including exposure/area.
    Optional source_spectra maps IDs to bin-integrated SourceSpectrum objects.
    """
    from ._unfold_plot import unfold_options
    config = unfold_options(power, power_scale, flux_unit, model_display)
    if source_spectra is not None and model_display != "source":
        raise ValueError("source_spectra requires model_display='source'")
    for key in ("rate", "density"):
        if kwargs.pop(key, None) is not None:
            raise ValueError(f"plot_unfold defines its own flux normalization; omit {key}")
    return _plot_spectra(ids, session=session, datasets=datasets, ax=ax, overplot=overplot,
                         clearwindow=clearwindow, styles=styles, res=res, models=models,
                         model_styles=model_styles, model_plot=model_plot, residual_styles=residual_styles,
                         residual_ax=residual_ax, residual_stat=residual_stat,
                         unfold=config, response_weights=response_weights, source_spectra=source_spectra,
                         rate=False, density=False, **kwargs)


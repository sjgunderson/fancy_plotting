"""Model-count evaluation and ISISscripts-style pointwise diagnostics.

Sherpa imports are confined to the PHA adapter. The numerical diagnostics
operate on count arrays and are independently testable without Sherpa.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.special import xlogy


@dataclass(frozen=True)
class ModelCurve:
    members: tuple
    lo: np.ndarray
    hi: np.ndarray
    y: np.ndarray
    label: str


@dataclass(frozen=True)
class ResidualCurve:
    members: tuple
    lo: np.ndarray
    hi: np.ndarray
    y: np.ndarray
    yerr: object
    valid: np.ndarray
    mode: int
    statistic: str
    group_index: int
    label: str


def statistic_kind(value):
    """Select ISIS's sigma/deviance family without changing the fit statistic."""
    name = str(getattr(value, "name", value) if value is not None else "data").lower()
    aliases = {"chi2datavar": "data", "chi2": "data", "chi": "data",
               "chi2gehrels": "gehrels", "chi2modvar": "model",
               "rebin_covariance_chi2": "data", "cstat": "cash",
               "cstatnegativepenalty": "cash"}
    return aliases.get(name, name)


def poisson_deviance(data, model):
    """Per-bin CStat deviance, including zero-count and exact-match limits."""
    data, model = np.asarray(data, float), np.asarray(model, float)
    out = np.full(data.shape, np.nan)
    valid = (data >= 0) & (model >= 0) & np.isfinite(data) & np.isfinite(model)
    zero = valid & (data == 0)
    out[zero] = 2*model[zero]
    boundary = valid & (data > 0) & (model == 0)
    out[boundary] = np.inf
    positive = valid & (data > 0) & (model > 0)
    d, m = data[positive], model[positive]
    delta = d-m
    near = np.abs(delta) < 1.e-4*m
    dev = np.empty_like(d)
    far = ~near
    # Log differences avoid overflowing d/m for a tiny positive model.
    with np.errstate(over="ignore", invalid="ignore"):
        dev[far] = 2*(d[far]*(np.log(d[far])-np.log(m[far]))-delta[far])
    # Avoid subtracting almost equal terms when data ~= model.
    u = delta[near]/m[near]
    dev[near] = m[near]*u**2*(1-u/3+u**2/6-u**3/10)
    out[positive] = np.maximum(dev, 0.)
    return out


def diagnostic(data, model, variance, *, mode, statistic="data", background=None):
    """Return y, asymmetric yerr (or None), and a finite diagnostic mask.

    Inputs are *summed counts*, never rate densities or display offsets.
    background is the same measured background subtracted from data and
    model for a net ratio. Standardized/deviance diagnostics use raw totals.
    variance is already propagated/coarsened, including background errors.
    For statistic='model' it is the propagated total-model count variance.
    """
    if mode not in range(1, 7):
        raise ValueError("res must be an integer from 0 through 6")
    d, m, var = (np.asarray(x, dtype=float) for x in (data, model, variance))
    if d.shape != m.shape or d.shape != var.shape or d.ndim != 1:
        raise ValueError("diagnostic counts, model, and variance must be matching vectors")
    family = statistic_kind(statistic)
    kind = (mode-1) % 3
    valid = np.isfinite(d) & np.isfinite(m) & np.isfinite(var) & (var >= 0)
    y = np.full(d.shape, np.nan)
    yerr = None
    if kind == 2:
        b = np.zeros_like(d) if background is None else np.asarray(background, float)
        if b.shape != d.shape:
            raise ValueError("background counts must match the diagnostic grid")
        numerator, denominator = d-b, m-b
        sigma = np.sqrt(var)
        valid &= np.isfinite(b) & (denominator > 0)
        if family == "model":
            valid &= denominator > sigma
        y[valid] = numerator[valid]/denominator[valid]
        yerr = np.full((2, len(d)), np.nan)
        if family == "model":
            left = numerator[valid]/(denominator[valid]+sigma[valid])
            right = numerator[valid]/(denominator[valid]-sigma[valid])
            yerr[0, valid] = y[valid]-np.minimum(left, right)
            yerr[1, valid] = np.maximum(left, right)-y[valid]
        else:
            yerr[:, valid] = sigma[valid]/denominator[valid]
    elif family in ("cash", "ml"):
        if np.any(d < 0):
            raise ValueError("Cash/ML plot diagnostics require raw nonnegative counts")
        if family == "cash":
            magnitude = poisson_deviance(d, m)
            signed = np.sign(d-m)*magnitude
        else:
            # The expression used by the supplied ISISscripts 'ml' branch.
            term = m+np.log1p(d)
            positive = m > 0
            term[positive] -= xlogy(d[positive], m[positive])
            signed = 2*np.sign(d-m)*term
        valid &= np.isfinite(signed)
        y[valid] = np.sign(signed[valid])*np.sqrt(np.abs(signed[valid])) if kind == 0 else signed[valid]
        y[valid & (d == m)] = 0.
    else:
        if family not in ("data", "gehrels", "model"):
            raise NotImplementedError(f"no ISIS-style standardized residual mapping for {statistic!r}; "
                                      "choose res=3/6 for a ratio or residual_stat='data'")
        valid &= var > 0
        chi = (d[valid]-m[valid])/np.sqrt(var[valid])
        y[valid] = chi if kind == 0 else np.sign(chi)*chi**2
        if kind == 0:
            yerr = np.full((2, len(d)), np.nan)
            yerr[:, valid] = 1.
    valid &= np.isfinite(y)
    return y, yerr, valid


def _expand_prediction(pha, model):
    """Restore a response-filtered result to original channel positions."""
    values = np.asarray(model(pha.channel), dtype=float)
    if values.shape == np.asarray(pha.channel).shape:
        return values
    noticed = np.isin(pha.channel, pha.get_noticed_channels())
    if values.shape != (int(noticed.sum()),):
        raise ValueError("folded model does not map to the PHA channels")
    full = np.zeros(len(pha.channel))
    full[noticed] = values
    return full


def evaluate_models(session, datasets, *, models=None):
    """Evaluate available models once per dataset; no session/data writes.

    Values returned are raw source-aperture predicted counts. Standard PHA
    source models with no background model use the measured, scaled background
    as ISIS does; modeled backgrounds are included through Sherpa's response
    construction. Explicit full models follow the PHA's subtraction convention.
    Offline model arrays must match the SpectrumData input bins exactly.
    """
    if models is not None and not isinstance(models, Mapping):
        raise TypeError("models must map dataset IDs to integrated predicted counts")
    if not datasets:
        return {}
    if models is not None:
        result = {}
        for key, data in datasets.items():
            if key not in models:
                continue
            values = np.asarray(models[key], dtype=float)
            expected = np.asarray(data.counts).shape
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(f"model counts for dataset {key!r} must be finite and match its native bins")
            result[key] = values.copy()
        return result
    if session is None or not hasattr(session, "list_model_ids"):
        return {}
    available = set(session.list_model_ids())
    if not available.intersection(datasets):
        return {}
    from sherpa.astro.background import get_response_for_pha
    from sherpa.utils.err import ModelErr, IdentifierErr
    jobs = {}
    for key, pha in datasets.items():
        if key not in available:
            continue
        additive = np.zeros(len(pha.channel))
        if not pha.background_ids:
            jobs[key] = (pha, session.get_model(key), additive)
            continue
        background = sum(np.asarray(pha.get_background_scale(bid, units="counts", group=False, filter=False)) *
                         np.asarray(pha.get_background(bid).counts) for bid in pha.background_ids)
        try:
            source = session.get_source(key)
        except (ModelErr, IdentifierErr):
            # Explicit set_full_model: respect the original PHA convention.
            if pha.subtracted:
                additive = background
            jobs[key] = (pha, session.get_model(key), additive)
            continue
        bkg_sources = {}
        for bid in pha.background_ids:
            try:
                bkg_sources[bid] = session.get_bkg_source(key, bkg_id=bid)
            except ModelErr:
                try:
                    session.get_bkg_model(key, bkg_id=bid)
                except ModelErr:
                    pass
                else:
                    raise NotImplementedError("background full models need explicit raw total count "
                                              "predictions via models= or a source set_full_model")
        pileup = None
        if key in getattr(session, "list_pileup_model_ids", lambda: [])():
            pileup = session.get_pileup_model(key)
        view = deepcopy(pha)
        view.unsubtract()
        convolved = get_response_for_pha(view, source, bkg_sources, pileup, key)
        if not bkg_sources:
            additive = background
        jobs[key] = (view, convolved, additive)
    # Start all responses before evaluating any: shared source caches survive
    # across dataset evaluations, including SharedGridSource wrappers.
    started = []
    try:
        for _, model, _ in jobs.values():
            model.startup(True)
            started.append(model)
        return {key: _expand_prediction(pha, model)+background
                for key, (pha, model, background) in jobs.items()}
    finally:
        for model in reversed(started):
            model.teardown()


def component_styles(value, count, name):
    """Validate Matplotlib model/residual style dictionaries before drawing."""
    from .plotting import _canonical_options, _style, _CURVE, _LINE_KEYS, _boolean
    if value is None:
        values = [{} for _ in range(count)]
    elif isinstance(value, Mapping):
        values = [value for _ in range(count)]
    elif isinstance(value, (list, tuple)) and len(value) == count:
        values = value
    else:
        raise ValueError(f"{name} requires a dict or one dict per displayed component ({count})")
    allowed = _LINE_KEYS | {"ecolor", "capsize", "capthick", "elinewidth", "barsabove", "errorevery",
                            "xerrorbars", "yerrorbars", "label"}
    if name == "model_styles":
        allowed = _LINE_KEYS | {"label"}  # model curves have no fitted-parameter error bars
    output = []
    for item in values:
        if not isinstance(item, Mapping):
            raise TypeError(f"{name} entries must be dictionaries")
        item = _canonical_options(item)
        if set(item)-allowed:
            raise TypeError(f"{name} accepts Matplotlib style options, not numerical or panel controls")
        for key in ("xerrorbars", "yerrorbars"):
            if key in item:
                item[key] = _boolean(item[key], key)
        _style({**_CURVE, **item})
        output.append(item)
    return output


def residual_label(mode, statistic):
    kind = (mode-1) % 3
    if kind == 2:
        return "Data / model"
    family = statistic_kind(statistic)
    if family in ("cash", "ml"):
        term = "C" if family == "cash" else "ML"
        return rf"Signed $\sqrt{{\Delta {term}}}$" if kind == 0 else rf"Signed $\Delta {term}$"
    return r"$\chi$" if kind == 0 else r"Signed $\chi^2$"


def prepare_axes(ax, residual_ax, *, res, reuse):
    """Create/reuse a lower panel without clearing unrelated subplots."""
    from matplotlib import pyplot as plt
    if isinstance(ax, (list, tuple, np.ndarray)):
        if len(ax) != 2 or residual_ax is not None:
            raise ValueError("provide ax=(data_ax, residual_ax) or separate axes arguments")
        ax, residual_ax = ax
    if not res and residual_ax is not None:
        raise ValueError("residual_ax requires res=1 through 6")
    if ax is None:
        if residual_ax is not None:
            raise ValueError("provide a data ax together with residual_ax")
        ax = plt.gca() if reuse else plt.subplots()[1]
    if not res:
        previous = getattr(ax, "_fancy_plotting_residual_ax", None)
        if previous is not None and not reuse:
            if getattr(ax, "_fancy_plotting_owns_residual_ax", False):
                previous.remove()
                original = getattr(ax, "_fancy_plotting_original_subplotspec", None)
                if original is not None:
                    ax.set_subplotspec(original)
                elif hasattr(ax, "_fancy_plotting_original_position"):
                    ax.set_axes_locator(ax._fancy_plotting_original_locator)
                    ax.set_position(ax._fancy_plotting_original_position)
            else:
                previous.clear()
            ax._fancy_plotting_residual_ax = None
            ax.tick_params(axis="x", labelbottom=True)
        return ax, None
    if residual_ax is None:
        residual_ax = getattr(ax, "_fancy_plotting_residual_ax", None)
        if residual_ax is None or residual_ax not in ax.figure.axes:
            spec = getattr(ax, "get_subplotspec", lambda: None)()
            if spec is None:
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                ax._fancy_plotting_original_position = ax.get_position().frozen()
                ax._fancy_plotting_original_locator = ax.get_axes_locator()
                residual_ax = make_axes_locatable(ax).append_axes("bottom", size="30%", pad=.1, sharex=ax)
            else:
                ax._fancy_plotting_original_subplotspec = spec
                sub = spec.subgridspec(2, 1, height_ratios=[3, 1], hspace=.05)
                ax.set_subplotspec(sub[0])
                residual_ax = ax.figure.add_subplot(sub[1], sharex=ax)
            ax._fancy_plotting_owns_residual_ax = True
    if residual_ax is ax or residual_ax.figure is not ax.figure:
        raise ValueError("data and residual axes must be distinct axes in the same figure")
    if not ax.get_shared_x_axes().joined(ax, residual_ax):
        residual_ax.sharex(ax)
    ax._fancy_plotting_residual_ax = residual_ax
    return ax, residual_ax


def preflight_axes(ax, residual_ax, res, statistic, signature, reuse):
    """Reject incompatible overplots before drawing or changing the layout."""
    from matplotlib import pyplot as plt
    if isinstance(ax, (list, tuple, np.ndarray)):
        if len(ax) != 2 or residual_ax is not None:
            raise ValueError("provide ax=(data_ax, residual_ax) or separate axes arguments")
        ax, residual_ax = ax
    if ax is None and reuse and plt.get_fignums():
        ax = plt.gca()
    if ax is None:
        if residual_ax is not None:
            raise ValueError("provide a data ax together with residual_ax")
        return
    if residual_ax is not None and (residual_ax is ax or residual_ax.figure is not ax.figure):
        raise ValueError("data and residual axes must be distinct axes in the same figure")
    if reuse and getattr(ax, "_fancy_plotting_units", signature) != signature:
        raise ValueError("overplotted spectra must have the same unit, rate, density, and power settings")
    lower = residual_ax if residual_ax is not None else getattr(ax, "_fancy_plotting_residual_ax", None)
    if res and reuse and lower is not None:
        expected = (signature[0], (res-1) % 3, statistic if (res-1) % 3 != 2 else statistic == "model")
        if getattr(lower, "_fancy_plotting_diagnostic", expected) != expected:
            raise ValueError("overplotted residual panels must use the same diagnostic and statistic")


def make_diagnostics(members, raw, net, models, options, mode, statistic, group_index, unit, channel_bounds):
    """Rebin counts first; modes 1-3 remain per-member, 4-6 sum members."""
    from .plotting import _aggregate
    blocks = [tuple(range(len(members)))] if mode > 3 else [(i,) for i in range(len(members))]
    output = []
    for block in blocks:
        ids = tuple(members[i] for i in block)
        opts = {**options, "combine": "sum", "scale": 1., "offset": 0., "label": None}
        data = _aggregate(ids, [raw[i] for i in block], opts, False, False, unit, channel_bounds)
        measured = _aggregate(ids, [net[i] for i in block], opts, False, False, unit, channel_bounds)
        model = _aggregate(ids, [models[i] for i in block], opts, False, False, unit, channel_bounds)
        variance = model.yerr**2 if statistic == "model" else measured.yerr**2
        background = data.y-measured.y if options["subtract_background"] or options.get("background_only", False) else None
        y, yerr, valid = diagnostic(data.y, model.y, variance, mode=mode,
                                    statistic=statistic, background=background)
        output.append(ResidualCurve(ids, data.lo, data.hi, y, yerr, valid, mode, statistic,
                                    group_index, measured.label))
    return output


def render_models(ax, curves, data_artists, data_options, styles, model_plots):
    from .plotting import _CURVE, _LINE_KEYS, _style, _histogram, _model_line
    output = []
    for curve, artists, data, override, presentation in zip(curves, data_artists, data_options, styles, model_plots):
        if curve is None or data["show_model"] is False:
            output.append(None)
            continue
        style = _style({**_CURVE, "color": artists[0].get_color(), "linestyle": "--",
                        "linewidth": 1.5, "marker": "None", **override})
        values = np.where(curve.y > 0, curve.y, np.nan) if ax.get_yscale() == "log" else curve.y
        x, y = (_histogram if presentation == "histogram" else _model_line)(curve.lo, curve.hi, values)
        line_options = {k: style[k] for k in _LINE_KEYS if style[k] is not None}
        line, = ax.plot(x, y, **line_options, label=override.get("label", curve.label))
        output.append(line)
    return output


def render_residuals(data_ax, ax, curves, styles, data_artists, mode, statistic, unit, reuse, xlabel):
    from .plotting import _CURVE, _LINE_KEYS, _style, _histogram, _place_errorbars
    signature = (unit, (mode-1) % 3, statistic if (mode-1) % 3 != 2 else statistic == "model")
    previous = getattr(ax, "_fancy_plotting_diagnostic", signature)
    if reuse and previous != signature:
        raise ValueError("overplotted residual panels must use the same diagnostic and statistic")
    if not reuse:
        ax.clear()
    ax._fancy_plotting_diagnostic = signature
    ax.set_yscale("linear")
    baseline = 1. if (mode-1) % 3 == 2 else 0.
    reference = getattr(ax, "_fancy_plotting_reference_line", None)
    if reference is None or reference not in ax.lines:
        reference = ax.axhline(baseline, color="0.45", linewidth=1, linestyle="--", label="_nolegend_")
        ax._fancy_plotting_reference_line = reference
    output = []
    histogram = (mode-1) % 3 == 1 or (statistic in ("cash", "ml") and (mode-1) % 3 != 2)
    for curve, override in zip(curves, styles):
        color = data_artists[curve.group_index][0].get_color()
        style = _style({**_CURVE, "color": color, "marker": "None" if histogram else "o",
                        "markersize": 3., "linestyle": "-" if histogram else "none", **override})
        values = np.where(curve.valid, curve.y, np.nan)
        x, y = _histogram(curve.lo, curve.hi, values)
        line_options = {k: style[k] for k in _LINE_KEYS - {"marker"} if style[k] is not None}
        line, = ax.plot(x, y, **line_options, marker="None", label="_nolegend_")
        marker_options = {k: style[k] for k in _LINE_KEYS - {"color", "linestyle", "drawstyle", "dashes"}
                          if style[k] is not None}
        marker = style["marker"]
        has_marker = marker is not None and str(marker).lower() not in ("none", "", " ")
        errors = ax.errorbar((curve.lo+curve.hi)/2, values,
                            xerr=(curve.hi-curve.lo)/2 if style["xerrorbars"] else None,
                            yerr=curve.yerr if style["yerrorbars"] else None,
                            fmt="" if has_marker else "none", color=style["color"],
                            ecolor=style["color"] if style["ecolor"] is None else style["ecolor"],
                            elinewidth=style["elinewidth"] if style["elinewidth"] is not None else style["linewidth"],
                            capsize=style["capsize"], capthick=style["capthick"],
                            barsabove=style["barsabove"], errorevery=style["errorevery"],
                            **marker_options, linestyle="none", label=override.get("label", curve.label))
        _place_errorbars(errors, line, style["barsabove"])
        output.append((line, errors))
    ax.set_ylabel(residual_label(mode, statistic))
    ax.set_xlabel(xlabel)
    ax.set_xlim(data_ax.get_xlim())
    data_ax.set_xlabel("")
    data_ax.tick_params(axis="x", labelbottom=False)
    return output, reference


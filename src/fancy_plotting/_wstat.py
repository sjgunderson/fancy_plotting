"""Profile-Poisson diagnostics for an on-source spectrum and measured backgrounds."""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ._fit_plot import poisson_deviance
from .grid import EnergyGrid
from .resample import OverlapRebin


def profile_deviance(data, source, background, alpha):
    """Return W and its sign for one on bin and independent off measurements.

    ``source`` is the predicted source count, ``background`` the unscaled
    off counts, and ``alpha`` converts their expected counts to the on region.
    Pooling equal-alpha measurements reduces exactly to the usual W statistic.
    Different alphas retain separate off-region nuisance parameters.
    """
    b, a = np.asarray(background, float), np.asarray(alpha, float)
    if b.ndim != 1 or b.shape != a.shape or not b.size:
        raise ValueError("WStat requires matching background counts and scales")
    if (not np.isfinite(data) or not np.isfinite(source) or
            not np.isfinite(b).all() or not np.isfinite(a).all() or
            data < 0 or source < 0 or np.any(b < 0) or np.any(a <= 0)):
        raise ValueError("WStat requires nonnegative raw/source counts and positive finite background scales")
    sign = np.sign(data-source-np.dot(a, b))
    if np.all(a == a[0]):
        # Stable positive root for the profiled on-region background q.
        off, scale = b.sum(), a[0]
        k = 1+1/scale
        z = data+off-k*source
        root = np.hypot(z, 2*np.sqrt(k)*np.sqrt(off)*np.sqrt(source))
        q = (z+root)/(2*k) if z >= 0 else 2*off*source/(root-z)
        on_mean, off_mean = source+q, np.array([q/scale])
        b = np.array([off])
    elif data == 0:
        off_mean = b/(1+a)
        on_mean = source+np.dot(a, off_mean)
    else:
        # Stationarity gives beta_j = B_j/(1+alpha_j*t), t=1-D/mu.
        # A zero-count background can attain the boundary t=-1/max(alpha).
        lower = -1/a.max()
        at_boundary = a == a.max()
        positive = b > 0

        def off_at(t):
            out = np.zeros_like(b)
            out[positive] = b[positive]/(1+a[positive]*t)
            return out

        def equation(t):
            return (1-t)*(source+np.dot(a, off_at(t)))-data

        if not np.any(positive & at_boundary) and equation(lower) <= 0:
            off_mean = off_at(lower)
            on_mean = data/(1-lower)
            index = np.flatnonzero(at_boundary)[0]
            off_mean[index] = max(0., (on_mean-source-np.dot(a, off_mean))/a[index])
        else:
            lo = np.nextafter(lower, 1.)
            t = brentq(equation, lo, 1., xtol=np.finfo(float).tiny,
                       rtol=4*np.finfo(float).eps)
            off_mean = off_at(t)
            on_mean = source+np.dot(a, off_mean)
    value = poisson_deviance(np.array([data]), np.array([on_mean]))[0]
    value += poisson_deviance(b, off_mean).sum()
    return value, sign


@dataclass(frozen=True)
class WStatInput:
    source: object
    background: object
    alpha: np.ndarray


def read_wstat_input(pha, total, unit, options, *, raw=False):
    """Read source/off counts with Sherpa's grouping, filters and scale rules."""
    from .plotting import SpectrumData, _read_pha, _convert

    if isinstance(pha, SpectrumData):
        raise ValueError("WStat residuals require DataPHA with an associated background; "
                         "use residual_stat='cstat' for source-only SpectrumData")
    if pha.subtracted:
        raise ValueError("WStat requires unsubtracted PHA data; unsubtract before plotting")
    if len(pha.background_ids) != 1:
        raise ValueError("WStat residuals require exactly one associated background per dataset")
    bid = pha.background_ids[0]
    bkg = pha.get_background(bid)
    if not np.array_equal(pha.channel, bkg.channel):
        raise ValueError("WStat background channels must match the source channels")
    measured = np.asarray(pha.get_background_scale(bid, units="counts", group=False, filter=False)) * bkg.counts
    source = _read_pha(pha, unit, options, raw=raw, values=np.asarray(total)-measured)
    background = _read_pha(pha, unit, options, raw=raw, values=bkg.counts)
    view = deepcopy(pha)
    if raw:
        noticed = view.get_noticed_channels()
        view.ungroup()
        view.quality_filter = None
        view.mask = np.isin(view.channel, noticed)
    view.set_analysis("energy" if unit != "channel" else "channel")
    dummy = np.ones(len(view.channel))

    def middle(value):
        return np.asarray(view.apply_filter(dummy*(1. if value is None else value), view._middle))

    if pha.exposure is None or bkg.exposure is None:
        raise ValueError("WStat requires source and background exposures")
    numerator = pha.exposure*middle(pha.areascal)*middle(pha.backscal)
    denominator = bkg.exposure*middle(bkg.areascal)*middle(bkg.backscal)
    if (not np.isfinite(numerator).all() or not np.isfinite(denominator).all() or
            np.any(numerator <= 0) or np.any(denominator <= 0)):
        raise ValueError("WStat requires positive finite exposure, BACKSCAL and AREASCAL values")
    alpha = numerator/denominator
    lo, hi = view._get_ebins(group=False)
    lo = np.asarray(view.apply_filter(lo, view._min))
    hi = np.asarray(view.apply_filter(hi, view._max))
    order = np.argsort(lo)
    # Convert/reorder scale values exactly as the corresponding count view.
    scales = _convert(SpectrumData(lo[order], hi[order], alpha[order],
                                  unit=view.units, hc=source.hc), unit)
    return WStatInput(source, background, scales.counts)


def pooled_diagnostic(data, inputs):
    """Reprofile on the displayed count bins, preserving off-region scalings."""
    sources = np.zeros(len(data.y))
    off = [[] for _ in data.y]
    scales = [[] for _ in data.y]
    for values in inputs:
        mapping = OverlapRebin(EnergyGrid(values.source.lo, values.source.hi),
                               EnergyGrid(data.lo, data.hi)).matrix.tocsr()
        sources += mapping @ values.source.counts
        for row in range(len(data.y)):
            start, stop = mapping.indptr[row:row+2]
            indices, weights = mapping.indices[start:stop], mapping.data[start:stop]
            off[row].extend(weights*values.background.counts[indices])
            scales[row].extend(values.alpha[indices])
    signed = np.full(len(data.y), np.nan)
    for row, (d, s, b, a) in enumerate(zip(data.y, sources, off, scales)):
        if not b:
            continue
        value, sign = profile_deviance(d, s, b, a)
        signed[row] = sign*value
    return signed

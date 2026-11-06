"""Count-display weighting from the supplied ISISscripts power_scale routine."""
from dataclasses import replace
from numbers import Integral

import numpy as np
from scipy import sparse


def counts_options(power, power_scale):
    if not isinstance(power, Integral) or isinstance(power, (bool, np.bool_)) or power not in range(4):
        raise ValueError("power must be an integer from 0 through 3; 1 gives counts/bin")
    names = {1: "arithmetic", 2: "geometric", 3: "powerlaw"}
    if isinstance(power_scale, Integral) and not isinstance(power_scale, (bool, np.bool_)):
        name = names.get(power_scale)
    elif isinstance(power_scale, str):
        name = {"arithmetic": "arithmetic", "midpoint": "arithmetic", "geometric": "geometric",
                "powerlaw": "powerlaw", "power-law": "powerlaw"}.get(power_scale.lower())
    else:
        name = None
    if name is None:
        raise ValueError("power_scale must be arithmetic, geometric, or powerlaw (ISIS settings 1, 2, 3)")
    return int(power), name


def count_weights(lo, hi, power=1, power_scale="arithmetic"):
    """Return bin weights, after grouping and display-coordinate conversion.

    The powerlaw power=0 branch deliberately follows the supplied source
    formula, rather than substituting a different within-bin integral.
    Factored/log1p forms preserve its limits for very narrow bins.
    """
    power, mode = counts_options(power, power_scale)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if lo.shape != hi.shape or lo.ndim != 1 or not np.isfinite(lo).all() or \
            not np.isfinite(hi).all() or np.any(hi <= lo):
        raise ValueError("count weights require matching increasing low/high bin boundaries")
    if power == 1:
        return np.ones_like(lo)
    if mode == "arithmetic":
        midpoint = lo/2+hi/2
        if power == 0 and np.any(midpoint <= 0):
            raise ValueError("power=0 requires positive arithmetic bin centers")
        weights = midpoint**(power-1)
    elif mode == "geometric":
        if np.any(lo < 0) or (power == 0 and np.any(lo <= 0)):
            raise ValueError("geometric count weighting requires nonnegative bounds, positive for power=0")
        weights = (np.sqrt(lo)*np.sqrt(hi))**(power-1)
    elif power == 0:
        if np.any(lo <= 0):
            raise ValueError("powerlaw power=0 requires positive bin boundaries")
        ratio = lo/hi
        weights = (1+ratio+ratio**2)/(3*lo)
    elif power == 2:
        if np.any(lo <= 0):
            raise ValueError("powerlaw power=2 requires positive bin boundaries")
        delta = (hi-lo)/lo
        weights = hi*np.log1p(delta)/delta
    else:
        weights = lo*hi
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("count weights must be finite and nonnegative")
    return weights


def weight_curve(curve, options, *, offset=0.):
    power, mode = options
    if power == 1:
        return curve
    weights = count_weights(curve.lo, curve.hi, power, mode)
    covariance = curve.covariance
    if covariance is not None:
        diagonal = sparse.diags(weights)
        covariance = (diagonal @ covariance @ diagonal).tocsr()
    return replace(curve, y=(curve.y-offset)*weights+offset, yerr=curve.yerr*weights,
                   covariance=covariance)


def counts_label(unit, power):
    from .units import quantity_label, unit_mathtext
    return quantity_label("Counts", (unit_mathtext(unit), power-1), (r"\mathrm{bin}", -1))


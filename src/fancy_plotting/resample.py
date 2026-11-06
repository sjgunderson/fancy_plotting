"""Conservative bin-overlap regridding of integrated spectra and errors."""
from dataclasses import dataclass
import weakref

import numpy as np
from scipy import sparse

from .grid import EnergyGrid
from .units import HC_KEV_ANGSTROM as HC, convert_bins

# An implementation detail, deliberately independent of DataPHA.units and
# set_analysis. ISIS likewise matches observed spectra in wavelength space.
_MATCH_COORDINATE = "wavelength"


def convert_grid(grid, source_unit, target_unit, hc=HC):
    """Return an increasing grid plus the permutation of its input bins."""
    lo, hi, order = convert_bins(grid.lo, grid.hi, source_unit, target_unit, hc=hc)
    return EnergyGrid(lo, hi), order


class OverlapRebin:
    """Redistribute bin integrals by overlap/source-bin-width.

    Overlap is evaluated in the chosen matching coordinate, e.g. wavelength
    for HEG/MEG. This assumes constant density within each source bin.
    Bins outside the target range are cropped; no extrapolation is used.
    """

    def __init__(self, source, target):
        self.source, self.target = source, target
        rows, cols, weights = [], [], []
        i = j = 0
        while i < source.size and j < target.size:
            lower, upper = max(source.lo[i], target.lo[j]), min(source.hi[i], target.hi[j])
            tolerance = 64*np.finfo(float).eps*max(abs(lower), abs(upper))
            if upper-lower > tolerance:
                rows.append(j)
                cols.append(i)
                weights.append((upper-lower)/(source.hi[i]-source.lo[i]))
            if source.hi[i] <= target.hi[j]:
                i += 1
            else:
                j += 1
        self.matrix = sparse.csr_matrix((weights, (rows, cols)), shape=(target.size, source.size))
        # Snap only floating-point round-trip noise, never physical offsets.
        close = np.isclose(self.matrix.data, 1., rtol=0, atol=64*np.finfo(float).eps)
        self.matrix.data[close] = 1.
        self.fractional = bool(np.any(self.matrix.data != 1.))
        self.source_coverage = np.asarray(self.matrix.sum(axis=0)).ravel()

    def __call__(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (self.source.size,):
            raise ValueError("values must match the source bins")
        return np.asarray(self.matrix @ values).ravel()

    def covariance(self, covariance):
        covariance = sparse.csr_matrix(covariance)
        if covariance.shape != (self.source.size, self.source.size):
            raise ValueError("covariance must match the source bins")
        return (self.matrix @ covariance @ self.matrix.T).tocsr()


def diagonal_covariance(variance):
    variance = np.asarray(variance, dtype=float)
    if variance.ndim != 1 or np.any(variance < 0) or not np.isfinite(variance).all():
        raise ValueError("variances must be finite, nonnegative, and one dimensional")
    return sparse.diags(variance, format="csr")


@dataclass
class ResampledCombination:
    grid: EnergyGrid
    projections: tuple
    counts: np.ndarray
    covariance: object
    fractional: bool


def resample_combination(grids, counts, *, grid=None, variances=None):
    """Regrid independent native spectra and sum their full covariances."""
    grids, counts = tuple(grids), tuple(np.asarray(c, dtype=float) for c in counts)
    if not grids or len(grids) != len(counts):
        raise ValueError("provide matching nonempty grids and count arrays")
    if variances is None:
        # A one-count variance floor keeps empty native count bins usable
        # in the Gaussian approximation used for fractionally rebinned fits.
        variances = [np.maximum(c, 1.) for c in counts]
    if len(variances) != len(grids):
        raise ValueError("provide one variance array per spectrum")
    grid = grids[0] if grid is None else grid
    projections = tuple(OverlapRebin(g, grid) for g in grids)
    total = np.zeros(grid.size)
    covariance = sparse.csr_matrix((grid.size, grid.size))
    for native, values, variance, projection in zip(grids, counts, variances, projections):
        if values.shape != (native.size,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("native counts must be finite and nonnegative, one per bin")
        total += projection(values)
        covariance += projection.covariance(diagonal_covariance(variance))
    return ResampledCombination(grid, projections, total, covariance,
                                any(p.fractional for p in projections))


@dataclass(frozen=True)
class DataGridMatch:
    """Shared observed-bin view; the original PHA arrays stay native."""
    reference: object
    reference_id: object

    @property
    def unit(self):
        return _MATCH_COORDINATE

    def reference_data(self):
        result = self.reference()
        if result is None:
            raise ValueError("reference dataset is no longer available; match the grids again")
        return result


_MATCHES = weakref.WeakKeyDictionary()


def register_data_match(datasets, *, reference=None):
    reference = next(iter(datasets)) if reference is None else reference
    if reference not in datasets:
        raise ValueError("reference must be one of the matched dataset IDs")
    ref = datasets[reference]
    match = DataGridMatch(weakref.ref(ref), reference)
    for data in datasets.values():
        _MATCHES[data] = match
    return match


def get_data_match(data):
    return _MATCHES.get(data)

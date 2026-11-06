"""NumPy-only integrated grids, projections, and shared evaluations."""
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from threading import RLock

import numpy as np


@dataclass(frozen=True, eq=False)
class EnergyGrid:
    """Increasing, non-overlapping energy bins, in keV; gaps are allowed."""

    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self):
        lo = np.array(self.lo, dtype=float, copy=True)
        hi = np.array(self.hi, dtype=float, copy=True)
        if lo.ndim != 1 or hi.shape != lo.shape or not lo.size:
            raise ValueError("lo and hi must be nonempty, equal-length 1D arrays")
        if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
            raise ValueError("grid boundaries must be finite")
        if np.any(lo < 0) or np.any(hi <= lo):
            raise ValueError("energy bins require 0 <= lo < hi")
        if np.any(lo[1:] < hi[:-1]):
            raise ValueError("energy bins must increase without overlap")
        lo.setflags(write=False)
        hi.setflags(write=False)
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @classmethod
    def from_edges(cls, edges):
        edges = np.asarray(edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError("edges must be a 1D array with at least two values")
        return cls(edges[:-1], edges[1:])

    @property
    def size(self):
        return self.lo.size


def merge_grids(grids):
    """Union every boundary without rounding or averaging calibration edges."""
    grids = tuple(grids)
    if not grids or not all(isinstance(g, EnergyGrid) for g in grids):
        raise ValueError("provide at least one EnergyGrid")
    edges = np.unique(np.concatenate([a for g in grids for a in (g.lo, g.hi)]))
    return EnergyGrid.from_edges(edges)


@dataclass(frozen=True, eq=False)
class GridPlan:
    """A common grid and native-grid mappings, independent of any model."""

    grid: EnergyGrid
    native: tuple
    projections: tuple


def prepare_grids(grids, *, grid=None):
    """Prepare bin geometry alone; no model, evaluator, or parameters needed."""
    native = tuple(grids)
    if not native or not all(isinstance(g, EnergyGrid) for g in native):
        raise ValueError("provide at least one EnergyGrid")
    if grid is None:
        grid = merge_grids(native)
    return GridPlan(grid, native, tuple(Projection(grid, target) for target in native))


def _indices(axis, values):
    """Exact edge lookup, allowing only floating-point round-trip noise."""
    values = np.asarray(values, dtype=float)
    idx = np.searchsorted(axis, values)
    right = np.clip(idx, 0, len(axis) - 1)
    left = np.clip(idx - 1, 0, len(axis) - 1)
    idx = np.where(abs(axis[right] - values) <= abs(axis[left] - values), right, left)
    tolerance = 16 * np.finfo(float).eps * np.maximum(abs(values), abs(axis[idx]))
    if np.any(abs(axis[idx] - values) > tolerance):
        raise ValueError("requested edge is absent from the shared grid; rebuild the grid")
    return idx


class Projection:
    """Sum whole integrated model bins onto a coarser, aligned grid."""

    def __init__(self, source, target):
        self.source_size = source.size
        self.target_size = target.size
        self.start = _indices(source.lo, target.lo)
        self.stop = _indices(source.hi, target.hi) + 1
        if np.any(self.stop <= self.start):
            raise ValueError("target bins cannot split source bins")
        gaps = np.r_[0, np.cumsum(source.lo[1:] != source.hi[:-1])]
        if np.any(gaps[self.stop - 1] != gaps[self.start]):
            raise ValueError("a target bin cannot bridge a gap in the source grid")

    def __call__(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (self.source_size,):
            raise ValueError("values must have one entry per source bin")
        # Avoid subtracting large cumulative totals for weak narrow features.
        boundaries = np.column_stack((self.start, self.stop)).ravel()
        return np.add.reduceat(np.r_[values, 0.0], boundaries)[::2]


def count_projection(native, target):
    """Conservative count coarsening: no splitting, loss, or double counting."""
    projection = Projection(native, target)
    if (projection.start[0] != 0 or projection.stop[-1] != native.size or
            np.any(projection.start[1:] != projection.stop[:-1])):
        raise ValueError("combined grid must use every input count bin exactly once")
    return projection


@dataclass(frozen=True, eq=False)
class CountCombination:
    """A model-independent sum on compatible selected/grouped count bins."""

    grid: EnergyGrid
    projections: tuple
    counts: np.ndarray


def prepare_count_combination(grids, counts, *, grid=None):
    """Sum already grouped/selected counts using whole-bin projections.

    The first input grid is the default target. Explicit targets may further
    coarsen it. All bins supplied here must be used exactly once; an earlier
    data-selection step may already have excluded channels. The native
    source/response input grids are independent of this count-bin operation.
    """
    grids, counts = tuple(grids), tuple(counts)
    if not grids or len(grids) != len(counts):
        raise ValueError("provide matching nonempty sequences of count grids and arrays")
    if not all(isinstance(g, EnergyGrid) for g in grids):
        raise ValueError("every count grid must be an EnergyGrid")
    arrays = []
    for native, values in zip(grids, counts):
        values = np.asarray(values, dtype=float)
        if values.shape != (native.size,) or not np.isfinite(values).all():
            raise ValueError("counts must have one finite value per selected/grouped bin")
        if np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError("Poisson combination requires nonnegative integer counts")
        arrays.append(values)
    grid = grids[0] if grid is None else grid
    try:
        projections = tuple(count_projection(native, grid) for native in grids)
    except ValueError as exc:
        raise ValueError("selected/grouped count bins are incompatible; use matching "
                         "boundaries or an explicit common coarser grid") from exc
    combined = sum(projection(values) for projection, values in zip(projections, arrays))
    combined.setflags(write=False)
    return CountCombination(grid, projections, combined)


CacheInfo = namedtuple("CacheInfo", "hits misses entries maxsize")


class SharedEvaluation:
    """Evaluate a deterministic integrated spectrum once per cached vector.

    evaluator(parameters, lo_keV, hi_keV) must return bin-integrated flux.
    token is a hashable caller-supplied configuration key. It is not a
    replacement for clearing any caches internal to evaluator itself.
    """

    def __init__(self, grid, evaluator, cache_size=1):
        if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size < 1:
            raise ValueError("cache_size must be a positive integer")
        self.grid = grid
        self.evaluator = evaluator
        self.maxsize = cache_size
        self.enabled = True
        self._cache = OrderedDict()
        self._lock = RLock()
        self._hits = self._misses = 0

    def evaluate(self, parameters, token=None):
        parameters = np.asarray(parameters, dtype=float)
        if parameters.ndim != 1 or not np.isfinite(parameters).all():
            raise ValueError("model parameters must be a finite 1D vector")
        key = (parameters.tobytes(), token)
        hash(key)
        with self._lock:
            if self.enabled and key in self._cache:
                self._hits += 1
                self._cache.move_to_end(key)
                return self._cache[key].copy()
            self._misses += 1
            values = np.array(self.evaluator(parameters.copy(), self.grid.lo, self.grid.hi),
                              dtype=float, copy=True)
            if values.shape != (self.grid.size,) or not np.isfinite(values).all():
                raise ValueError("source model must return one finite integrated value per bin")
            if self.enabled:
                values.setflags(write=False)
                self._cache[key] = values
                if len(self._cache) > self.maxsize:
                    self._cache.popitem(last=False)
            return values.copy()

    def project(self, parameters, target, token=None):
        # Validate before doing potentially expensive model work.
        projection = Projection(self.grid, target)
        return projection(self.evaluate(parameters, token))

    def clear(self, reset_stats=True):
        with self._lock:
            self._cache.clear()
            if reset_stats:
                self._hits = self._misses = 0

    def cache_info(self):
        with self._lock:
            return CacheInfo(self._hits, self._misses, len(self._cache), self.maxsize)

"""Sherpa adapters. Source models in this module are defined in keV."""
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

try:
    from sherpa.astro import hc
    from sherpa.astro.data import DataPHA
    from sherpa.astro.instrument import Response1D, ARFModel, RMFModel, RSPModel
    from sherpa.data import Data1DInt, DataSimulFit
    from sherpa.fit import Fit
    from sherpa.models.model import ArithmeticModel, CompositeModel, SimulFitModel
    from sherpa.stats import CStat
except ImportError as exc:
    raise ImportError("Run the Sherpa adapter inside a CIAO/Sherpa Python environment") from exc

from .grid import (EnergyGrid, GridPlan, Projection, SharedEvaluation,
                   prepare_count_combination, prepare_grids, _indices)
from .resample import register_data_match, convert_grid, resample_combination
from .units import axis_label


def _parts(model):
    yield model
    for part in getattr(model, "parts", ()):
        yield from _parts(part)


class SharedGridSource(CompositeModel, ArithmeticModel):
    """Cache a full source expression on one integrated energy grid.

    ``source.calc(p, lo, hi)`` must return photon flux integrated over each
    energy bin. Configuration changes outside Parameter objects (e.g.
    XSPEC abundances or xset values) require ``clear_cache()`` afterwards.
    """

    ndim = 1

    def __init__(self, source, grid, name="shared_grid", cache_size=1):
        if getattr(source, "ndim", None) not in (None, 1):
            raise ValueError("only one-dimensional source models are supported")
        if any(isinstance(part, (ARFModel, RMFModel, RSPModel)) for part in _parts(source)):
            raise ValueError("supply an unconvolved source expression, without ARF/RMF models")
        self.source = source
        self.grid = grid
        self._evaluation = SharedEvaluation(grid, source.calc, cache_size)
        self._projections = OrderedDict()
        CompositeModel.__init__(self, name, (source,))

    def calc(self, p, xlo, xhi=None, *args, **kwargs):
        if xhi is None or args:
            raise ValueError("shared evaluation requires integrated energy bins")
        if set(kwargs) - {"integrate"} or not kwargs.get("integrate", True):
            raise ValueError("only integrated model evaluation is supported")
        if getattr(self.source, "integrate", True) is False:
            raise ValueError("the source must return bin-integrated flux")
        target = EnergyGrid(xlo, xhi)
        key = (target.lo.tobytes(), target.hi.tobytes())
        if key not in self._projections:
            self._projections[key] = Projection(self.grid, target)
            if len(self._projections) > 128:
                self._projections.popitem(last=False)
        self._projections.move_to_end(key)
        # Include integrate flags in addition to every explicit parameter.
        state = tuple(getattr(part, "integrate", None) for part in _parts(self.source))
        return self._projections[key](self._evaluation.evaluate(p, state))

    def startup(self, cache=False):
        self._evaluation.enabled = bool(cache)
        self._evaluation.clear()
        self.source.startup(cache)

    def teardown(self):
        self.source.teardown()

    def clear_cache(self):
        """Clear shared AND native source caches after changing model settings."""
        self._evaluation.clear()
        self.source.cache_clear()

    cache_clear = clear_cache

    def cache_info(self):
        return self._evaluation.cache_info()


class _DatasetSource(CompositeModel, ArithmeticModel):
    """Translate Sherpa's response-coordinate convention into energy."""

    ndim = 1

    def __init__(self, shared, pha, name):
        self.shared = shared
        self.pha = pha
        CompositeModel.__init__(self, name, (shared,))

    def calc(self, p, xlo, xhi=None, *args, **kwargs):
        if xhi is None:
            raise ValueError("a PHA response must supply low and high bin edges")
        if self.pha.units == "wavelength":
            xlo, xhi = hc / np.asarray(xhi), hc / np.asarray(xlo)
        return self.shared.calc(p, xlo, xhi, *args, **kwargs)

    def startup(self, cache=False):
        self.shared.startup(cache)

    def teardown(self):
        self.shared.teardown()


def _datasets(datasets):
    if isinstance(datasets, Mapping):
        result = dict(datasets)
    else:
        result = dict(enumerate(datasets, 1))
    if not result or not all(isinstance(p, DataPHA) for p in result.values()):
        raise ValueError("provide DataPHA objects, either as a sequence or an ID mapping")
    if len({id(p) for p in result.values()}) != len(result):
        raise ValueError("the same DataPHA object cannot occur more than once")
    for pha in result.values():
        if len(pha.response_ids) != 1:
            raise ValueError("this prototype requires exactly one response per PHA dataset")
        arf, rmf = pha.get_response()
        if arf is not None and rmf is not None:
            if not (np.array_equal(arf.energ_lo, rmf.energ_lo) and
                    np.array_equal(arf.energ_hi, rmf.energ_hi)):
                raise ValueError("this prototype requires matching ARF/RMF input bins within each dataset")
    return result


def _response_grids(datasets):
    for pha in datasets.values():
        arf, rmf = pha.get_response()
        for response in (arf, rmf):
            if response is not None:
                yield EnergyGrid(response.energ_lo, response.energ_hi)


@dataclass
class DatasetGroup:
    """Prepared dataset grids. A source can be attached or replaced later."""

    datasets: dict
    plan: GridPlan
    name: str = "shared_grid"
    session: object = None
    shared: object = field(default=None, init=False)
    sources: dict = field(default_factory=dict, init=False)
    models: dict = field(default_factory=dict, init=False)
    data_match: object = field(default=None, init=False)
    combination_mode: str = field(default="joint", init=False)

    @property
    def grid(self):
        return self.plan.grid

    def _check_data(self):
        if self.session is not None:
            for key, pha in self.datasets.items():
                if self.session.get_data(key) is not pha:
                    raise ValueError("session datasets changed; match the grids again")
        current = tuple(_response_grids(_datasets(self.datasets)))
        if (len(current) != len(self.plan.native) or
                any(not (np.array_equal(a.lo, b.lo) and np.array_equal(a.hi, b.hi))
                    for a, b in zip(current, self.plan.native))):
            raise ValueError("response grids changed; match the grids again")

    def _require_source(self):
        if self.shared is None:
            raise RuntimeError("no source assigned; call group.set_source(source) before fitting")

    def set_source(self, source, *, cache_size=1):
        """Attach any compatible source to this prepared geometry.

        A UI-backed group installs the wrappers in its session. Repeating
        this operation replaces the model/cache but reuses the grid plan.
        Source expressions are defined in keV and return integrated flux.
        """
        self._check_data()
        if self.session is not None:
            if set(self.datasets).intersection(self.session.list_pileup_model_ids()):
                raise ValueError("pileup models are not supported by this prototype")
        shared = SharedGridSource(source, self.grid, self.name, cache_size)
        # The response mappings were already computed without a model.
        for native, projection in zip(self.plan.native, self.plan.projections):
            shared._projections[(native.lo.tobytes(), native.hi.tobytes())] = projection
        sources = {key: _DatasetSource(shared, pha, f"{self.name}_{key}")
                   for key, pha in self.datasets.items()}
        models = {key: Response1D(pha)(sources[key]) for key, pha in self.datasets.items()}
        if self.session is not None:
            for key in self.datasets:
                self.session.set_source(key, sources[key])
            models = {key: self.session.get_model(key) for key in self.datasets}
        self.shared, self.sources, self.models = shared, sources, models
        return self

    def cache_info(self):
        self._require_source()
        return self.shared.cache_info()

    def clear_cache(self):
        self._require_source()
        self.shared.clear_cache()

    def make_fit(self, stat=None, method=None):
        """Construct a Fit using the current combination mode and selection.

        After combine_datasets(mode="isis"), sum matched data and folded
        models before calculating the statistic. The returned Fit freezes
        the current grouping/filtering; call make_fit again after changes.
        Ordinary ui.fit remains joint. The following settings describe
        mode="joint" (also used when no combination has been requested).

        UI-backed groups inherit the session's statistic, optimizer, and
        iterative-fit settings, including simultaneous background models.
        Explicit ``stat`` and ``method`` override only the returned Fit.
        Calling its ``fit()`` does not update ``ui.get_fit_results()``;
        use ``ui.fit(*group.datasets)`` for normal UI result bookkeeping.

        Without a session, the default statistic remains CStat for raw
        background-free data. Background-bearing or subtracted data need
        an explicit statistic, e.g. ``WStat()`` for unsubtracted Poisson
        backgrounds or an appropriate chi-square statistic after subtraction.
        """
        if self.combination_mode == "isis":
            return _make_isis_fit(self, stat=stat, method=method)
        self._require_source()
        self._check_data()
        if self.session is not None:
            self.models = {key: self.session.get_model(key) for key in self.datasets}
            ids = tuple(self.datasets)
            # Use the same preparation as ui.fit: get_model alone includes
            # background flux in the source region, but does not add the
            # background-region likelihoods. _get_fit handles both, including
            # full background models, subtraction, grouping, and filtering.
            _, fit = self.session._get_fit(ids[0], ids[1:])
            if stat is not None:
                fit.stat = stat
            if method is not None:
                fit.method = method
            return fit
        if stat is None and any(p.background_ids or p.subtracted
                                for p in self.datasets.values()):
            raise ValueError("background-bearing DataPHA fits require an explicit statistic: "
                             "use stat=WStat() for unsubtracted Poisson backgrounds, "
                             "or an appropriate chi-square statistic after subtraction; "
                             "use match_dataset_grids(..., session=...) for session models")
        return _make_fit(
            DataSimulFit("shared_grid_data", tuple(self.datasets.values())),
            SimulFitModel("shared_grid_models", tuple(self.models.values())),
            stat, method)


def _make_fit(data, model, stat, method):
    options = {"stat": CStat() if stat is None else stat}
    if method is not None:
        options["method"] = method
    return Fit(data, model, **options)


def match_pha_grids(datasets, *, grid=None, name="shared_grid", reference=None):
    """Prepare common response-bin geometry, without a source model.

    ``grid`` may be an EnergyGrid refining every response bin boundary.
    Otherwise use the union of all ARF/RMF boundaries. This operation
    does not inspect existing models, build model wrappers, or change
    counts, responses, grouping, filters, or a Sherpa UI session.
    Also register a common observed-bin view on the reference dataset for
    plotting and combination. The first ID is the default reference;
    matching uses a fixed internal coordinate independent of set_analysis.
    There is no unit option. Native PHA/response arrays
    remain intact. Later call ``group.set_source(source)`` for shared models.
    """
    datasets = _datasets(datasets)
    plan = prepare_grids(_response_grids(datasets), grid=grid)
    group = DatasetGroup(datasets, plan, name)
    group.data_match = register_data_match(datasets, reference=reference)
    return group


def match_dataset_grids(ids, *, session=None, grid=None, name="shared_grid",
                       reference=None):
    """Prepare shared grids for loaded PHA datasets before defining a model.

    Only datasets and response-bin boundaries are read. No existing model
    is required, inspected, inferred, or modified. Later use
    ``group.set_source(source)`` to assign a model in this UI session.
    Register observed-bin regridding onto reference as well as a shared
    source grid. Plotting/combination use the matched view; the underlying
    native PHA counts and calibration arrays remain intact.
    Matching is independent of Sherpa's current analysis setting, before
    or after preparation. No unit argument or user conversion is needed.
    """
    if session is None:
        from sherpa.astro import ui as session
    ids = tuple(ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("provide a nonempty sequence of distinct dataset IDs")
    datasets = {key: session.get_data(key) for key in ids}
    group = match_pha_grids(datasets, grid=grid, name=name,
                            reference=reference)
    group.session = session
    return group


def _channel_grid(pha):
    """Return detector-energy bins in increasing order and channel permutation."""
    _, rmf = pha.get_response()
    if rmf is None or rmf.e_min is None or rmf.e_max is None:
        raise ValueError("count combination requires RMF EBOUNDS for every dataset")
    channel = np.asarray(pha.channel)
    indices = channel.astype(int) - rmf.offset
    if (np.any(channel != channel.astype(int)) or np.any(indices < 0) or
            np.any(indices >= len(rmf.e_min))):
        raise ValueError("PHA channels do not map into RMF EBOUNDS")
    lo, hi = np.asarray(rmf.e_min)[indices], np.asarray(rmf.e_max)[indices]
    order = np.argsort(lo)
    return EnergyGrid(lo[order], hi[order]), order


def _selected_count_grid(pha, unit="energy"):
    """Detector bins and counts after the PHA's actual grouping/filtering.

    Keep the permutation of GROUPED values for model predictions. Responses
    are still evaluated on their native input grids and folded to channels;
    the same PHA selection/grouping is then applied to those predictions.
    """
    if len(pha.get_noticed_channels()) == 0:
        raise ValueError("each combined dataset must have at least one selected bin")
    native, order = _channel_grid(pha)
    # Restore native channel order before asking Sherpa to apply its grouping.
    lo, hi = np.empty(native.size), np.empty(native.size)
    lo[order], hi[order] = native.lo, native.hi
    lo = np.asarray(pha.apply_filter(lo, pha._min))
    hi = np.asarray(pha.apply_filter(hi, pha._max))
    counts = np.asarray(pha.get_dep(filter=True))
    if not lo.size:
        raise ValueError("each combined dataset must have at least one selected bin")
    order = np.argsort(lo)
    grid, permutation = convert_grid(EnergyGrid(lo[order], hi[order]), "energy", unit, hc=hc)
    order = order[permutation]
    return grid, order, counts[order]


def _raw_selected_snapshot(pha):
    """Keep channel selections but postpone grouping until after regrid."""
    selected = np.array(pha.get_noticed_channels(), copy=True)
    snapshot = deepcopy(pha)
    snapshot.ungroup()
    snapshot.quality_filter = None
    snapshot.mask = np.isin(snapshot.channel, selected)
    return snapshot


class _CombinedCountsModel(CompositeModel, ArithmeticModel):
    ndim = 1

    def __init__(self, models, phas, projections, orders, grid, name, additive=None, divisors=None):
        self.models = tuple(models)
        self.phas = tuple(phas)
        self.projections = tuple(projections)
        self.orders = tuple(orders)
        self.grid = grid
        self.additive = tuple(np.zeros(len(pha.channel)) for pha in phas) if additive is None else tuple(additive)
        self.divisors = (1.,)*len(self.phas) if divisors is None else tuple(divisors)
        CompositeModel.__init__(self, name, self.models)

    def calc(self, p, xlo, xhi=None, *args, **kwargs):
        if xhi is None:
            raise ValueError("combined count models require integrated bins")
        total = np.zeros(self.grid.size)
        offset = 0
        for model, pha, projection, order, additive, divisor in zip(
                self.models, self.phas, self.projections, self.orders, self.additive, self.divisors):
            size = len(model.pars)
            native = model.calc(p[offset:offset + size], pha.channel)
            # Group and select model counts EXACTLY as for the observed data.
            # apply_filter accepts either full or response-filtered channels.
            selected = pha.apply_filter((np.asarray(native) + additive)/divisor)
            total += projection(np.asarray(selected)[order])
            offset += size
        # Data1DInt passes only noticed bins here. These must be original
        # combined bins, never partial/fractional bins.
        loidx = _indices(self.grid.lo, xlo)
        hiidx = _indices(self.grid.hi, xhi)
        if np.any(loidx != hiidx):
            raise ValueError("requested bins must match the combined dataset")
        return total[loidx]

    def startup(self, cache=False):
        for model in self.models:
            model.startup(cache)

    def teardown(self):
        for model in self.models:
            model.teardown()


@dataclass
class CombinedDataset:
    data: Data1DInt
    members: tuple
    group: DatasetGroup
    projections: tuple
    orders: tuple
    grid: EnergyGrid
    covariance: object = None
    fractional: bool = False
    unit: str = "energy"
    model: object = field(default=None, init=False)
    _session: object = field(default=None, init=False, repr=False)
    _id: object = field(default=None, init=False, repr=False)

    def set_source(self, source, *, cache_size=1):
        """Attach or replace a source after counts and response maps exist."""
        if self._session is not None and self._session.get_data(self._id) is not self.data:
            raise ValueError("registered dataset changed; register the combination again")
        self.group.set_source(source, cache_size=cache_size)
        self.model = _CombinedCountsModel(
            tuple(self.group.models.values()), tuple(self.group.datasets.values()),
            self.projections, self.orders, self.grid, self.data.name + "_model")
        if self._session is not None:
            self._session.set_full_model(self._id, self.model)
        return self

    def make_fit(self, stat=None, method=None):
        if self.model is None:
            raise RuntimeError("no source assigned; call combined.set_source(source) before fitting")
        if self.fractional or self.data.staterror is not None:
            from .statistics import CovarianceChi2
            from sherpa.stats import Likelihood
            if stat is None:
                stat = CovarianceChi2(self.data, self.covariance)
            elif self.fractional and isinstance(stat, Likelihood):
                raise ValueError("fractionally rebinned counts are correlated; use the default "
                                 "covariance fit or fit native data with a Poisson statistic")
        return _make_fit(self.data, self.model, stat, method)

    def cache_info(self):
        """Source evaluations/reuse for this combination's assigned model."""
        return self.group.cache_info()

    def register(self, id, *, session=None):
        """Register data now; attach the full model now or at set_source()."""
        if session is None:
            from sherpa.astro import ui as session
        if id in session.list_data_ids():
            raise ValueError("choose a new dataset ID for the combined spectrum")
        session.set_data(id, self.data)
        if self.model is not None:
            session.set_full_model(id, self.model)
        if self.fractional or self.data.staterror is not None:
            from .statistics import CovarianceChi2
            session.set_stat(CovarianceChi2(self.data, self.covariance))
        else:
            session.set_stat(CStat())
        self._session, self._id = session, id


def combine_datasets(group, *, mode="isis", grid=None, name="combined"):
    """Prepare an ISIS-style combined fit, a joint fit, or a count snapshot.

    The default sums matched data and individually folded models BEFORE
    computing the statistic. Call ``group.make_fit()`` after grouping and
    filtering; each call reads the current selection. ``mode="joint"``
    explicitly retains Sherpa's independent dataset likelihoods, including
    WStat. Ordinary ``ui.fit`` is always a native Sherpa joint fit.
    ``mode="sum"`` retains the covariance-aware background-free snapshot.
    An output ``grid`` and ``name`` apply only to sum mode.
    """
    if mode in ("isis", "joint"):
        if grid is not None or name != "combined":
            raise ValueError("grid and name are only supported with mode='sum'")
        group._check_data()
        group.combination_mode = mode
        return group
    if mode != "sum":
        raise ValueError("mode must be 'isis', 'joint', or 'sum'")
    return _combine_datasets_sum(group, grid=grid, name=name)


def _make_isis_fit(group, *, stat=None, method=None):
    """Freeze the current matched/grouped view for one summed-statistic fit.

    Native channel data and responses stay untouched. Fold each source through
    its own unfiltered response, then apply exactly the plotting operator.
    """
    from scipy import sparse
    from sherpa.stats import Likelihood, WStat
    from .plotting import _matched_target, _read_pha, _alignment_projection, _areascal_divisor
    from .resample import diagonal_covariance

    group._check_data()
    if group.session is None:
        group._require_source()
    if stat is None:
        stat = CStat() if group.session is None else group.session.get_stat()
    if method is None and group.session is not None:
        method = group.session.get_method()
    if isinstance(stat, WStat):
        raise ValueError("ISIS combined-count fitting does not use Sherpa's per-spectrum WStat "
                         "background likelihoods; use mode='joint' for WStat, or explicitly "
                         "choose CStat for fixed measured backgrounds")
    subtract = {bool(pha.subtracted) for pha in group.datasets.values()}
    if len(subtract) != 1:
        raise ValueError("ISIS combination requires consistent background-subtraction settings")
    subtract = subtract.pop()
    if subtract and isinstance(stat, Likelihood):
        raise ValueError("Poisson statistics require unsubtracted counts")
    options = dict(subtract_background=subtract, background_only=False,
                   error_method="sqrt", exposure_source="pha")
    reference = group.data_match.reference_data()
    unit = group.data_match.unit
    target = _matched_target(reference, unit, options)
    grid = EnergyGrid(target.lo, target.hi)
    phas, orders, projections, models, additive, divisors = [], [], [], [], [], []
    counts = np.zeros(grid.size)
    covariance = sparse.csr_matrix((grid.size, grid.size))
    for key, original in group.datasets.items():
        pha = deepcopy(original)
        pha.ungroup()
        pha.quality_filter = None
        pha.mask = True
        for bid in pha.background_ids:
            background = pha.get_background(bid)
            background.ungroup()
            background.quality_filter = None
            background.mask = True
        native, order, _ = _selected_count_grid(pha, unit)
        projection = _alignment_projection(native, target)
        spectrum = _read_pha(pha, unit, options, raw=True, unfiltered=True, isis_areascal=True)
        counts += projection(spectrum.counts)
        covariance += projection.covariance(diagonal_covariance(spectrum.staterror**2))
        offset = np.zeros(len(pha.channel))
        if group.session is None:
            if pha.background_ids:
                raise ValueError("ISIS fits with backgrounds require a session; use mode='joint' "
                                 "with an explicit statistic for session-free backgrounds")
            source = group.sources[key]
            model = Response1D(pha)(source)
        else:
            from sherpa.astro.background import get_response_for_pha
            from sherpa.utils.err import ModelErr, IdentifierErr
            if key in group.session.list_pileup_model_ids():
                raise ValueError("ISIS matched-count combination does not support pileup models")
            try:
                source = group.session.get_source(key)
            except (ModelErr, IdentifierErr) as exc:
                raise ValueError("ISIS combined fitting requires source expressions; "
                                 "use set_source rather than set_full_model, or mode='joint'") from exc
            bkg_sources = {}
            measured = np.zeros(len(pha.channel))
            for bid in pha.background_ids:
                measured += np.asarray(pha.get_background_scale(
                    bid, units="counts", group=False, filter=False))*pha.get_background(bid).counts
                try:
                    bkg_sources[bid] = group.session.get_bkg_source(key, bkg_id=bid)
                except ModelErr:
                    try:
                        group.session.get_bkg_model(key, bkg_id=bid)
                    except ModelErr:
                        pass
                    else:
                        raise ValueError("full background models require mode='joint'")
            if bkg_sources and len(bkg_sources) != len(pha.background_ids):
                raise ValueError("model either all or none of a dataset's backgrounds")
            model = get_response_for_pha(pha, source, bkg_sources, None, key)
            if bkg_sources:
                offset = -measured if subtract else offset
            elif not subtract:
                offset = measured
        phas.append(pha)
        orders.append(order)
        projections.append(projection)
        models.append(model)
        additive.append(offset)
        divisors.append(_areascal_divisor(pha))
    data = Data1DInt(group.name+"_isis_counts", grid.lo, grid.hi, counts,
                     staterror=np.sqrt(covariance.diagonal()))
    data.set_xlabel(axis_label(unit))
    data.set_ylabel("Counts")
    model = _CombinedCountsModel(models, phas, projections, orders, grid,
                                 group.name+"_isis_model", additive, divisors)
    return _make_fit(data, model, stat, method)


def _combine_datasets_sum(group, *, grid=None, name="combined"):
    """Combine counts and response geometry without requiring any model.

    ``grid`` is a detector-bin grid in the matching coordinate, not the
    source evaluation grid. Default: the reference's selected/grouped bins.
    Native selected channels and folded predictions are conservatively
    rebinned onto it. Fractional overlaps carry their full covariance; the
    default fit uses covariance-aware Gaussian chi-square in that case.
    Combined background likelihoods and subtracted/weighted input counts
    are not supported. Later call
    ``combined.set_source(source)`` to build the folded model. Any model
    attached to the original group is not inspected or inherited.
    """
    group._check_data()
    match = group.data_match
    unit = "energy" if match is None else match.unit
    target = None
    if grid is None and match is not None:
        from .plotting import _matched_target
        target = _matched_target(match.reference_data(), unit,
                                 dict(subtract_background=False, background_only=False,
                                      error_method="sqrt", exposure_source="pha"))
        grid = EnergyGrid(target.lo, target.hi)
    # Regrid native channels BEFORE reference grouping. Independent grouping
    # of HEG/MEG first would lose resolution needed by the matched view.
    phas = tuple(_raw_selected_snapshot(pha) if match is not None else deepcopy(pha)
                 for pha in group.datasets.values())
    count_grids, orders, selected_counts, variances = [], [], [], []
    explicit_errors = False
    for pha in phas:
        if pha.background_ids or pha.subtracted:
            raise ValueError("count combination currently supports background-free raw counts")
        counts = np.asarray(pha.counts)
        if (not np.isfinite(counts).all() or np.any(counts < 0) or
                np.any(counts != np.floor(counts))):
            raise ValueError("Poisson combination requires nonnegative integer counts")
        native, order, counts = _selected_count_grid(pha, unit)
        count_grids.append(native)
        orders.append(order)
        selected_counts.append(counts)
        variance = np.maximum(counts, 1.)
        if pha.staterror is not None:
            variance = np.asarray(pha.get_staterror(filter=True))[order]**2
            explicit_errors = True
        if pha.syserror is not None:
            variance = variance + np.asarray(pha.get_syserror(filter=True))[order]**2
            explicit_errors = True
        variances.append(variance)
    if match is None:
        combination = prepare_count_combination(count_grids, selected_counts, grid=grid)
        from .resample import diagonal_covariance
        covariance = diagonal_covariance(sum(p(v) for p, v in zip(combination.projections, variances)))
        fractional = False
    else:
        combination = resample_combination(count_grids, selected_counts, grid=grid, variances=variances)
        if target is not None:
            from .plotting import _alignment_projection
            from .resample import ResampledCombination, diagonal_covariance
            projections = tuple(_alignment_projection(native, target) for native in count_grids)
            combination = ResampledCombination(
                grid, projections, sum(p(c) for p, c in zip(projections, selected_counts)),
                sum((p.covariance(diagonal_covariance(v)) for p, v in zip(projections, variances))),
                any(p.fractional for p in projections))
        covariance, fractional = combination.covariance, combination.fractional
    grid = combination.grid
    members = tuple(group.datasets)
    snapshot_group = match_pha_grids(dict(zip(members, phas)), grid=group.grid,
                                     name=name + "_shared",
                                     reference=None if match is None else match.reference_id)
    errors = np.sqrt(covariance.diagonal()) if fractional or explicit_errors else None
    data = Data1DInt(name, grid.lo.copy(), grid.hi.copy(), combination.counts.copy(), staterror=errors)
    data.set_xlabel(axis_label(unit))
    data.set_ylabel("Counts")
    return CombinedDataset(data, members, snapshot_group, combination.projections,
                           tuple(orders), grid, covariance, fractional, unit)


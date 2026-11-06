"""Real Sherpa integration tests. Skipped when Sherpa is not installed."""
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

HAS_SHERPA = importlib.util.find_spec("sherpa") is not None

if HAS_SHERPA:
    from sherpa.astro.data import DataARF, DataPHA, DataRMF
    from sherpa.astro.instrument import Response1D
    from sherpa.astro.ui.utils import Session
    from sherpa.data import DataSimulFit
    from sherpa.fit import Fit
    from sherpa.models.model import ArithmeticModel, SimulFitModel
    from sherpa.models.parameter import Parameter
    from sherpa.optmethods import NelderMead
    from sherpa.stats import CStat
    from fancy_plotting import (EnergyGrid, Projection, combine_datasets,
                             match_dataset_grids, match_pha_grids, plot_data, plot_counts)

    class CountingPowerLaw(ArithmeticModel):
        ndim = 1

        def __init__(self, name="src"):
            self.gamma = Parameter(name, "gamma", 1.7, .5, 3.5, frozen=True)
            self.ampl = Parameter(name, "ampl", .08, .001, 1)
            self.calls = 0
            super().__init__(name, (self.gamma, self.ampl))

        def calc(self, p, lo, hi=None, *args, **kwargs):
            self.calls += 1
            gamma, ampl = p
            lo, hi = np.asarray(lo), np.asarray(hi)
            if abs(gamma - 1) < 1e-10:
                return ampl * np.log(hi / lo)
            return ampl * (hi**(1-gamma) - lo**(1-gamma)) / (1-gamma)

    def make_pha(name, nenergy=32, nchan=8, exposure=500):
        edges = np.linspace(.5, 7, nenergy+1)
        channels = np.linspace(.5, 7, nchan+1)
        energy = (edges[:-1] + edges[1:]) / 2
        centres = (channels[:-1] + channels[1:]) / 2
        matrix = np.exp(-.5*((energy[:, None]-centres)/.35)**2)
        matrix /= matrix.sum(axis=1)[:, None]
        rmf = DataRMF(name+"_rmf", nchan, edges[:-1], edges[1:],
                      np.ones(nenergy, dtype=int), np.ones(nenergy, dtype=int),
                      np.full(nenergy, nchan, dtype=int), matrix.ravel(),
                      offset=1, e_min=channels[:-1], e_max=channels[1:])
        arf = DataARF(name+"_arf", edges[:-1], edges[1:],
                      30*(energy/2)**(-.2))
        pha = DataPHA(name, np.arange(1, nchan+1), np.zeros(nchan),
                      exposure=exposure, areascal=np.linspace(.9, 1.1, nchan))
        pha.set_response(arf, rmf)
        pha.set_analysis("energy")
        return pha


@unittest.skipUnless(HAS_SHERPA, "requires a real CIAO/Sherpa installation")
class SherpaPlottingTests(unittest.TestCase):
    def setUp(self):
        self.phas = [make_pha("a", 32, 8, 500), make_pha("b", 48, 8, 800)]
        self.session = Session()
        self.session.set_stat("chi2datavar")
        for key, pha in enumerate(self.phas, 1):
            pha.counts = np.arange(5, 13, dtype=float) * key
            self.session.set_data(key, pha)

    def tearDown(self):
        from matplotlib import pyplot as plt
        plt.close("all")

    def test_plotting_uses_no_model_and_preserves_input(self):
        error = AssertionError("data plotting must not access models")
        before = [(p.counts.copy(), p.units, p.grouped, p.subtracted) for p in self.phas]
        with patch.object(Session, "get_source", side_effect=error), \
                patch.object(Session, "get_model", side_effect=error), \
                patch.object(Session, "set_source", side_effect=error):
            result = plot_data([[1, 2]], session=self.session, combine="sum")
        expected = (self.phas[0].counts/500 + self.phas[1].counts/800)/(6.5/8)
        assert_allclose(result.curves[0].y, expected)
        for pha, (counts, units, grouped, subtracted) in zip(self.phas, before):
            assert_array_equal(pha.counts, counts)
            self.assertEqual((pha.units, pha.grouped, pha.subtracted), (units, grouped, subtracted))

    def test_background_subtraction_and_scaled_variance(self):
        source = self.phas[0]
        source.counts = np.full(8, 100.)
        source.backscal = 2.
        background = make_pha("background", 32, 8, 500)
        background.counts = np.full(8, 10.)
        background.backscal = 4.
        source.set_background(background)
        result = plot_counts(1, session=self.session)
        assert_allclose(result.curves[0].y, 95)
        assert_allclose(result.curves[0].yerr, np.sqrt(102.5))
        self.assertFalse(source.subtracted)
        raw = plot_counts(1, session=self.session, subtract_background=False)
        assert_allclose(raw.curves[0].y, 100)

    def test_fancy_plot_prefers_arf_exposure_with_pha_override_and_fallback(self):
        pha = self.phas[0]
        arf, _ = pha.get_response()
        arf.exposure = 1000.
        default = plot_data(1, session=self.session, density=False).curves[0]
        native = plot_data(1, session=self.session, density=False, exposure_source="pha").curves[0]
        assert_allclose(default.y, pha.counts/1000)
        assert_allclose(native.y, pha.counts/500)
        for missing in (None, 0., -1.):
            arf.exposure = missing
            fallback = plot_data(1, session=self.session, density=False).curves[0]
            assert_allclose(fallback.y, pha.counts/500)

    def test_identical_grouping_and_filters(self):
        for pha in self.phas:
            pha.grouping = np.tile([1, -1], 4)
            pha.group()
            pha.notice(1.3, 6)
        masks = [np.array(p.mask, copy=True) for p in self.phas]
        result = plot_counts([[1, 2]], session=self.session)
        assert_allclose(result.curves[0].y, sum(p.get_dep(filter=True) for p in self.phas))
        for pha, mask in zip(self.phas, masks):
            assert_array_equal(pha.mask, mask)
            self.assertTrue(pha.grouped)

    def test_wavelength_uses_detector_bounds(self):
        result = plot_data(1, session=self.session, unit="Angstrom")
        curve = result.curves[0]
        assert_allclose(curve.y*(curve.hi-curve.lo), self.phas[0].counts[::-1]/500)
        self.assertEqual(self.phas[0].units, "energy")

    def test_different_noticed_bins_cannot_be_combined(self):
        self.phas[1].notice(2, 4)
        with self.assertRaisesRegex(ValueError, "detector bins"):
            plot_data([[1, 2]], session=self.session)

    def test_native_plot_preferences_are_inherited_without_modification(self):
        prefs = self.session.get_data_plot_prefs(1)
        before = dict(prefs)
        try:
            prefs.update(color="navy", marker="s", linestyle="dotted", capsize=3)
            result = plot_data([1, 2], session=self.session, styles=[{}, {"color": "orange"}])
            self.assertEqual(result.lines[0].get_color(), "navy")
            self.assertEqual(result.lines[1].get_color(), "orange")
            self.assertEqual(result.errorbars[0].lines[0].get_marker(), "s")
            self.assertEqual(result.errorbars[0].lines[1][0].get_markersize(), 6)
            self.assertEqual(prefs["color"], "navy")
        finally:
            prefs.clear()
            prefs.update(before)

    def test_default_dataset_and_matplotlib_subplot_workflow(self):
        from matplotlib import pyplot as plt
        self.session.set_default_id(2)
        figure, axes = plt.subplots(2, 1)
        old, = axes[0].plot([1, 2], [3, 4])
        plt.sca(axes[1])
        result = plot_data(session=self.session, clearwindow=False,
                           marker="o", mfc="none", capsize=2, xscale="log")
        self.assertEqual(result.curves[0].members, (2,))
        self.assertIs(result.ax, axes[1])
        self.assertIn(old, axes[0].lines)
        figure.canvas.draw()


@unittest.skipUnless(HAS_SHERPA, "requires a real CIAO/Sherpa installation")
class SherpaCountsPlottingTests(unittest.TestCase):
    def setUp(self):
        self.pha = make_pha("counts", 32, 8, 500)
        self.pha.counts = np.arange(10, 18, dtype=float)
        self.session = Session()
        self.session.set_data(1, self.pha)
        self.session.set_stat("chi2datavar")

    def tearDown(self):
        from matplotlib import pyplot as plt
        plt.close("all")

    def attach_background(self):
        self.pha.backscal = 2.
        background = make_pha("counts_background", 32, 8, 250)
        background.backscal = 4.
        background.counts = np.arange(1, 9, dtype=float)
        self.pha.set_background(background)
        return background

    def test_counts_ignore_rate_and_factor_but_follow_analysis_coordinate(self):
        for coordinate in ("energy", "wavelength", "channel"):
            self.session.set_analysis(coordinate, type="rate", factor=3)
            result = plot_counts(1, session=self.session)
            expected = self.pha.counts[::-1] if coordinate == "wavelength" else self.pha.counts
            assert_allclose(result.curves[0].y, expected)
            self.assertEqual(result.ax.get_ylabel(), r"Counts ($\mathrm{bin}^{-1}$)")
            self.assertEqual(self.pha.units, coordinate)
            self.assertEqual(self.pha.plot_fac, 3)

    def test_response_folded_model_uses_display_power_and_residuals_do_not(self):
        self.session.set_source(1, CountingPowerLaw())
        for coordinate in ("energy", "wavelength", "channel"):
            self.session.set_analysis(coordinate)
            base = plot_counts(1, session=self.session, res=4, rebin=2)
            weighted = plot_counts(1, session=self.session, res=4, rebin=2, power=2)
            weights = (base.curves[0].lo+base.curves[0].hi)/2
            assert_allclose(weighted.curves[0].y, base.curves[0].y*weights)
            assert_allclose(weighted.model_curves[0].y, base.model_curves[0].y*weights)
            assert_allclose(weighted.residuals[0].y, base.residuals[0].y)

    def test_background_only_scales_then_groups_and_filters_with_errors(self):
        background = self.attach_background()
        self.pha.grouping = np.tile([1, -1], 4)
        self.pha.group()
        self.pha.notice(1.3, 6)
        scale = self.pha.get_background_scale(1, units="counts", group=False, filter=False)
        expected = self.pha.apply_filter(scale*background.counts)
        variance = self.pha.apply_filter(scale**2*background.counts)
        counts, mask = self.pha.counts.copy(), np.array(self.pha.mask, copy=True)
        result = plot_counts(1, session=self.session, background="only")
        assert_allclose(result.curves[0].y, expected)
        assert_allclose(result.curves[0].yerr, np.sqrt(variance))
        assert_array_equal(self.pha.counts, counts)
        assert_array_equal(self.pha.mask, mask)
        self.assertTrue(self.pha.grouped)
        self.assertFalse(self.pha.subtracted)

    def test_background_only_suppresses_model_and_uses_source_net_ratio(self):
        self.attach_background()
        self.session.set_source(1, CountingPowerLaw())
        net = plot_counts(1, session=self.session, res=6)
        background = plot_counts(1, session=self.session, background="only", res=6)
        raw = plot_counts(1, session=self.session, background="include", res=6)
        assert_allclose(background.curves[0].y, raw.curves[0].y-net.curves[0].y)
        assert_allclose(background.residuals[0].y, net.residuals[0].y)
        self.assertIsNone(background.model_artists[0])
        self.assertIsNone(background.model_curves[0])

    def test_background_only_without_residuals_never_looks_up_model(self):
        self.attach_background()
        error = AssertionError("background-only display must not look up source models")
        with patch.object(Session, "get_source", side_effect=error), \
                patch.object(Session, "get_model", side_effect=error):
            result = plot_counts(1, session=self.session, background="only", show_model=True)
        self.assertIsNone(result.model_artists[0])

    def test_counts_do_not_require_exposure_metadata(self):
        self.pha.exposure = None
        result = plot_counts(1, session=self.session)
        assert_allclose(result.curves[0].y, self.pha.counts)


@unittest.skipUnless(HAS_SHERPA, "requires a real CIAO/Sherpa installation")
class ModelFreePreparationTests(unittest.TestCase):
    def setUp(self):
        self.phas = [make_pha("a", 32, 8), make_pha("b", 48, 16)]

    def test_matching_and_combining_never_access_models(self):
        session = Session()
        for key, pha in enumerate(self.phas, 1):
            session.set_data(key, pha)
        error = AssertionError("preparation must not access or construct a model")
        with patch.object(Session, "get_source", side_effect=error), \
                patch.object(Session, "get_model", side_effect=error), \
                patch.object(Session, "set_source", side_effect=error), \
                patch("fancy_plotting.adapter.SharedGridSource", side_effect=error), \
                patch("fancy_plotting.adapter.Response1D", side_effect=error):
            group = match_dataset_grids([1, 2], session=session)
            combined = combine_datasets(group)
        self.assertIsNone(group.shared)
        self.assertEqual(group.models, {})
        self.assertIs(combined, group)
        self.assertGreater(group.grid.size, 48)

    def test_matching_preserves_distinct_existing_models(self):
        session = Session()
        existing = [CountingPowerLaw("first"), CountingPowerLaw("second")]
        for key, (pha, source) in enumerate(zip(self.phas, existing), 1):
            session.set_data(key, pha)
            session.set_source(key, source)
        group = match_dataset_grids([1, 2], session=session)
        for key, source in enumerate(existing, 1):
            self.assertIs(session.get_source(key), source)
            self.assertEqual(source.calls, 0)
        self.assertIsNone(group.shared)

    def test_late_model_replacement_reuses_geometry(self):
        group = match_pha_grids(self.phas)
        plan = group.plan
        first = CountingPowerLaw("first")
        group.set_source(first)
        original_shared = group.shared
        prediction = group.models[1](self.phas[0].channel)
        second = CountingPowerLaw("second")
        second.ampl = .16
        group.set_source(second)
        self.assertIs(group.plan, plan)
        self.assertIsNot(group.shared, original_shared)
        assert_allclose(group.models[1](self.phas[0].channel), 2*prediction)

    def test_register_combination_before_creating_source(self):
        combined = combine_datasets(match_pha_grids(self.phas), mode="sum")
        session = Session()
        combined.register("sum", session=session)
        self.assertIs(session.get_data("sum"), combined.data)
        self.assertIsNone(combined.model)
        self.assertEqual(session.get_stat().name, "cstat")
        source = CountingPowerLaw()
        combined.set_source(source)
        assert_allclose(session.get_model("sum")(combined.data.xlo, combined.data.xhi),
                        combined.data.eval_model(combined.model))

    def test_unbound_fits_fail_with_actionable_message(self):
        group = match_pha_grids(self.phas)
        with self.assertRaisesRegex(RuntimeError, "set_source"):
            group.make_fit()
        with self.assertRaisesRegex(RuntimeError, "set_source"):
            group.cache_info()
        combined = combine_datasets(group)
        with self.assertRaisesRegex(RuntimeError, "set_source"):
            combined.make_fit()

    def test_changed_geometry_requires_preparation_again(self):
        group = match_pha_grids(self.phas)
        arf, rmf = self.phas[0].get_response()
        for response in (arf, rmf):
            response.energ_lo = response.energ_lo + .001
            response.energ_hi = response.energ_hi + .001
        with self.assertRaisesRegex(ValueError, "match the grids again"):
            group.set_source(CountingPowerLaw())


@unittest.skipUnless(HAS_SHERPA, "requires a real CIAO/Sherpa installation")
class SherpaTests(unittest.TestCase):
    def setUp(self):
        self.source = CountingPowerLaw()
        self.phas = [make_pha("a", 32, 8), make_pha("b", 48, 16, 800)]

    def test_native_predictions_non_diagonal_responses(self):
        native = [Response1D(p)(self.source)(p.channel) for p in self.phas]
        group = match_pha_grids(self.phas).set_source(self.source)
        for pha, model, expected in zip(self.phas, group.models.values(), native):
            assert_allclose(model(pha.channel), expected, rtol=2e-12, atol=1e-10)

    def test_one_source_call_and_changed_parameters(self):
        group = match_pha_grids(self.phas).set_source(self.source)
        for pha, model in zip(self.phas, group.models.values()):
            model(pha.channel)
        self.assertEqual(self.source.calls, 1)
        self.assertEqual(group.cache_info().hits, 1)
        self.source.ampl = .12
        for pha, model in zip(self.phas, group.models.values()):
            model(pha.channel)
        self.assertEqual(self.source.calls, 2)

    def test_changed_frozen_parameter_and_clear(self):
        group = match_pha_grids(self.phas).set_source(self.source)
        model = group.models[1]
        first = model(self.phas[0].channel)
        self.source.gamma = 2.2
        changed = model(self.phas[0].channel)
        self.assertFalse(np.allclose(first, changed))
        self.assertEqual(self.source.calls, 2)
        group.clear_cache()
        model(self.phas[0].channel)
        self.assertEqual(self.source.calls, 3)

    def test_linked_parameter(self):
        external = Parameter("external", "normalization", .08)
        self.source.ampl.link = external
        group = match_pha_grids(self.phas).set_source(self.source)
        first = group.models[1](self.phas[0].channel)
        external.val = .16
        second = group.models[1](self.phas[0].channel)
        assert_allclose(second, 2*first, rtol=1e-12)

    def test_wavelength_analysis_keeps_energy_source(self):
        expected = [Response1D(p)(self.source)(p.channel) for p in self.phas]
        for pha in self.phas:
            pha.set_analysis("wavelength")
        group = match_pha_grids(self.phas).set_source(self.source)
        for pha, model, values in zip(self.phas, group.models.values(), expected):
            assert_allclose(model(pha.channel), values, rtol=2e-12, atol=1e-10)
        self.assertEqual(group.cache_info().misses, 1)

    def test_grouping_and_filters_match_native(self):
        for pha in self.phas:
            pha.grouping = np.tile([1, -1], len(pha.channel)//2)
            pha.group()
            pha.notice(1, 5)
        native = [Response1D(p)(self.source) for p in self.phas]
        group = match_pha_grids(self.phas).set_source(self.source)
        models = native + list(group.models.values())
        for model in models:
            model.startup(True)
        try:
            for pha, baseline, shared in zip(self.phas, native, group.models.values()):
                assert_allclose(pha.eval_model_to_fit(shared),
                                pha.eval_model_to_fit(baseline), rtol=2e-12, atol=1e-10)
        finally:
            for model in reversed(models):
                model.teardown()

    def test_cache_false_respected(self):
        group = match_pha_grids(self.phas).set_source(self.source)
        for model in group.models.values():
            model.startup(False)
        try:
            for pha, model in zip(self.phas, group.models.values()):
                model(pha.channel)
            self.assertEqual(self.source.calls, 2)
        finally:
            for model in group.models.values():
                model.teardown()

    def test_joint_fit_agrees_with_ordinary_sherpa(self):
        native_models = [Response1D(p)(self.source) for p in self.phas]
        for pha, model in zip(self.phas, native_models):
            pha.counts = np.rint(model(pha.channel))
        self.source.ampl = .04
        baseline = Fit(DataSimulFit("native", tuple(self.phas)),
                       SimulFitModel("native_models", tuple(native_models)),
                       stat=CStat(), method=NelderMead()).fit()
        self.source.ampl = .04
        group = match_pha_grids(self.phas).set_source(self.source)
        combined = combine_datasets(group)
        self.assertIs(combined, group)
        shared = combined.make_fit(method=NelderMead()).fit()
        assert_allclose(shared.parvals, baseline.parvals, rtol=1e-5, atol=1e-8)
        assert_allclose(shared.statval, baseline.statval, atol=1e-7)

    def test_joint_combination_rejects_sum_only_options(self):
        group = match_pha_grids(self.phas)
        with self.assertRaisesRegex(ValueError, "only supported with mode='sum'"):
            combine_datasets(group, grid=EnergyGrid.from_edges([.5, 1.1, 7]))
        with self.assertRaisesRegex(ValueError, "only supported with mode='sum'"):
            combine_datasets(group, name="custom")
        with self.assertRaisesRegex(ValueError, "mode must be"):
            combine_datasets(group, mode="invalid")

    def test_ui_attachment_and_identity(self):
        session = Session()
        for i, pha in enumerate(self.phas, 1):
            session.set_data(i, pha)
            session.set_source(i, self.source)
        group = match_dataset_grids([1, 2], session=session).set_source(self.source)
        for i, pha in enumerate(self.phas, 1):
            self.assertIs(session.get_source(i), group.sources[i])
            session.get_model(i)(pha.channel)
        self.assertEqual(group.cache_info().misses, 1)

    def test_combination_matches_sum_and_filters(self):
        for pha in self.phas:
            pha.counts = np.arange(len(pha.channel)) + 1
        group = match_pha_grids(self.phas).set_source(self.source)
        combined = combine_datasets(group, mode="sum").set_source(self.source)
        expected = self.phas[0].counts + self.phas[1].counts.reshape(8, 2).sum(axis=1)
        assert_array_equal(combined.data.y, expected[::-1])
        ordinary = [Response1D(p)(self.source)(p.channel) for p in self.phas]
        expected_model = (ordinary[0] + ordinary[1].reshape(8, 2).sum(axis=1))[::-1]
        assert_allclose(combined.data.eval_model(combined.model), expected_model, rtol=2e-12)
        combined.data.notice(1.4, 5)
        assert_allclose(combined.data.eval_model_to_fit(combined.model),
                        combined.data.apply_filter(expected_model), rtol=2e-12)

    def test_combined_fit_has_one_free_normalization(self):
        for pha in self.phas:
            pha.counts = np.rint(Response1D(pha)(self.source)(pha.channel))
        group = match_pha_grids(self.phas).set_source(self.source)
        combined = combine_datasets(group, mode="sum").set_source(self.source)
        self.source.ampl = .04
        result = combined.make_fit(method=NelderMead()).fit()
        self.assertEqual(len(result.parvals), 1)
        self.assertTrue(result.succeeded)
        self.assertAlmostEqual(result.parvals[0], .08, delta=2e-4)

    def test_combination_snapshot_and_ui_registration(self):
        for pha in self.phas:
            pha.counts = np.arange(len(pha.channel)) + 1
        combined = combine_datasets(match_pha_grids(self.phas), mode="sum").set_source(self.source)
        before = combined.data.eval_model(combined.model)
        self.phas[0].exposure = 1
        self.phas[0].counts[:] = 0
        assert_allclose(combined.data.eval_model(combined.model), before)
        session = Session()
        combined.register("sum", session=session)
        assert_allclose(session.get_model("sum")(combined.data.xlo, combined.data.xhi), before)
        with self.assertRaises(ValueError):
            combined.register("sum", session=session)

    def test_combination_projects_fractional_bins_and_validates_native_counts(self):
        group = match_pha_grids(self.phas).set_source(self.source)
        fractional = combine_datasets(group, mode="sum", grid=EnergyGrid.from_edges([.5, 1.1, 7]))
        self.assertTrue(fractional.fractional)
        self.phas[0].counts[0] = .5
        with self.assertRaises(ValueError):
            combine_datasets(group, mode="sum")
        self.phas[0].counts[0] = 0
        self.phas[0].notice(2, 4)
        selected = combine_datasets(group, mode="sum")
        self.assertGreater(len(selected.data.y), 0)

    def test_combination_rejects_background(self):
        self.phas[0].set_background(make_pha("background", 32, 8))
        group = match_pha_grids(self.phas).set_source(self.source)
        with self.assertRaises(ValueError):
            combine_datasets(group, mode="sum")

    def test_combination_does_not_inherit_ui_source(self):
        session = Session()
        for i, pha in enumerate(self.phas, 1):
            session.set_data(i, pha)
        group = match_dataset_grids([1, 2], session=session).set_source(self.source)
        session.set_source(2, 1.1 * group.sources[2])
        combined = combine_datasets(group, mode="sum")
        self.assertIsNone(combined.model)
        self.assertIsNone(combined.group.shared)

    def test_mismatched_arf_rmf_and_folded_sources_rejected(self):
        folded = Response1D(self.phas[0])(self.source)
        with self.assertRaises(ValueError):
            match_pha_grids(self.phas).set_source(folded)
        arf, rmf = self.phas[0].get_response()
        shifted = DataARF("shifted", arf.energ_lo + .001,
                          arf.energ_hi + .001, arf.specresp)
        self.phas[0].set_response(arf=shifted, rmf=rmf)
        with self.assertRaises(ValueError):
            match_pha_grids(self.phas).set_source(self.source)


@unittest.skipUnless(HAS_SHERPA, "requires a real CIAO/Sherpa installation")
class GroupedSherpaCombinationTests(unittest.TestCase):
    def setUp(self):
        self.phas = [make_pha("a", 32, 12, 500), make_pha("b", 48, 12, 800)]
        for i, pha in enumerate(self.phas, 1):
            pha.counts = i*np.arange(1, 13, dtype=float)
        self.group = match_pha_grids(self.phas)  # prepare before grouping/model creation

    def group_width(self, width):
        from sherpa.astro import data as astro_data
        if not getattr(astro_data, "groupstatus", False):
            self.skipTest("Sherpa optional dynamic grouping library is unavailable")
        for pha in self.phas:
            pha.group_width(width)

    def test_match_then_group_width_then_combine_without_model(self):
        self.group_width(3)
        combined = combine_datasets(self.group, mode="sum")
        expected = sum(pha.counts.reshape(4, 3).sum(axis=1) for pha in self.phas)
        assert_array_equal(combined.data.y, expected[::-1])
        self.assertIsNone(combined.model)
        self.assertIs(combined.group.grid, self.group.grid)
        self.assertEqual(len(combined.data.y), 4)

    def test_nonuniform_grouping(self):
        flags = np.array([1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1])
        for pha in self.phas:
            pha.grouping = flags.copy()
            pha.group()
        combined = combine_datasets(self.group, mode="sum")
        assert_allclose(combined.data.y, sum(pha.get_dep(filter=True) for pha in self.phas)[::-1])
        self.assertEqual(len(combined.data.y), 4)

    def test_grouped_predictions_and_single_source_evaluation(self):
        self.group_width(3)
        source = CountingPowerLaw()
        ordinary = sum(pha.apply_filter(Response1D(pha)(source)(pha.channel)) for pha in self.phas)
        source.calls = 0
        combined = combine_datasets(self.group, mode="sum").set_source(source)
        combined.model.startup(True)
        try:
            predicted = combined.data.eval_model_to_fit(combined.model)
            assert_allclose(predicted, ordinary[::-1], rtol=2e-12)
            self.assertEqual(source.calls, 1)
            self.assertEqual(combined.cache_info().misses, 1)
        finally:
            combined.model.teardown()

    def test_grouped_fit_with_one_free_normalization(self):
        self.group_width(3)
        source = CountingPowerLaw()
        expected_unit = 0.
        for pha in self.phas:
            full = Response1D(pha)(source)(pha.channel)
            pha.counts = np.rint(full)
            expected_unit += full.sum()/source.ampl.val
        combined = combine_datasets(self.group, mode="sum").set_source(source)
        expected = combined.data.y.sum()/expected_unit
        source.ampl = .04
        result = combined.make_fit(method=NelderMead()).fit()
        self.assertEqual(result.numpoints, 4)
        self.assertEqual(len(result.parvals), 1)
        assert_allclose(result.parvals[0], expected, rtol=2e-4)

    def test_compatible_grouping_and_selection(self):
        self.group_width(3)
        for pha in self.phas:
            pha.notice(2, 6)
        combined = combine_datasets(self.group, mode="sum")
        assert_allclose(combined.data.y, sum(pha.get_dep(filter=True) for pha in self.phas)[::-1])
        source = CountingPowerLaw()
        models = [Response1D(pha)(source) for pha in self.phas]
        for model in models:
            model.startup(True)
        try:
            expected = sum(pha.eval_model_to_fit(model) for pha, model in zip(self.phas, models))
        finally:
            for model in models:
                model.teardown()
        combined.set_source(source)
        combined.model.startup(True)
        try:
            assert_allclose(combined.data.eval_model_to_fit(combined.model), expected[::-1], rtol=2e-12)
        finally:
            combined.model.teardown()

    def test_quality_exclusion_of_incomplete_groups(self):
        self.group_width(5)
        for pha in self.phas:
            pha.ignore_bad()
        combined = combine_datasets(self.group, mode="sum")
        assert_allclose(combined.data.y, sum(pha.get_dep(filter=True) for pha in self.phas)[::-1])

    def test_reference_grouping_defines_common_matched_view(self):
        from sherpa.astro import data as astro_data
        if not getattr(astro_data, "groupstatus", False):
            self.skipTest("Sherpa optional dynamic grouping library is unavailable")
        self.phas[0].group_width(2)
        self.phas[1].group_width(3)
        reference_grouped = combine_datasets(self.group, mode="sum")
        assert_allclose(reference_grouped.data.y,
                        sum(pha.counts.reshape(6, 2).sum(axis=1) for pha in self.phas)[::-1])
        from sherpa.astro import hc
        combined = combine_datasets(self.group, mode="sum", grid=EnergyGrid.from_edges(hc/np.array([7, 3.75, .5])))
        assert_allclose(combined.data.y, sum(pha.counts.reshape(2, 6).sum(axis=1) for pha in self.phas)[::-1])

    def test_regrouping_originals_preserves_combination_snapshot(self):
        self.group_width(3)
        combined = combine_datasets(self.group, mode="sum")
        before = combined.data.y.copy()
        self.group_width(6)
        assert_array_equal(combined.data.y, before)
        rebuilt = combine_datasets(self.group, mode="sum")
        self.assertEqual(len(rebuilt.data.y), 2)
        self.assertEqual(len(combined.data.y), 4)

    def test_descending_detector_energy_order(self):
        for pha in self.phas:
            _, rmf = pha.get_response()
            rmf.e_min = rmf.e_min[::-1].copy()
            rmf.e_max = rmf.e_max[::-1].copy()
        self.group_width(3)
        combined = combine_datasets(self.group, mode="sum")
        expected = sum(pha.get_dep(filter=True) for pha in self.phas)
        assert_allclose(combined.data.y, expected)

    def test_fully_ignored_dataset_rejected(self):
        self.phas[0].ignore()
        with self.assertRaisesRegex(ValueError, "selected bin"):
            combine_datasets(self.group, mode="sum")


@unittest.skipUnless(HAS_SHERPA, "requires a real CIAO/Sherpa installation")
class GratingReferenceMatchTests(unittest.TestCase):
    def setUp(self):
        from sherpa.astro import hc
        self.heg = make_pha("heg", 32, 10, 500)
        self.meg = make_pha("meg", 48, 17, 800)
        for pha, shift in ((self.heg, 0.), (self.meg, .006)):
            wave = np.linspace(5+shift, 25+shift, len(pha.channel)+1)
            _, rmf = pha.get_response()
            rmf.e_min, rmf.e_max = hc/wave[1:], hc/wave[:-1]
            pha.header["GRATING"] = "HETG"
            pha.counts = np.arange(len(pha.channel), dtype=float)+20
        self.session = Session()
        self.session.set_data(1, self.heg)
        self.session.set_data(2, self.meg)

    def tearDown(self):
        from matplotlib import pyplot as plt
        plt.close("all")

    def match(self):
        return match_dataset_grids([1, 2], session=self.session,
                                    reference=2)

    def test_reported_heg_meg_plot_failure(self):
        self.match()
        result = plot_counts([[1, 2]], session=self.session, unit="Angstrom")
        self.assertEqual(len(result.curves[0].y), 17)
        self.assertTrue(np.isfinite(result.curves[0].y).all())
        self.assertEqual(len(self.heg.counts), 10)  # native data are preserved

    def test_analysis_coordinate_does_not_select_matching_coordinate(self):
        group = match_dataset_grids([1, 2], session=self.session, reference=2)
        baseline = combine_datasets(group, mode="sum")
        self.assertEqual(group.data_match.unit, "wavelength")
        self.meg.set_analysis("wavelength")
        group = match_dataset_grids([1, 2], session=self.session, reference=2)
        self.assertEqual(group.data_match.unit, "wavelength")
        assert_allclose(combine_datasets(group, mode="sum").data.y, baseline.data.y)

    def test_matching_api_has_no_unit_argument(self):
        import inspect
        for function in (match_dataset_grids, match_pha_grids):
            self.assertNotIn("data_unit", inspect.signature(function).parameters)
        with self.assertRaises(TypeError):
            match_dataset_grids([1, 2], session=self.session, data_unit="wavelength")

    def test_analysis_switch_changes_display_without_rematching(self):
        group = self.match()
        before = plot_counts([[1, 2]], session=self.session).curves[0]
        self.heg.set_analysis("wavelength")
        after = plot_counts([[1, 2]], session=self.session).curves[0]
        assert_allclose(after.y, before.y[::-1])
        from fancy_plotting.resample import get_data_match
        self.assertIs(group.data_match, get_data_match(self.meg))

    def test_frequency_plot_uses_calibrated_energy_bounds(self):
        from fancy_plotting.units import H_KEV_S
        self.match()
        energy = plot_data([[1, 2]], session=self.session, unit="keV").curves[0]
        freq = plot_data([[1, 2]], session=self.session, unit="Hz").curves[0]
        assert_allclose(freq.lo, energy.lo/H_KEV_S)
        assert_allclose(freq.y, energy.y*H_KEV_S)
        self.assertEqual(self.heg.units, "energy")

    def test_reference_grouping_controls_every_matched_curve(self):
        self.match()
        self.meg.grouping = np.r_[np.tile([1, -1], 8), 1]
        self.meg.group()
        result = plot_counts([1, 2, [1, 2]], session=self.session, unit="Angstrom")
        self.assertEqual([len(c.y) for c in result.curves], [9, 9, 9])
        assert_allclose(result.curves[2].y, result.curves[0].y+result.curves[1].y)

    def test_matched_background_plotting(self):
        for pha in (self.heg, self.meg):
            background = make_pha("background", len(pha.get_rmf().energ_lo), len(pha.channel), pha.exposure)
            background.counts = np.full(len(pha.channel), 2.)
            pha.set_background(background)
        self.match()
        result = plot_data([[1, 2]], session=self.session, unit="Angstrom", combine="sum")
        self.assertTrue(np.isfinite(result.curves[0].yerr).all())
        self.assertFalse(self.heg.subtracted)
        self.assertFalse(self.meg.subtracted)

    def test_combination_rebins_data_and_response_predictions_identically(self):
        group = self.match()
        source = CountingPowerLaw()
        combined = combine_datasets(group, mode="sum").set_source(source)
        from fancy_plotting.adapter import _raw_selected_snapshot, _selected_count_grid
        expected = np.zeros(17)
        for pha, projection in zip((self.heg, self.meg), combined.projections):
            snapshot = _raw_selected_snapshot(pha)
            _, order, _ = _selected_count_grid(snapshot, "wavelength")
            expected += projection(snapshot.apply_filter(Response1D(snapshot)(source)(snapshot.channel))[order])
        combined.group.clear_cache()
        source.calls = 0
        actual = combined.data.eval_model(combined.model)
        assert_allclose(actual, expected, rtol=2e-12)
        self.assertEqual(source.calls, 1)
        self.assertTrue(combined.fractional)

    def test_fractional_fit_uses_full_covariance(self):
        combined = combine_datasets(self.match(), mode="sum").set_source(CountingPowerLaw())
        fit = combined.make_fit()
        self.assertEqual(fit.stat.name, "rebin_covariance_chi2")
        residual = combined.data.y-combined.data.eval_model(combined.model)
        expected = residual @ np.linalg.solve(combined.covariance.toarray(), residual)
        assert_allclose(fit.calc_stat(), expected, rtol=1e-10)
        with self.assertRaisesRegex(ValueError, "correlated"):
            combined.make_fit(stat=CStat())

    def test_registered_fractional_fit_selects_covariance_statistic(self):
        combined = combine_datasets(self.match(), mode="sum").set_source(CountingPowerLaw())
        combined.register("sum", session=self.session)
        self.assertEqual(self.session.get_stat().name, "rebin_covariance_chi2")


@unittest.skipUnless(HAS_SHERPA, "requires a real CIAO/Sherpa installation")
class SherpaAnalysisIndependenceTests(unittest.TestCase):
    """The public workflow uses only set_analysis to choose display units."""

    def setUp(self):
        self.session = Session()
        self.phas = [make_pha("spectrum_a", 32, 7, 500), make_pha("spectrum_b", 48, 11, 800)]
        for key, pha in enumerate(self.phas, 1):
            pha.counts = key*np.arange(20, 20+len(pha.channel), dtype=float)
            self.session.set_data(key, pha)

    def tearDown(self):
        from matplotlib import pyplot as plt
        plt.close("all")

    def combination_values(self, group):
        source = CountingPowerLaw()
        combined = combine_datasets(group, mode="sum").set_source(source)
        combined.model.startup(True)
        try:
            prediction = combined.data.eval_model(combined.model)
            self.assertEqual(source.calls, 1)
        finally:
            combined.model.teardown()
        return (combined.grid.lo, combined.grid.hi, combined.data.y,
                combined.covariance.toarray(), prediction)

    def test_all_mixed_analysis_settings_produce_identical_matches(self):
        from itertools import product
        baseline = None
        baseline_source_grid = None
        for settings in product(("energy", "wavelength", "channel"), repeat=2):
            with self.subTest(settings=settings):
                for key, setting in enumerate(settings, 1):
                    self.session.set_analysis(key, setting)
                group = match_dataset_grids([1, 2], session=self.session)
                values = self.combination_values(group)
                if baseline is None:
                    baseline = values
                    baseline_source_grid = (group.grid.lo.copy(), group.grid.hi.copy())
                for actual, expected in zip(values, baseline):
                    assert_allclose(actual, expected, rtol=2e-12)
                assert_array_equal(group.grid.lo, baseline_source_grid[0])
                assert_array_equal(group.grid.hi, baseline_source_grid[1])
                self.assertEqual(tuple(pha.units for pha in self.phas), settings)

    def test_existing_match_survives_analysis_changes(self):
        group = match_dataset_grids([1, 2], session=self.session)
        baseline = self.combination_values(group)
        plan = group.plan
        for setting in ("channel", "wavelength", "energy"):
            self.session.set_analysis(setting)
            for actual, expected in zip(self.combination_values(group), baseline):
                assert_allclose(actual, expected, rtol=2e-12)
            self.assertIs(group.plan, plan)
            self.assertEqual([p.units for p in self.phas], [setting, setting])

    def test_joint_cstat_and_predictions_survive_analysis_changes(self):
        group = match_dataset_grids([1, 2], session=self.session).set_source(CountingPowerLaw())
        baseline = None
        for setting in ("energy", "wavelength", "channel"):
            self.session.set_analysis(setting)
            fit = group.make_fit(stat=CStat())
            fit.model.startup(True)
            try:
                values = (fit.calc_stat(), *(self.session.get_data(key).eval_model_to_fit(model)
                           for key, model in group.models.items()))
            finally:
                fit.model.teardown()
            if baseline is None:
                baseline = values
            for actual, expected in zip(values, baseline):
                assert_allclose(actual, expected, rtol=2e-12)

    def test_channel_display_uses_reference_labels_after_matching_and_rebin(self):
        group = match_dataset_grids([1, 2], session=self.session)
        energy = plot_counts([[1, 2]], session=self.session, rebin=3).curves[0]
        self.session.set_analysis("channel")
        result = plot_counts([[1, 2]], session=self.session, rebin=3)
        channel = result.curves[0]
        self.assertEqual(result.ax.get_xlabel(), "Channel")
        assert_allclose(channel.lo, [1, 2, 5])
        assert_allclose(channel.hi, [2, 5, 8])
        assert_allclose(channel.y, energy.y)
        assert_allclose(channel.covariance.toarray(), energy.covariance.toarray())
        self.assertEqual(group.data_match.reference_id, 1)
        self.assertEqual([p.units for p in self.phas], ["channel", "channel"])

    def test_plot_data_obeys_sherpa_counts_and_factor_defaults(self):
        match_dataset_grids([1, 2], session=self.session)
        self.session.set_analysis("energy", type="counts", factor=0)
        baseline = plot_data([[1, 2]], session=self.session).curves[0]
        self.session.set_analysis("energy", type="counts", factor=2)
        result = plot_data([[1, 2]], session=self.session)
        multiplier = ((baseline.lo+baseline.hi)/2)**2
        assert_allclose(result.curves[0].y, baseline.y*multiplier)
        assert_allclose(result.curves[0].yerr, baseline.yerr*multiplier)
        self.assertNotIn("/ s", result.ax.get_ylabel())
        self.assertEqual([p.plot_fac for p in self.phas], [2, 2])
        self.assertFalse(any(p.rate for p in self.phas))


@unittest.skipUnless(HAS_SHERPA, "requires a real CIAO/Sherpa installation")
class SherpaFitPlottingTests(unittest.TestCase):
    def setUp(self):
        self.session = Session()
        self.session.set_stat("chi2datavar")
        self.phas = [make_pha("a", 32, 8, 500), make_pha("b", 48, 11, 800)]
        self.source = CountingPowerLaw()
        for key, pha in enumerate(self.phas, 1):
            pha.counts = np.arange(20, 20+len(pha.channel), dtype=float)*key
            self.session.set_data(key, pha)
            self.session.set_source(key, self.source)

    def tearDown(self):
        from matplotlib import pyplot as plt
        plt.close("all")

    def native_prediction(self, key):
        pha = self.session.get_data(key)
        model = self.session.get_model(key)
        model.startup(True)
        try:
            return np.asarray(pha.eval_model_to_fit(model)).copy()
        finally:
            model.teardown()

    def test_auto_model_matches_native_response_folded_counts(self):
        predicted = self.native_prediction(1)
        result = plot_data(1, session=self.session)
        widths = result.curves[0].hi-result.curves[0].lo
        assert_allclose(result.model_curves[0].y*widths*500, predicted, rtol=2e-12)
        self.assertIsNone(result.residual_ax)

    def test_grouping_filtering_and_standardized_residuals(self):
        pha = self.phas[0]
        pha.grouping = np.tile([1, -1], 4)
        pha.group()
        pha.notice(1.3, 6)
        predicted = self.native_prediction(1)
        expected = (pha.get_dep(filter=True)-predicted)/np.sqrt(pha.get_dep(filter=True))
        result = plot_data(1, session=self.session, res=1)
        assert_allclose(result.residuals[0].y, expected, rtol=2e-12)

    def test_matched_models_share_one_source_evaluation(self):
        match_dataset_grids([1, 2], session=self.session).set_source(self.source)
        self.source.calls = 0
        result = plot_data([[1, 2]], session=self.session, res=4)
        self.assertEqual(self.source.calls, 1)
        self.assertEqual(len(result.residuals), 1)
        self.assertEqual(len(result.model_curves[0].y), 8)

    def test_all_modes_with_mismatched_response_and_detector_grids(self):
        match_dataset_grids([1, 2], session=self.session)
        for mode in range(1, 7):
            with self.subTest(mode=mode):
                result = plot_data([[1, 2]], session=self.session, res=mode, rebin=2)
                self.assertEqual(len(result.residuals), 2 if mode < 4 else 1)
                self.assertEqual(len(result.model_curves[0].y), 4)
                result.figure.canvas.draw()

    def test_cash_statistic_is_selected_from_session(self):
        from fancy_plotting._fit_plot import poisson_deviance
        self.session.set_stat("cstat")
        predicted = self.native_prediction(1)
        expected = np.sign(self.phas[0].counts-predicted)*np.sqrt(poisson_deviance(self.phas[0].counts, predicted))
        result = plot_data(1, session=self.session, res=1)
        self.assertEqual(result.residuals[0].statistic, "cash")
        assert_allclose(result.residuals[0].y, expected, rtol=2e-12)

    def test_background_subtraction_is_consistent_for_data_model_and_residual(self):
        pha = self.phas[0]
        background = make_pha("bg", 32, 8, 500)
        background.counts = np.full(8, 10.)
        pha.set_background(background)
        pha.backscal, background.backscal = 2., 4.
        pha.counts = np.full(8, 100.)
        source = self.native_prediction(1)
        expected = (95-source)/np.sqrt(102.5)
        for subtracted in (False, True):
            pha.subtract() if subtracted else pha.unsubtract()
            result = plot_data(1, session=self.session, res=1)
            widths = result.curves[0].hi-result.curves[0].lo
            assert_allclose(result.model_curves[0].y*widths*500, source, rtol=2e-12)
            assert_allclose(result.residuals[0].y, expected, rtol=2e-12)
            self.assertEqual(pha.subtracted, subtracted)

    def test_modeled_background_uses_raw_total_and_same_measured_subtraction(self):
        pha = self.phas[0]
        background = make_pha("bg", 32, 8, 500)
        background.counts = np.full(8, 10.)
        pha.set_background(background)
        pha.backscal, background.backscal = 2., 4.
        pha.counts = np.full(8, 100.)
        self.session.set_bkg_source(1, CountingPowerLaw("bg_source"))
        total = self.native_prediction(1)
        result = plot_data(1, session=self.session, res=1)
        widths = result.curves[0].hi-result.curves[0].lo
        assert_allclose(result.model_curves[0].y*widths*500, total-5, rtol=2e-12)
        assert_allclose(result.residuals[0].y, (100-total)/np.sqrt(102.5), rtol=2e-12)

    def test_explicit_full_model_with_background(self):
        pha = self.phas[0]
        background = make_pha("bg", 32, 8, 500)
        background.counts = np.full(8, 4.)
        pha.set_background(background)
        self.session.set_full_model(1, Response1D(pha)(self.source))
        total = self.native_prediction(1)
        result = plot_data(1, session=self.session, res=3)
        assert_allclose(result.residuals[0].y, (pha.counts-4)/(total-4), rtol=2e-12)

    def test_units_can_change_after_matching_with_models_and_lower_panel(self):
        match_dataset_grids([1, 2], session=self.session).set_source(self.source)
        baseline = plot_data([[1, 2]], session=self.session, res=4)
        for quantity in ("wavelength", "channel", "energy"):
            with self.subTest(quantity=quantity):
                self.session.set_analysis(quantity)
                result = plot_data([[1, 2]], session=self.session, res=4)
                expected = baseline.residuals[0].y[::-1] if quantity == "wavelength" else baseline.residuals[0].y
                assert_allclose(result.residuals[0].y, expected, rtol=2e-12)
                result.figure.canvas.draw()

    def test_plotting_never_fits_changes_parameters_or_changes_original_arrays(self):
        before = [(p.counts.copy(), p.units, p.grouped, p.subtracted, np.array(p.mask)) for p in self.phas]
        parameters = tuple(p.val for p in self.source.pars)
        with patch.object(Session, "fit", side_effect=AssertionError("plot must not fit")), \
                patch.object(Session, "set_source", side_effect=AssertionError("plot must not assign models")):
            plot_data([1, 2], session=self.session, res=1)
        self.assertEqual(tuple(p.val for p in self.source.pars), parameters)
        for pha, (counts, units, grouped, subtracted, mask) in zip(self.phas, before):
            assert_array_equal(pha.counts, counts)
            assert_array_equal(pha.mask, mask)
            self.assertEqual((pha.units, pha.grouped, pha.subtracted), (units, grouped, subtracted))

    def test_explicit_data_only_option_never_looks_up_models(self):
        with patch.object(Session, "get_model", side_effect=AssertionError("data only")), \
                patch.object(Session, "list_model_ids", side_effect=AssertionError("data only")):
            result = plot_data([1, 2], session=self.session, show_model=False)
        self.assertEqual(result.model_artists, (None, None))


if __name__ == "__main__":
    unittest.main()


"""Count normalization, ISIS power_scale formulas, and real Matplotlib output."""
import unittest

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from fancy_plotting import SpectrumData, plot_counts, plot_data
from fancy_plotting._counts_plot import count_weights
from fancy_plotting.plotting import _background_arrays, _curve_options


class CountsPlottingTests(unittest.TestCase):
    def setUp(self):
        self.covariance = np.array([[16, 2, 0], [2, 25, 3], [0, 3, 36.]])
        self.data = {1: SpectrumData([1, 2, 4], [2, 4, 7], [16, 25, 36], None,
                                    covariance=sparse.csr_matrix(self.covariance)),
                     2: SpectrumData([1, 2, 4], [2, 4, 7], [9, 16, 25], 200)}
        self.models = {1: [15, 30, 35], 2: [10, 15, 30]}

    def tearDown(self):
        plt.close("all")

    def plot(self, ids=1, **kwargs):
        return plot_counts(ids, datasets=self.data, **kwargs)

    def test_default_counts_do_not_require_exposure_or_divide_by_width(self):
        result = self.plot()
        assert_array_equal(result.curves[0].y, [16, 25, 36])
        assert_allclose(result.curves[0].covariance.toarray(), self.covariance)
        self.assertEqual(result.ax.get_ylabel(), r"Counts ($\mathrm{bin}^{-1}$)")
        result.figure.canvas.draw()

    def test_sherpa_rate_and_plot_factor_do_not_change_counts_default(self):
        # Input-array protocol test; real DataPHA coverage is in test_sherpa.
        object.__setattr__(self.data[1], "rate", True)
        object.__setattr__(self.data[1], "plot_fac", 3)
        result = self.plot(models=self.models)
        assert_allclose(result.curves[0].y, [16, 25, 36])
        assert_allclose(result.model_curves[0].y, self.models[1])

    def test_counts_combination_sums_counts_and_covariance(self):
        result = self.plot([[1, 2]])
        assert_array_equal(result.curves[0].y, [25, 41, 61])
        assert_allclose(result.curves[0].covariance.toarray(), self.covariance+np.diag([9, 16, 25]))
        mean = self.plot([[1, 2]], combine="mean").curves[0]
        assert_allclose(mean.y, result.curves[0].y/2)
        assert_allclose(mean.covariance.toarray(), result.curves[0].covariance.toarray()/4)

    def test_each_power_matches_arithmetic_source_formula(self):
        for power in range(4):
            with self.subTest(power=power):
                result = self.plot(power=power, models=self.models)
                weights = np.array([1.5, 3, 5.5])**(power-1)
                assert_allclose(result.curves[0].y, np.array([16, 25, 36])*weights)
                assert_allclose(result.model_curves[0].y, np.array(self.models[1])*weights)
                assert_allclose(result.curves[0].yerr, np.sqrt([16, 25, 36])*weights)

    def test_geometric_weights_match_source_for_all_powers(self):
        for power in range(4):
            weights = count_weights([1, 2, 4], [2, 4, 7], power, "geometric")
            assert_allclose(weights, np.sqrt([2, 8, 28])**(power-1))

    def test_powerlaw_weights_match_literal_isisscripts_formula(self):
        lo, hi = np.array([1, 2, 4.]), np.array([2, 4, 7.])
        expected = [(1-(lo/hi)**3)/(1-lo/hi)/lo/3, np.ones(3),
                    lo*hi/(hi-lo)*np.log(hi/lo), lo*hi]
        for power in range(4):
            assert_allclose(count_weights(lo, hi, power, "powerlaw"), expected[power])

    def test_powerlaw_narrow_bin_limits_are_stable(self):
        lo = np.array([1., 1000., .001])
        hi = np.nextafter(lo, np.inf)
        for power in range(4):
            assert_allclose(count_weights(lo, hi, power, "powerlaw"), lo**(power-1), rtol=1e-14)

    def test_isis_power_scale_numeric_modes_and_names_agree(self):
        for number, name in enumerate(("arithmetic", "geometric", "powerlaw"), 1):
            assert_allclose(self.plot(power=2, power_scale=number).curves[0].y,
                            self.plot(power=2, power_scale=name).curves[0].y)

    def test_full_covariance_and_offset_are_weighted_in_correct_order(self):
        result = self.plot(power=2, scale=3, offset=10, models=self.models)
        weights = np.array([1.5, 3, 5.5])
        assert_allclose(result.curves[0].y, 3*np.array([16, 25, 36])*weights+10)
        assert_allclose(result.model_curves[0].y, 3*np.array(self.models[1])*weights+10)
        assert_allclose(result.curves[0].covariance.toarray(),
                        9*weights[:, None]*self.covariance*weights[None, :])

    def test_rebin_precedes_power_weighting(self):
        result = self.plot(rebin=2, power=2, models=self.models)
        assert_allclose(result.curves[0].y, [(16+25)*2.5, 36*5.5])
        assert_allclose(result.model_curves[0].y, [(15+30)*2.5, 35*5.5])
        assert_allclose(result.curves[0].yerr, np.sqrt([16+25+4, 36])*[2.5, 5.5])

    def test_display_unit_conversion_precedes_power_weighting(self):
        result = self.plot(unit="Angstrom", power=2, models=self.models)
        curve = result.curves[0]
        weights = (curve.lo+curve.hi)/2
        assert_allclose(curve.y, np.array([36, 25, 16])*weights)
        assert_allclose(result.model_curves[0].y, np.array(self.models[1])[::-1]*weights)
        self.assertIn(r"\mathrm{\AA}", result.ax.get_ylabel())

    def test_counts_without_power_preserve_bins_under_unit_changes(self):
        base = self.plot(rebin=2).curves[0]
        for unit in ("Angstrom", "nm"):
            converted = self.plot(rebin=2, unit=unit).curves[0]
            assert_allclose(converted.y, base.y[::-1])
            assert_allclose(converted.covariance.toarray(), base.covariance.toarray()[::-1, ::-1])

    def test_all_residual_modes_are_independent_of_power_scale_and_offset(self):
        for mode in range(1, 7):
            with self.subTest(mode=mode):
                baseline = self.plot([[1, 2]], models=self.models, res=mode, rebin=2)
                weighted = self.plot([[1, 2]], models=self.models, res=mode, rebin=2,
                                     power=3, power_scale="powerlaw", scale=2, offset=100)
                for first, second in zip(baseline.residuals, weighted.residuals):
                    assert_allclose(first.y, second.y)
                    if first.yerr is not None:
                        assert_allclose(first.yerr, second.yerr)
                weighted.figure.canvas.draw()

    def test_cash_diagnostics_remain_unweighted(self):
        baseline = self.plot(models=self.models, res=4, residual_stat="cstat")
        weighted = self.plot(models=self.models, res=4, residual_stat="cstat", power=0)
        assert_allclose(weighted.residuals[0].y, baseline.residuals[0].y)

    def test_model_and_residual_styles_use_matplotlib_names(self):
        result = self.plot(models=self.models, res=4, color="navy", marker="o", lw=1.2,
                           model_styles={"color": "orange", "linestyle": "--"},
                           residual_styles={"color": "green", "marker": "s"})
        self.assertEqual(result.lines[0].get_color(), "navy")
        self.assertEqual(result.model_artists[0].get_color(), "orange")
        self.assertEqual(result.residual_artists[0][1].lines[0].get_marker(), "s")

    def test_invalid_options_preserve_existing_axes(self):
        fig, ax = plt.subplots()
        original, = ax.plot([1, 2], [3, 4])
        for kwargs in ({"power": -1}, {"power": 4}, {"power": True}, {"power": 1.5},
                       {"power_scale": "bogus"}, {"rate": True}, {"density": True},
                       {"background": "bogus"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.plot(ax=ax, **kwargs)
            self.assertIn(original, ax.lines)
            self.assertEqual(len(fig.axes), 1)

    def test_invalid_coordinate_weights_preserve_existing_axes(self):
        fig, ax = plt.subplots()
        original, = ax.plot([1, 2], [3, 4])
        data = {1: SpectrumData([0], [1], [10], unit="channel")}
        with self.assertRaisesRegex(ValueError, "positive"):
            plot_counts(1, datasets=data, ax=ax, power=0, power_scale="geometric")
        self.assertIn(original, ax.lines)
        assert_allclose(plot_counts(1, datasets=data).curves[0].y, [10])

    def test_overplot_rejects_incompatible_power_before_drawing(self):
        result = self.plot(power=2, res=4, models=self.models)
        old = tuple(result.ax.lines), tuple(result.residual_ax.lines)
        with self.assertRaisesRegex(ValueError, "power"):
            self.plot(ax=result.ax, overplot=True, power=3, res=4, models=self.models)
        self.assertEqual((tuple(result.ax.lines), tuple(result.residual_ax.lines)), old)
        self.plot(ax=result.ax, overplot=True, power=2, power_scale=1, res=4, models=self.models)

    def test_background_only_has_no_top_model_but_retains_source_residuals(self):
        # SpectrumData has no attached background, so its background is zero.
        result = self.plot(background="only", res=6, models=self.models)
        baseline = self.plot(res=6, models=self.models)
        assert_allclose(result.curves[0].y, 0)
        assert_allclose(result.curves[0].yerr, 0)
        self.assertIsNone(result.model_curves[0])
        self.assertIsNone(result.model_artists[0])
        assert_allclose(result.residuals[0].y, baseline.residuals[0].y)

    def test_background_only_does_not_require_a_model(self):
        result = self.plot(background="only", show_model=True)
        assert_allclose(result.curves[0].y, 0)
        self.assertIsNone(result.model_curves[0])

    def test_background_views_have_separate_cache_entries(self):
        result = self.plot([1, 1], background=["include", "only"], models=self.models)
        assert_allclose(result.curves[0].y, [16, 25, 36])
        assert_allclose(result.curves[1].y, 0)
        self.assertIsNotNone(result.model_curves[0])
        self.assertIsNone(result.model_curves[1])

    def test_background_style_overrides_global_legacy_flag_and_vice_versa(self):
        _, options = _curve_options(2, {"background": "only"},
                                    [{"subtract_background": False}, {}])
        self.assertFalse(options[0]["background_only"])
        self.assertTrue(options[1]["background_only"])
        _, options = _curve_options(1, {"subtract_background": False}, [{"background": "subtract"}])
        self.assertTrue(options[0]["subtract_background"])
        with self.assertRaisesRegex(ValueError, "choose background"):
            self.plot(background="only", subtract_background=False)

    def test_plot_data_counts_can_overplot_default_counts(self):
        result = self.plot()
        other = plot_data(2, datasets=self.data, rate=False, density=False, ax=result.ax, overplot=True)
        self.assertIn(result.lines[0], other.ax.lines)


class BackgroundArrayTests(unittest.TestCase):
    """Numerical source-aperture scaling through a minimal data-method protocol."""
    class Background:
        channel = np.array([1, 2])
        def __init__(self, counts, staterror=None, syserror=None):
            self.counts = np.asarray(counts, float)
            self.staterror = staterror
            self.syserror = syserror
        def ungroup(self):
            pass
        def unsubtract(self):
            pass
        def get_syserror(self, filter=False):
            return self.syserror

    class Source:
        channel = np.array([1, 2])
        def __init__(self, backgrounds, scales):
            self.backgrounds, self.scales = backgrounds, scales
            self.background_ids = tuple(backgrounds)
        def get_background(self, key):
            return self.backgrounds[key]
        def get_background_scale(self, key, **kwargs):
            return self.scales[key]

    def test_vector_scales_multiple_backgrounds_and_systematic_errors(self):
        a = self.Background([10, 20], [2, 3], [1, 2])
        b = self.Background([5, 8], [1, 2])
        source = self.Source({1: a, 2: b}, {1: [.5, 2], 2: .25})
        counts, variance = _background_arrays(source, {"error_method": "sqrt"})
        assert_allclose(counts, np.array([.5, 2])*[10, 20]+.25*np.array([5, 8]))
        assert_allclose(variance, np.array([.5, 2])**2*[5, 13]+.25**2*np.array([1, 4]))
        assert_array_equal(a.counts, [10, 20])

    def test_missing_stored_errors_use_requested_poisson_rule(self):
        source = self.Source({1: self.Background([0, 9])}, {1: 2})
        _, sqrt_variance = _background_arrays(source, {"error_method": "sqrt"})
        _, gehrels_variance = _background_arrays(source, {"error_method": "gehrels"})
        assert_allclose(sqrt_variance, [0, 36])
        assert_allclose(gehrels_variance, 4*(1+np.sqrt([.75, 9.75]))**2)
        with self.assertRaisesRegex(ValueError, "stored"):
            _background_arrays(source, {"error_method": "stored"})

    def test_misaligned_background_channels_are_rejected(self):
        background = self.Background([10, 20])
        background.channel = np.array([2, 3])
        with self.assertRaisesRegex(ValueError, "channels"):
            _background_arrays(self.Source({1: background}, {1: 1}), {"error_method": "sqrt"})

    def test_nonfinite_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            _background_arrays(self.Source({1: self.Background([10, 20])}, {1: np.nan}),
                               {"error_method": "sqrt"})


if __name__ == "__main__":
    unittest.main()


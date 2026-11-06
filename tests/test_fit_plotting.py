"""ISIS-style model/residual behavior using real numerical and plotting code."""
import unittest
import warnings

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from fancy_plotting import SpectrumData, plot_data, plot_counts
from fancy_plotting._fit_plot import diagnostic, poisson_deviance
from fancy_plotting.resample import register_data_match


class FitPlottingTests(unittest.TestCase):
    def setUp(self):
        self.data = {1: SpectrumData([1, 2], [2, 3], [20, 10], 10),
                     2: SpectrumData([1, 2], [2, 3], [10, 30], 20),
                     3: SpectrumData([1, 2], [2, 3], [4, 9], 30),
                     4: SpectrumData([1, 2], [2, 3], [6, 7], 40)}
        self.models = {1: [10, 20], 2: [20, 20], 3: [5, 7], 4: [4, 8]}

    def tearDown(self):
        plt.close("all")

    def plot(self, ids=1, **kwargs):
        kwargs.setdefault("models", self.models)
        return plot_data(ids, datasets=self.data, **kwargs)

    def test_automatic_model_overlay_without_automatic_residuals(self):
        result = self.plot([[1, 2]])
        self.assertEqual(len(result.figure.axes), 1)
        self.assertIsNone(result.residual_ax)
        self.assertEqual(result.residuals, ())
        assert_allclose(result.model_curves[0].y, [1, 40/30])
        self.assertIsNotNone(result.model_artists[0])

    def test_each_residual_mode_creates_a_shared_lower_panel(self):
        for mode in range(1, 7):
            with self.subTest(mode=mode):
                result = self.plot([[1, 2], [3, 4]], res=mode, ylog=True)
                self.assertEqual(len(result.figure.axes), 2)
                self.assertTrue(result.ax.get_shared_x_axes().joined(result.ax, result.residual_ax))
                self.assertEqual(result.residual_ax.get_yscale(), "linear")
                expected = [(1,), (2,), (3,), (4,)] if mode <= 3 else [(1, 2), (3, 4)]
                self.assertEqual([r.members for r in result.residuals], expected)
                assert_allclose(result.reference_line.get_ydata(), [1, 1] if mode in (3, 6) else [0, 0])
                result.figure.canvas.draw()

    def test_combined_residuals_can_hide_individual_discrepancies(self):
        combined = self.plot([[1, 2]], res=4)
        assert_allclose(combined.residuals[0].y, [0, 0])
        individual = self.plot([[1, 2]], res=1)
        assert_allclose(individual.residuals[0].y, [10/np.sqrt(20), -10/np.sqrt(10)])
        assert_allclose(individual.residuals[1].y, [-10/np.sqrt(10), 10/np.sqrt(30)])

    def test_residuals_sum_counts_even_when_upper_panel_sums_individual_rates(self):
        result = self.plot([[1, 2]], combine="sum", res=4)
        assert_allclose(result.curves[0].y, [2.5, 2.5])
        assert_allclose(result.model_curves[0].y, [2, 3])
        assert_allclose(result.residuals[0].y, [0, 0])

    def test_plot_only_rebinning_precedes_standardization(self):
        result = self.plot(1, res=4, rebin=2)
        assert_allclose(result.residuals[0].y, [0])
        assert_allclose(result.residuals[0].yerr, [[1], [1]])
        self.assertEqual(len(result.model_curves[0].y), 1)

    def test_upper_model_uses_same_normalization_grouping_scale_and_offset(self):
        result = self.plot([[1, 2]], combine="sum", rebin=2, scale=3, offset=5, res=6)
        assert_allclose(result.model_curves[0].y, [(30/10+40/20)/2*3+5])
        # Density uses the new width=2. Residual ratio uses raw summed counts.
        assert_allclose(result.residuals[0].y, [1])

    def test_residuals_do_not_depend_on_display_units_or_offset(self):
        baseline = self.plot(1, res=1)
        converted = self.plot(1, res=1, unit="Angstrom", scale=4, offset=20)
        assert_allclose(converted.residuals[0].y, baseline.residuals[0].y[::-1])
        assert_allclose(converted.residuals[0].yerr, baseline.residuals[0].yerr[:, ::-1])

    def test_signed_chi_square_is_not_plain_difference(self):
        result = self.plot(1, res=2)
        assert_allclose(result.residuals[0].y, [5, -10])
        self.assertIsNone(result.residuals[0].yerr)

    def test_ratio_data_errors(self):
        result = self.plot(1, res=3)
        assert_allclose(result.residuals[0].y, [2, .5])
        assert_allclose(result.residuals[0].yerr, np.tile(np.sqrt([20, 10])/[10, 20], (2, 1)))

    def test_model_variance_changes_sigma_without_changing_data_errors(self):
        data = self.plot(1, res=1)
        model = self.plot(1, res=1, residual_stat="chi2modvar")
        assert_allclose(model.residuals[0].y, [10/np.sqrt(10), -10/np.sqrt(20)])
        assert_allclose(model.curves[0].yerr, data.curves[0].yerr)

    def test_model_variance_ratio_has_asymmetric_errors(self):
        y, errors, valid = diagnostic([9], [16], [16], mode=3, statistic="model")
        assert_allclose(y, [9/16])
        assert_allclose(errors, [[9/16-9/20], [9/12-9/16]])
        assert_array_equal(valid, [True])

    def test_cash_and_cstat_use_signed_deviance(self):
        expected = np.sign(np.array([20, 10])-np.array([10, 20])) * np.sqrt(
            2*(np.array([10, 20])-np.array([20, 10])+np.array([20, 10])*np.log(np.array([20, 10])/[10, 20])))
        for stat in ("cash", "cstat"):
            result = self.plot(1, res=1, residual_stat=stat)
            assert_allclose(result.residuals[0].y, expected)
            self.assertIsNone(result.residuals[0].yerr)
            squared = self.plot(1, res=2, residual_stat=stat)
            assert_allclose(squared.residuals[0].y, np.sign(expected)*expected**2)

    def test_cash_zero_counts_exact_matches_and_zero_model_limits(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", RuntimeWarning)
            y, errors, valid = diagnostic([0, 0, 4, 4], [0, 4, 0, 4], [0, 0, 4, 4], mode=1, statistic="cash")
        assert_allclose(y[[0, 1, 3]], [0, -np.sqrt(8), 0])
        assert_array_equal(valid, [True, True, False, True])
        self.assertTrue(np.isnan(y[2]))
        self.assertIsNone(errors)
        self.assertFalse(captured)

    def test_cash_nearly_equal_large_counts_are_numerically_stable(self):
        result = poisson_deviance(np.array([1.e9+1, 1.e9-1]), np.array([1.e9, 1.e9]))
        assert_allclose(result, [1.e-9, 1.e-9], rtol=1.e-8)
        tiny_model = poisson_deviance(np.array([1.e9]), np.array([1.e-300]))
        assert_allclose(tiny_model, [2*(1.e-300-1.e9+1.e9*(np.log(1.e9)-np.log(1.e-300)))])

    def test_ml_branch_follows_supplied_isisscripts_formula(self):
        y, _, valid = diagnostic([4, 9], [2, 16], [4, 9], mode=2, statistic="ml")
        expected = np.sign(np.array([4, 9])-[2, 16])*2*(
            np.array([2, 16])+np.log(np.array([4, 9])+1)-np.array([4, 9])*np.log([2, 16]))
        assert_allclose(y, expected)
        self.assertTrue(valid.all())
        root, _, _ = diagnostic([4, 9], [2, 16], [4, 9], mode=1, statistic="ml")
        assert_allclose(root, np.sign(expected)*np.sqrt(np.abs(expected)))

    def test_gehrels_sigma_is_selected_by_statistic(self):
        result = self.plot(1, res=1, residual_stat="chi2gehrels")
        assert_allclose(result.residuals[0].y, np.array([10, -10])/(1+np.sqrt([20.75, 10.75])))

    def test_undefined_ratios_are_masked_without_modifying_upper_data(self):
        result = self.plot(1, res=3, models={1: [0, -1]})
        self.assertFalse(result.residuals[0].valid.any())
        self.assertTrue(np.isnan(result.residuals[0].y).all())
        assert_allclose(result.curves[0].y, [2, 1])
        result.figure.canvas.draw()

    def test_background_ratio_and_standardized_residual_use_distinct_counts(self):
        ratio, errors, _ = diagnostic([30], [25], [34], mode=3, background=[5])
        assert_allclose(ratio, [25/20])
        assert_allclose(errors, np.full((2, 1), np.sqrt(34)/20))
        chi, _, _ = diagnostic([30], [25], [34], mode=1, background=[5])
        assert_allclose(chi, [5/np.sqrt(34)])

    def test_fractional_matching_uses_same_operator_for_data_and_model(self):
        data = {1: SpectrumData([1], [3], [12], 10, unit="Angstrom"),
                2: SpectrumData([1, 2], [2, 3], [3, 5], 20, unit="Angstrom")}
        register_data_match(data, reference=2)
        result = plot_counts([[1, 2]], datasets=data, models={1: [8], 2: [4, 6]}, res=4)
        assert_allclose(result.curves[0].y, [9, 11])
        assert_allclose(result.model_curves[0].y, [8, 10])
        assert_allclose(result.residuals[0].y, [1/np.sqrt(6), 1/np.sqrt(8)])
        regrouped = plot_counts([[1, 2]], datasets=data, models={1: [8], 2: [4, 6]}, res=4, rebin=2)
        assert_allclose(regrouped.residuals[0].y, [2/np.sqrt(20)])

    def test_missing_models_raise_before_clearing_existing_axes(self):
        fig, ax = plt.subplots()
        old, = ax.plot([1, 2], [3, 4])
        with self.assertRaisesRegex(ValueError, "no model"):
            self.plot([[1, 2]], ax=ax, res=4, models={1: [10, 20]})
        self.assertIn(old, ax.lines)
        self.assertEqual(len(fig.axes), 1)

    def test_partial_auto_model_does_not_draw_a_mismatched_partial_sum(self):
        result = self.plot([[1, 2]], models={1: [10, 20]})
        self.assertIsNone(result.model_curves[0])
        self.assertIsNone(result.model_artists[0])

    def test_show_model_false_keeps_data_only_and_can_hide_model_above_residuals(self):
        only_data = self.plot(1, show_model=False)
        self.assertIsNone(only_data.model_artists[0])
        residuals = self.plot(1, show_model=False, res=1)
        self.assertIsNone(residuals.model_artists[0])
        self.assertEqual(len(residuals.residuals), 1)

    def test_component_styles_are_independent_matplotlib_styles(self):
        result = self.plot([[1, 2]], res=1, color="navy", marker="o",
                           model_styles={"color": "red", "linewidth": 2},
                           residual_styles=[{"color": "green", "marker": "s"},
                                            {"color": "purple", "marker": "^"}])
        self.assertEqual(result.lines[0].get_color(), "navy")
        self.assertEqual(result.model_artists[0].get_color(), "red")
        self.assertEqual(result.model_artists[0].get_linewidth(), 2)
        self.assertEqual(result.residual_artists[0][1].lines[0].get_marker(), "s")
        self.assertEqual(result.residual_artists[1][1].lines[0].get_color(), "purple")

    def test_model_and_residual_style_lengths_use_their_component_counts(self):
        with self.assertRaisesRegex(ValueError, "model_styles"):
            self.plot([[1, 2]], model_styles=[{}, {}])
        with self.assertRaisesRegex(ValueError, "residual_styles"):
            self.plot([[1, 2]], res=1, residual_styles=[{}])
        with self.assertRaisesRegex(TypeError, "Matplotlib"):
            self.plot(1, model_styles={"combine": "mean"})

    def test_explicit_axes_preserve_unrelated_panels(self):
        fig, axes = plt.subplots(3, 1)
        previous, = axes[0].plot([1, 2], [3, 4])
        result = self.plot(1, ax=(axes[1], axes[2]), res=1)
        self.assertEqual(fig.axes, list(axes))
        self.assertIn(previous, axes[0].lines)
        self.assertIs(result.residual_ax, axes[2])
        self.assertEqual(result.ax.get_xlabel(), "")
        self.assertEqual(result.residual_ax.get_xlabel(), r"Energy ($\mathrm{keV}$)")

    def test_lower_panel_can_be_added_inside_one_existing_subplot(self):
        fig, axes = plt.subplots(1, 2)
        previous, = axes[1].plot([1, 2], [3, 4])
        result = self.plot(1, ax=axes[0], res=1)
        self.assertEqual(len(fig.axes), 3)
        self.assertIn(previous, axes[1].lines)
        result.ax.set_xlim(1.2, 2.8)
        assert_allclose(result.residual_ax.get_xlim(), [1.2, 2.8])

    def test_overplot_reuses_lower_panel_and_reference_line(self):
        first = self.plot(1, res=1)
        second = self.plot(2, ax=first.ax, overplot=True, res=1)
        self.assertEqual(len(second.figure.axes), 2)
        self.assertIs(first.residual_ax, second.residual_ax)
        self.assertIs(first.reference_line, second.reference_line)
        self.assertIn(first.residual_artists[0][0], second.residual_ax.lines)

    def test_incompatible_residual_overplot_does_not_mutate_existing_artists(self):
        first = self.plot(1, res=1)
        upper, lower = tuple(first.ax.lines), tuple(first.residual_ax.lines)
        with self.assertRaisesRegex(ValueError, "same diagnostic"):
            self.plot(2, ax=first.ax, overplot=True, res=3)
        self.assertEqual(tuple(first.ax.lines), upper)
        self.assertEqual(tuple(first.residual_ax.lines), lower)

    def test_data_only_replacement_removes_automatically_created_lower_panel(self):
        first = self.plot(1, res=1)
        second = self.plot(1, ax=first.ax, res=0)
        self.assertEqual(len(second.figure.axes), 1)
        self.assertIsNone(second.residual_ax)
        self.assertTrue(second.ax.get_subplotspec().is_last_row())

    def test_manual_axes_layout_is_restored_after_removing_residuals(self):
        fig = plt.figure()
        ax = fig.add_axes([.15, .15, .7, .7])
        position = ax.get_position().bounds
        self.plot(1, ax=ax, res=1)
        fig.canvas.draw()
        self.plot(1, ax=ax, res=0)
        fig.canvas.draw()
        self.assertEqual(len(fig.axes), 1)
        assert_allclose(ax.get_position().bounds, position)

    def test_invalid_modes_and_model_arrays_do_not_create_figures(self):
        for value in (True, -1, 7, 1.5):
            with self.subTest(mode=value), self.assertRaises(ValueError):
                self.plot(1, res=value)
        with self.assertRaisesRegex(ValueError, "model counts"):
            self.plot(1, models={1: [3]})
        with self.assertRaisesRegex(ValueError, "residual"):
            self.plot(1, res=0, residual_styles={"color": "red"})
        self.assertFalse(plt.get_fignums())

    def test_unsupported_statistic_does_not_masquerade_as_cash(self):
        with self.assertRaisesRegex(NotImplementedError, "wstat"):
            self.plot(1, res=1, residual_stat="wstat")
        ratio = self.plot(1, res=3, residual_stat="wstat")
        assert_allclose(ratio.residuals[0].y, [2, .5])

    def test_upper_data_and_offline_inputs_are_not_modified_by_models(self):
        before = {key: value.counts.copy() for key, value in self.data.items()}
        arrays = {key: np.asarray(value).copy() for key, value in self.models.items()}
        result = self.plot([[1, 2], [3, 4]], res=6, rebin=2)
        for key, values in before.items():
            assert_array_equal(self.data[key].counts, values)
            assert_array_equal(self.models[key], arrays[key])
        self.assertEqual([r.members for r in result.residuals], [(1, 2), (3, 4)])


if __name__ == "__main__":
    unittest.main()


"""Exercise real aggregation and Matplotlib; no Sherpa substitutes."""
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from fancy_plotting import SpectrumData, plot_data, plot_counts
from fancy_plotting.plotting import HC_KEV_ANGSTROM, normalize_groups


class PlottingTests(unittest.TestCase):
    def setUp(self):
        self.data = {1: SpectrumData([1, 2], [2, 4], [10, 40], exposure=10),
                     2: SpectrumData([1, 2], [2, 4], [60, 80], exposure=20),
                     3: SpectrumData([1, 2], [2, 4], [20, 60], exposure=10),
                     4: SpectrumData([1, 2], [2, 4], [80, 120], exposure=20)}

    def tearDown(self):
        plt.close("all")

    def plot(self, groups, **kwargs):
        return plot_data(groups, datasets=self.data, **kwargs)

    def test_grouping_contract(self):
        self.assertEqual(normalize_groups([[1, 2], [3, 4]]), ((1, 2), (3, 4)))
        self.assertEqual(normalize_groups([1, 2, 3, 4]), ((1,), (2,), (3,), (4,)))
        self.assertEqual(normalize_groups([[1, 2], 3]), ((1, 2), (3,)))
        self.assertEqual(normalize_groups("rgs1"), (("rgs1",),))
        self.assertEqual(normalize_groups(np.array([1, 2])), ((1,), (2,)))

    def test_invalid_groups(self):
        for ids in ([], [[]], [[1, 1]], [[[1, 2]]], {1, 2}, [True], [1.5], [""]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                normalize_groups(ids)

    def test_sum_rates_and_error_propagation(self):
        curve = self.plot([[1, 2]], combine="sum").curves[0]
        assert_allclose(curve.y, [4, 4])
        assert_allclose(curve.yerr, np.sqrt([.25, .15]))

    def test_arithmetic_mean_of_rates(self):
        curve = self.plot([[1, 2]], combine="mean").curves[0]
        assert_allclose(curve.y, [2, 2])
        assert_allclose(curve.yerr, np.sqrt([.25, .15])/2)

    def test_default_exposure_coadd_is_not_arithmetic_mean(self):
        curve = self.plot([[1, 2]]).curves[0]
        self.assertEqual(curve.combine, "exposure")
        assert_allclose(curve.y, [70/30, 2])
        assert_allclose(curve.yerr, np.sqrt([70, 120]) / 30 / [1, 2])

    def test_sum_exp_alias_is_explicit(self):
        a = self.plot([[1, 2]], sum_exp=False).curves[0]
        b = self.plot([[1, 2]], sum_exp=True).curves[0]
        self.assertEqual(a.combine, "mean_exposure")
        assert_allclose(a.y, [70/15, 4])
        assert_allclose(a.yerr, np.sqrt([70, 120])/15/[1, 2])
        assert_allclose(b.y, [70/30, 2])
        with self.assertRaises(ValueError):
            self.plot([[1, 2]], combine="mean", sum_exp=True)

    def test_mean_exposure_differs_from_sum_of_rates_for_unequal_exposures(self):
        isis = self.plot([[1, 2]], combine="mean_exposure").curves[0]
        summed = self.plot([[1, 2]], combine="sum").curves[0]
        self.assertNotAlmostEqual(isis.y[0], summed.y[0])
        assert_allclose(isis.y, self.plot([[1, 2]], sum_exp=False).curves[0].y)

    def test_sum_exp_per_curve_and_equal_exposures(self):
        curves = self.plot([[1, 2], [1, 3]], sum_exp=[True, False]).curves
        assert_allclose(curves[0].y, [70/30, 2])
        assert_allclose(curves[1].y, [3, 5])
        assert_allclose(curves[1].y, self.plot([[1, 3]], combine="sum").curves[0].y)

    def test_sum_exp_does_not_normalize_integrated_plot_counts(self):
        for setting in (True, False):
            curve = plot_counts([[1, 2]], datasets=self.data, sum_exp=setting).curves[0]
            assert_allclose(curve.y, [70, 120])
            assert_allclose(curve.yerr, np.sqrt([70, 120]))

    def test_individual_spectra_and_common_axes(self):
        result = self.plot([1, 2, 3, 4])
        self.assertEqual(len(result.curves), 4)
        self.assertEqual(len(result.artists), 4)
        assert_allclose(result.curves[0].y, [1, 2])
        self.assertTrue(all(line.axes is result.ax for line, _ in result.artists))

    def test_per_group_normalization_and_style(self):
        result = self.plot([[1, 2], [3, 4]], combine=["sum", "mean"],
                           color=["red", "blue"], label=["sum", "mean"])
        assert_allclose(result.curves[0].y, [4, 4])
        assert_allclose(result.curves[1].y, [3, 3])
        self.assertEqual([line.get_color() for line, _ in result.artists], ["red", "blue"])
        self.assertEqual([c.label for c in result.curves], ["sum", "mean"])

    def test_scalar_options_broadcast(self):
        result = self.plot([[1, 2], [3, 4]], color="purple", linewidth=2)
        for line, _ in result.artists:
            self.assertEqual(line.get_color(), "purple")
            self.assertEqual(line.get_linewidth(), 2)

    def test_lengths_are_number_of_curves_not_input_ids(self):
        with self.assertRaisesRegex(ValueError, "plotted curve"):
            self.plot([[1, 2], [3, 4]], color=["red", "blue", "green", "black"])

    def test_styles_overrides(self):
        result = self.plot([[1, 2], [3, 4]], color="black", sum_exp=True,
                           styles=[{"dcol": "red", "combine": "sum"},
                                   {"color": "blue", "combine": "mean"}])
        self.assertEqual([line.get_color() for line, _ in result.artists], ["red", "blue"])
        self.assertEqual([c.combine for c in result.curves], ["sum", "mean"])

    def test_standard_pgplot_aliases(self):
        result = self.plot([1, 2], dcol=[2, 4], dsym=[4, 6], dstyle=[2, 4],
                           decol="gray", d_width=2)
        self.assertEqual([line.get_color() for line, _ in result.artists], ["red", "blue"])
        self.assertEqual(result.artists[0][0].get_linestyle(), "--")
        self.assertEqual(result.artists[0][1].lines[0].get_marker(), "o")
        self.assertEqual(result.artists[1][1].lines[0].get_marker(), "s")

    def test_dsym_zero_selects_histogram_without_square_markers(self):
        result = self.plot([1, 2], dsym=[0, 4])
        self.assertIsNone(result.artists[0][1].lines[0])
        self.assertEqual(result.artists[0][0].get_linestyle(), "-")
        self.assertEqual(result.artists[1][1].lines[0].get_marker(), "o")

    def test_alias_conflicts_and_unknown_options(self):
        with self.assertRaises(ValueError):
            self.plot(1, color="red", dcol="blue")
        with self.assertRaisesRegex(TypeError, "unsupported"):
            self.plot(1, colour="red")
        with self.assertRaisesRegex(ValueError, "no model"):
            self.plot(1, res=1)

    def test_counts_sum_and_mean_without_exposure(self):
        data = {key: SpectrumData(value.lo, value.hi, value.counts) for key, value in self.data.items()}
        summed = plot_counts([[1, 2]], datasets=data).curves[0]
        averaged = plot_counts([[1, 2]], datasets=data, combine="mean").curves[0]
        assert_allclose(summed.y, [70, 120])
        assert_allclose(averaged.y, [35, 60])
        assert_allclose(summed.yerr, np.sqrt([70, 120]))

    def test_missing_exposure_and_exposure_mode_counts_rejected(self):
        with self.assertRaisesRegex(ValueError, "exposure"):
            plot_data(1, datasets={1: SpectrumData([1], [2], [1])})
        with self.assertRaises(ValueError):
            self.plot(1, rate=False, combine="exposure")

    def test_scale_and_offset_propagate_errors(self):
        baseline = self.plot(1).curves[0]
        curve = self.plot(1, scale=3, offset=-2).curves[0]
        assert_allclose(curve.y, 3*baseline.y-2)
        assert_allclose(curve.yerr, 3*baseline.yerr)

    def test_rebin_before_binwidth_normalization(self):
        curve = self.plot(1, rebin=2).curves[0]
        assert_allclose(curve.lo, [1])
        assert_allclose(curve.hi, [4])
        assert_allclose(curve.y, [50/10/3])
        assert_allclose(curve.yerr, [np.sqrt(50)/10/3])

    def test_rebin_never_bridges_a_gap(self):
        data = SpectrumData([1, 2, 5, 6], [2, 3, 6, 7], [1, 2, 3, 4], exposure=1)
        result = plot_data(1, datasets={1: data}, rebin=20)
        assert_allclose(result.curves[0].lo, [1, 5])
        assert_allclose(result.curves[0].hi, [3, 7])
        self.assertTrue(np.isnan(result.artists[0][0].get_ydata()).any())

    def test_mismatched_group_bins_raise_before_clearing_axes(self):
        ax = plt.subplots()[1]
        original, = ax.plot([1, 2], [3, 4])
        other = SpectrumData([1, 2], [2, 5], [1, 1], exposure=1)
        with self.assertRaisesRegex(ValueError, "detector bins"):
            plot_data([[1, 2]], datasets={1: self.data[1], 2: other}, ax=ax)
        self.assertIn(original, ax.lines)

    def test_wavelength_conversion_and_density(self):
        curve = self.plot(1, unit="Angstrom").curves[0]
        assert_allclose(curve.lo, HC_KEV_ANGSTROM/np.array([4, 2]))
        assert_allclose(curve.hi, HC_KEV_ANGSTROM/np.array([2, 1]))
        assert_allclose(curve.y*(curve.hi-curve.lo), [4, 1])

    def test_channels_auto_detected_but_require_response_for_physical_conversion(self):
        data = SpectrumData([0, 1], [1, 2], [2, 3], exposure=1, unit="channel")
        with self.assertRaises(ValueError):
            plot_data(1, datasets={1: data}, unit="keV")
        self.assertEqual(plot_data(1, datasets={1: data}).ax.get_xlabel(), "Channel")
        result = plot_data(1, datasets={1: data}, unit="channel")
        self.assertEqual(result.ax.get_xlabel(), "Channel")

    def test_stored_errors_and_gehrels(self):
        supplied = SpectrumData([1, 2], [2, 4], [10, 40], exposure=10, staterror=[2, 3])
        curve = plot_data(1, datasets={1: supplied}, error_method="stored").curves[0]
        assert_allclose(curve.yerr, [.2, .15])
        gehrels = self.plot(1, error_method="gehrels").curves[0]
        assert_allclose(gehrels.yerr, (1+np.sqrt([10.75, 40.75]))/10/[1, 2])
        with self.assertRaises(ValueError):
            self.plot(1, error_method="stored")

    def test_log_masks_negative_values_only_for_rendering(self):
        data = SpectrumData([1, 2], [2, 4], [-2, 4], exposure=1, staterror=[2, 2])
        result = plot_data(1, datasets={1: data}, ylog=True)
        self.assertEqual(result.curves[0].y[0], -2)
        self.assertTrue(np.isnan(result.artists[0][0].get_ydata()[0]))
        self.assertEqual(result.ax.get_yscale(), "log")

    def test_panel_controls(self):
        result = self.plot(1, xlog=True, ylog=True, xrange=[1, 4], yrange=[.1, 10],
                           xlabel="E", ylabel="R", title="Spectrum", legend=False)
        self.assertEqual(result.ax.get_xlabel(), "E")
        self.assertEqual(result.ax.get_ylabel(), "R")
        self.assertEqual(result.ax.get_title(), "Spectrum")
        self.assertIsNone(result.ax.get_legend())
        assert_allclose(result.ax.get_xlim(), [1, 4])

    def test_overplot_and_clear(self):
        a = self.plot(1)
        first_line = a.artists[0][0]
        b = self.plot(2, ax=a.ax, oplt=True)
        self.assertIs(a.ax, b.ax)
        self.assertIn(first_line, b.ax.lines)
        self.plot(3, ax=a.ax)
        self.assertNotIn(first_line, a.ax.lines)

    def test_overplot_rejects_unit_mismatch(self):
        a = self.plot(1)
        with self.assertRaises(ValueError):
            self.plot(2, ax=a.ax, oplt=True, unit="wavelength")

    def test_invalid_curve_options(self):
        for option in ({"rebin": 0}, {"rebin": 1.5}, {"alpha": 2}, {"scale": 0},
                       {"linewidth": -1}, {"dcol": 99}, {"dsym": 99},
                       {"combine": "median"}, {"xlim": [np.nan, 1]}):
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.plot(1, **option)

    def test_no_sherpa_import_or_model_needed_for_arrays(self):
        with patch.dict("sys.modules", {"sherpa": None}):
            result = self.plot([[1, 2], 3])
        self.assertEqual(len(result.curves), 2)

    def test_inputs_not_modified(self):
        before = {key: (value.lo.copy(), value.hi.copy(), value.counts.copy())
                  for key, value in self.data.items()}
        self.plot([[1, 2], [3, 4]], rebin=[1, 2], unit="wavelength")
        for key, values in before.items():
            for a, b in zip(values, (self.data[key].lo, self.data[key].hi, self.data[key].counts)):
                assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()

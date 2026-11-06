import unittest
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from fancy_plotting import SpectrumData, plot_data, plot_counts
from fancy_plotting.resample import register_data_match


class MatchedPlottingTests(unittest.TestCase):
    def setUp(self):
        self.heg = SpectrumData([1, 2], [2, 3], [10, 20], 10, unit="Angstrom")
        self.meg = SpectrumData([1, 1.5, 2.5], [1.5, 2.5, 3], [2, 3, 4], 20, unit="Angstrom")
        self.data = {1: self.heg, 2: self.meg}

    def tearDown(self):
        plt.close("all")

    def test_matched_heg_meg_sum_no_longer_raises(self):
        register_data_match(self.data, reference=2)
        result = plot_counts([[1, 2]], datasets=self.data, unit="Angstrom")
        curve = result.curves[0]
        assert_allclose(curve.lo, self.meg.lo)
        assert_allclose(curve.y, [7, 18, 14])
        assert_allclose(curve.yerr**2, [4.5, 10.5, 9])
        self.assertGreater(curve.covariance[0, 1], 0)

    def test_matching_is_independent_of_requested_plot_unit(self):
        register_data_match(self.data, reference=2)
        wave = plot_counts([[1, 2]], datasets=self.data, unit="Angstrom").curves[0]
        energy = plot_counts([[1, 2]], datasets=self.data, unit="keV").curves[0]
        assert_allclose(energy.y, wave.y[::-1])
        assert_allclose(energy.covariance.toarray(), wave.covariance.toarray()[::-1, ::-1])

    def test_on_the_fly_reference_option_without_registration(self):
        result = plot_counts([[1, 2]], datasets=self.data, align=True, reference=2, unit="Angstrom")
        assert_allclose(result.curves[0].y, [7, 18, 14])

    def test_numpy_boolean_alignment_flag(self):
        result = plot_counts([[1, 2]], datasets=self.data, align=np.bool_(True),
                             reference=2, unit="Angstrom")
        assert_allclose(result.curves[0].y, [7, 18, 14])

    def test_average_and_exposure_normalization_after_regrid(self):
        register_data_match(self.data, reference=2)
        summed = plot_data([[1, 2]], datasets=self.data, unit="Angstrom", combine="sum", density=False).curves[0]
        mean = plot_data([[1, 2]], datasets=self.data, unit="Angstrom", combine="mean", density=False).curves[0]
        pooled = plot_data([[1, 2]], datasets=self.data, unit="Angstrom", density=False).curves[0]
        assert_allclose(summed.y, np.array([5, 15, 10])/10+self.meg.counts/20)
        assert_allclose(mean.y, summed.y/2)
        assert_allclose(pooled.y, [7/30, 18/30, 14/30])

    def test_grouped_display_propagates_cross_bin_covariance(self):
        register_data_match(self.data, reference=2)
        result = plot_counts([[1, 2]], datasets=self.data, unit="Angstrom", rebin=3)
        assert_allclose(result.curves[0].y, [39])
        assert_allclose(result.curves[0].yerr**2, [39])

    def test_isis_sum_exp_false_uses_mean_exposure_on_matched_bins(self):
        register_data_match(self.data, reference=2)
        result = plot_data([[1, 2]], datasets=self.data, unit="Angstrom",
                           sum_exp=False, density=False).curves[0]
        assert_allclose(result.y, np.array([7, 18, 14])/15)
        assert_allclose(result.yerr**2, np.array([4.5, 10.5, 9])/15**2)

    def test_single_matched_spectrum_uses_reference_bins(self):
        register_data_match(self.data, reference=2)
        result = plot_counts(1, datasets=self.data, unit="Angstrom")
        assert_allclose(result.curves[0].y, [5, 15, 10])

    def test_inputs_remain_native(self):
        register_data_match(self.data, reference=2)
        plot_counts([[1, 2]], datasets=self.data, unit="Angstrom")
        assert_array_equal(self.heg.counts, [10, 20])
        assert_array_equal(self.heg.lo, [1, 2])

    def test_alignment_can_be_explicitly_disabled(self):
        register_data_match(self.data, reference=2)
        with self.assertRaises(ValueError):
            plot_counts([[1, 2]], datasets=self.data, unit="Angstrom", align=False)

    def test_multiple_displayed_groups_are_rebinned_independently(self):
        register_data_match(self.data, reference=2)
        result = plot_counts([[1, 2], 1], datasets=self.data, unit="Angstrom", color=["blue", "red"])
        self.assertEqual(len(result.curves), 2)
        assert_allclose(result.curves[0].y, [7, 18, 14])
        assert_allclose(result.curves[1].y, [5, 15, 10])


if __name__ == "__main__":
    unittest.main()

"""General response integration, flux normalization, covariance, and rendering."""
from copy import deepcopy
import unittest

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from fancy_plotting import SpectrumData, SourceSpectrum, UnfoldedCurve, plot_unfold, plot_counts
from fancy_plotting._unfold_plot import (response_weights, KEV_TO_ERG, unfold_options,
                                      unfold_curve, source_curve, incident_grid)
from fancy_plotting.plotting import _convert, _curve_options
from fancy_plotting.resample import register_data_match
from fancy_plotting.units import HC_KEV_ANGSTROM as HC, convert_bins


class ArrayARF:
    """Response-method protocol for numerical tests; not a Sherpa replacement."""
    def __init__(self, lo, hi, area, exposure=None):
        self.energ_lo, self.energ_hi = np.array(lo), np.array(hi)
        self.specresp, self.exposure = np.array(area), exposure


class ArrayRMF:
    def __init__(self, lo, hi, matrix):
        self.energ_lo, self.energ_hi = np.array(lo), np.array(hi)
        self.matrix = np.array(matrix)
        self.filtered = True
    def notice(self):
        self.filtered = False
    def apply_rmf(self, flux):
        if self.filtered:
            raise AssertionError("calibration must use the full response")
        return self.matrix.T @ flux


class ArrayPHA:
    def __init__(self, responses, channels, exposure=10, areascal=None):
        self.responses = responses
        self.response_ids = tuple(responses)
        self.channel = np.arange(1, channels+1)
        self.exposure, self.areascal = exposure, areascal
    def get_response(self, key):
        return self.responses[key]


class ResponseWeightTests(unittest.TestCase):
    def test_diagonal_response_weights_integrate_wavelength(self):
        arf = ArrayARF([1, 2], [2, 4], [10, 20])
        rmf = ArrayRMF([1, 2], [2, 4], np.eye(2))
        pha = ArrayPHA({1: (arf, rmf)}, 2, exposure=100, areascal=[.5, 2])
        assert_allclose(response_weights(pha), 100*np.array([.5, 2])*[10, 20]*HC*np.array([.5, .25]))
        self.assertTrue(rmf.filtered)

    def test_broad_response_includes_incident_photons_from_all_bins(self):
        matrix = np.array([[.8, .2], [.3, .7], [.1, .9]])
        arf = ArrayARF([1, 2, 4], [2, 4, 8], [10, 20, 30])
        rmf = ArrayRMF([1, 2, 4], [2, 4, 8], matrix)
        result = response_weights(ArrayPHA({1: (arf, rmf)}, 2))
        assert_allclose(result, 10*matrix.T @ (np.array([10, 20, 30])*HC*np.array([.5, .25, .125])))

    def test_mismatched_arf_rmf_bins_are_integrated_before_redistribution(self):
        # RMF bin [1,4] straddles all of ARF [1,2], plus part of [2,8].
        arf = ArrayARF([1, 2], [2, 8], [10, 20])
        rmf = ArrayRMF([1, 4], [4, 8], np.eye(2))
        result = response_weights(ArrayPHA({1: (arf, rmf)}, 2))
        expected = 10*HC*np.array([10*(1-1/2)+20*(1/2-1/4), 20*(1/4-1/8)])
        assert_allclose(result, expected)

    def test_same_length_different_arf_rmf_edges_are_not_assumed_identical(self):
        arf = ArrayARF([1, 3], [3, 8], [2, 10])
        rmf = ArrayRMF([1, 4], [4, 8], np.eye(2))
        weights = response_weights(ArrayPHA({1: (arf, rmf)}, 2, exposure=1))
        assert_allclose(weights, HC*np.array([2*(1-1/3)+10*(1/3-1/4), 10*(1/4-1/8)]))

    def test_gapped_arf_and_rmf_do_not_interpolate_through_missing_incident_bins(self):
        arf = ArrayARF([1, 4], [2, 8], [10, 20])
        rmf = ArrayRMF([1, 2, 4], [2, 4, 8], np.eye(3))
        pha = ArrayPHA({1: (arf, rmf)}, 3, exposure=1)
        assert_allclose(response_weights(pha), HC*np.array([5, 0, 2.5]))
        # A full RSP has no separate ARF; its missing detector column stays zero.
        full_rsp = ArrayRMF([1, 4], [2, 8], [[4, 0, 0], [0, 0, 9]])
        assert_allclose(response_weights(ArrayPHA({1: (None, full_rsp)}, 3, exposure=2)),
                        2*HC*np.array([2, 0, 9/8]))

    def test_zero_areascal_does_not_disable_adjacent_response_channels(self):
        arf = ArrayARF([1, 2, 4], [2, 4, 8], [10, 20, 30])
        rmf = ArrayRMF([1, 2, 4], [2, 4, 8], np.eye(3))
        full = response_weights(ArrayPHA({1: (arf, rmf)}, 3, exposure=3))
        assert_allclose(response_weights(ArrayPHA({1: (arf, rmf)}, 3, exposure=3,
                                                   areascal=[1, 0, 1])), full*[1, 0, 1])

    def test_multiple_response_components_sum_their_own_exposures(self):
        a = ArrayARF([1, 2], [2, 4], [10, 20], exposure=100)
        b = ArrayARF([1, 2], [2, 4], [30, 40], exposure=200)
        rmf = ArrayRMF([1, 2], [2, 4], np.eye(2))
        pha = ArrayPHA({1: (a, rmf), 5: (b, rmf)}, 2, exposure=50, areascal=.5)
        widths = HC*np.array([.5, .25])
        assert_allclose(response_weights(pha), .5*(100*np.array([10, 20])+200*np.array([30, 40]))*widths)
        assert_allclose(response_weights(pha, "pha"), .5*50*np.array([40, 60])*widths)

    def test_arf_exposure_fallback_and_pha_override(self):
        for arf_exp in (None, 0, -2):
            arf = ArrayARF([1], [2], [10], exposure=arf_exp)
            pha = ArrayPHA({1: (arf, None)}, 1, exposure=50)
            assert_allclose(response_weights(pha), [50*10*HC/2])
        arf.exposure = 100
        assert_allclose(response_weights(pha), [100*10*HC/2])
        assert_allclose(response_weights(pha, "pha"), [50*10*HC/2])

    def test_arf_only_and_full_rsp_only_are_supported(self):
        arf = ArrayARF([1, 2], [2, 4], [10, 20])
        rsp = ArrayRMF([1, 2], [2, 4], [[10, 0], [0, 20]])
        first = response_weights(ArrayPHA({1: (arf, None)}, 2))
        second = response_weights(ArrayPHA({1: (None, rsp)}, 2))
        assert_allclose(first, second)

    def test_response_integral_handles_gaps_and_crops_without_extrapolation(self):
        arf = ArrayARF([2, 4], [3, 8], [10, 20])
        rmf = ArrayRMF([1, 2, 3, 4], [2, 3, 4, 8], np.eye(4))
        weights = response_weights(ArrayPHA({1: (arf, rmf)}, 4, exposure=1))
        assert_allclose(weights, [0, 10*HC*(1/2-1/3), 0, 20*HC*(1/4-1/8)])

    def test_response_native_row_order_is_respected(self):
        arf = ArrayARF([2, 1], [4, 2], [20, 10])
        rmf = ArrayRMF([2, 1], [4, 2], [[0, 1], [1, 0]])
        result = response_weights(ArrayPHA({1: (arf, rmf)}, 2))
        assert_allclose(result, 10*HC*np.array([10*.5, 20*.25]))

    def test_rmf_channel_offset_maps_weights_to_pha_channel_labels(self):
        rsp = ArrayRMF([1], [2], [[2, 4, 6, 8]])
        rsp.offset = 0
        pha = ArrayPHA({1: (None, rsp)}, 2)
        pha.channel = np.array([0, 3])
        assert_allclose(response_weights(pha), 10*HC/2*np.array([2, 8]))

    def test_source_grid_preserves_both_arf_and_rmf_boundaries(self):
        arf = ArrayARF([1, 2], [2, 4], [10, 20])
        rmf = ArrayRMF([1, 3], [3, 4], np.eye(2))
        grid = incident_grid(ArrayPHA({1: (arf, rmf)}, 2))
        assert_allclose(grid.lo, [1, 2, 3])
        assert_allclose(grid.hi, [2, 3, 4])

    def test_invalid_calibration_is_rejected(self):
        arf = ArrayARF([1], [2], [10])
        pha = ArrayPHA({1: (arf, None)}, 1)
        for mutation in (lambda p: setattr(p, "exposure", None),
                         lambda p: setattr(p, "areascal", -1),
                         lambda p: setattr(p.responses[1][0], "specresp", [-2]),
                         lambda p: setattr(p.responses[1][0], "energ_lo", [0])):
            value = deepcopy(pha)
            mutation(value)
            with self.assertRaises(ValueError):
                response_weights(value)
        with self.assertRaisesRegex(ValueError, "requires an ARF"):
            response_weights(ArrayPHA({}, 1))


class UnfoldPlottingTests(unittest.TestCase):
    def setUp(self):
        # Arrays start in energy order; correction weights include all
        # response/exposure/area terms, for unit density per Angstrom.
        self.data = {1: SpectrumData([1, 2, 4], [2, 4, 8], [10, 20, 40], 100),
                     2: SpectrumData([1, 2, 4], [2, 4, 8], [20, 30, 50], 1000)}
        self.weights = {1: np.array([2, 3, 4.]), 2: np.array([10, 20, 30.])}
        self.models = {1: [12, 18, 44], 2: [18, 32, 45]}

    def tearDown(self):
        plt.close("all")

    def plot(self, ids=1, **kwargs):
        kwargs.setdefault("response_weights", self.weights)
        return plot_unfold(ids, datasets=self.data, **kwargs)

    def test_photon_density_matches_counts_over_weights_and_bin_jacobian(self):
        result = self.plot()
        expected = np.array([10, 20, 40])/self.weights[1]*HC*np.array([.5, .25, .125])/[1, 2, 4]
        assert_allclose(result.curves[0].y, expected)
        self.assertIsInstance(result.curves[0], UnfoldedCurve)
        assert_allclose(result.curves[0].weights, self.weights[1])
        self.assertIn("keV", result.ax.get_ylabel())
        result.figure.canvas.draw()

    def test_combination_pools_counts_and_weights_before_division(self):
        combined = self.plot([[1, 2]]).curves[0]
        widths = HC*np.array([.5, .25, .125])/[1, 2, 4]
        expected = np.array([30, 50, 90])/(self.weights[1]+self.weights[2])*widths
        assert_allclose(combined.y, expected)
        assert_allclose(combined.weights, self.weights[1]+self.weights[2])
        mean = (self.plot(1).curves[0].y+self.plot(2).curves[0].y)/2
        self.assertFalse(np.allclose(combined.y, mean))

    def test_plot_rebinning_sums_counts_and_weights_before_division(self):
        # Canonical wavelength grouping first combines energy bins [2,4],[4,8].
        result = self.plot(rebin=2).curves[0]
        assert_allclose(result.lo, [1, 2])
        assert_allclose(result.hi, [2, 8])
        assert_allclose(result.weights, [2, 7])
        assert_allclose(result.y, np.array([10/2, 60/7])*HC*np.array([.5, .375])/[1, 6])

    def test_full_covariance_is_transformed_with_the_flux_factors(self):
        covariance = sparse.csr_matrix([[10, 2, 1], [2, 20, 3], [1, 3, 40]])
        self.data[1] = SpectrumData([1, 2, 4], [2, 4, 8], [10, 20, 40], covariance=covariance)
        curve = self.plot(scale=3, offset=10).curves[0]
        expected = 9*curve.factor[:, None]*covariance.toarray()*curve.factor[None, :]
        assert_allclose(curve.covariance.toarray(), expected)
        assert_allclose(curve.yerr, np.sqrt(expected.diagonal()))
        assert_allclose(curve.y, np.array([10, 20, 40])*curve.factor*3+10)

    def test_source_model_parameters_do_not_enter_data_correction(self):
        first = self.plot(models=self.models)
        second = self.plot(models={1: np.array(self.models[1])*10})
        assert_allclose(first.curves[0].y, second.curves[0].y)
        assert_allclose(second.model_curves[0].y, first.model_curves[0].y*10)

    def test_unit_changes_preserve_bin_integrated_photon_flux(self):
        base = self.plot().curves[0]
        integral = base.y*(base.hi-base.lo)
        for unit in ("Angstrom", "nm", "eV", "Hz", "GHz"):
            with self.subTest(unit=unit):
                curve = self.plot(unit=unit).curves[0]
                _, _, order = convert_bins(base.lo, base.hi, "keV", unit)
                assert_allclose(curve.y*(curve.hi-curve.lo), integral[order])
                expected = base.covariance.toarray()*np.diff([1, 2, 4, 8])[:, None]*np.diff([1, 2, 4, 8])[None, :]
                actual = curve.covariance.toarray()*(curve.hi-curve.lo)[:, None]*(curve.hi-curve.lo)[None, :]
                assert_allclose(actual, expected[order][:, order])

    def test_energy_erg_default_is_isis_power_three(self):
        photon = self.plot().curves[0]
        result = self.plot(flux_unit="ergs")
        assert_allclose(result.curves[0].y, photon.y*np.array([1.5, 3, 6])**2*KEV_TO_ERG)
        self.assertEqual(result.ax.get_ylabel(), r"Energy flux ($\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}$)")

    def test_wavelength_erg_default_is_isis_power_one(self):
        photon = self.plot(unit="Angstrom").curves[0]
        result = self.plot(unit="Angstrom", flux_unit="ergs")
        assert_allclose(result.curves[0].y, photon.y*HC*KEV_TO_ERG)
        self.assertEqual(result.ax.get_ylabel(), r"Energy flux ($\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}$)")

    def test_scaled_energy_units_preserve_erg_sed(self):
        base = self.plot(flux_unit="ergs").curves[0]
        for unit in ("eV", "MeV", "erg", "Hz", "GHz"):
            curve = self.plot(flux_unit="ergs", unit=unit).curves[0]
            assert_allclose(curve.y, base.y)

    def test_scaled_wavelength_units_preserve_erg_sed(self):
        base = self.plot(flux_unit="ergs", unit="Angstrom").curves[0]
        curve = self.plot(flux_unit="ergs", unit="nm").curves[0]
        assert_allclose(curve.y, base.y)

    def test_geometric_erg_sed_agrees_between_energy_and_wavelength(self):
        energy = self.plot(flux_unit="ergs", power_scale="geometric").curves[0]
        wave = self.plot(flux_unit="ergs", power_scale="geometric", unit="Angstrom").curves[0]
        assert_allclose(energy.y, wave.y[::-1])
        assert_allclose(energy.yerr, wave.yerr[::-1])

    def test_arithmetic_erg_sed_keeps_literal_isis_finite_bin_weighting(self):
        energy = self.plot(flux_unit="ergs").curves[0]
        wave = self.plot(flux_unit="ergs", unit="Angstrom").curves[0]
        assert_allclose(energy.y/wave.y[::-1], np.array([1.5, 3, 6])**2/np.array([2, 8, 32]))

    def test_power_and_power_scale_apply_equally_to_data_and_model(self):
        from fancy_plotting._counts_plot import count_weights
        base = self.plot(models=self.models)
        for power in range(4):
            for mode in ("arithmetic", "geometric", "powerlaw"):
                result = self.plot(models=self.models, power=power, power_scale=mode)
                weights = count_weights([1, 2, 4], [2, 4, 8], power, mode)
                assert_allclose(result.curves[0].y, base.curves[0].y*weights)
                assert_allclose(result.model_curves[0].y, base.model_curves[0].y*weights)
                plt.close(result.figure)

    def test_all_residual_modes_retain_count_space_diagnostics(self):
        for mode in range(1, 7):
            baseline = plot_counts([[1, 2]], datasets=self.data, models=self.models, res=mode)
            result = self.plot([[1, 2]], models=self.models, res=mode, flux_unit="ergs", scale=5, offset=10)
            self.assertEqual(len(result.figure.axes), 2)
            for a, b in zip(baseline.residuals, result.residuals):
                assert_allclose(a.y, b.y)
                if a.yerr is not None:
                    assert_allclose(a.yerr, b.yerr)
            result.figure.canvas.draw()
            plt.close(result.figure)
            plt.close(baseline.figure)

    def test_cash_residuals_are_independent_of_flux_display(self):
        baseline = self.plot(models=self.models, res=4, residual_stat="cstat")
        result = self.plot(models=self.models, res=4, residual_stat="cstat", flux_unit="ergs", power=0)
        assert_allclose(result.residuals[0].y, baseline.residuals[0].y)

    def test_zero_sensitivity_bins_are_masked_with_covariance_metadata(self):
        result = self.plot(response_weights={1: [2, 0, 4]}, models=self.models)
        curve = result.curves[0]
        assert_array_equal(curve.valid, [True, False, True])
        self.assertTrue(np.isnan(curve.y[1]))
        self.assertTrue(np.isnan(curve.yerr[1]))
        assert_allclose(curve.covariance.toarray()[1], 0)
        self.assertTrue(np.isnan(result.model_curves[0].y[1]))
        result.figure.canvas.draw()

    def test_group_sensitivity_can_be_supplied_by_only_one_member(self):
        result = self.plot([[1, 2]], response_weights={1: [0, 3, 4], 2: [2, 0, 0]})
        assert_array_equal(result.curves[0].valid, True)
        assert_allclose(result.curves[0].weights, [2, 3, 4])

    def test_missing_exposure_is_allowed_for_explicit_complete_weights(self):
        self.data[1] = SpectrumData([1, 2, 4], [2, 4, 8], [10, 20, 40])
        assert_allclose(self.plot().curves[0].weights, self.weights[1])

    def test_matching_applies_the_same_fractional_operator_to_counts_and_weights(self):
        data = {1: SpectrumData([1], [3], [12], unit="Angstrom"),
                2: SpectrumData([1, 2], [2, 3], [3, 5], unit="Angstrom")}
        register_data_match(data, reference=2)
        result = plot_unfold([[1, 2]], datasets=data, response_weights={1: [8], 2: [2, 4]},
                             models={1: [10], 2: [3, 4]}, res=4)
        assert_allclose(result.curves[0].weights, [6, 8])
        assert_allclose(result.curves[0].y, [9/6, 11/8])
        assert_allclose(result.model_curves[0].y, [8/6, 9/8])
        assert_allclose(result.curves[0].covariance.toarray(), [[6/36, 3/48], [3/48, 8/64]])

    def test_odd_rebin_groups_are_identical_across_display_coordinates(self):
        energy = self.plot(rebin=2).curves[0]
        wave = self.plot(rebin=2, unit="Angstrom").curves[0]
        assert_allclose(wave.weights, energy.weights[::-1])
        assert_allclose(wave.y*(wave.hi-wave.lo), (energy.y*(energy.hi-energy.lo))[::-1])

    def test_rebinning_never_bridges_an_ignored_gap(self):
        self.data[1] = SpectrumData([1, 2, 5], [2, 3, 6], [10, 20, 30], unit="Angstrom")
        result = self.plot(rebin=10).curves[0]
        assert_allclose(result.lo, [1, 5])
        assert_allclose(result.hi, [3, 6])
        assert_allclose(result.weights, [5, 4])
        assert_allclose(result.y, [30/5, 30/4])

    def test_rebinning_splits_zero_sensitivity_runs_in_data_model_and_residuals(self):
        data = {1: SpectrumData([10, 11, 12, 13, 14], [11, 12, 13, 14, 15],
                                [20, 20, 99, 20, 20], unit="Angstrom")}
        result = plot_unfold(1, datasets=data, response_weights={1: [2, 2, 0, 2, 2]},
                             models={1: [20, 20, 90, 20, 20]}, unit="Angstrom", rebin=5, res=4)
        curve, model, residual = result.curves[0], result.model_curves[0], result.residuals[0]
        assert_allclose(curve.lo, [10, 12, 13])
        assert_allclose(curve.hi, [12, 13, 15])
        assert_allclose(curve.weights, [4, 0, 4])
        assert_array_equal(curve.valid, [True, False, True])
        assert_allclose(curve.y[[0, 2]], [10, 10])
        self.assertTrue(np.isnan(curve.y[1]))
        self.assertTrue(np.isnan(model.y[1]))
        assert_allclose(residual.lo, curve.lo)
        assert_allclose(residual.hi, curve.hi)
        assert_allclose(residual.y[1], 9/np.sqrt(99))
        result.figure.canvas.draw()

    def test_other_dataset_can_supply_sensitivity_in_a_missing_channel(self):
        data = {1: SpectrumData([1, 2, 3], [2, 3, 4], [2, 0, 2], unit="Angstrom"),
                2: SpectrumData([1, 2, 3], [2, 3, 4], [0, 3, 0], unit="Angstrom")}
        result = plot_unfold([[1, 2]], datasets=data,
                             response_weights={1: [2, 0, 2], 2: [0, 3, 0]}, rebin=3)
        curve = result.curves[0]
        assert_allclose(curve.lo, [1])
        assert_allclose(curve.hi, [4])
        assert_allclose(curve.weights, [7])
        assert_allclose(curve.y, [1])

    def test_positive_response_with_zero_counts_is_transparently_zero(self):
        data = {1: SpectrumData([1, 2], [2, 3], [0, 0], unit="Angstrom")}
        curve = plot_unfold(1, datasets=data, response_weights={1: [3, 4]}).curves[0]
        assert_array_equal(curve.valid, [True, True])
        assert_allclose(curve.y, [0, 0])
        self.assertTrue((curve.factor > 0).all())

    def test_sherpa_rate_and_plot_factor_do_not_affect_unfolding(self):
        baseline = self.plot().curves[0]
        object.__setattr__(self.data[1], "plot_fac", 3)
        object.__setattr__(self.data[1], "rate", False)
        assert_allclose(self.plot().curves[0].y, baseline.y)

    def test_background_only_hides_model_but_preserves_source_residuals(self):
        baseline = self.plot(models=self.models, res=6)
        result = self.plot(background="only", models=self.models, res=6)
        assert_allclose(result.curves[0].y, 0)
        self.assertIsNone(result.model_curves[0])
        assert_allclose(result.residuals[0].y, baseline.residuals[0].y)

    def test_matplotlib_styles_and_explicit_axes_are_shared_with_counts(self):
        fig, axes = plt.subplots(2, 1, sharex=True)
        result = self.plot(models=self.models, ax=axes, res=4, color="navy", marker="o", ls="none",
                           model_styles={"color": "orange", "lw": 2},
                           residual_styles={"color": "green", "marker": "s"})
        self.assertIs(result.ax, axes[0])
        self.assertEqual(result.lines[0].get_color(), "navy")
        self.assertEqual(result.model_artists[0].get_color(), "orange")
        self.assertEqual(result.residual_artists[0][1].lines[0].get_marker(), "s")

    def test_invalid_options_and_missing_weights_preserve_existing_axes(self):
        fig, ax = plt.subplots()
        original, = ax.plot([1, 2], [3, 4])
        for kwargs in ({"power": 4}, {"flux_unit": "mJy"}, {"model_display": "wrong"},
                       {"response_weights": {}}, {"response_weights": {1: [1, -1, 2]}},
                       {"response_weights": {1: [0, 0, 0]}}, {"response_weights": None},
                       {"combine": "mean"}, {"rate": True}, {"density": False}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.plot(ax=ax, **kwargs)
            self.assertIn(original, ax.lines)
        self.assertEqual(len(fig.axes), 1)

    def test_overplot_validates_flux_units_and_power_before_mutation(self):
        result = self.plot(res=4, models=self.models)
        upper, lower = tuple(result.ax.lines), tuple(result.residual_ax.lines)
        with self.assertRaisesRegex(ValueError, "same unit"):
            self.plot(ax=result.ax, overplot=True, res=4, models=self.models, flux_unit="ergs")
        self.assertEqual(tuple(result.ax.lines), upper)
        self.assertEqual(tuple(result.residual_ax.lines), lower)
        self.plot(2, ax=result.ax, overplot=True, res=4, models=self.models)
        self.assertEqual(len(result.figure.axes), 2)


class SourceDisplayTests(unittest.TestCase):
    def setUp(self):
        self.data = {1: SpectrumData([1, 2], [2, 3], [10, 20], unit="Angstrom"),
                     2: SpectrumData([1, 2], [2, 3], [30, 40], unit="Angstrom")}
        self.sources = {1: SourceSpectrum([1, 1.5, 2, 2.5], [1.5, 2, 2.5, 3], [.5, 1, 1.5, 2], "Angstrom"),
                        2: SourceSpectrum([1, 2], [2, 3], [3, 7], "Angstrom")}
        self.weights = {1: [2, 4], 2: [10, 20]}

    def tearDown(self):
        plt.close("all")

    def plot(self, ids=1, **kwargs):
        return plot_unfold(ids, datasets=self.data, response_weights=self.weights,
                           source_spectra=self.sources, model_display="source", **kwargs)

    def test_source_can_use_finer_bins_than_data(self):
        result = self.plot()
        self.assertEqual(len(result.curves[0].y), 2)
        self.assertEqual(len(result.model_curves[0].y), 4)
        assert_allclose(result.model_curves[0].y, [1, 2, 3, 4])

    def test_combined_source_follows_isis_arithmetic_mean_on_first_source_grid(self):
        result = self.plot([[1, 2]])
        assert_allclose(result.model_curves[0].y, ([1, 2, 3, 4]+np.array([3, 3, 7, 7]))/2)
        assert_allclose(result.curves[0].y, [40/12, 60/24])

    def test_erg_source_display_uses_the_same_power_definition(self):
        photon = self.plot().model_curves[0]
        result = self.plot(flux_unit="ergs").model_curves[0]
        assert_allclose(result.y, photon.y*HC*KEV_TO_ERG)

    def test_source_display_removes_models_in_ignored_gaps(self):
        self.data[1] = SpectrumData([1, 2.5], [1.5, 3], [10, 20], unit="Angstrom")
        result = self.plot()
        assert_array_equal(np.isfinite(result.model_curves[0].y), [True, False, False, True])
        result.figure.canvas.draw()

    def test_source_model_does_not_bridge_zero_sensitivity_gaps(self):
        self.weights[1] = [2, 0]
        result = self.plot()
        assert_array_equal(np.isfinite(result.model_curves[0].y), [True, True, False, False])

    def test_missing_source_for_a_folded_model_is_explicit(self):
        self.sources.pop(1)
        with self.assertRaisesRegex(ValueError, "unsmeared source"):
            self.plot(models={1: [10, 20]})

    def test_source_display_retains_count_residuals_when_requested(self):
        result = self.plot(models={1: [10, 25]}, res=4)
        assert_allclose(result.residuals[0].y, [0, -5/np.sqrt(20)])

    def test_source_spectrum_requires_physical_integrated_flux(self):
        with self.assertRaisesRegex(ValueError, "physical"):
            SourceSpectrum([1], [2], [1], "channel")
        with self.assertRaises(ValueError):
            SourceSpectrum([1], [2], [np.nan])


class ChannelUnfoldingTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_calibrated_channels_preserve_integrated_flux(self):
        # Exercise the real numerical path with physical bins and explicit
        # reference channel calibration. Real PHA coverage is separate.
        data = SpectrumData([1, 2, 4], [2, 4, 8], [10, 20, 30], unit="Angstrom")
        weights = SpectrumData(data.lo, data.hi, [2, 4, 5], staterror=[0, 0, 0], unit="Angstrom")
        _, opts = _curve_options(1, {"rate": False, "density": False}, None)
        channels = (np.array([8., 4., 0.]), np.array([10., 8., 4.]))
        config = unfold_options(None, "arithmetic", "photons", "convolved").resolved("channel")
        curve = unfold_curve((1,), [data], [weights], opts[0], config, "channel", channels)
        assert_allclose(curve.y*(curve.hi-curve.lo), (np.array([10, 20, 30])/[2, 4, 5]*[1, 2, 4])[::-1])
        assert_allclose(curve.weights, [5, 4, 2])

    def test_channel_erg_density_uses_physical_energy_not_channel_number(self):
        data = SpectrumData([1, 2], [2, 4], [10, 20], unit="Angstrom")
        weights = SpectrumData(data.lo, data.hi, [2, 4], staterror=[0, 0], unit="Angstrom")
        _, opts = _curve_options(1, {"rate": False, "density": False}, None)
        channels = ([100, 0], [200, 100])
        photons = unfold_options(None, "arithmetic", "photons", "convolved").resolved("channel")
        ergs = unfold_options(None, "arithmetic", "ergs", "convolved").resolved("channel")
        p = unfold_curve((1,), [data], [weights], opts[0], photons, "channel", channels)
        e = unfold_curve((1,), [data], [weights], opts[0], ergs, "channel", channels)
        assert_allclose(e.y/p.y, KEV_TO_ERG*HC*np.array([(.25+.5)/2, (.5+1)/2]))

    def test_channel_source_is_averaged_onto_calibrated_display_bins(self):
        data = SpectrumData([1, 2], [2, 4], [10, 20], unit="Angstrom")
        weights = SpectrumData(data.lo, data.hi, [2, 4], staterror=[0, 0], unit="Angstrom")
        source = SourceSpectrum([1, 1.5, 2, 3], [1.5, 2, 3, 4], [1, 2, 3, 4], "Angstrom")
        _, opts = _curve_options(1, {"rate": False, "density": False}, None)
        config = unfold_options(None, "arithmetic", "photons", "source").resolved("channel")
        channels = ([5, 0], [10, 5])
        curve = unfold_curve((1,), [data], [weights], opts[0], config, "channel", channels)
        model = source_curve((1,), [source], [data], opts[0], config, "channel", channels, curve)
        assert_allclose(model.y, [7/5, 3/5])

    def test_channel_rebinning_preserves_the_same_physical_groups(self):
        data = SpectrumData([1, 2, 4], [2, 4, 8], [10, 20, 30], unit="Angstrom")
        weights = SpectrumData(data.lo, data.hi, [2, 4, 5], staterror=[0, 0, 0], unit="Angstrom")
        _, opts = _curve_options(1, {"rate": False, "density": False, "rebin": 2}, None)
        config = unfold_options(None, "arithmetic", "photons", "convolved").resolved("channel")
        channels = ([8, 4, 0], [10, 8, 4])
        curve = unfold_curve((1,), [data], [weights], opts[0], config, "channel", channels)
        assert_allclose(curve.lo, [0, 4])
        assert_allclose(curve.hi, [4, 10])
        assert_allclose(curve.weights, [5, 6])
        assert_allclose(curve.y, [30/5, 30/6*3/6])


if __name__ == "__main__":
    unittest.main()


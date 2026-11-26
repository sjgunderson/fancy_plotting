"""Matched channel membership and sum-before-statistic regressions."""
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.optimize import minimize_scalar

from fancy_plotting import EnergyGrid, SpectrumData
from fancy_plotting._fit_plot import poisson_deviance
from fancy_plotting.plotting import (_MatchedTarget, _align_spectrum,
                                     _apply_isis_areascal, _areascal_divisor,
                                     _matched_target, _convert)
from fancy_plotting.units import HC_KEV_ANGSTROM as HC
from fancy_plotting.resample import GroupedOverlapRebin, OverlapRebin
from test_sherpa import HAS_SHERPA


class MatchedMembershipTests(unittest.TestCase):
    def test_compiled_isis_area_rebin_and_group_reference(self):
        fixtures = json.loads(Path(__file__).with_name("isis_grouped_reference.json").read_text())
        for case in fixtures["cases"]:
            with self.subTest(name=case["name"]):
                source = EnergyGrid(case["source_lo"], case["source_hi"])
                reference = EnergyGrid(case["reference_lo"], case["reference_hi"])
                target = EnergyGrid(case["target_lo"], case["target_hi"])
                pha = SimpleNamespace(channel=np.arange(source.size), counts=np.array(case["counts"]),
                                       staterror=None, areascal=np.array(case["areascal"]))
                _apply_isis_areascal(pha)
                groups = case["groups"]
                matrix = sparse.csr_matrix((np.ones(sum(map(len, groups))),
                                            (np.repeat(np.arange(len(groups)), list(map(len, groups))),
                                             np.concatenate(groups))),
                                           shape=(target.size, reference.size))
                actual = GroupedOverlapRebin(source, reference, target, matrix)(pha.counts)
                assert_allclose(actual, case["isis_counts"], rtol=3e-13, atol=1e-10)

    def test_rgs_area_variations_do_not_become_narrow_count_peaks(self):
        area = np.full(120, .1)
        area[[15, 47, 82]] = 1.
        pha = SimpleNamespace(channel=np.arange(120), counts=100*area,
                               staterror=np.sqrt(100*area), areascal=area)
        original_errors = pha.staterror.copy()
        divisor = _apply_isis_areascal(pha)
        assert_allclose(pha.counts, 100)
        assert_allclose(pha.staterror, original_errors/area)
        assert_allclose(divisor, area)
        self.assertEqual(pha.areascal, 1.)

    def test_areascal_is_applied_before_matching_and_grouping(self):
        grid = EnergyGrid.from_edges([1, 2, 3, 4, 5])
        target = EnergyGrid.from_edges([1, 3, 5])
        grouping = sparse.csr_matrix([[1, 1, 0, 0], [0, 0, 1, 1]])
        mapping = GroupedOverlapRebin(grid, grid, target, grouping)
        area = np.array([.1, 1., .5, .25])
        flux = np.array([10., 50., 20., 40.])
        pha = SimpleNamespace(channel=np.arange(4), counts=flux*area,
                               staterror=None, areascal=area)
        _apply_isis_areascal(pha)
        assert_allclose(mapping(pha.counts), [60, 60])
        self.assertFalse(np.allclose(mapping(flux*area)/(grouping@area/2), [60, 60]))

    def test_zero_or_nonfinite_area_uses_isis_fallback(self):
        pha = SimpleNamespace(channel=np.arange(4), areascal=[0., np.nan, np.inf, .5])
        assert_allclose(_areascal_divisor(pha), [1, 1, 1, .5])

    def test_group_bounds_do_not_reintroduce_excluded_channel_spike(self):
        native = EnergyGrid.from_edges([1, 2, 3, 4, 5, 6, 7])
        output = EnergyGrid([1, 4], [4, 7])
        # Channel 1 is bad, but channels 0 and 2 remain in the same group.
        grouping = sparse.csr_matrix([[1, 0, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
        counts = np.array([10, 10000, 10, 10, 10, 10])
        mapping = GroupedOverlapRebin(native, native, output, grouping)
        assert_array_equal(mapping(counts), [20, 30])
        # This was the previous bounding-interval shortcut.
        assert_array_equal(OverlapRebin(native, output)(counts), [10020, 30])

    def test_reference_membership_builder_handles_reversed_energy_bounds(self):
        # Exercise the membership/permutation builder independently of Sherpa's
        # compiled PHA reader. Real DataPHA integration is tested below.
        wave = np.arange(1., 8.)
        groups = (np.array([0, 2]), np.array([3, 4, 5]))
        reference = SimpleNamespace(
            channel=np.arange(1, 7),
            _get_ebins=lambda group=False: (HC/wave[1:], HC/wave[:-1]),
            apply_filter=lambda values, function: np.array([function(np.asarray(values)[g]) for g in groups]),
            get_noticed_channels=lambda: np.array([1, 3, 4, 5, 6]),
            set_analysis=lambda value: None, _min=np.min, _max=np.max)
        full = SpectrumData(wave[:-1], wave[1:], [10, 10000, 10, 10, 10, 10], unit="Angstrom")
        grouped = SpectrumData([1, 4], [4, 7], [20, 30], unit="Angstrom")
        for unit, expected in (("Angstrom", [20, 30]), ("energy", [30, 20])):
            with self.subTest(unit=unit):
                def read(pha, unit, options, raw=False, **kwargs):
                    return _convert(full if raw else grouped, unit)
                with patch("fancy_plotting.plotting._read_pha", side_effect=read):
                    target = _matched_target(reference, unit, {})
                result = _align_spectrum(full, target, unit, {"error_method": "sqrt"})
                assert_allclose(result.counts, expected)

    def test_reference_calibration_gap_is_not_filled_by_other_member(self):
        source = EnergyGrid.from_edges([1, 2, 3, 4])
        reference = EnergyGrid([1, 3], [2, 4])
        target = EnergyGrid([1], [4])
        mapping = GroupedOverlapRebin(source, reference, target, [[1, 1]])
        assert_array_equal(mapping([10, 10000, 10]), [20])
        assert_array_equal(mapping.source_coverage, [1, 0, 1])

    def test_plot_alignment_and_covariance_use_exact_membership(self):
        native = EnergyGrid.from_edges([1, 2, 3, 4])
        target = _MatchedTarget(SpectrumData([1], [4], [0]), native,
                                sparse.csr_matrix([[1, 0, 1]]))
        source = SpectrumData(native.lo, native.hi, [10, 10000, 20], 100)
        result = _align_spectrum(source, target, "energy", {"error_method": "sqrt"})
        assert_allclose(result.counts, [30])
        assert_allclose(result.covariance.toarray(), [[30]])

    def test_shifted_rgs_grids_match_then_group_not_group_then_match(self):
        reference = EnergyGrid.from_edges(6 + .01*np.arange(41))
        source = EnergyGrid.from_edges(5.995 + .01*np.arange(42))
        grouping = sparse.kron(sparse.eye(4), np.ones((1, 10)), format="csr")
        output = EnergyGrid(reference.lo[::10], reference.hi[9::10])
        counts = np.ones(source.size)*10
        counts[[9, 10]] = [30, 70]
        mapping = GroupedOverlapRebin(source, reference, output, grouping)
        expected = grouping @ OverlapRebin(source, reference)(counts)
        assert_allclose(mapping(counts), expected, rtol=1e-13)
        assert_allclose(GroupedOverlapRebin(source, reference, output, grouping)(
            12*(source.hi-source.lo)), 12*(output.hi-output.lo), rtol=1e-13)

    def test_model_folding_then_matching_commutes_with_output_rmf_rebin(self):
        rng = np.random.default_rng(26)
        source = EnergyGrid.from_edges([1, 2.2, 3.1, 4])
        reference = EnergyGrid.from_edges([1, 2, 3, 4])
        target = EnergyGrid.from_edges([1, 3, 4])
        mapping = GroupedOverlapRebin(source, reference, target, [[1, 1, 0], [0, 0, 1]])
        rmf = rng.uniform(0, 1, (source.size, 17))
        flux = rng.uniform(0, 1, 17)
        assert_allclose(mapping(rmf @ flux), (mapping.matrix @ rmf) @ flux, rtol=1e-14)

    def test_summed_cstat_is_distinct_from_joint_cstat(self):
        data = np.array([[10., 30.], [30., 10.]])
        models = np.full((2, 2), 20.)
        self.assertGreater(poisson_deviance(data.ravel(), models.ravel()).sum(), 0)
        assert_allclose(poisson_deviance(data.sum(axis=0), models.sum(axis=0)), 0)

    def test_summed_fit_recovers_analytic_normalization(self):
        counts = [np.array([9, 18, 31.]), np.array([7, 12, 29.])]
        predictions = [np.array([1, 2, 3.]), np.array([2, 3, 4.])]
        reference = EnergyGrid.from_edges([1, 2, 3, 4])
        target = EnergyGrid.from_edges([1, 3, 4])
        mapping = GroupedOverlapRebin(reference, reference, target, [[1, 1, 0], [0, 0, 1]])
        data = sum(mapping(c) for c in counts)
        unit_model = sum(mapping(m) for m in predictions)
        result = minimize_scalar(lambda a: poisson_deviance(data, a*unit_model).sum(),
                                 bounds=(.1, 20), method="bounded",
                                 options={"xatol": 1e-11})
        assert_allclose(result.x, data.sum()/unit_model.sum(), rtol=1e-7)

    def test_membership_validation_rejects_double_counting(self):
        grid = EnergyGrid.from_edges([1, 2, 3])
        for matrix in ([[1, 1], [0, 1]], [[.5, 0], [0, 1]], [[1, 1]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                GroupedOverlapRebin(grid, grid, grid, matrix)


if HAS_SHERPA:
    from test_sherpa import make_pha, CountingPowerLaw
    from sherpa.astro.instrument import Response1D
    from sherpa.astro.ui.utils import Session
    from sherpa.stats import CStat, WStat
    from sherpa.optmethods import NelderMead
    from fancy_plotting import match_dataset_grids, combine_datasets, plot_counts


@unittest.skipUnless(HAS_SHERPA, "requires a real CIAO/Sherpa installation")
class ISISSherpaCombinationTests(unittest.TestCase):
    def setUp(self):
        self.session = Session()
        self.session.set_stat("cstat")
        self.source = CountingPowerLaw()
        self.phas = [make_pha("rgs_a", 32, 12), make_pha("rgs_b", 48, 12, 800)]
        for key, pha in enumerate(self.phas, 1):
            pha.counts = np.arange(1, 13, dtype=float)*key
            self.session.set_data(key, pha)
            self.session.set_source(key, self.source)
        self.group = combine_datasets(match_dataset_grids([1, 2], session=self.session))

    def tearDown(self):
        from matplotlib import pyplot as plt
        plt.close("all")

    def test_default_combination_pools_before_cstat(self):
        fit = self.group.make_fit()
        expected = sum(Response1D(p)(self.source)(p.channel)/p.areascal for p in self.phas)
        expected_data = sum(p.counts/p.areascal for p in self.phas)
        assert_allclose(fit.data.y, expected_data[::-1])
        assert_allclose(fit.data.eval_model_to_fit(fit.model), expected[::-1])
        assert_allclose(fit.calc_stat(), poisson_deviance(expected_data, expected).sum(), rtol=1e-12)
        self.assertEqual(len(fit.data.y), 12)
        self.assertNotAlmostEqual(fit.calc_stat(), self.session.calc_stat(1, 2))

    def test_sherpa_fit_recovers_summed_cstat_normalization(self):
        self.group.set_source(self.source)
        fit = self.group.make_fit(stat=CStat(), method=NelderMead())
        unit_prediction = fit.data.eval_model_to_fit(fit.model)/self.source.ampl.val
        expected = fit.data.y.sum()/unit_prediction.sum()
        result = fit.fit()
        self.assertTrue(result.succeeded)
        assert_allclose(result.parvals, [expected], rtol=2e-4)

    def test_grouping_and_selection_after_combination_are_read_at_make_fit(self):
        self.session.group_width(1, 3)
        self.session.group_width(2, 3)
        self.session.notice(2.2, 5.2)
        fit = self.group.make_fit()
        displayed = plot_counts([[1, 2]], session=self.session,
                                 unit="Angstrom", subtract_background=False).curves[0]
        assert_allclose(fit.data.y, displayed.y)
        assert_allclose(fit.data.eval_model_to_fit(fit.model),
                        plot_counts([[1, 2]], session=self.session, unit="Angstrom",
                                    subtract_background=False).model_curves[0].y)
        previous = fit.data.y.copy()
        self.session.notice()
        self.assertGreater(self.group.make_fit().data.y.size, previous.size)

    def test_explicit_joint_mode_preserves_native_likelihood(self):
        joint = combine_datasets(self.group, mode="joint")
        joint.set_source(self.source)
        assert_allclose(joint.make_fit().calc_stat(), self.session.calc_stat(1, 2), rtol=1e-12)

    def test_wstat_requires_explicit_joint_mode(self):
        with self.assertRaisesRegex(ValueError, "mode='joint'"):
            self.group.make_fit(stat=WStat())

    def test_fit_does_not_change_native_data_or_filters(self):
        before = [(p.counts.copy(), p.grouped, p.mask) for p in self.phas]
        self.group.make_fit(stat=CStat()).calc_stat()
        for pha, (counts, grouped, mask) in zip(self.phas, before):
            assert_array_equal(pha.counts, counts)
            self.assertEqual(pha.grouped, grouped)
            assert_array_equal(pha.mask, mask)

    def test_rgs_area_correction_agrees_for_plot_data_and_combined_model(self):
        for pha in self.phas:
            pha.areascal = np.array([.1, .1, 1., .1, .2, .2, 1., .2, .1, .1, 1., .1])
            pha.counts = 100*pha.areascal
        fit = self.group.make_fit()
        curve = plot_counts([[1, 2]], session=self.session, unit="Angstrom",
                             subtract_background=False).curves[0]
        assert_allclose(curve.y, 200)
        assert_allclose(fit.data.y, curve.y)
        self.assertEqual(self.phas[0].counts[0], 10)

    def test_bad_channel_inside_reference_group_is_excluded_from_every_member(self):
        for pha in self.phas:
            pha.areascal = 1.
            pha.counts = np.full(12, 10.)
            pha.counts[1] = 10000.
            pha.grouping = np.tile([1, -1, -1], 4)
            pha.group()
        self.phas[0].quality = np.zeros(12, dtype=int)
        self.phas[0].quality[1] = 5
        self.phas[0].ignore_bad()
        displayed = plot_counts([[1, 2]], session=self.session,
                                 unit="Angstrom", subtract_background=False).curves[0]
        assert_allclose(displayed.y, [60, 60, 60, 40])
        assert_allclose(self.group.make_fit().data.y, displayed.y)


if __name__ == "__main__":
    unittest.main()

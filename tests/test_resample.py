"""HEG/MEG-like overlap matching, covariance, and response consistency."""
import unittest
import json
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from fancy_plotting import EnergyGrid, SharedEvaluation, prepare_grids
from fancy_plotting.resample import (OverlapRebin, diagonal_covariance,
                                  register_data_match, resample_combination)
from fancy_plotting.covariance import CovarianceWhitening


class RegridTests(unittest.TestCase):
    def test_upstream_isis_c_rebin_reference_vectors(self):
        fixtures = json.loads(Path(__file__).with_name("isis_rebin_reference.json").read_text())
        for case in fixtures["cases"]:
            with self.subTest(name=case["name"]):
                source = EnergyGrid(case["source_lo"], case["source_hi"])
                target = EnergyGrid(case["target_lo"], case["target_hi"])
                actual = OverlapRebin(source, target)(case["counts"])
                assert_allclose(actual, case["isis_counts"], rtol=2e-13, atol=1e-12)

    def test_shifted_grid_fractional_counts_and_covariance(self):
        source = EnergyGrid.from_edges([1, 2, 3])
        target = EnergyGrid.from_edges([1, 1.5, 2.5, 3])
        mapping = OverlapRebin(source, target)
        assert_allclose(mapping([10, 20]), [5, 15, 10])
        covariance = mapping.covariance(diagonal_covariance([10, 20]))
        assert_allclose(covariance.toarray(), [[2.5, 2.5, 0], [2.5, 7.5, 5], [0, 5, 5]])
        self.assertTrue(mapping.fractional)

    def test_round_trip_grouping_restores_native_variance(self):
        original = EnergyGrid.from_edges([1, 2, 3])
        split = EnergyGrid.from_edges([1, 1.5, 2, 2.5, 3])
        first = OverlapRebin(original, split)
        second = OverlapRebin(split, original)
        covariance = second.covariance(first.covariance(diagonal_covariance([10, 20])))
        assert_allclose(covariance.toarray(), np.diag([10, 20]))
        assert_allclose(second(first([10, 20])), [10, 20])

    def test_coarsening_keeps_independent_poisson_counts(self):
        mapping = OverlapRebin(EnergyGrid.from_edges([1, 2, 3, 4, 5]),
                               EnergyGrid.from_edges([1, 3, 5]))
        self.assertFalse(mapping.fractional)
        assert_allclose(mapping([1, 2, 3, 4]), [3, 7])
        assert_allclose(mapping.covariance(diagonal_covariance([1, 2, 3, 4])).toarray(), np.diag([3, 7]))

    def test_realistic_shifted_grating_bins_conserve_density(self):
        heg_edges = 5 + .012*np.arange(1671)
        meg_edges = 5.006 + .024*np.arange(834)
        heg = EnergyGrid.from_edges(heg_edges)
        meg = EnergyGrid.from_edges(meg_edges)
        mapping = OverlapRebin(heg, meg)
        density = 9.3
        result = mapping(density*(heg.hi-heg.lo))
        assert_allclose(result/(meg.hi-meg.lo), density, rtol=3e-12)
        self.assertLess(mapping.matrix.nnz, 4*meg.size)

    def test_no_extrapolation_and_explicit_cropping(self):
        source = EnergyGrid.from_edges([1, 2, 3])
        wider = OverlapRebin(source, EnergyGrid.from_edges([0, 1, 2, 3, 4]))
        assert_array_equal(wider([10, 20]), [0, 10, 20, 0])
        crop = OverlapRebin(source, EnergyGrid.from_edges([1.5, 2.5]))
        assert_allclose(crop([10, 20]), [15])
        assert_allclose(crop.source_coverage, [.5, .5])

    def test_gaps_do_not_receive_extrapolated_counts(self):
        mapping = OverlapRebin(EnergyGrid([1, 3], [2, 4]), EnergyGrid.from_edges([1, 2, 3, 4]))
        assert_array_equal(mapping([10, 20]), [10, 0, 20])

    def test_combined_fractional_counts_have_full_rank_covariance(self):
        reference = EnergyGrid.from_edges([1, 1.5, 2.5, 3])
        other = EnergyGrid.from_edges([1, 2, 3])
        result = resample_combination([reference, other], [[2, 3, 4], [10, 20]])
        assert_allclose(result.counts, [7, 18, 14])
        self.assertTrue(result.fractional)
        residual = np.array([1, 2, -1.])
        whitened = CovarianceWhitening(result.covariance)(residual)
        expected = residual @ np.linalg.solve(result.covariance.toarray(), residual)
        assert_allclose(whitened @ whitened, expected)

    def test_banded_whitening_scales_to_grating_channel_counts(self):
        n = 8192
        covariance = sparse.diags([np.ones(n-1)*.2, np.ones(n)*2, np.ones(n-1)*.2],
                                   [-1, 0, 1], format="csr")
        whitener = CovarianceWhitening(covariance)
        self.assertEqual(whitener.factor.shape, (2, n))
        result = whitener(np.ones(n))
        self.assertTrue(np.isfinite(result).all())

    def test_singular_covariance_fit_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "coarser grid"):
            CovarianceWhitening([[1, 1], [1, 1]])

    def test_rebinned_model_uses_same_operator_as_data(self):
        source_grids = [EnergyGrid.from_edges(np.linspace(.5, 7, n)) for n in (17, 25)]
        plan = prepare_grids(source_grids)
        calls = []
        def source(p, lo, hi):
            calls.append(1)
            return p[0]*np.log(hi/lo)
        shared = SharedEvaluation(plan.grid, source)
        detector = [EnergyGrid.from_edges([1, 2, 3]), EnergyGrid.from_edges([1, 1.5, 2.5, 3])]
        target = detector[1]
        responses = [np.arange(2*g.size, dtype=float).reshape(2, g.size)+1 for g in source_grids[:1]]
        responses.append(np.arange(3*source_grids[1].size, dtype=float).reshape(3, -1)+1)
        total, baseline = np.zeros(target.size), np.zeros(target.size)
        for g, model_projection, output_grid, response in zip(source_grids, plan.projections, detector, responses):
            mapping = OverlapRebin(output_grid, target)
            total += mapping(response @ model_projection(shared.evaluate([2.])))
            baseline += mapping(response @ (2*np.log(g.hi/g.lo)))
        assert_allclose(total, baseline, rtol=1e-13)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()

"""Numerical checks of grouped fitting/count geometry without Sherpa."""
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from fancy_plotting import (EnergyGrid, SharedEvaluation, prepare_grids,
                         prepare_count_combination)


def group_arrays(grid, values, starts):
    starts = np.asarray(starts)
    stops = np.r_[starts[1:], grid.size]
    return EnergyGrid(grid.lo[starts], grid.hi[stops-1]), np.add.reduceat(values, starts)


class GroupedCombinationTests(unittest.TestCase):
    def setUp(self):
        self.grid = EnergyGrid.from_edges(np.linspace(.5, 7, 13))
        self.counts = [np.arange(1, 13), 3*np.arange(1, 13)]

    def test_fixed_width_grouping_at_every_scale(self):
        for width in (1, 2, 3, 4, 5, 6, 7, 12, 20):
            with self.subTest(width=width):
                starts = np.arange(0, 12, width)
                grouped = [group_arrays(self.grid, values, starts) for values in self.counts]
                combination = prepare_count_combination([g for g, _ in grouped], [c for _, c in grouped])
                assert_array_equal(combination.counts, sum(c for _, c in grouped))
                self.assertEqual(combination.grid.size, len(starts))
                self.assertEqual(combination.counts.sum(), sum(c.sum() for c in self.counts))

    def test_nonuniform_grouping_is_supported(self):
        grouped = [group_arrays(self.grid, values, [0, 1, 4, 8]) for values in self.counts]
        combination = prepare_count_combination([g for g, _ in grouped], [c for _, c in grouped])
        assert_array_equal(combination.counts, [4, 36, 104, 168])

    def test_different_groupings_with_explicit_common_coarsening(self):
        a, ac = group_arrays(self.grid, self.counts[0], np.arange(0, 12, 2))
        b, bc = group_arrays(self.grid, self.counts[1], np.arange(0, 12, 3))
        with self.assertRaisesRegex(ValueError, "incompatible"):
            prepare_count_combination([a, b], [ac, bc])
        target, expected = group_arrays(self.grid, sum(self.counts), [0, 6])
        result = prepare_count_combination([a, b], [ac, bc], grid=target)
        assert_array_equal(result.counts, expected)

    def test_compatible_native_groupings_use_first_target(self):
        a, ac = group_arrays(self.grid, self.counts[0], np.arange(0, 12, 4))
        b, bc = group_arrays(self.grid, self.counts[1], np.arange(0, 12, 2))
        result = prepare_count_combination([a, b], [ac, bc])
        assert_array_equal(result.counts, np.add.reduceat(sum(self.counts), [0, 4, 8]))

    def test_identical_filtered_intervals_and_gaps(self):
        grid = EnergyGrid([1, 2, 5, 6], [2, 3, 6, 7])
        result = prepare_count_combination([grid, grid], [[1, 2, 3, 4], [10, 20, 30, 40]])
        assert_array_equal(result.counts, [11, 22, 33, 44])
        with self.assertRaises(ValueError):
            prepare_count_combination([grid, grid], [[1, 2, 3, 4], [10, 20, 30, 40]],
                                      grid=EnergyGrid.from_edges([1, 7]))

    def test_different_selections_cannot_drop_counts_silently(self):
        subset = EnergyGrid(self.grid.lo[1:], self.grid.hi[1:])
        with self.assertRaisesRegex(ValueError, "incompatible"):
            prepare_count_combination([subset, self.grid], [self.counts[0][1:], self.counts[1]])

    def test_fractional_and_invalid_counts_rejected(self):
        for values in ([1, 2.5], [-1, 2], [np.nan, 2], [1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                prepare_count_combination([EnergyGrid.from_edges([1, 2, 3])], [values])
        with self.assertRaises(ValueError):
            prepare_count_combination([], [])

    def test_count_snapshot_and_read_only_result(self):
        counts = np.arange(1, 13, dtype=float)
        result = prepare_count_combination([self.grid], [counts])
        counts[:] = 0
        assert_array_equal(result.counts, np.arange(1, 13))
        with self.assertRaises(ValueError):
            result.counts[0] = 0

    def test_one_source_evaluation_with_grouped_non_diagonal_responses(self):
        input_grids = [EnergyGrid.from_edges(np.linspace(.5, 7, n+1)) for n in (24, 32)]
        plan = prepare_grids(input_grids)
        calls = []

        def source(p, lo, hi):
            calls.append(len(lo))
            return p[0] * np.log(hi/lo)

        shared = SharedEvaluation(plan.grid, source)
        centers = (self.grid.lo+self.grid.hi)/2
        responses = []
        for i, grid in enumerate(input_grids):
            energy = (grid.lo+grid.hi)/2
            matrix = np.exp(-.5*((centers[:, None]-energy)/.3)**2)
            matrix /= matrix.sum(axis=0)
            responses.append(matrix*(500+100*i))
        for width in (1, 2, 3, 4, 6, 12):
            starts = np.arange(0, 12, width)
            grouped_grids = [group_arrays(self.grid, c, starts)[0] for c in self.counts]
            prepared = prepare_count_combination(grouped_grids,
                [group_arrays(self.grid, c, starts)[1] for c in self.counts])
            predicted = []
            baseline = []
            for response, native, mapping in zip(responses, input_grids, plan.projections):
                folded = response @ mapping(shared.evaluate([2.]))
                predicted.append(np.add.reduceat(folded, starts))
                native_flux = 2*np.log(native.hi/native.lo)
                baseline.append(np.add.reduceat(response @ native_flux, starts))
            grouped_model = sum(proj(y) for proj, y in zip(prepared.projections, predicted))
            assert_allclose(grouped_model, sum(baseline), rtol=2e-13)
            self.assertEqual(len(grouped_model), len(starts))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], plan.grid.size)


if __name__ == "__main__":
    unittest.main()

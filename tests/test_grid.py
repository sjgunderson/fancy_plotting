import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from fancy_plotting.grid import (EnergyGrid, Projection, SharedEvaluation,
                              count_projection, merge_grids, prepare_grids)


class ModelFreePlanTests(unittest.TestCase):
    def test_prepare_before_defining_any_evaluator(self):
        grids = [EnergyGrid.from_edges([1, 2, 4]),
                 EnergyGrid.from_edges([1, 1.5, 3, 4])]
        plan = prepare_grids(grids)
        assert_array_equal(plan.grid.lo, [1, 1.5, 2, 3])
        assert_array_equal(plan.grid.hi, [1.5, 2, 3, 4])
        values = plan.grid.hi - plan.grid.lo
        for native, projection in zip(plan.native, plan.projections):
            assert_allclose(projection(values), native.hi - native.lo)

    def test_same_plan_supports_unrelated_later_models(self):
        plan = prepare_grids([EnergyGrid.from_edges([1, 2, 4]),
                              EnergyGrid.from_edges([1, 1.5, 3, 4])])
        # Model functions are defined AFTER all matching has finished.
        def power(p, lo, hi):
            return p[0] * (hi**3-lo**3)/3

        def exponential(p, lo, hi):
            return p[0] * (np.exp(-lo)-np.exp(-hi))

        for function in (power, exponential):
            evaluation = SharedEvaluation(plan.grid, function)
            for native, projection in zip(plan.native, plan.projections):
                assert_allclose(projection(evaluation.evaluate([2])),
                                function([2], native.lo, native.hi), rtol=2e-14)
            self.assertEqual(evaluation.cache_info().misses, 1)

    def test_custom_refinement_has_precomputed_mappings(self):
        native = EnergyGrid.from_edges([1, 2, 4])
        fine = EnergyGrid.from_edges([1, 1.5, 2, 3, 4])
        plan = prepare_grids([native], grid=fine)
        self.assertIs(plan.grid, fine)
        assert_array_equal(plan.projections[0]([1, 2, 3, 4]), [3, 7])

    def test_unusable_custom_grid_rejected_during_preparation(self):
        with self.assertRaises(ValueError):
            prepare_grids([EnergyGrid.from_edges([1, 2, 4])],
                          grid=EnergyGrid.from_edges([1, 3, 4]))

    def test_empty_preparation_rejected(self):
        with self.assertRaises(ValueError):
            prepare_grids([])


class GridTests(unittest.TestCase):
    def test_union_and_integrated_flux(self):
        a = EnergyGrid.from_edges([1, 2, 4])
        b = EnergyGrid.from_edges([1, 1.5, 3, 4])
        common = merge_grids([a, b])
        values = (common.hi ** 3 - common.lo ** 3) / 3
        for grid in [a, b]:
            assert_allclose(Projection(common, grid)(values),
                            (grid.hi ** 3 - grid.lo ** 3) / 3, rtol=1e-14)

    def test_random_grids_against_analytic_integrals(self):
        rng = np.random.default_rng(9)
        grids = [EnergyGrid.from_edges(np.r_[0.1, np.sort(rng.uniform(.1, 8, n)), 8])
                 for n in (23, 57, 89)]
        common = merge_grids(grids)
        flux = np.log(common.hi / common.lo)
        for grid in grids:
            assert_allclose(Projection(common, grid)(flux),
                            np.log(grid.hi / grid.lo), rtol=5e-13, atol=2e-16)

    def test_invalid_grids(self):
        for lo, hi in [([], []), ([1], [1]), ([2], [1]), ([-1], [1]),
                       ([1, 1.5], [2, 3]), ([np.nan], [2]), ([1], [np.inf]),
                       ([[1]], [[2]]), ([1], [2, 3])]:
            with self.subTest(lo=lo), self.assertRaises(ValueError):
                EnergyGrid(lo, hi)

    def test_copies_and_read_only_coordinates(self):
        lo, hi = np.array([1.]), np.array([2.])
        grid = EnergyGrid(lo, hi)
        lo[0] = 9
        self.assertEqual(grid.lo[0], 1)
        with self.assertRaises(ValueError):
            grid.lo[0] = 3

    def test_requested_edge_must_exist(self):
        master = EnergyGrid.from_edges([1, 2, 3])
        with self.assertRaises(ValueError):
            Projection(master, EnergyGrid.from_edges([1, 1.5, 3]))

    def test_round_trip_noise(self):
        master = EnergyGrid.from_edges([.5, 1., 2.])
        edges = np.array([.5, 1., 2.])
        edges = np.nextafter(edges, np.inf)
        assert_allclose(Projection(master, EnergyGrid.from_edges(edges))([2, 3]), [2, 3])

    def test_gaps_not_silently_filled(self):
        native = EnergyGrid([1, 3], [2, 4])
        with self.assertRaises(ValueError):
            Projection(native, EnergyGrid.from_edges([1, 4]))
        assert_array_equal(Projection(native, native)([2, 3]), [2, 3])

    def test_weak_bin_not_lost_to_cumulative_cancellation(self):
        grid = EnergyGrid.from_edges([1, 2, 3, 4])
        assert_array_equal(Projection(grid, grid)([1e25, 1e-20, 7]), [1e25, 1e-20, 7])

    def test_count_coarsening_and_rejection(self):
        native = EnergyGrid.from_edges([0, 1, 2, 3, 4])
        target = EnergyGrid.from_edges([0, 2, 4])
        assert_array_equal(count_projection(native, target)([2, 3, 5, 7]), [5, 12])
        for edges in ([0, 1.5, 4], [1, 2, 4], [0, 2, 3]):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                count_projection(native, EnergyGrid.from_edges(edges))

    def test_reject_wrong_value_shape(self):
        grid = EnergyGrid.from_edges([1, 2, 3])
        with self.assertRaises(ValueError):
            Projection(grid, grid)([1])


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.grid = merge_grids([EnergyGrid.from_edges([1, 2, 4]),
                                 EnergyGrid.from_edges([1, 1.5, 3, 4])])
        self.calls = 0

        def evaluator(p, lo, hi):
            self.calls += 1
            return p[0] * (hi - lo) + p[1] * (hi**2 - lo**2) / 2

        self.evaluator = evaluator
        self.cache = SharedEvaluation(self.grid, evaluator)

    def test_once_for_multiple_requested_grids(self):
        for edges in ([1, 2, 4], [1, 1.5, 3, 4], [2, 3]):
            grid = EnergyGrid.from_edges(edges)
            assert_allclose(self.cache.project([3, 2], grid),
                            3 * (grid.hi-grid.lo) + grid.hi**2-grid.lo**2)
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.cache.cache_info().hits, 2)

    def test_every_parameter_in_key(self):
        self.cache.evaluate([3, 2])
        self.cache.evaluate([3, 4])
        self.cache.evaluate([7, 4])
        self.assertEqual(self.calls, 3)

    def test_lru_eviction(self):
        cache = SharedEvaluation(self.grid, self.evaluator, cache_size=2)
        for p in ([1, 2], [2, 2], [1, 2], [3, 2], [2, 2]):
            cache.evaluate(p)
        self.assertEqual(self.calls, 4)
        self.assertEqual(cache.cache_info().entries, 2)

    def test_returned_arrays_cannot_poison_cache(self):
        result = self.cache.evaluate([3, 2])
        expected = result.copy()
        result[:] = -999
        assert_array_equal(self.cache.evaluate([3, 2]), expected)

    def test_disabled_and_clear(self):
        self.cache.enabled = False
        self.cache.evaluate([1, 2])
        self.cache.evaluate([1, 2])
        self.assertEqual(self.calls, 2)
        self.cache.clear()
        self.assertEqual(self.cache.cache_info().misses, 0)

    def test_configuration_token(self):
        self.cache.evaluate([1, 2], token="a")
        self.cache.evaluate([1, 2], token="b")
        self.assertEqual(self.calls, 2)

    def test_error_not_cached(self):
        cache = SharedEvaluation(self.grid, lambda p, lo, hi: [np.nan] * len(lo))
        for _ in range(2):
            with self.assertRaises(ValueError):
                cache.evaluate([1])
        self.assertEqual(cache.cache_info().entries, 0)
        self.assertEqual(cache.cache_info().misses, 2)

    def test_invalid_parameters_and_size(self):
        for p in ([[1, 2]], [np.inf], [np.nan]):
            with self.assertRaises(ValueError):
                self.cache.evaluate(p)
        for size in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                SharedEvaluation(self.grid, self.evaluator, cache_size=size)

    def test_parallel_requests_share_one_computation(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.cache.evaluate([2, 3]), range(20)))
        self.assertEqual(self.calls, 1)
        for result in results:
            assert_array_equal(result, results[0])


if __name__ == "__main__":
    unittest.main()

"""Joint-fit regressions against the real Sherpa UI; skipped without Sherpa."""
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from test_sherpa import HAS_SHERPA

if HAS_SHERPA:
    from test_sherpa import CountingPowerLaw, make_pha
    from sherpa.astro.instrument import Response1D
    from sherpa.astro.ui.utils import Session
    from sherpa.data import DataSimulFit
    from sherpa.fit import Fit
    from sherpa.models.model import SimulFitModel
    from sherpa.optmethods import NelderMead
    from sherpa.stats import CStat, WStat, Chi2DataVar
    from sherpa.utils.err import FitErr, ModelErr
    from fancy_plotting import combine_datasets, match_dataset_grids, match_pha_grids


@unittest.skipUnless(HAS_SHERPA, "requires a real CIAO/Sherpa installation")
class SherpaJointFittingTests(unittest.TestCase):
    def setUp(self):
        self.source = CountingPowerLaw("joint_source")
        self.session = Session()
        self.session.set_stat("wstat")
        self.session.set_method("neldermead")
        self.phas = {}
        for key, ne, nc, exposure in (("a", 32, 8, 500), (3, 48, 16, 800)):
            pha = make_pha(str(key), ne, nc, exposure)
            background = make_pha(str(key)+"_bg", ne, nc, exposure/2)
            pha.backscal = 2.
            background.backscal = np.linspace(4., 6., nc)
            background.counts = np.arange(12., 12.+nc)
            pha.set_background(background)
            scale = pha.get_background_scale(1, units="counts", group=False, filter=False)
            pha.counts = np.rint(Response1D(pha)(self.source)(pha.channel) +
                                 scale*background.counts)
            self.session.set_data(key, pha)
            self.session.set_source(key, self.source)
            self.phas[key] = pha

    def group(self):
        group = match_dataset_grids(list(self.phas), session=self.session).set_source(self.source)
        return combine_datasets(group)

    def compare_with_ui(self):
        # Compare ordinary response folding with the shared-grid adapter,
        # resetting every free parameter to exactly the same starting point.
        self.source.ampl = .04
        params = list(self.source.pars)
        for key, pha in self.phas.items():
            if not pha.subtracted:
                for bid in pha.background_ids:
                    try:
                        params.extend(self.session.get_bkg_model(key, bid).pars)
                    except ModelErr:
                        # An unmodeled WStat background has no parameters.
                        pass
        start = [(p, p.val) for p in params if not p.frozen]
        self.session.fit(*self.phas)
        baseline = self.session.get_fit_results()
        for par, value in start:
            par.val = value
        group = self.group()
        before = [(p.counts.copy(), p.mask.copy() if isinstance(p.mask, np.ndarray) else p.mask,
                   p.grouped, p.subtracted, p.get_background().counts.copy())
                  for p in self.phas.values()]
        fit = group.make_fit()
        assert_allclose(fit.calc_stat(), self.session.calc_stat(*self.phas), rtol=2e-12)
        result = fit.fit()
        self.assertTrue(result.succeeded)
        assert_allclose(result.parvals, baseline.parvals, rtol=2e-5, atol=1e-8)
        assert_allclose(result.statval, baseline.statval, rtol=2e-8, atol=1e-7)
        self.assertEqual((result.numpoints, result.dof), (baseline.numpoints, baseline.dof))
        for pha, (counts, mask, grouped, subtracted, bg_counts) in zip(self.phas.values(), before):
            assert_array_equal(pha.counts, counts)
            assert_array_equal(pha.mask, mask)
            assert_array_equal(pha.get_background().counts, bg_counts)
            self.assertEqual((pha.grouped, pha.subtracted), (grouped, subtracted))
        return fit, result

    def test_wstat_joint_fit_matches_ui_with_grouping_and_filters(self):
        for pha in self.phas.values():
            pha.grouping = np.tile([1, -1], len(pha.channel)//2)
            pha.group()
            pha.notice(1., 5.)
        fit, result = self.compare_with_ui()
        self.assertEqual(result.statname, "wstat")
        self.assertIs(fit.stat, self.session.get_stat())
        self.assertIs(fit.method, self.session.get_method())

    def test_subtracted_chi_square_joint_fit_matches_ui(self):
        self.session.set_stat("chi2datavar")
        for pha in self.phas.values():
            pha.subtract()
        _, result = self.compare_with_ui()
        self.assertEqual(result.statname, "chi2datavar")

    def test_modeled_background_likelihoods_are_included(self):
        self.session.set_stat("cstat")
        for key in self.phas:
            self.session.set_bkg_source(key, CountingPowerLaw("bkg_"+str(key)))
        fit, result = self.compare_with_ui()
        self.assertEqual(len(fit.data.datasets), 4)
        self.assertEqual(len(result.parvals), 3)

    def test_full_background_models_are_included(self):
        self.session.set_stat("cstat")
        group = self.group()
        for key, pha in self.phas.items():
            background = pha.get_background()
            bgmodel = Response1D(background)(CountingPowerLaw("full_bkg_"+str(key)))
            self.session.set_bkg_full_model(key, bgmodel)
            # Sherpa requires an explicit source-region full model when a
            # background is itself specified as a full model.
            scale = pha.get_background_scale(1, units="counts", group=False, filter=False)
            self.session.set_full_model(key, Response1D(pha)(group.sources[key]) + scale*bgmodel)
        fit = group.make_fit()
        self.assertEqual(len(fit.data.datasets), 4)
        assert_allclose(fit.calc_stat(), self.session.calc_stat(*self.phas), rtol=2e-12)
        self.assertTrue(fit.fit().succeeded)

    def test_explicit_options_do_not_change_session(self):
        self.session.set_stat("chi2datavar")
        self.session.set_method("levmar")
        group = self.group()
        stat, method = WStat(), NelderMead()
        original_stat, original_method = self.session.get_stat(), self.session.get_method()
        fit = group.make_fit(stat=stat, method=method)
        self.assertIs(fit.stat, stat)
        self.assertIs(fit.method, method)
        self.assertIs(self.session.get_stat(), original_stat)
        self.assertIs(self.session.get_method(), original_method)
        self.assertEqual(fit.fit().statname, "wstat")

    def test_group_fit_excludes_unrelated_session_datasets(self):
        other = make_pha("unrelated")
        other.counts = np.full(8, 100.)
        self.session.set_data("unrelated", other)
        self.session.set_source("unrelated", CountingPowerLaw("unrelated_source"))
        fit = self.group().make_fit()
        self.assertEqual(tuple(fit.data.datasets), tuple(self.phas.values()))
        assert_allclose(fit.calc_stat(), self.session.calc_stat(*self.phas), rtol=2e-12)

    def test_statistic_changes_after_group_creation_are_respected(self):
        group = self.group()
        self.session.set_stat("chi2datavar")
        for pha in self.phas.values():
            pha.subtract()
        fit = group.make_fit()
        self.assertIs(fit.stat, self.session.get_stat())
        self.assertTrue(fit.fit().succeeded)

    def test_incompatible_poisson_statistic_still_rejects_subtracted_data(self):
        for pha in self.phas.values():
            pha.subtract()
        with self.assertRaises(FitErr):
            self.group().make_fit(stat=CStat()).fit()

    def test_session_free_wstat_agrees_with_native_fit(self):
        models = tuple(Response1D(p)(self.source) for p in self.phas.values())
        self.source.ampl = .04
        baseline = Fit(DataSimulFit("native", tuple(self.phas.values())),
                       SimulFitModel("native_models", models),
                       stat=WStat(), method=NelderMead()).fit()
        self.source.ampl = .04
        group = match_pha_grids(self.phas).set_source(self.source)
        result = group.make_fit(stat=WStat(), method=NelderMead()).fit()
        assert_allclose(result.parvals, baseline.parvals, rtol=2e-5, atol=1e-8)
        assert_allclose(result.statval, baseline.statval, rtol=2e-8, atol=1e-7)

    def test_session_free_subtracted_data_support_chi_square(self):
        for pha in self.phas.values():
            pha.subtract()
        group = match_pha_grids(self.phas).set_source(self.source)
        result = group.make_fit(stat=Chi2DataVar(), method=NelderMead()).fit()
        self.assertTrue(result.succeeded)

    def test_session_free_background_data_require_statistic(self):
        group = match_pha_grids(self.phas).set_source(self.source)
        with self.assertRaisesRegex(ValueError, "explicit statistic"):
            group.make_fit()


if __name__ == "__main__":
    unittest.main()

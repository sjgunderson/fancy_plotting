"""Real DataPHA/ARF/RMF integration checks; skipped without Sherpa."""
from copy import deepcopy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from test_sherpa import HAS_SHERPA

if HAS_SHERPA:
    from test_sherpa import make_pha, CountingPowerLaw
    from sherpa.astro import hc
    from sherpa.astro.data import DataARF
    from sherpa.astro.instrument import Response1D
    from sherpa.astro.ui.utils import Session
    from sherpa.models.model import ArithmeticModel
    from fancy_plotting import plot_unfold, match_dataset_grids
    from fancy_plotting._unfold_plot import response_weights


@unittest.skipUnless(HAS_SHERPA, "requires a real CIAO/Sherpa installation")
class SherpaUnfoldTests(unittest.TestCase):
    def setUp(self):
        self.pha = make_pha("unfold", 32, 8, 500)
        self.pha.counts = np.arange(20, 28, dtype=float)
        self.source = CountingPowerLaw()
        self.session = Session()
        self.session.set_stat("chi2datavar")
        self.session.set_data(1, self.pha)
        self.session.set_source(1, self.source)

    def tearDown(self):
        from matplotlib import pyplot as plt
        plt.close("all")

    def attach_background(self):
        background = make_pha("bg", 32, 8, 500)
        background.counts = np.full(8, 6.)
        self.pha.backscal, background.backscal = 2., 4.
        self.pha.set_background(background)
        return background

    def test_reference_weights_match_a_real_folded_flat_wavelength_spectrum(self):
        class FlatWavelength(ArithmeticModel):
            ndim = 1
            def __init__(self):
                ArithmeticModel.__init__(self, "flat_wave", ())
            def calc(self, p, lo, hi=None, *args, **kwargs):
                return hc*(1/lo-1/hi)
        model = Response1D(self.pha)(FlatWavelength())
        model.startup(True)
        try:
            expected = model(self.pha.channel)
        finally:
            model.teardown()
        assert_allclose(response_weights(self.pha, hc=hc), expected, rtol=3e-12)

    def test_no_source_model_is_required(self):
        session = Session()
        session.set_data(1, self.pha)
        result = plot_unfold(1, session=session)
        self.assertIsNone(result.model_curves[0])
        self.assertTrue(result.curves[0].valid.all())

    def test_explicit_data_only_mode_never_looks_up_a_model(self):
        error = AssertionError("flux correction must not access models")
        with patch.object(Session, "get_source", side_effect=error), \
                patch.object(Session, "get_model", side_effect=error), \
                patch.object(Session, "list_model_ids", side_effect=error):
            result = plot_unfold(1, session=self.session, show_model=False)
        self.assertTrue(result.curves[0].valid.all())

    def test_source_parameter_changes_leave_data_fixed_and_move_model(self):
        first = plot_unfold(1, session=self.session)
        self.source.ampl = self.source.ampl.val*2
        second = plot_unfold(1, session=self.session)
        assert_allclose(second.curves[0].y, first.curves[0].y)
        assert_allclose(second.model_curves[0].y, 2*first.model_curves[0].y)

    def test_grouping_and_notice_are_applied_to_counts_and_weights(self):
        native_weights = response_weights(self.pha, hc=hc)
        self.pha.grouping = np.tile([1, -1], 4)
        self.pha.group()
        self.pha.notice(1.3, 6)
        expected = self.pha.apply_filter(native_weights)
        result = plot_unfold(1, session=self.session, res=4)
        assert_allclose(result.curves[0].weights, expected)
        self.assertTrue(self.pha.grouped)

    def test_filtering_the_response_cannot_change_full_calibration_weights(self):
        expected = response_weights(self.pha, hc=hc)
        _, rmf = self.pha.get_response()
        rmf.notice(np.array([3, 4]))
        filtered = rmf.get_indep()[0].copy()
        assert_allclose(response_weights(self.pha, hc=hc), expected)
        assert_array_equal(rmf.get_indep()[0], filtered)

    def test_mismatched_arf_rmf_incident_grids_use_the_general_integral(self):
        _, rmf = self.pha.get_response()
        edges = np.linspace(.5, 7, 65)
        arf = DataARF("fine_arf", edges[:-1], edges[1:], np.full(64, 10.))
        self.pha.set_response(arf, rmf)
        expected = self.pha.exposure*self.pha.areascal*(
            rmf.matrix.reshape(32, 8).T @ (10*hc*(1/rmf.energ_lo-1/rmf.energ_hi)))
        result = plot_unfold(1, session=self.session, show_model=False)
        assert_allclose(result.curves[0].weights, expected, rtol=3e-12)

    def test_multiple_response_components_all_contribute(self):
        extra = make_pha("extra", 48, 8, 500)
        arf, rmf = extra.get_response()
        self.pha.set_response(arf, rmf, id=2)
        expected = np.zeros(8)
        for key in self.pha.response_ids:
            a, r = self.pha.get_response(key)
            expected += self.pha.exposure*(r.matrix.reshape(len(r.energ_lo), 8).T @ (
                a.specresp*hc*(1/a.energ_lo-1/a.energ_hi)))
        expected *= self.pha.areascal
        result = plot_unfold(1, session=self.session, show_model=False)
        assert_allclose(result.curves[0].weights, expected, rtol=3e-12)

    def test_arf_exposure_precedence_can_be_overridden(self):
        arf, _ = self.pha.get_response()
        arf.exposure = 1000
        default = plot_unfold(1, session=self.session, show_model=False).curves[0]
        pha = plot_unfold(1, session=self.session, show_model=False, exposure_source="pha").curves[0]
        assert_allclose(default.weights, 2*pha.weights)
        assert_allclose(default.y, pha.y/2)

    def test_background_views_use_the_same_flux_conversion_and_count_residuals(self):
        self.attach_background()
        net = plot_unfold(1, session=self.session, res=6)
        raw = plot_unfold(1, session=self.session, background="include", res=6)
        background = plot_unfold(1, session=self.session, background="only", res=6)
        assert_allclose(raw.curves[0].y-net.curves[0].y, background.curves[0].y)
        assert_allclose(background.residuals[0].y, net.residuals[0].y)
        assert_allclose(net.curves[0].yerr, np.sqrt(self.pha.counts+1.5)*net.curves[0].factor)
        self.assertIsNone(background.model_curves[0])

    def test_zero_areascal_is_a_masked_flux_bin(self):
        self.pha.areascal = np.array([0, 1, 1, 1, 1, 1, 1, 1.])
        result = plot_unfold(1, session=self.session)
        assert_array_equal(result.curves[0].valid, [False, True, True, True, True, True, True, True])
        self.assertTrue(np.isnan(result.curves[0].y[0]))

    def test_optional_real_rgs_full_rsp_has_nonzero_valid_unfolded_bins(self):
        """Run with SHERPA_TEST_DATA pointing to Sherpa's public test-data tree."""
        from sherpa.astro import io
        root = Path(os.environ.get("SHERPA_TEST_DATA", "/no-sherpa-test-data"))
        names = ("P0112880201R1S004SRSPEC1003.FTZ",
                 "P0112880201R1S004RSPMAT1003.FTZ")
        paths = next(((parent/name for name in names)
                      for parent in (root/"xmmrgs", root/"sherpatest"/"xmmrgs")
                      if all((parent/name).exists() for name in names)), None)
        if paths is None:
            self.skipTest("Sherpa XMM RGS PHA and RSPMAT fixtures are not installed")
        pha_path, rmf_path = paths
        pha = io.read_pha(str(pha_path), use_errors=True)
        rmf = io.read_rmf(str(rmf_path))
        pha.set_response(None, rmf)  # RSPMAT has the effective area in the matrix.
        weights = response_weights(pha, hc=hc)
        self.assertGreater(np.count_nonzero(weights), 1000)
        session = Session()
        session.set_data(1, pha)
        curve = plot_unfold(1, session=session, show_model=False, rebin=1).curves[0]
        self.assertGreater(np.count_nonzero(np.isfinite(curve.y) & (curve.y != 0)), 100)
        self.assertGreater(np.count_nonzero(~curve.valid), 0)

    def test_intrinsic_model_uses_its_incident_grid_without_response_smearing(self):
        result = plot_unfold(1, session=self.session, model_display="source")
        model = result.model_curves[0]
        self.assertEqual(len(model.y), 32)
        expected = self.source(model.lo, model.hi)/(model.hi-model.lo)
        assert_allclose(model.y, expected, rtol=3e-12)
        self.assertEqual(len(result.curves[0].y), 8)

    def test_source_display_retains_a_finer_arf_grid(self):
        _, rmf = self.pha.get_response()
        edges = np.linspace(.5, 7, 65)
        self.pha.set_response(DataARF("fine_arf", edges[:-1], edges[1:], np.full(64, 10.)), rmf)
        result = plot_unfold(1, session=self.session, model_display="source")
        self.assertEqual(len(result.model_curves[0].y), 64)
        model = result.model_curves[0]
        assert_allclose(model.y, self.source(model.lo, model.hi)/(model.hi-model.lo), rtol=3e-12)

    def test_shared_source_still_evaluates_once_for_convolved_group(self):
        other = make_pha("other", 48, 11, 800)
        other.counts = np.full(11, 30.)
        self.session.set_data(2, other)
        match_dataset_grids([1, 2], session=self.session).set_source(self.source)
        self.source.calls = 0
        result = plot_unfold([[1, 2]], session=self.session, res=4)
        self.assertEqual(self.source.calls, 1)
        self.assertEqual(len(result.residuals), 1)

    def test_analysis_changes_preserve_flux_integrals_without_rematching(self):
        other = make_pha("other", 48, 11, 800)
        other.counts = np.full(11, 30.)
        self.session.set_data(2, other)
        match_dataset_grids([1, 2], session=self.session).set_source(self.source)
        base = plot_unfold([[1, 2]], session=self.session, res=4, rebin=2)
        integral = base.curves[0].y*(base.curves[0].hi-base.curves[0].lo)
        for coordinate in ("wavelength", "channel", "energy"):
            self.session.set_analysis(coordinate, type="counts", factor=3)
            result = plot_unfold([[1, 2]], session=self.session, res=4, rebin=2)
            curve = result.curves[0]
            assert_allclose(curve.y*(curve.hi-curve.lo), integral[::-1] if coordinate == "wavelength" else integral)
            assert_allclose(result.residuals[0].y,
                            base.residuals[0].y[::-1] if coordinate == "wavelength" else base.residuals[0].y)

    def test_channel_source_display_has_a_calibrated_density(self):
        self.session.set_analysis("channel")
        result = plot_unfold(1, session=self.session, model_display="source", flux_unit="ergs", res=4)
        self.assertEqual(len(result.model_curves[0].y), 8)
        self.assertIn("channel", result.ax.get_ylabel())
        result.figure.canvas.draw()

    def test_unfolding_does_not_mutate_inputs_or_assign_a_model(self):
        self.attach_background()
        self.pha.grouping = np.tile([1, -1], 4)
        self.pha.group()
        self.pha.notice(1.3, 6)
        self.pha.set_analysis("wavelength")
        before = deepcopy(self.pha)
        params = tuple(p.val for p in self.source.pars)
        with patch.object(Session, "fit", side_effect=AssertionError("no fit")), \
                patch.object(Session, "set_source", side_effect=AssertionError("no assignment")):
            plot_unfold(1, session=self.session, model_display="source", res=4)
        assert_array_equal(self.pha.counts, before.counts)
        assert_array_equal(self.pha.mask, before.mask)
        self.assertEqual((self.pha.units, self.pha.grouped, self.pha.subtracted),
                         (before.units, before.grouped, before.subtracted))
        self.assertEqual(tuple(p.val for p in self.source.pars), params)
        assert_array_equal(self.pha.get_response()[1].matrix, before.get_response()[1].matrix)

    def test_pileup_has_no_source_independent_flux_normalization(self):
        from sherpa.astro.models import JDPileup
        self.session.set_pileup_model(1, JDPileup())
        with self.assertRaisesRegex(NotImplementedError, "linear response"):
            plot_unfold(1, session=self.session)


if __name__ == "__main__":
    unittest.main()


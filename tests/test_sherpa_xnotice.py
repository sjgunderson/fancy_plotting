"""Real DataPHA filter behavior; skipped when Sherpa is unavailable."""
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from test_sherpa import HAS_SHERPA

if HAS_SHERPA:
    from sherpa.astro.ui.utils import Session
    from fancy_plotting import xnotice, plot_unfold
    from test_sherpa import make_pha


@unittest.skipUnless(HAS_SHERPA, "requires a real CIAO/Sherpa installation")
class SherpaXnoticeTests(unittest.TestCase):
    def setUp(self):
        self.session = Session()
        for key in (1, 2):
            pha = make_pha(f"filter{key}")
            pha.counts = np.arange(10, 18, dtype=float)
            self.session.set_data(key, pha)

    def test_exclusive_filter_respects_current_energy_wavelength_and_channel_analysis(self):
        for analysis, lo, hi in (("energy", 1.5, 5.),
                                 ("wavelength", 5., 15.),
                                 ("channel", 2., 5.)):
            with self.subTest(analysis=analysis):
                self.session.set_analysis(1, analysis)
                mid = int((lo+hi)/2) if analysis == "channel" else (lo+hi)/2
                self.session.notice_id(1, lo, mid)
                xnotice(1, lo, hi, session=self.session)
                actual = self.session.get_data(1).get_noticed_channels().copy()
                expected = Session()
                other = make_pha("comparison")
                other.set_analysis(analysis)
                expected.set_data(1, other)
                expected.ignore_id(1)
                expected.notice_id(1, lo, hi)
                assert_array_equal(actual, expected.get_data(1).get_noticed_channels())
                self.assertGreater(len(actual), 0)
                self.assertLess(len(actual), 8)

    def test_unrelated_data_and_previous_filter_are_not_retained(self):
        self.session.set_analysis(1, "energy")
        self.session.set_analysis(2, "wavelength")
        self.session.ignore_id(2, 5, 12)
        untouched = self.session.get_data(2).get_noticed_channels().copy()
        xnotice(1, 1.5, 2.5, session=self.session)
        first = self.session.get_data(1).get_noticed_channels().copy()
        xnotice(1, 3.5, 5.5, session=self.session)
        second = self.session.get_data(1).get_noticed_channels()
        self.assertFalse(np.intersect1d(first, second).size)
        assert_array_equal(untouched, self.session.get_data(2).get_noticed_channels())
        result = plot_unfold(1, session=self.session, show_model=False)
        self.assertEqual(len(result.curves[0].y), len(second))

    def test_mixed_analysis_ids_each_use_their_own_current_x_axis(self):
        self.session.set_analysis(1, "energy")
        self.session.set_analysis(2, "wavelength")
        xnotice([1, 2], 4, 6, session=self.session)
        first = self.session.get_data(1).get_noticed_channels()
        second = self.session.get_data(2).get_noticed_channels()
        self.assertGreater(len(first), 0)
        self.assertGreater(len(second), 0)
        self.assertFalse(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()


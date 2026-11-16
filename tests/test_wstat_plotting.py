"""WStat numerics and pooled plotting diagnostics, independent of Sherpa."""
import json
from pathlib import Path
import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import minimize
from scipy.special import xlogy

from fancy_plotting import SpectrumData
from fancy_plotting._fit_plot import make_diagnostics, signed_root
from fancy_plotting._wstat import WStatInput, profile_deviance
from fancy_plotting.plotting import _CURVE


class WStatPlottingTests(unittest.TestCase):
    def test_matches_compiled_upstream_sherpa_reference(self):
        reference = json.loads(Path(__file__).with_name("wstat_reference.json").read_text())
        for row in reference["cases"]:
            with self.subTest(row=row):
                d, s, b, a = row["input"]
                value, sign = profile_deviance(d, s, [b], [a])
                assert_allclose(value, row["wstat"], rtol=2e-10, atol=2e-10)
                self.assertEqual(sign, np.sign(d-s-a*b))

    def test_zero_and_exact_match_limits_have_no_warnings(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", RuntimeWarning)
            assert_allclose(profile_deviance(0, 0, [0], [1])[0], 0)
            assert_allclose(profile_deviance(10, 8, [4], [.5])[0], 0)
            assert_allclose(signed_root([0., -4., 9.]), [0., -2., 3.])
        self.assertFalse(captured)

    def test_equal_scales_reduce_to_one_pooled_off_measurement(self):
        assert_allclose(profile_deviance(34, 8, [4, 7, 9], [.3]*3),
                        profile_deviance(34, 8, [20], [.3]))

    def test_unequal_scales_match_independent_nuisance_minimization(self):
        for d, s, b, a in ((23., 4., [9., 7.], [.1, .8]),
                           (3., 2., [0., 8.], [2., .2]),
                           (40., 1., [0., 0.], [.2, 3.]),
                           (0., 3., [5., 8.], [.2, .9])):
            b, a = np.array(b), np.array(a)

            def objective(beta):
                mu = s+a @ beta
                return 2*(mu-d+xlogy(d, d/mu) +
                          np.sum(beta-b+xlogy(b, b/np.maximum(beta, 1e-300))))

            fit = minimize(objective, np.maximum(b, 1.), method="Nelder-Mead",
                           bounds=[(0, None)]*len(b),
                           options={"xatol":1e-10, "fatol":1e-10, "maxiter":3000})
            self.assertTrue(fit.success)
            value, _ = profile_deviance(d, s, b, a)
            assert_allclose(value, fit.fun, rtol=1e-9, atol=1e-9)

    def test_combination_and_rebinning_profile_counts_before_residual_transform(self):
        lo, hi = [1., 2., 3.], [2., 3., 4.]
        raw = [SpectrumData(lo, hi, [12, 7, 3]), SpectrumData(lo, hi, [8, 5, 1])]
        source = [SpectrumData(lo, hi, [2, 4, 3]), SpectrumData(lo, hi, [3, 1, 2])]
        bkg = [SpectrumData(lo, hi, [4, 3, 1]), SpectrumData(lo, hi, [5, 1, 2])]
        scales = [np.full(3, .2), np.full(3, .8)]
        contexts = [WStatInput(s, b, a) for s, b, a in zip(source, bkg, scales)]
        opts = {**_CURVE, "combine":"sum", "rebin":2, "scale":5, "offset":100}
        # WStat source/off input is independent of upper-panel model and variance.
        for mode in (1, 2, 4, 5):
            for unit in ("keV", "Angstrom"):
                curves = make_diagnostics((1, 2), raw, raw, source, opts, mode, "wstat", 0,
                                          unit, None, wstat_inputs=contexts)
                blocks = [(0,), (1,)] if mode < 4 else [(0, 1)]
                for curve, block in zip(curves, blocks):
                    expected = []
                    for indices in ([0, 1], [2]):
                        d = sum(raw[i].counts[indices].sum() for i in block)
                        s = sum(source[i].counts[indices].sum() for i in block)
                        b = np.concatenate([bkg[i].counts[indices] for i in block])
                        a = np.concatenate([scales[i][indices] for i in block])
                        value, sign = profile_deviance(d, s, b, a)
                        expected.append(sign*value)
                    expected = np.array(expected)
                    if mode in (1, 4):
                        expected = signed_root(expected)
                    if unit == "Angstrom":
                        expected = expected[::-1]
                    assert_allclose(curve.y, expected)
                    assert_array_equal(curve.valid, [True, True])
                    self.assertIsNone(curve.yerr)

    def test_invalid_poisson_inputs_fail_explicitly(self):
        for args in ((-1, 2, [3], [.1]), (1, -2, [3], [.1]),
                     (1, 2, [-3], [.1]), (1, 2, [3], [0]),
                     (1, 2, [3], [np.nan])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                profile_deviance(*args)

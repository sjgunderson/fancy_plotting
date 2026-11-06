"""Physical checks of conversion, density/covariance, and fixed-bin identity."""
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from fancy_plotting import EnergyGrid, SpectrumData, plot_data, plot_counts
from fancy_plotting.units import H_KEV_S, HC_KEV_ANGSTROM, normalize_unit, supported_units, convert_bins
from fancy_plotting.plotting import _convert
from fancy_plotting.resample import convert_grid, register_data_match, OverlapRebin


class SpectralUnitTests(unittest.TestCase):
    def setUp(self):
        self.data = SpectrumData([1, 2, 4], [2, 4, 7], [10, 30, 60], 10,
                                 covariance=sparse.csr_matrix([[10, 1, 0], [1, 30, 2], [0, 2, 60]]))

    def tearDown(self):
        plt.close("all")

    def test_physical_anchors(self):
        for unit, expected in (("eV", 1000.), ("MeV", .001), ("erg", 1.602176634e-9),
                               ("J", 1.602176634e-16), ("Hz", 1/H_KEV_S)):
            with self.subTest(unit=unit):
                lo, hi, order = convert_bins([1], [2], "keV", unit)
                assert_allclose(lo, [expected], rtol=2e-15)
                assert_allclose(hi, [2*expected], rtol=2e-15)
                assert_array_equal(order, [0])
        lo, hi, _ = convert_bins([1], [2], "keV", "nm")
        assert_allclose(lo, [HC_KEV_ANGSTROM/20])
        assert_allclose(hi, [HC_KEV_ANGSTROM/10])

    def test_all_physical_unit_pairs_round_trip(self):
        units = [u for u in supported_units() if u != "channel"]
        for source in units:
            start = _convert(self.data, source)
            for target in units:
                with self.subTest(source=source, target=target):
                    final = _convert(_convert(start, target), source)
                    assert_allclose(final.lo, start.lo, rtol=2e-15)
                    assert_allclose(final.hi, start.hi, rtol=2e-15)
                    assert_array_equal(final.counts, start.counts)
                    assert_allclose(final.covariance.toarray(), start.covariance.toarray())

    def test_aliases(self):
        for text, canonical in ((" keV ", "energy"), ("Å", "wavelength"), ("A", "wavelength"),
                                ("µm", "um"), ("μm", "um"), ("frequency", "Hz"), ("wave", "wavelength")):
            self.assertEqual(normalize_unit(text), canonical)

    def test_reciprocal_conversion_requires_positive_edges(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            convert_bins([0, 1], [1, 2], "keV", "nm")
        lo, _, _ = convert_bins([0, 1], [1, 2], "keV", "eV")
        assert_array_equal(lo, [0, 1000])

    def test_channel_calibration_not_inferred(self):
        for unit in ("eV", "Angstrom", "Hz"):
            with self.subTest(unit=unit), self.assertRaisesRegex(ValueError, "response"):
                convert_bins([1], [2], "channel", unit)
        with self.assertRaises(TypeError):
            register_data_match({1: self.data}, data_unit="channel")

    def test_unknown_units_are_rejected(self):
        for value in ("furlong", "km/s", 3, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "spectral unit"):
                normalize_unit(value)

    def test_grid_conversion_and_permutation(self):
        grid = EnergyGrid([1, 2, 4], [2, 4, 7])
        wave, order = convert_grid(grid, "keV", "nm")
        assert_array_equal(order, [2, 1, 0])
        back, permutation = convert_grid(wave, "nm", "keV")
        assert_allclose(back.lo, grid.lo)
        assert_allclose(back.hi, grid.hi)
        assert_array_equal(permutation, [2, 1, 0])

    def test_density_integrals_and_covariance_are_unit_invariant(self):
        for unit in ("eV", "MeV", "erg", "Angstrom", "nm", "um", "Hz", "GHz"):
            with self.subTest(unit=unit):
                result = plot_data(1, datasets={1: self.data}, unit=unit)
                curve = result.curves[0]
                widths = curve.hi-curve.lo
                _, _, order = convert_bins(self.data.lo, self.data.hi, self.data.unit, unit)
                assert_allclose(curve.y*widths, self.data.counts[order]/10)
                integrated_cov = curve.covariance.toarray()*widths[:, None]*widths[None, :]
                assert_allclose(integrated_cov, self.data.covariance.toarray()[order][:, order]/100)
                self.assertIn({"Angstrom": r"\mathrm{\AA}", "um": r"\mu\mathrm{m}"}.get(unit, unit), result.ax.get_xlabel())
                self.assertIn({"Angstrom": r"\mathrm{\AA}", "um": r"\mu\mathrm{m}"}.get(unit, unit), result.ax.get_ylabel())

    def test_unmatched_plot_rebin_keeps_odd_partial_group_on_conversion(self):
        base = plot_counts(1, datasets={1: self.data}, rebin=2, unit="keV").curves[0]
        wave = plot_counts(1, datasets={1: self.data}, rebin=2, unit="nm").curves[0]
        assert_array_equal(base.y, [40, 60])
        assert_array_equal(wave.y, [60, 40])
        assert_allclose(wave.covariance.toarray(), base.covariance.toarray()[::-1, ::-1])

    def test_matched_sum_rebin_unchanged_by_display_unit(self):
        other = SpectrumData([1, 3], [3, 7], [20, 80], exposure=20)
        datasets = {1: self.data, 2: other}
        match = register_data_match(datasets, reference=1)
        base = plot_counts([[1, 2]], datasets=datasets, rebin=2, unit="keV").curves[0]
        for unit in ("eV", "MeV", "Angstrom", "nm", "Hz"):
            with self.subTest(unit=unit):
                curve = plot_counts([[1, 2]], datasets=datasets, rebin=2, unit=unit).curves[0]
                _, _, order = convert_bins(base.lo, base.hi, "keV", unit)
                assert_allclose(curve.y, base.y[order])
                assert_allclose(curve.covariance.toarray(), base.covariance.toarray()[order][:, order])
                self.assertEqual(match.unit, "wavelength")

    def test_scaled_matching_coordinate_preserves_overlap_operator(self):
        first = EnergyGrid([1, 3], [3, 7])
        second = EnergyGrid([1, 2, 4], [2, 4, 7])
        base = OverlapRebin(first, second).matrix.toarray()
        for unit in ("eV", "MeV", "Hz", "erg"):
            a = convert_grid(first, "keV", unit)[0]
            b = convert_grid(second, "keV", unit)[0]
            assert_allclose(OverlapRebin(a, b).matrix.toarray(), base, rtol=2e-14)

    def test_nonlinear_matching_coordinate_is_a_distinct_explicit_choice(self):
        first = EnergyGrid([1], [3])
        second = EnergyGrid([1, 2], [2, 3])
        energy = OverlapRebin(first, second)([12])
        wave_a, _ = convert_grid(first, "keV", "Angstrom")
        wave_b, _ = convert_grid(second, "keV", "Angstrom")
        wave = OverlapRebin(wave_a, wave_b)([12])[::-1]
        assert_allclose(energy, [6, 6])
        assert_allclose(wave, [9, 3])

    def test_default_display_infers_input_unit(self):
        data = _convert(self.data, "nm")
        default = plot_data(1, datasets={1: data})
        explicit = plot_data(1, datasets={1: data}, unit="nm")
        assert_allclose(default.curves[0].y, explicit.curves[0].y)
        self.assertEqual(default.ax.get_xlabel(), r"Wavelength ($\mathrm{nm}$)")

    def test_mixed_input_units_describe_identical_bins(self):
        datasets = {1: self.data, 2: _convert(self.data, "eV"), 3: _convert(self.data, "nm")}
        curve = plot_counts([[1, 2, 3]], datasets=datasets, unit="Hz").curves[0]
        assert_allclose(curve.y, self.data.counts*3)
        assert_allclose(curve.covariance.toarray(), self.data.covariance.toarray()*3)

    def test_matching_coordinate_is_fixed_without_reading_analysis_or_headers(self):
        # Unknown instrument names are deliberately irrelevant.
        object.__setattr__(self.data, "header", {"GRATING": "HETG", "INSTRUME": "unknown"})
        self.assertEqual(register_data_match({1: self.data}).unit, "wavelength")
        wave = _convert(self.data, "nm")
        self.assertEqual(register_data_match({1: wave}).unit, "wavelength")

        class Reference:
            @property
            def units(self):
                raise AssertionError("matching must not read the analysis setting")
        reference = Reference()
        self.assertEqual(register_data_match({1: reference}).unit, "wavelength")

    def test_matching_numerics_unchanged_by_input_coordinate(self):
        source = SpectrumData([1], [3], [12], exposure=10)
        reference = SpectrumData([1, 2], [2, 3], [2, 4], exposure=20)
        baseline_covariance = None
        for source_unit in ("keV", "Angstrom", "nm"):
            for reference_unit in ("keV", "Angstrom", "nm"):
                with self.subTest(source=source_unit, reference=reference_unit):
                    datasets = {1: _convert(source, source_unit), 2: _convert(reference, reference_unit)}
                    register_data_match(datasets, reference=2)
                    curve = plot_counts([[1, 2]], datasets=datasets, unit="keV").curves[0]
                    assert_allclose(curve.y, [11, 7])
                    if baseline_covariance is None:
                        baseline_covariance = curve.covariance.toarray()
                    assert_allclose(curve.covariance.toarray(), baseline_covariance)

    def test_sherpa_counts_and_plot_factor_defaults(self):
        object.__setattr__(self.data, "rate", False)
        object.__setattr__(self.data, "grouped", False)
        object.__setattr__(self.data, "plot_fac", 2)
        curve = plot_data(1, datasets={1: self.data}).curves[0]
        multiplier = ((self.data.lo+self.data.hi)/2)**2
        assert_allclose(curve.y, self.data.counts*multiplier)
        assert_allclose(curve.yerr**2, self.data.covariance.diagonal()*multiplier**2)

    def test_calibrated_channel_display_keeps_matched_groups(self):
        from fancy_plotting.plotting import _aggregate, _CURVE
        data = SpectrumData([1, 2, 3], [2, 3, 4], [10, 20, 60], unit="Angstrom")
        curve = _aggregate((1,), [data], {**_CURVE, "combine": "sum", "rebin": 2},
                           False, False, "channel", ([3, 2, 1], [4, 3, 2]))
        assert_allclose(curve.lo, [1, 2])
        assert_allclose(curve.hi, [2, 4])
        assert_allclose(curve.y, [60, 30])
        assert_allclose(curve.yerr**2, [60, 30])

    def test_overplot_equivalent_alias_allowed_different_scale_rejected(self):
        result = plot_data(1, datasets={1: self.data}, unit="energy")
        plot_data(1, datasets={1: self.data}, unit="keV", ax=result.ax, overplot=True)
        with self.assertRaisesRegex(ValueError, "same unit"):
            plot_data(1, datasets={1: self.data}, unit="eV", ax=result.ax, overplot=True)

    def test_custom_sherpa_hc_is_preserved(self):
        data = SpectrumData([1, 2], [2, 3], [10, 20], exposure=10, hc=12.39841874)
        converted = _convert(data, "nm")
        assert_allclose(converted.lo, data.hc/10/data.hi[::-1])
        assert_allclose(_convert(converted, "keV").lo, data.lo)


if __name__ == "__main__":
    unittest.main()


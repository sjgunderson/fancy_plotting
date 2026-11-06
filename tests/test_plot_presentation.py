"""Rendered model styles, drawing order, and scientific axis labels."""
import unittest

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from fancy_plotting import SpectrumData, SourceSpectrum, plot_data, plot_counts, plot_unfold
from fancy_plotting.units import supported_units


class PlotPresentationTests(unittest.TestCase):
    def setUp(self):
        self.data = {key: SpectrumData([1, 2, 4, 5], [2, 3, 5, 6],
                                      np.array([20, 32, 15, 25])*key, exposure=100*key)
                     for key in (1, 2)}
        self.models = {key: np.array([18, 29, 17, 26])*key for key in self.data}
        self.weights = {key: np.full(4, 500.*key) for key in self.data}

    def tearDown(self):
        plt.close("all")

    def plot(self, function, ids=1, **kwargs):
        if function is plot_unfold:
            kwargs["response_weights"] = self.weights
        return function(ids, datasets=self.data, models=self.models, **kwargs)

    def test_all_public_functions_style_models_independently_without_changing_values(self):
        for function in (plot_data, plot_counts, plot_unfold):
            with self.subTest(function=function.__name__):
                baseline = self.plot(function, res=4)
                styled = self.plot(function, res=4, color="navy", ecolor="gray", model_plot="line",
                                   model_styles={"color": "crimson", "ls": ":", "lw": 2.5,
                                                 "alpha": .65, "marker": "o", "label": "Fit"})
                model = styled.model_artists[0]
                self.assertEqual(model.get_color(), "crimson")
                self.assertEqual(model.get_linestyle(), ":")
                self.assertEqual(model.get_linewidth(), 2.5)
                self.assertEqual(model.get_marker(), "o")
                self.assertEqual(model.get_label(), "Fit")
                self.assertEqual(styled.lines[0].get_color(), "navy")
                assert_allclose(styled.curves[0].y, baseline.curves[0].y)
                assert_allclose(styled.model_curves[0].y, baseline.model_curves[0].y)
                assert_allclose(styled.residuals[0].y, baseline.residuals[0].y)
                styled.figure.canvas.draw()

    def test_line_models_preserve_detector_gaps_and_use_bin_centers_once(self):
        for function in (plot_data, plot_counts, plot_unfold):
            with self.subTest(function=function.__name__):
                result = self.plot(function, model_plot="line")
                line = result.model_artists[0]
                assert_allclose(line.get_xdata(), [1.5, 2.5, np.nan, 4.5, 5.5])
                self.assertTrue(np.isnan(line.get_ydata()[2]))
                assert_allclose(line.get_ydata()[[0, 1, 3, 4]], result.model_curves[0].y)

    def test_each_displayed_model_can_choose_its_own_presentation(self):
        result = self.plot(plot_counts, [1, 2], model_plot=["line", "histogram"],
                           model_styles=[{"color": "red"}, {"color": "green", "ls": "-"}])
        self.assertEqual(len(result.model_artists[0].get_xdata()), 5)
        self.assertEqual(len(result.model_artists[1].get_xdata()), 12)
        self.assertEqual(result.model_artists[1].get_color(), "green")

    def test_source_model_presentation_also_preserves_incident_grid_gaps(self):
        source = SourceSpectrum([1, 2, 4, 5], [2, 3, 5, 6], [.2, .3, .15, .25])
        result = self.plot(plot_unfold, model_display="source", model_plot="line",
                           source_spectra={1: source}, model_styles={"color": "red"})
        assert_allclose(result.model_artists[0].get_xdata(), [1.5, 2.5, np.nan, 4.5, 5.5])

    def test_default_caps_and_bars_are_behind_histograms_with_or_without_markers(self):
        for function in (plot_data, plot_counts, plot_unfold):
            for marker in ("none", "o"):
                with self.subTest(function=function.__name__, marker=marker):
                    result = self.plot(function, [1, 2], marker=marker, capsize=3, ecolor="red",
                                       xerrorbars=True, zorder=5)
                    for line, errors in result.artists:
                        self.assertEqual(line.get_zorder(), 5)
                        for artist in (*errors.lines[1], *errors.lines[2]):
                            self.assertLess(artist.get_zorder(), line.get_zorder())
                    result.figure.canvas.draw()

    def test_explicit_barsabove_can_put_bars_above_histograms(self):
        result = self.plot(plot_counts, barsabove=True, capsize=4, zorder=7)
        line, errors = result.artists[0]
        for artist in (*errors.lines[1], *errors.lines[2]):
            self.assertGreater(artist.get_zorder(), line.get_zorder())

    def test_residual_errorbars_obey_the_same_layer_order(self):
        result = self.plot(plot_data, res=1,
                           residual_styles={"linestyle": "-", "capsize": 3, "ecolor": "red"})
        line, errors = result.residual_artists[0]
        for artist in (*errors.lines[1], *errors.lines[2]):
            self.assertLess(artist.get_zorder(), line.get_zorder())

    def test_invalid_presentation_or_layer_preserves_existing_axes(self):
        for kwargs in ({"model_plot": "smooth"}, {"model_plot": ["line", "line"]},
                       {"model_plot": False}, {"zorder": float("nan")}):
            with self.subTest(kwargs=kwargs):
                _, ax = plt.subplots()
                old, = ax.plot([1, 2], [3, 4])
                with self.assertRaises(ValueError):
                    self.plot(plot_data, ax=ax, **kwargs)
                self.assertIn(old, ax.lines)

    def test_labels_render_for_every_physical_unit(self):
        for unit in supported_units():
            if unit == "channel":
                continue
            with self.subTest(unit=unit):
                result = self.plot(plot_unfold, unit=unit, power=3)
                self.assertIn("$", result.ax.get_xlabel())
                self.assertIn(r"\mathrm{cm}^{-2}", result.ax.get_ylabel())
                self.assertNotIn("^2 /", result.ax.get_ylabel())
                result.figure.canvas.draw()
                plt.close(result.figure)

    def test_angstrom_micron_and_exponents_have_mathtext(self):
        angstrom = self.plot(plot_counts, unit="Angstrom", power=3)
        self.assertEqual(angstrom.ax.get_xlabel(), r"Wavelength ($\mathrm{\AA}$)")
        self.assertEqual(angstrom.ax.get_ylabel(), r"Counts ($\mathrm{\AA}^{2}\,\mathrm{bin}^{-1}$)")
        micron = self.plot(plot_unfold, unit="um")
        self.assertIn(r"\mu\mathrm{m}", micron.ax.get_xlabel())
        self.assertIn(r"{\mu\mathrm{m}}^{-1}", micron.ax.get_ylabel())
        self.assertIn(r"\mathrm{s}^{-1}", micron.ax.get_ylabel())

    def test_user_axis_labels_are_preserved_verbatim(self):
        result = self.plot(plot_unfold, res=4, xlabel=r"$E / E_0$", ylabel="My flux")
        self.assertEqual(result.residual_ax.get_xlabel(), r"$E / E_0$")
        self.assertEqual(result.ax.get_ylabel(), "My flux")


if __name__ == "__main__":
    unittest.main()

"""Exercise the public plotting API against real Matplotlib artists."""
import unittest
import warnings

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from fancy_plotting import SpectrumData, plot_data


class ArraySession:
    """Only the public data/default/preference interface; no model API."""
    def __init__(self, datasets):
        self.datasets = datasets
        self.prefs = {key: {"color": color, "marker": "s", "linestyle": "dotted", "capsize": 3,
                            "xlog": False, "ylog": False, "linecolor": None}
                      for key, color in zip(datasets, ("navy", "darkorange"))}

    def get_default_id(self):
        return 2

    def get_data(self, key):
        return self.datasets[key]

    def get_data_plot_prefs(self, key):
        return self.prefs[key]


class MatplotlibAPITests(unittest.TestCase):
    def setUp(self):
        self.data = {1: SpectrumData([1, 2, 3], [2, 3, 4], [10, 40, 60], exposure=10),
                     2: SpectrumData([1, 2, 3], [2, 3, 4], [20, 50, 70], exposure=20)}

    def tearDown(self):
        plt.close("all")

    def plot(self, ids=1, **kwargs):
        return plot_data(ids, datasets=self.data, **kwargs)

    def test_matplotlib_shorthand_and_named_line_styles(self):
        result = self.plot([1, 2], c=["navy", "orange"], ls=["dotted", "dashdot"],
                           lw=2.5, ms=7, marker="o", mfc="none", mec="black", mew=.8)
        self.assertEqual([l.get_linestyle() for l in result.lines], [":", "-."])
        self.assertEqual([l.get_linewidth() for l in result.lines], [2.5, 2.5])
        for container in result.errorbars:
            marker = container.lines[0]
            self.assertEqual(marker.get_markersize(), 7)
            self.assertEqual(marker.get_markerfacecolor(), "none")
            self.assertEqual(marker.get_markeredgecolor(), "black")
        result.figure.canvas.draw()

    def test_rgb_and_rgba_are_scalar_colors_even_with_multiple_curves(self):
        for color in ((.1, .2, .8), [.1, .2, .8, .5], np.array([.1, .2, .8])):
            with self.subTest(color=color):
                result = self.plot([1, 2], color=color, ecolor=(.8, .1, .2), marker="o")
                for line, errors in result.artists:
                    assert_allclose(to_rgba(line.get_color()), to_rgba(color))
                    assert_allclose(errors.lines[2][0].get_color()[0], to_rgba((.8, .1, .2)))

    def test_custom_dash_tuple_is_one_style(self):
        result = self.plot([1, 2], linestyle=(2, (6, 2, 1, 2)))
        self.assertTrue(all(line.is_dashed() for line in result.lines))
        result.figure.canvas.draw()

    def test_custom_marker_tuple_is_one_marker(self):
        result = self.plot([1, 2], marker=(5, 1, 0), linestyle="none")
        for errors in result.errorbars:
            self.assertEqual(errors.lines[0].get_marker(), (5, 1, 0))
        result.figure.canvas.draw()

    def test_numeric_marker_uses_matplotlib_semantics(self):
        result = self.plot(marker=0)
        self.assertEqual(result.errorbars[0].lines[0].get_marker(), 0)

    def test_errorbar_controls_do_not_change_errors_or_counts(self):
        baseline = self.plot().curves[0]
        result = self.plot(marker="o", ecolor="red", capsize=4, capthick=2,
                           elinewidth=.75, errorevery=2, barsabove=True)
        errors = result.errorbars[0]
        self.assertEqual(len(errors.lines[2][0].get_segments()), 2)
        assert_allclose(errors.lines[2][0].get_linewidths(), [.75])
        self.assertEqual(errors.lines[1][0].get_markersize(), 8)
        self.assertEqual(errors.lines[1][0].get_markeredgewidth(), 2)
        assert_allclose(result.curves[0].y, baseline.y)
        assert_allclose(result.curves[0].yerr, baseline.yerr)
        result.figure.canvas.draw()

    def test_errorevery_tuple_is_not_broadcast(self):
        result = self.plot([1, 2], errorevery=(1, 2))
        self.assertEqual([len(e.lines[2][0].get_segments()) for e in result.errorbars], [1, 1])

    def test_per_curve_dicts_override_shared_styles(self):
        result = self.plot([[1, 2], 1], color="gray", marker="o", capsize=1,
                           styles=[{"color": "navy", "combine": "mean", "mfc": "none"},
                                   {"color": "orange", "capsize": 5, "marker": "s"}])
        self.assertEqual([line.get_color() for line in result.lines], ["navy", "orange"])
        self.assertEqual([c.combine for c in result.curves], ["mean", "exposure"])
        self.assertEqual(result.errorbars[1].lines[1][0].get_markersize(), 10)

    def test_sherpa_preferences_are_defaults_and_are_not_mutated(self):
        session = ArraySession(self.data)
        session.prefs[1]["label"] = "Reference"
        before = {key: dict(value) for key, value in session.prefs.items()}
        result = plot_data([1, 2], session=session, linewidth=2,
                           styles=[{}, {"color": "black", "marker": "o"}])
        self.assertEqual([line.get_color() for line in result.lines], ["navy", "black"])
        self.assertEqual([line.get_linestyle() for line in result.lines], [":", ":"])
        self.assertEqual(result.errorbars[0].lines[0].get_marker(), "s")
        self.assertEqual(result.curves[0].label, "Reference")
        self.assertEqual(session.prefs, before)

    def test_default_dataset_matches_sherpa_default_id(self):
        result = plot_data(session=ArraySession(self.data))
        self.assertEqual(result.curves[0].members, (2,))
        result = plot_data(datasets=self.data)
        self.assertEqual(result.curves[0].members, (1,))
        result = plot_data(id=1, session=ArraySession(self.data))
        self.assertEqual(result.curves[0].members, (1,))

    def test_explicit_scale_overrides_inherited_sherpa_log_preference(self):
        result = plot_data(1, session=ArraySession(self.data), xscale="log", markerfacecolor="auto")
        self.assertEqual(result.ax.get_xscale(), "log")
        result.figure.canvas.draw()

    def test_clearwindow_false_uses_current_subplot_and_preserves_other_panels(self):
        fig, axes = plt.subplots(2, 1)
        previous, = axes[0].plot([1, 2], [3, 4])
        plt.sca(axes[1])
        result = self.plot(clearwindow=False)
        self.assertIs(result.ax, axes[1])
        self.assertEqual(fig.axes, list(axes))
        self.assertIn(previous, axes[0].lines)

    def test_overplot_consumes_one_cycle_color_per_spectrum(self):
        with plt.rc_context({"axes.prop_cycle": matplotlib.cycler(color=["red", "green", "blue"])}):
            first = self.plot(marker="o")
            second = self.plot(2, ax=first.ax, overplot=True, marker="s")
            self.assertEqual(first.lines[0].get_color(), "red")
            self.assertEqual(second.lines[0].get_color(), "green")
            self.assertIn(first.lines[0], second.ax.lines)

    def test_rcparams_supply_sizes_when_not_overridden(self):
        with plt.rc_context({"lines.linewidth": 2.75, "lines.markersize": 8, "errorbar.capsize": 3,
                             "lines.linestyle": "--", "lines.marker": "o"}):
            result = self.plot()
            self.assertEqual(result.lines[0].get_linewidth(), 2.75)
            self.assertEqual(result.lines[0].get_linestyle(), "--")
            self.assertEqual(result.errorbars[0].lines[0].get_markersize(), 8)
            self.assertEqual(result.errorbars[0].lines[1][0].get_markersize(), 6)

    def test_native_axis_options_support_inversion_and_partial_limits(self):
        result = self.plot(xscale="log", yscale="log", xlim=(4, 1), ylim=(.1, None))
        assert_allclose(result.ax.get_xlim(), [4, 1])
        self.assertEqual(result.ax.get_ylim()[0], .1)
        self.assertEqual(result.ax.get_xscale(), "log")
        self.assertEqual(result.ax.get_yscale(), "log")

    def test_legend_options_and_result_artists_are_customizable(self):
        result = self.plot([1, 2], label=["A", "B"], legend={"loc": "lower left", "frameon": False},
                           alpha=.7, zorder=4, rasterized=True)
        self.assertFalse(result.ax.get_legend().get_frame_on())
        self.assertEqual([text.get_text() for text in result.ax.get_legend().get_texts()], ["A", "B"])
        for line in result.lines:
            self.assertEqual(line.get_alpha(), .7)
            self.assertEqual(line.get_zorder(), 4)
            self.assertTrue(line.get_rasterized())
        result.lines[0].set_linewidth(4)
        result.figure.canvas.draw()

    def test_invalid_styles_preserve_existing_artists(self):
        for option in ({"capsize": -1}, {"linestyle": "wavy"}, {"marker": "unknown_marker"},
                       {"errorevery": 0}, {"xscale": "unknown"}, {"xlog": True, "xscale": "linear"}):
            with self.subTest(option=option):
                fig, ax = plt.subplots()
                old, = ax.plot([1, 2], [3, 4])
                with self.assertRaises(ValueError):
                    self.plot(ax=ax, **option)
                self.assertIn(old, ax.lines)

    def test_zero_line_and_marker_widths_are_valid_matplotlib_values(self):
        result = self.plot(linewidth=0, markersize=0, capsize=0, elinewidth=0, marker="o")
        self.assertEqual(result.lines[0].get_linewidth(), 0)
        result.figure.canvas.draw()

    def test_modern_syntax_has_no_deprecation_warnings(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            self.plot([1, 2], color=["navy", "orange"], linestyle="dashed", ecolor="gray",
                      capsize=2, xlim=(1, 4), overplot=False)
        self.assertFalse([w for w in caught if issubclass(w.category, DeprecationWarning)])

    def test_legacy_alias_warns_and_preserves_old_interpretation(self):
        with self.assertWarns(DeprecationWarning):
            old = self.plot(dcol=2, dsym=4, dstyle=2, decol="gray")
        modern = self.plot(color="red", marker="o", linestyle="--", ecolor="gray")
        self.assertEqual(old.lines[0].get_color(), modern.lines[0].get_color())
        self.assertEqual(old.errorbars[0].lines[0].get_marker(), modern.errorbars[0].lines[0].get_marker())
        assert_allclose(old.curves[0].y, modern.curves[0].y)


if __name__ == "__main__":
    unittest.main()

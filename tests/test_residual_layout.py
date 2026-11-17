"""Residual panel access, logarithmic tick labels, and Matplotlib layout."""
import unittest

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np
from numpy.testing import assert_allclose

from fancy_plotting import SpectrumData, plot_data, plot_counts, plot_unfold


class ResidualLayoutTests(unittest.TestCase):
    def setUp(self):
        edges = np.geomspace(.2, 6, 31)
        self.data = {1: SpectrumData(edges[:-1], edges[1:], np.arange(30)+10, exposure=100)}
        self.models = {1: np.arange(30)+11}
        self.weights = {1: np.full(30, 500.)}

    def tearDown(self):
        plt.close("all")

    def plot(self, function=plot_data, **kwargs):
        if function is plot_unfold:
            kwargs["response_weights"] = self.weights
        return function(1, datasets=self.data, models=self.models, **kwargs)

    def test_new_residual_figures_use_constrained_layout_and_ordered_axes(self):
        for function in (plot_data, plot_counts, plot_unfold):
            with self.subTest(function=function.__name__):
                result = self.plot(function, res=4)
                self.assertTrue(result.figure.get_constrained_layout())
                self.assertEqual(result.axes, (result.ax, result.residual_ax))
                result.figure.canvas.draw()
                upper, lower = (ax.get_position() for ax in result.axes)
                assert_allclose([upper.x0, upper.x1], [lower.x0, lower.x1])
                self.assertGreater(upper.y0, lower.y1)
                self.assertGreater(upper.height, lower.height)

    def test_data_only_figures_keep_matplotlib_layout_defaults(self):
        with plt.rc_context({"figure.constrained_layout.use": False, "figure.autolayout": False}):
            result = self.plot()
            self.assertIsNone(result.figure.get_layout_engine())
            self.assertEqual(result.axes, (result.ax,))

    def test_log_major_and_minor_labels_are_hidden_only_on_upper_panel(self):
        for function in (plot_data, plot_counts, plot_unfold):
            with self.subTest(function=function.__name__):
                with plt.rc_context({"xtick.labeltop": True}):
                    result = self.plot(function, res=4, xlog=True, xlabel="My energy")
                    # The formatter is shared, but label visibility must remain local.
                    result.residual_ax.xaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10)))
                    result.residual_ax.xaxis.set_minor_formatter(ScalarFormatter())
                    result.figure.canvas.draw()
                self.assertEqual(result.ax.get_xlabel(), "")
                self.assertEqual(result.ax.get_xticklabels(which="both"), [])
                self.assertTrue(result.residual_ax.get_xticklabels())
                self.assertTrue(result.residual_ax.get_xticklabels(minor=True))
                self.assertEqual(result.residual_ax.get_xlabel(), "My energy")

    def test_side_by_side_plots_expose_independent_panels(self):
        fig, slots = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        engine = fig.get_layout_engine()
        left = self.plot(ax=slots[0], res=4, xlog=True)
        right = self.plot(ax=slots[1], res=4, xlog=True)
        data_ax, residual_ax = right.axes
        data_ax.set_ylabel("")
        residual_ax.set(ylabel="", xlabel="Right energy", ylim=(-3, 3))
        data_ax.tick_params(axis="y", which="both", labelleft=False)
        residual_ax.tick_params(axis="y", which="both", labelleft=False)
        left.residual_ax.set_ylim(-5, 5)
        fig.canvas.draw()
        self.assertIs(fig.get_layout_engine(), engine)
        self.assertEqual(len(fig.axes), 4)
        self.assertEqual(left.axes, (slots[0], left.residual_ax))
        self.assertEqual(right.axes, (slots[1], right.residual_ax))
        self.assertTrue(left.ax.get_ylabel())
        self.assertTrue(left.residual_ax.get_ylabel())
        self.assertEqual(right.residual_ax.get_xlabel(), "Right energy")
        assert_allclose(left.residual_ax.get_ylim(), [-5, 5])
        self.assertLess(left.ax.get_position().x1, right.ax.get_position().x0)
        self.assertFalse(left.ax.get_shared_x_axes().joined(left.ax, right.ax))

    def test_supplied_figures_retain_their_layout_engine(self):
        for layout in (None, "tight", "constrained"):
            with self.subTest(layout=layout):
                fig, axes = plt.subplots(2, 1, sharex=True, layout=layout)
                engine = fig.get_layout_engine()
                result = self.plot(ax=axes, res=4)
                self.assertEqual(result.axes, tuple(axes))
                self.assertIs(fig.get_layout_engine(), engine)
                fig.canvas.draw()

    def test_replacing_residual_plot_restores_minor_tick_labels(self):
        result = self.plot(res=4, xlog=True)
        result = self.plot(ax=result.ax, xlog=True)
        result.ax.xaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10)))
        result.ax.xaxis.set_minor_formatter(ScalarFormatter())
        result.figure.canvas.draw()
        self.assertEqual(result.axes, (result.ax,))
        self.assertTrue(result.ax.get_xticklabels())
        self.assertTrue(result.ax.get_xticklabels(minor=True))
        self.assertTrue(result.ax.get_xlabel())


if __name__ == "__main__":
    unittest.main()

"""Render an offline example of model styles, error-bar layers, and Mathtext."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from fancy_plotting import SpectrumData, plot_counts, plot_data, plot_unfold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("model-styling-demo.png"))
    args = parser.parse_args()
    edges = np.linspace(.5, 3.5, 181)
    center = (edges[:-1]+edges[1:])/2
    model = 35*np.exp(-center/.9)+14*np.exp(-.5*((center-1.4)/.06)**2)+4
    counts = np.random.default_rng(8).poisson(model)
    data = {1: SpectrumData(edges[:-1], edges[1:], counts, exposure=1000)}
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    for function, ax, title in zip((plot_counts, plot_data, plot_unfold), axes,
                                  ("Integrated counts", "Count-rate density", "Response-corrected spectrum")):
        kwargs = {"response_weights": {1: np.full(len(center), 500.)}, "unit": "Angstrom"} \
            if function is plot_unfold else {}
        function(1, datasets=data, models={1: model}, ax=ax, ecolor="#b5bdcc", linewidth=1.2,
                 color="#243b53", model_plot="line", label="Data",
                 model_styles={"color": "#b93539", "ls": "-", "lw": 1.8, "label": "Model"},
                 title=title, legend={"frameon": False}, **kwargs)
        ax.grid(alpha=.12)
    fig.savefig(args.output, dpi=160)
    print(args.output)


if __name__ == "__main__":
    main()

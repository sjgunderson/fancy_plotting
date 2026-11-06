"""Grid matching, combined fitting, and presentation tools for Sherpa spectra."""

from .grid import (EnergyGrid, Projection, SharedEvaluation, merge_grids,
                   GridPlan, prepare_grids, CountCombination,
                   prepare_count_combination)
from .resample import OverlapRebin

__version__ = "0.1.0"
__all__ = [
    "EnergyGrid", "Projection", "SharedEvaluation", "merge_grids",
    "GridPlan", "prepare_grids", "SharedGridSource", "DatasetGroup",
    "match_pha_grids", "match_dataset_grids", "combine_datasets",
    "CombinedDataset", "plot_data", "plot_counts", "plot_unfold",
    "xnotice", "SpectrumData", "PlotResult", "PlotCurve",
    "CountCombination", "prepare_count_combination", "OverlapRebin",
    "CovarianceChi2", "ModelCurve", "ResidualCurve", "UnfoldedCurve",
    "SourceSpectrum",
]

_ADAPTER_EXPORTS = {
    "SharedGridSource", "DatasetGroup", "match_pha_grids",
    "match_dataset_grids", "combine_datasets", "CombinedDataset",
}
_PLOTTING_EXPORTS = {
    "plot_data", "plot_counts", "plot_unfold", "SpectrumData", "PlotResult",
    "PlotCurve",
}


def __getattr__(name):
    if name in _ADAPTER_EXPORTS:
        from . import adapter
        return getattr(adapter, name)
    if name in _PLOTTING_EXPORTS:
        from . import plotting
        return getattr(plotting, name)
    if name == "xnotice":
        from .filters import xnotice
        return xnotice
    if name in ("UnfoldedCurve", "SourceSpectrum"):
        from . import _unfold_plot
        return getattr(_unfold_plot, name)
    if name in ("ModelCurve", "ResidualCurve"):
        from . import _fit_plot
        return getattr(_fit_plot, name)
    if name == "CovarianceChi2":
        from .statistics import CovarianceChi2
        return CovarianceChi2
    raise AttributeError(name)

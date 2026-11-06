"""Sherpa statistic for fractionally rebinned, correlated count spectra."""
import numpy as np
from scipy import sparse
from sherpa.models.model import SimulFitModel
from sherpa.stats import Stat

from .covariance import CovarianceWhitening
from .grid import _indices


class CovarianceChi2(Stat):
    """Gaussian chi-square using the full propagated count covariance."""
    _can_calculate_rstat = True

    def __init__(self, data, covariance):
        self.data = data
        self.covariance = sparse.csr_matrix(covariance)
        self._whiteners = {}
        super().__init__("rebin_covariance_chi2")

    @staticmethod
    def calc_staterror(data):
        return np.sqrt(np.maximum(data, 1.))

    def calc_stat(self, data, model):
        datasets = getattr(data, "datasets", (data,))
        models = model.parts if isinstance(model, SimulFitModel) else (model,)
        if len(datasets) != 1 or datasets[0] is not self.data or len(models) != 1:
            raise ValueError("this covariance statistic belongs to its combined dataset; fit it alone")
        lo, hi = self.data.get_indep(filter=True)
        indices = _indices(self.data.xlo, lo)
        if np.any(indices != _indices(self.data.xhi, hi)):
            raise ValueError("fit bins do not match the combined covariance")
        key = indices.tobytes()
        if key not in self._whiteners:
            self._whiteners[key] = CovarianceWhitening(self.covariance[indices][:, indices])
        residual = self.data.get_dep(filter=True)-self.data.eval_model_to_fit(models[0])
        whitened = self._whiteners[key](residual)
        return float(whitened @ whitened), whitened

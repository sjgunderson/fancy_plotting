"""Sparse, banded Gaussian residual whitening for rebinned spectra."""
import numpy as np
from scipy import sparse
from scipy.linalg import cholesky_banded, solve_banded


class CovarianceWhitening:
    def __init__(self, covariance):
        matrix = sparse.csr_matrix(covariance, dtype=float)
        if matrix.shape[0] != matrix.shape[1] or not matrix.shape[0]:
            raise ValueError("covariance must be nonempty and square")
        if not np.isfinite(matrix.data).all():
            raise ValueError("covariance must be finite")
        difference = matrix-matrix.T
        if difference.nnz and np.max(abs(difference.data)) > 1e-10 * max(1., abs(matrix.data).max()):
            raise ValueError("covariance must be symmetric")
        lower = sparse.tril(matrix, format="coo")
        self.size = matrix.shape[0]
        self.bandwidth = int(np.max(lower.row-lower.col)) if lower.nnz else 0
        if (self.bandwidth+1)*self.size > 20_000_000:
            raise ValueError("covariance bandwidth is too large; use a coarser reference grid")
        bands = np.zeros((self.bandwidth+1, self.size))
        bands[lower.row-lower.col, lower.col] = lower.data
        try:
            self.factor = cholesky_banded(bands, lower=True, check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise ValueError("rebinned covariance is singular; use the reference native grid "
                             "or a coarser grid for fitting") from exc

    def __call__(self, residual):
        residual = np.asarray(residual, dtype=float)
        if residual.shape != (self.size,):
            raise ValueError("residual must match covariance size")
        return solve_banded((self.bandwidth, 0), self.factor, residual, check_finite=True)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.base import FloatArray, KernelEvaluation
from .scalar import _median_heuristic, _validate_particles


def _as_metric_matrix(metric: FloatArray, dimension: int) -> FloatArray:
    array = np.asarray(metric, dtype=float)
    if array.ndim == 1:
        if array.shape[0] != dimension:
            raise ValueError("Diagonal metric must match the particle dimension.")
        if np.any(array <= 0.0):
            raise ValueError("Diagonal metric entries must be positive.")
        return np.diag(array)
    if array.shape != (dimension, dimension):
        raise ValueError("Metric must have shape (d,) or (d, d).")
    eigenvalues = np.linalg.eigvalsh(0.5 * (array + array.T))
    if np.any(eigenvalues <= 0.0):
        raise ValueError("Metric must be positive definite.")
    return 0.5 * (array + array.T)


@dataclass(frozen=True)
class MatrixRBFKernel:
    """Matrix-valued RBF kernel with a fixed positive definite metric."""

    metric: FloatArray | None = None
    bandwidth: float | None = None

    def evaluate(
        self,
        source_particles: FloatArray,
        target_particles: FloatArray,
    ) -> KernelEvaluation:
        _validate_particles(source_particles, target_particles)
        dimension = source_particles.shape[1]
        if self.metric is None:
            metric = np.eye(dimension, dtype=float)
        else:
            metric = _as_metric_matrix(self.metric, dimension)
        diff = source_particles[:, None, :] - target_particles[None, :, :]
        mahalanobis_distance = np.einsum(
            "sad,df,saf->sa",
            diff,
            metric,
            diff,
            optimize=True,
        )
        particle_count = max(source_particles.shape[0], target_particles.shape[0])
        bandwidth = (
            float(self.bandwidth)
            if self.bandwidth is not None
            else _median_heuristic(mahalanobis_distance, particle_count)
        )
        if bandwidth <= 0.0:
            raise ValueError("bandwidth must be positive.")

        scalar_matrix = np.exp(-mahalanobis_distance / bandwidth)
        metric_diff = np.einsum("ij,saj->sai", metric, diff, optimize=True)
        scalar_grad = (-2.0 / bandwidth) * metric_diff * scalar_matrix[:, :, None]
        divergence = np.einsum("ij,saj->sai", metric, scalar_grad, optimize=True)
        operator = scalar_matrix[:, :, None, None] * metric[None, None, :, :]
        return KernelEvaluation(
            matrix=None,
            grad_source=divergence,
            operator=operator,
            bandwidth=bandwidth,
        )

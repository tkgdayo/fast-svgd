from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core.base import FloatArray, HessianArray, HessianFunction, ScoreFunction


@dataclass(frozen=True)
class FunctionTarget:
    """Bind score and optional Hessian callables into a target object."""

    score_function: ScoreFunction
    hessian_function: HessianFunction | None = None

    def score(self, particles: FloatArray) -> FloatArray:
        return np.asarray(self.score_function(particles), dtype=float)

    def hessian(self, particles: FloatArray) -> HessianArray:
        if self.hessian_function is None:
            raise ValueError("This target does not provide a hessian_function.")
        return np.asarray(self.hessian_function(particles), dtype=float)


@dataclass(frozen=True)
class GaussianTarget:
    """Multivariate Gaussian target with exact score and Hessian."""

    mean: FloatArray
    covariance: FloatArray | None = None
    precision: FloatArray | None = None

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        if mean.ndim != 1:
            raise ValueError("mean must be a one-dimensional array.")
        if (self.covariance is None) == (self.precision is None):
            raise ValueError("Provide exactly one of covariance or precision.")

        if self.precision is not None:
            precision = np.asarray(self.precision, dtype=float)
        else:
            covariance = np.asarray(self.covariance, dtype=float)
            precision = np.linalg.inv(covariance)

        if precision.shape != (mean.shape[0], mean.shape[0]):
            raise ValueError("precision/covariance must match the mean dimension.")
        precision = 0.5 * (precision + precision.T)

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "covariance", None)

    def score(self, particles: FloatArray) -> FloatArray:
        centered = np.asarray(particles, dtype=float) - self.mean[None, :]
        return -(centered @ self.precision)

    def hessian(self, particles: FloatArray) -> HessianArray:
        return np.repeat((-self.precision)[None, :, :], particles.shape[0], axis=0)

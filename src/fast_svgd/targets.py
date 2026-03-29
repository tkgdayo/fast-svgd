from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core.base import FloatArray, HessianArray, HessianFunction, ScoreFunction


def _as_diagonal_covariance(
    values: FloatArray,
    *,
    mean_dimension: int,
    name: str,
) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        scalar = float(array)
        if scalar <= 0.0:
            raise ValueError(f"{name} must be positive.")
        return np.eye(mean_dimension, dtype=float) * scalar
    if array.ndim == 1:
        if array.shape[0] != mean_dimension:
            raise ValueError(f"{name} must match the mean dimension.")
        if np.any(array <= 0.0):
            raise ValueError(f"{name} entries must be positive.")
        return np.diag(array)
    raise ValueError(f"{name} must be a scalar or one-dimensional array.")


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
    std: FloatArray | None = None
    var: FloatArray | None = None
    covariance: FloatArray | None = None
    precision: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        if mean.ndim != 1:
            raise ValueError("mean must be a one-dimensional array.")
        provided = sum(
            option is not None for option in (self.std, self.var, self.covariance)
        )
        if provided != 1:
            raise ValueError("Provide exactly one of std, var, or covariance.")

        if self.std is not None:
            std = np.asarray(self.std, dtype=float)
            covariance = _as_diagonal_covariance(
                std * std,
                mean_dimension=mean.shape[0],
                name="std",
            )
        elif self.var is not None:
            covariance = _as_diagonal_covariance(
                self.var,
                mean_dimension=mean.shape[0],
                name="var",
            )
        else:
            covariance = np.asarray(self.covariance, dtype=float)
            if covariance.shape != (mean.shape[0], mean.shape[0]):
                raise ValueError("covariance must match the mean dimension.")
            covariance = 0.5 * (covariance + covariance.T)
            eigenvalues = np.linalg.eigvalsh(covariance)
            if np.any(eigenvalues <= 0.0):
                raise ValueError("covariance must be positive definite.")

        precision = np.linalg.inv(covariance)

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "precision", precision)

    def score(self, particles: FloatArray) -> FloatArray:
        centered = np.asarray(particles, dtype=float) - self.mean[None, :]
        return -(centered @ self.precision)

    def hessian(self, particles: FloatArray) -> HessianArray:
        return np.repeat((-self.precision)[None, :, :], particles.shape[0], axis=0)

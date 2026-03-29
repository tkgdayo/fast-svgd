from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.base import FloatArray, KernelEvaluation

_MIN_BANDWIDTH = 1e-12


def _validate_particles(source_particles: FloatArray, target_particles: FloatArray) -> None:
    if source_particles.ndim != 2 or target_particles.ndim != 2:
        raise ValueError("Particles must be two-dimensional arrays of shape (n, d).")
    if source_particles.shape[1] != target_particles.shape[1]:
        raise ValueError("Source and target particles must share the same dimension.")


def _squared_distances(
    source_particles: FloatArray,
    target_particles: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    diff = source_particles[:, None, :] - target_particles[None, :, :]
    squared_distance = np.sum(diff * diff, axis=-1)
    return diff, squared_distance


def _median_heuristic(squared_distance: FloatArray, particle_count: int) -> float:
    positive = squared_distance[squared_distance > 0.0]
    if positive.size == 0:
        return 1.0
    bandwidth = float(np.median(positive)) / np.log(particle_count + 1.0)
    return max(bandwidth, _MIN_BANDWIDTH)


@dataclass(frozen=True)
class RBFKernel:
    """Exponentiated quadratic kernel with optional median heuristic bandwidth."""

    bandwidth: float | None = None

    def evaluate(
        self,
        source_particles: FloatArray,
        target_particles: FloatArray,
    ) -> KernelEvaluation:
        _validate_particles(source_particles, target_particles)
        diff, squared_distance = _squared_distances(source_particles, target_particles)
        particle_count = max(source_particles.shape[0], target_particles.shape[0])
        bandwidth = (
            float(self.bandwidth)
            if self.bandwidth is not None
            else _median_heuristic(squared_distance, particle_count)
        )
        if bandwidth <= 0.0:
            raise ValueError("bandwidth must be positive.")

        matrix = np.exp(-squared_distance / bandwidth)
        grad_source = (-2.0 / bandwidth) * diff * matrix[:, :, None]
        return KernelEvaluation(
            matrix=matrix,
            grad_source=grad_source,
            operator=None,
            bandwidth=bandwidth,
        )


@dataclass(frozen=True)
class IMQKernel:
    """Inverse multiquadric kernel often used in particle methods."""

    scale: float = 1.0
    beta: float = -0.5
    offset: float = 1.0

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError("scale must be positive.")
        if self.offset <= 0.0:
            raise ValueError("offset must be positive.")

    def evaluate(
        self,
        source_particles: FloatArray,
        target_particles: FloatArray,
    ) -> KernelEvaluation:
        _validate_particles(source_particles, target_particles)
        diff, squared_distance = _squared_distances(source_particles, target_particles)
        base = self.offset + squared_distance / self.scale
        matrix = np.power(base, self.beta)
        grad_source = (
            (2.0 * self.beta / self.scale)
            * np.power(base, self.beta - 1.0)[:, :, None]
            * diff
        )
        return KernelEvaluation(
            matrix=matrix,
            grad_source=grad_source,
            operator=None,
            bandwidth=None,
        )


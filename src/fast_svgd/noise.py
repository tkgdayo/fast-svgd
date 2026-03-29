from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import FloatArray


@dataclass(frozen=True)
class NoNoise:
    def sample(
        self,
        particles: FloatArray,
        step_size: float,
        rng: np.random.Generator,
    ) -> FloatArray:
        del step_size, rng
        return np.zeros_like(particles)


@dataclass(frozen=True)
class GaussianNoise:
    """Gaussian perturbation used by SPOS-style particle updates."""

    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative.")

    def sample(
        self,
        particles: FloatArray,
        step_size: float,
        rng: np.random.Generator,
    ) -> FloatArray:
        if step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        if self.temperature == 0.0:
            return np.zeros_like(particles)
        scale = np.sqrt(2.0 * self.temperature * step_size)
        return rng.normal(loc=0.0, scale=scale, size=particles.shape)


SPOSNoise = GaussianNoise


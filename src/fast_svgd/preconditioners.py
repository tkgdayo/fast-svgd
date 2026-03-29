from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import FloatArray


@dataclass(frozen=True)
class IdentityPreconditioner:
    def apply(
        self,
        particles: FloatArray,
        update: FloatArray,
        scores: FloatArray,
    ) -> FloatArray:
        del particles, scores
        return update


@dataclass(frozen=True)
class DiagonalPreconditioner:
    """Multiply updates by a fixed diagonal scaling."""

    diagonal: FloatArray

    def apply(
        self,
        particles: FloatArray,
        update: FloatArray,
        scores: FloatArray,
    ) -> FloatArray:
        del particles, scores
        diagonal = np.asarray(self.diagonal, dtype=float)
        if diagonal.ndim == 1:
            if diagonal.shape[0] != update.shape[1]:
                raise ValueError("One-dimensional diagonal must match particle dimension.")
            return update * diagonal[None, :]
        if diagonal.shape != update.shape:
            raise ValueError("Two-dimensional diagonal must match the update shape.")
        return update * diagonal


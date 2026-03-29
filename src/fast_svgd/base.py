from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ScoreFunction = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True)
class KernelEvaluation:
    matrix: FloatArray
    grad_source: FloatArray
    bandwidth: float | None = None


@dataclass(frozen=True)
class InteractionBlock:
    active_indices: NDArray[np.int64]
    source_indices: NDArray[np.int64]


class Kernel(Protocol):
    def evaluate(
        self,
        source_particles: FloatArray,
        target_particles: FloatArray,
    ) -> KernelEvaluation: ...


class InteractionApproximator(Protocol):
    def blocks(
        self,
        n_particles: int,
        rng: np.random.Generator,
    ) -> tuple[InteractionBlock, ...]: ...


class Preconditioner(Protocol):
    def apply(
        self,
        particles: FloatArray,
        update: FloatArray,
        scores: FloatArray,
    ) -> FloatArray: ...


class NoiseInjector(Protocol):
    def sample(
        self,
        particles: FloatArray,
        step_size: float,
        rng: np.random.Generator,
    ) -> FloatArray: ...


@dataclass(frozen=True)
class StepDiagnostics:
    mean_score_norm: float
    mean_update_norm: float
    max_update_norm: float
    mean_particle_norm: float


@dataclass(frozen=True)
class StepResult:
    particles: FloatArray
    update: FloatArray
    scores: FloatArray
    noise: FloatArray
    diagnostics: StepDiagnostics


@dataclass(frozen=True)
class RunResult:
    particles: FloatArray
    diagnostics: tuple[StepDiagnostics, ...]
    trajectory: tuple[FloatArray, ...]


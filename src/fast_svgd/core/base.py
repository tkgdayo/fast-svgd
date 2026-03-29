from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
HessianArray = NDArray[np.float64]
ScoreFunction = Callable[[FloatArray], FloatArray]
HessianFunction = Callable[[FloatArray], HessianArray]


@dataclass(frozen=True)
class KernelEvaluation:
    grad_source: FloatArray
    matrix: FloatArray | None = None
    operator: NDArray[np.float64] | None = None
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
        particles: FloatArray,
        rng: np.random.Generator,
    ) -> tuple[InteractionBlock, ...]: ...


class Target(Protocol):
    def score(self, particles: FloatArray) -> FloatArray: ...


class SecondOrderTarget(Target, Protocol):
    def hessian(self, particles: FloatArray) -> HessianArray: ...


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
    particle_spread: float
    mean_step_displacement: float
    max_step_displacement: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mean_score_norm": self.mean_score_norm,
            "mean_update_norm": self.mean_update_norm,
            "max_update_norm": self.max_update_norm,
            "mean_particle_norm": self.mean_particle_norm,
            "particle_spread": self.particle_spread,
            "mean_step_displacement": self.mean_step_displacement,
            "max_step_displacement": self.max_step_displacement,
        }


@dataclass(frozen=True)
class StepResult:
    particles: FloatArray
    update: FloatArray
    scores: FloatArray
    noise: FloatArray
    diagnostics: StepDiagnostics


@dataclass(frozen=True)
class RunResult:
    initial_particles: FloatArray
    particles: FloatArray
    diagnostics: tuple[StepDiagnostics, ...]
    trajectory: tuple[FloatArray, ...]

    @property
    def has_trajectory(self) -> bool:
        return len(self.trajectory) >= 2

    def summary(self) -> Any:
        from ..diagnostics import summarize_run

        return summarize_run(self)

    def diagnostic_series(self) -> Any:
        from ..diagnostics import build_diagnostic_series

        return build_diagnostic_series(self)

    def plot_diagnostics(self, *, axes: Any = None) -> Any:
        from ..diagnostics import plot_diagnostics

        return plot_diagnostics(self, axes=axes)

    def plot_particles(
        self,
        *,
        ax: Any = None,
        step: int = -1,
        dims: tuple[int, int] | None = None,
        bins: int = 40,
    ) -> Any:
        from ..diagnostics import plot_particles

        return plot_particles(
            self,
            ax=ax,
            step=step,
            dims=dims,
            bins=bins,
        )

    def plot_particle_paths(
        self,
        *,
        ax: Any = None,
        dims: tuple[int, int] = (0, 1),
        max_particles: int = 128,
        alpha: float = 0.25,
    ) -> Any:
        from ..diagnostics import plot_particle_paths

        return plot_particle_paths(
            self,
            ax=ax,
            dims=dims,
            max_particles=max_particles,
            alpha=alpha,
        )

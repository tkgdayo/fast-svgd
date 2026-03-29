from __future__ import annotations

import numpy as np

from ..core.base import (
    FloatArray,
    HessianFunction,
    Kernel,
    RunResult,
    ScoreFunction,
    StepDiagnostics,
    StepResult,
)
from ..core.engine import (
    FastSVGD,
    _as_hessian_array,
    _as_particle_array,
    _as_score_array,
    _build_diagnostics,
    _score_term_from_evaluation,
    _validate_kernel_evaluation,
)
from ..interactions import FullBatchInteraction
from ..kernels import MatrixRBFKernel, RBFKernel
from ..noise import NoNoise
from ..preconditioners import IdentityPreconditioner


def _regularize_positive_matrix(matrix: FloatArray, jitter: float) -> FloatArray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, jitter)
    return (eigenvectors * clipped[None, :]) @ eigenvectors.T


class MatrixSVGD(FastSVGD):
    def __init__(
        self,
        *,
        metric: FloatArray | None = None,
        kernel: Kernel | None = None,
    ) -> None:
        super().__init__(
            kernel=kernel if kernel is not None else MatrixRBFKernel(metric=metric),
            interaction=FullBatchInteraction(),
            preconditioner=IdentityPreconditioner(),
            noise=NoNoise(),
        )


class SteinVariationalNewton:
    """Block-diagonal Stein variational Newton approximation.

    This implementation follows the local block solve spirit of Eq. (17) in
    Detommaso et al. (2018): each particle uses a curvature matrix built from
    Hessian information and kernel gradients, then solves a local Newton system.
    """

    def __init__(
        self,
        *,
        kernel: Kernel | None = None,
        regularization: float = 1e-6,
    ) -> None:
        self.kernel = kernel if kernel is not None else RBFKernel()
        self.regularization = float(regularization)
        if self.regularization <= 0.0:
            raise ValueError("regularization must be positive.")

    def _base_drift(
        self,
        particles: FloatArray,
        scores: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        evaluation = self.kernel.evaluate(particles, particles)
        n_particles, dimension = particles.shape
        _validate_kernel_evaluation(
            evaluation,
            source_size=n_particles,
            active_size=n_particles,
            dimension=dimension,
        )
        if evaluation.matrix is None:
            raise ValueError(
                "SteinVariationalNewton currently requires a scalar-valued kernel."
            )
        score_term = _score_term_from_evaluation(evaluation, scores) / n_particles
        repulsive_term = evaluation.grad_source.mean(axis=0)
        return score_term + repulsive_term, evaluation.matrix, evaluation.grad_source

    def _local_newton_direction(
        self,
        particles: FloatArray,
        scores: FloatArray,
        hessians: FloatArray,
    ) -> FloatArray:
        base_drift, kernel_matrix, kernel_gradients = self._base_drift(particles, scores)
        n_particles, dimension = particles.shape
        stabilized_precisions = np.empty_like(hessians)
        for particle_index in range(n_particles):
            stabilized_precisions[particle_index] = _regularize_positive_matrix(
                -hessians[particle_index],
                self.regularization,
            )

        direction = np.zeros_like(base_drift)
        identity = np.eye(dimension, dtype=float)
        for target_index in range(n_particles):
            local_hessian = np.zeros((dimension, dimension), dtype=float)
            for source_index in range(n_particles):
                gradient = kernel_gradients[source_index, target_index]
                local_hessian += (
                    stabilized_precisions[source_index]
                    * kernel_matrix[source_index, target_index] ** 2
                    + np.outer(gradient, gradient)
                )
            local_hessian /= n_particles
            local_hessian += self.regularization * identity
            direction[target_index] = np.linalg.solve(
                local_hessian,
                base_drift[target_index],
            )

        return direction

    def step(
        self,
        particles: FloatArray,
        score_function: ScoreFunction,
        *,
        hessian_function: HessianFunction,
        step_size: float,
        rng: np.random.Generator | None = None,
    ) -> StepResult:
        del rng
        if step_size <= 0.0:
            raise ValueError("step_size must be positive.")

        particles_array = _as_particle_array(particles)
        scores = _as_score_array(
            score_function(particles_array),
            particles_array.shape,
        )
        hessians = _as_hessian_array(
            hessian_function(particles_array),
            particles_array.shape,
        )
        drift = self._local_newton_direction(particles_array, scores, hessians)
        noise = np.zeros_like(particles_array)
        next_particles = particles_array + step_size * drift
        diagnostics = _build_diagnostics(next_particles, scores, drift)
        return StepResult(
            particles=next_particles,
            update=drift,
            scores=scores,
            noise=noise,
            diagnostics=diagnostics,
        )

    def run(
        self,
        particles: FloatArray,
        score_function: ScoreFunction,
        *,
        hessian_function: HessianFunction,
        n_steps: int,
        step_size: float,
        seed: int | None = None,
        store_trajectory: bool = False,
    ) -> RunResult:
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative.")

        generator = np.random.default_rng(seed)
        current_particles = _as_particle_array(particles).copy()
        diagnostics: list[StepDiagnostics] = []
        trajectory: list[FloatArray] = []

        if store_trajectory:
            trajectory.append(current_particles.copy())

        for _ in range(n_steps):
            step_result = self.step(
                current_particles,
                score_function,
                hessian_function=hessian_function,
                step_size=step_size,
                rng=generator,
            )
            current_particles = step_result.particles
            diagnostics.append(step_result.diagnostics)
            if store_trajectory:
                trajectory.append(current_particles.copy())

        return RunResult(
            particles=current_particles,
            diagnostics=tuple(diagnostics),
            trajectory=tuple(trajectory),
        )


SVN = SteinVariationalNewton

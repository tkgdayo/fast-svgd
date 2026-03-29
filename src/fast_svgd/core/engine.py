from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..interactions import FullBatchInteraction
from ..noise import NoNoise
from ..preconditioners import IdentityPreconditioner
from .base import (
    FloatArray,
    HessianArray,
    InteractionApproximator,
    Kernel,
    KernelEvaluation,
    NoiseInjector,
    Preconditioner,
    RunResult,
    ScoreFunction,
    StepDiagnostics,
    StepResult,
)


def _as_particle_array(particles: FloatArray) -> FloatArray:
    array = np.asarray(particles, dtype=float)
    if array.ndim != 2:
        raise ValueError("particles must have shape (n_particles, dimension).")
    if array.shape[0] == 0:
        raise ValueError("At least one particle is required.")
    return array


def _as_score_array(scores: FloatArray, shape: tuple[int, int]) -> FloatArray:
    array = np.asarray(scores, dtype=float)
    if array.shape != shape:
        raise ValueError(
            "score_function must return an array with the same shape as particles."
        )
    return array


def _as_hessian_array(hessians: HessianArray, shape: tuple[int, int]) -> HessianArray:
    array = np.asarray(hessians, dtype=float)
    expected_shape = (shape[0], shape[1], shape[1])
    if array.shape != expected_shape:
        raise ValueError(
            "hessian_function must return an array with shape "
            f"{expected_shape}, got {array.shape}."
        )
    return array


def _build_diagnostics(
    previous_particles: FloatArray,
    particles: FloatArray,
    scores: FloatArray,
    update: FloatArray,
) -> StepDiagnostics:
    score_norms = np.linalg.norm(scores, axis=1)
    update_norms = np.linalg.norm(update, axis=1)
    particle_norms = np.linalg.norm(particles, axis=1)
    centered_particles = particles - np.mean(particles, axis=0, keepdims=True)
    particle_spread = np.linalg.norm(centered_particles, axis=1)
    step_displacement = np.linalg.norm(particles - previous_particles, axis=1)
    return StepDiagnostics(
        mean_score_norm=float(np.mean(score_norms)),
        mean_update_norm=float(np.mean(update_norms)),
        max_update_norm=float(np.max(update_norms)),
        mean_particle_norm=float(np.mean(particle_norms)),
        particle_spread=float(np.mean(particle_spread)),
        mean_step_displacement=float(np.mean(step_displacement)),
        max_step_displacement=float(np.max(step_displacement)),
    )


def _validate_kernel_evaluation(
    evaluation: KernelEvaluation,
    *,
    source_size: int,
    active_size: int,
    dimension: int,
) -> None:
    expected_grad_shape = (source_size, active_size, dimension)
    if evaluation.grad_source.shape != expected_grad_shape:
        raise ValueError(
            "Kernel divergence has incompatible shape. "
            f"Expected {expected_grad_shape}, got {evaluation.grad_source.shape}."
        )

    if evaluation.operator is not None:
        expected_operator_shape = (source_size, active_size, dimension, dimension)
        if evaluation.operator.shape != expected_operator_shape:
            raise ValueError(
                "Kernel operator has incompatible shape. "
                f"Expected {expected_operator_shape}, got {evaluation.operator.shape}."
            )
    elif evaluation.matrix is not None:
        expected_kernel_shape = (source_size, active_size)
        if evaluation.matrix.shape != expected_kernel_shape:
            raise ValueError(
                "Kernel matrix has incompatible shape. "
                f"Expected {expected_kernel_shape}, got {evaluation.matrix.shape}."
            )
    else:
        raise ValueError("Kernel evaluation must provide either a matrix or an operator.")


def _score_term_from_evaluation(
    evaluation: KernelEvaluation,
    source_scores: FloatArray,
) -> FloatArray:
    if evaluation.operator is not None:
        return np.einsum(
            "sadb,sb->ad",
            evaluation.operator,
            source_scores,
            optimize=True,
        )
    if evaluation.matrix is not None:
        return np.einsum(
            "sa,sd->ad",
            evaluation.matrix,
            source_scores,
            optimize=True,
        )
    raise ValueError("Kernel evaluation must provide either a matrix or an operator.")


@dataclass
class FastSVGD:
    """Composable particle solver for SVGD-style updates."""

    kernel: Kernel
    interaction: InteractionApproximator = field(default_factory=FullBatchInteraction)
    preconditioner: Preconditioner = field(default_factory=IdentityPreconditioner)
    noise: NoiseInjector = field(default_factory=NoNoise)

    def _raw_drift(
        self,
        particles: FloatArray,
        scores: FloatArray,
        rng: np.random.Generator,
    ) -> FloatArray:
        n_particles, dimension = particles.shape
        update = np.zeros_like(particles)
        update_counts = np.zeros((n_particles, 1), dtype=float)

        for block in self.interaction.blocks(particles, rng):
            active_indices = np.asarray(block.active_indices, dtype=np.int64)
            source_indices = np.asarray(block.source_indices, dtype=np.int64)

            if active_indices.ndim != 1 or source_indices.ndim != 1:
                raise ValueError("Interaction indices must be one-dimensional.")
            if active_indices.size == 0 or source_indices.size == 0:
                raise ValueError("Interaction blocks must not be empty.")

            active_particles = particles[active_indices]
            source_particles = particles[source_indices]
            evaluation = self.kernel.evaluate(source_particles, active_particles)
            _validate_kernel_evaluation(
                evaluation,
                source_size=source_indices.size,
                active_size=active_indices.size,
                dimension=dimension,
            )

            score_term = (
                _score_term_from_evaluation(evaluation, scores[source_indices])
                / source_indices.size
            )
            repulsive_term = evaluation.grad_source.mean(axis=0)
            block_update = score_term + repulsive_term

            update[active_indices] += block_update
            update_counts[active_indices] += 1.0

        if np.any(update_counts == 0.0):
            raise ValueError(
                "interaction approximator must update every particle at least once."
            )

        return update / update_counts

    def _drift(
        self,
        particles: FloatArray,
        scores: FloatArray,
        rng: np.random.Generator,
    ) -> FloatArray:
        raw_update = self._raw_drift(particles, scores, rng)
        return self.preconditioner.apply(particles, raw_update, scores)

    def step(
        self,
        particles: FloatArray,
        score_function: ScoreFunction,
        *,
        step_size: float,
        rng: np.random.Generator | None = None,
    ) -> StepResult:
        if step_size <= 0.0:
            raise ValueError("step_size must be positive.")

        particles_array = _as_particle_array(particles)
        scores = _as_score_array(
            score_function(particles_array),
            particles_array.shape,
        )
        generator = np.random.default_rng() if rng is None else rng
        drift = np.asarray(self._drift(particles_array, scores, generator), dtype=float)
        if drift.shape != particles_array.shape:
            raise ValueError("preconditioner must preserve the particle shape.")

        noise = np.asarray(
            self.noise.sample(particles_array, step_size, generator),
            dtype=float,
        )
        if noise.shape != particles_array.shape:
            raise ValueError("noise injector must preserve the particle shape.")

        next_particles = particles_array + step_size * drift + noise
        diagnostics = _build_diagnostics(
            particles_array,
            next_particles,
            scores,
            drift,
        )
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
                step_size=step_size,
                rng=generator,
            )
            current_particles = step_result.particles
            diagnostics.append(step_result.diagnostics)
            if store_trajectory:
                trajectory.append(current_particles.copy())

        return RunResult(
            initial_particles=_as_particle_array(particles).copy(),
            particles=current_particles,
            diagnostics=tuple(diagnostics),
            trajectory=tuple(trajectory),
        )

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .core.base import FloatArray, RunResult

IntArray = NDArray[np.int64]


def _empty_float_array() -> FloatArray:
    return np.array([], dtype=float)


def _particle_spread(particles: FloatArray) -> float:
    centered = particles - np.mean(particles, axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(centered, axis=1)))


def _trajectory_stack(result: RunResult) -> FloatArray:
    if len(result.trajectory) < 2:
        raise ValueError(
            "Trajectory diagnostics require store_trajectory=True when running the solver."
        )
    return np.stack(result.trajectory, axis=0)


def _motion_arrays(result: RunResult) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    diagnostics = result.diagnostics
    mean_step_displacement = np.array(
        [item.mean_step_displacement for item in diagnostics],
        dtype=float,
    )
    max_step_displacement = np.array(
        [item.max_step_displacement for item in diagnostics],
        dtype=float,
    )
    cumulative_mean_path_length = np.cumsum(mean_step_displacement)
    particle_spread = np.array(
        [item.particle_spread for item in diagnostics],
        dtype=float,
    )
    return (
        mean_step_displacement,
        max_step_displacement,
        cumulative_mean_path_length,
        particle_spread,
    )


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting helpers require matplotlib. Install the optional 'plot' extra."
        ) from exc
    return plt


def _particles_for_step(result: RunResult, step: int) -> FloatArray:
    if step == -1:
        return result.particles
    if not result.trajectory:
        raise ValueError(
            "Selecting a specific step requires store_trajectory=True when running the solver."
        )
    return np.asarray(result.trajectory[step], dtype=float)


def _initial_to_final_displacement(result: RunResult) -> FloatArray:
    return result.particles - result.initial_particles


@dataclass(frozen=True)
class DiagnosticSeries:
    step: IntArray
    mean_score_norm: FloatArray
    mean_update_norm: FloatArray
    max_update_norm: FloatArray
    mean_particle_norm: FloatArray
    mean_step_displacement: FloatArray
    max_step_displacement: FloatArray
    cumulative_mean_path_length: FloatArray
    particle_spread: FloatArray

    def as_dict(self) -> dict[str, FloatArray | IntArray]:
        return {
            "step": self.step,
            "mean_score_norm": self.mean_score_norm,
            "mean_update_norm": self.mean_update_norm,
            "max_update_norm": self.max_update_norm,
            "mean_particle_norm": self.mean_particle_norm,
            "mean_step_displacement": self.mean_step_displacement,
            "max_step_displacement": self.max_step_displacement,
            "cumulative_mean_path_length": self.cumulative_mean_path_length,
            "particle_spread": self.particle_spread,
        }


@dataclass(frozen=True)
class RunSummary:
    n_steps: int
    mean_score_norm_last: float | None
    mean_update_norm_last: float | None
    max_update_norm_last: float | None
    mean_particle_norm_last: float | None
    final_particle_spread: float
    mean_final_displacement: float | None
    max_final_displacement: float | None
    mean_path_length: float | None
    max_path_length: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "n_steps": self.n_steps,
            "mean_score_norm_last": self.mean_score_norm_last,
            "mean_update_norm_last": self.mean_update_norm_last,
            "max_update_norm_last": self.max_update_norm_last,
            "mean_particle_norm_last": self.mean_particle_norm_last,
            "final_particle_spread": self.final_particle_spread,
            "mean_final_displacement": self.mean_final_displacement,
            "max_final_displacement": self.max_final_displacement,
            "mean_path_length": self.mean_path_length,
            "max_path_length": self.max_path_length,
        }


def build_diagnostic_series(result: RunResult) -> DiagnosticSeries:
    diagnostics = result.diagnostics
    step = np.arange(1, len(diagnostics) + 1, dtype=np.int64)
    mean_score_norm = np.array(
        [item.mean_score_norm for item in diagnostics],
        dtype=float,
    )
    mean_update_norm = np.array(
        [item.mean_update_norm for item in diagnostics],
        dtype=float,
    )
    max_update_norm = np.array(
        [item.max_update_norm for item in diagnostics],
        dtype=float,
    )
    mean_particle_norm = np.array(
        [item.mean_particle_norm for item in diagnostics],
        dtype=float,
    )
    (
        mean_step_displacement,
        max_step_displacement,
        cumulative_mean_path_length,
        particle_spread,
    ) = _motion_arrays(result)
    return DiagnosticSeries(
        step=step,
        mean_score_norm=mean_score_norm,
        mean_update_norm=mean_update_norm,
        max_update_norm=max_update_norm,
        mean_particle_norm=mean_particle_norm,
        mean_step_displacement=mean_step_displacement,
        max_step_displacement=max_step_displacement,
        cumulative_mean_path_length=cumulative_mean_path_length,
        particle_spread=particle_spread,
    )


def summarize_run(result: RunResult) -> RunSummary:
    last = result.diagnostics[-1] if result.diagnostics else None
    final_particle_spread = _particle_spread(result.particles)
    final_displacements = np.linalg.norm(_initial_to_final_displacement(result), axis=1)
    mean_final_displacement = float(np.mean(final_displacements))
    max_final_displacement = float(np.max(final_displacements))
    if result.diagnostics:
        mean_path_length = float(
            np.sum([item.mean_step_displacement for item in result.diagnostics])
        )
        max_path_length = float(
            np.sum([item.max_step_displacement for item in result.diagnostics])
        )
    else:
        mean_path_length = 0.0
        max_path_length = 0.0

    return RunSummary(
        n_steps=len(result.diagnostics),
        mean_score_norm_last=None if last is None else last.mean_score_norm,
        mean_update_norm_last=None if last is None else last.mean_update_norm,
        max_update_norm_last=None if last is None else last.max_update_norm,
        mean_particle_norm_last=None if last is None else last.mean_particle_norm,
        final_particle_spread=final_particle_spread,
        mean_final_displacement=mean_final_displacement,
        max_final_displacement=max_final_displacement,
        mean_path_length=mean_path_length,
        max_path_length=max_path_length,
    )


def plot_diagnostics(result: RunResult, *, axes: Any = None) -> Any:
    plt = _require_matplotlib()
    series = build_diagnostic_series(result)

    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    else:
        fig = axes[0].figure

    score_ax, motion_ax = axes
    score_ax.plot(series.step, series.mean_score_norm, label="mean_score_norm")
    score_ax.plot(series.step, series.mean_update_norm, label="mean_update_norm")
    score_ax.plot(series.step, series.max_update_norm, label="max_update_norm")
    score_ax.set_ylabel("norm")
    score_ax.set_title("Score and Update Diagnostics")
    score_ax.grid(alpha=0.2)
    score_ax.legend()

    motion_ax.plot(series.step, series.mean_particle_norm, label="mean_particle_norm")
    if series.mean_step_displacement.size:
        motion_ax.plot(
            series.step,
            series.mean_step_displacement,
            label="mean_step_displacement",
        )
        motion_ax.plot(
            series.step,
            series.cumulative_mean_path_length,
            label="cumulative_mean_path_length",
        )
        motion_ax.plot(series.step, series.particle_spread, label="particle_spread")
    motion_ax.set_xlabel("step")
    motion_ax.set_ylabel("magnitude")
    motion_ax.set_title("Particle Motion Diagnostics")
    motion_ax.grid(alpha=0.2)
    motion_ax.legend()

    fig.tight_layout()
    return fig, axes


def plot_particles(
    result: RunResult,
    *,
    ax: Any = None,
    step: int = -1,
    dims: tuple[int, int] | None = None,
    bins: int = 40,
) -> Any:
    plt = _require_matplotlib()
    particles = _particles_for_step(result, step)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    if particles.shape[1] == 1:
        ax.hist(particles[:, 0], bins=bins, alpha=0.85)
        ax.set_xlabel("particle value")
        ax.set_ylabel("count")
    else:
        active_dims = (0, 1) if dims is None else dims
        d0, d1 = active_dims
        if d0 == d1:
            raise ValueError("dims must refer to two different coordinates.")
        ax.scatter(particles[:, d0], particles[:, d1], s=18, alpha=0.8)
        ax.set_xlabel(f"dim {d0}")
        ax.set_ylabel(f"dim {d1}")
    ax.set_title("Particle Snapshot")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, ax


def plot_particle_paths(
    result: RunResult,
    *,
    ax: Any = None,
    dims: tuple[int, int] = (0, 1),
    max_particles: int = 128,
    alpha: float = 0.25,
) -> Any:
    plt = _require_matplotlib()
    if result.has_trajectory:
        stacked = _trajectory_stack(result)
    else:
        stacked = np.stack((result.initial_particles, result.particles), axis=0)

    if stacked.shape[2] == 1:
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        else:
            fig = ax.figure

        particle_count = stacked.shape[1]
        keep = min(max_particles, particle_count)
        particle_indices = np.linspace(0, particle_count - 1, num=keep, dtype=int)
        x = np.arange(stacked.shape[0], dtype=float)

        for particle_index in particle_indices:
            ax.plot(
                x,
                stacked[:, particle_index, 0],
                alpha=alpha,
                linewidth=1.0,
            )

        ax.set_xlabel("step")
        ax.set_ylabel("particle value")
        ax.set_title(
            "Particle Paths" if result.has_trajectory else "Initial-to-final particle displacement"
        )
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig, ax

    d0, d1 = dims
    if d0 == d1:
        raise ValueError("dims must refer to two different coordinates.")
    if d0 >= stacked.shape[2] or d1 >= stacked.shape[2]:
        raise ValueError("dims are out of range for the particle dimension.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    particle_count = stacked.shape[1]
    keep = min(max_particles, particle_count)
    particle_indices = np.linspace(0, particle_count - 1, num=keep, dtype=int)

    for particle_index in particle_indices:
        ax.plot(
            stacked[:, particle_index, d0],
            stacked[:, particle_index, d1],
            alpha=alpha,
            linewidth=1.0,
        )

    ax.scatter(
        stacked[0, particle_indices, d0],
        stacked[0, particle_indices, d1],
        s=20,
        alpha=0.35,
        label="initial",
    )
    ax.scatter(
        stacked[-1, particle_indices, d0],
        stacked[-1, particle_indices, d1],
        s=20,
        alpha=0.85,
        label="final",
    )
    ax.set_xlabel(f"dim {d0}")
    ax.set_ylabel(f"dim {d1}")
    ax.set_title(
        "Particle Paths" if result.has_trajectory else "Initial-to-final particle displacement"
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig, ax


__all__ = [
    "DiagnosticSeries",
    "RunSummary",
    "build_diagnostic_series",
    "summarize_run",
    "plot_diagnostics",
    "plot_particles",
    "plot_particle_paths",
]

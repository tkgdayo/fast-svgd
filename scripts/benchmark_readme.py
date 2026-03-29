from __future__ import annotations

from collections.abc import Callable
import platform
import statistics
import time
from dataclasses import dataclass

import numpy as np

from fast_svgd import (
    MatrixRBFKernel,
    MatrixSVGD,
    RandomBatchSVGD,
    RBFKernel,
    SPOS,
    SVGD,
    SteinVariationalNewton,
)

SEEDS = tuple(range(5))
STANDARD_NORMAL_PARTICLES = 256
STANDARD_NORMAL_SCALING_PARTICLES = (128, 256, 512)
STANDARD_NORMAL_STEP_SIZE = 0.03
STANDARD_NORMAL_STEPS = 250
STANDARD_NORMAL_INIT_MIN = -2.0
STANDARD_NORMAL_INIT_MAX = 2.0
RANDOM_BATCH_SIZE = 32
SPOS_TEMPERATURE = 0.01
ANISOTROPIC_PARTICLES = 64
ANISOTROPIC_INIT_MIN = -4.0
ANISOTROPIC_INIT_MAX = 4.0
ANISOTROPIC_PRECISION = np.diag([1.0, 25.0])


@dataclass(frozen=True)
class TrialSummary:
    mean: float
    variance: float
    wall_time_s: float


@dataclass(frozen=True)
class CurvatureTrialSummary:
    energy: float
    variance_x1: float
    variance_x2: float
    wall_time_s: float


@dataclass(frozen=True)
class SolverConfig:
    make: Callable[[], object]
    steps: int
    step_size: float


def standard_normal_score(particles: np.ndarray) -> np.ndarray:
    return -particles


def anisotropic_gaussian_score(particles: np.ndarray) -> np.ndarray:
    return -(particles @ ANISOTROPIC_PRECISION)


def anisotropic_gaussian_hessian(particles: np.ndarray) -> np.ndarray:
    return np.repeat((-ANISOTROPIC_PRECISION)[None, :, :], particles.shape[0], axis=0)


def initial_particles(
    seed: int,
    n_particles: int,
    *,
    dimension: int,
    lower: float,
    upper: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(lower, upper, size=(n_particles, dimension))


def standard_normal_solvers() -> dict[str, SolverConfig]:
    return {
        "SVGD": SolverConfig(
            make=lambda: SVGD(kernel=RBFKernel()),
            steps=STANDARD_NORMAL_STEPS,
            step_size=STANDARD_NORMAL_STEP_SIZE,
        ),
        "RandomBatchSVGD": SolverConfig(
            make=lambda: RandomBatchSVGD(
                kernel=RBFKernel(),
                batch_size=RANDOM_BATCH_SIZE,
            ),
            steps=STANDARD_NORMAL_STEPS,
            step_size=STANDARD_NORMAL_STEP_SIZE,
        ),
        "SPOS": SolverConfig(
            make=lambda: SPOS(
                kernel=RBFKernel(),
                temperature=SPOS_TEMPERATURE,
            ),
            steps=STANDARD_NORMAL_STEPS,
            step_size=STANDARD_NORMAL_STEP_SIZE,
        ),
    }


def anisotropic_solvers() -> dict[str, SolverConfig]:
    return {
        "SVGD": SolverConfig(
            make=lambda: SVGD(kernel=RBFKernel(bandwidth=5.0)),
            steps=40,
            step_size=0.03,
        ),
        "RandomBatchSVGD": SolverConfig(
            make=lambda: RandomBatchSVGD(
                batch_size=16,
                kernel=RBFKernel(bandwidth=5.0),
            ),
            steps=40,
            step_size=0.03,
        ),
        "SPOS": SolverConfig(
            make=lambda: SPOS(
                kernel=RBFKernel(bandwidth=5.0),
                temperature=0.01,
            ),
            steps=40,
            step_size=0.03,
        ),
        "MatrixSVGD": SolverConfig(
            make=lambda: MatrixSVGD(
                kernel=MatrixRBFKernel(metric=np.array([1.0, 5.0]), bandwidth=2.0),
            ),
            steps=40,
            step_size=0.04,
        ),
        "SteinVariationalNewton": SolverConfig(
            make=lambda: SteinVariationalNewton(
                kernel=RBFKernel(bandwidth=1.0),
                regularization=1e-4,
            ),
            steps=6,
            step_size=0.5,
        ),
    }


def run_trial(method_name: str, n_particles: int, seed: int) -> TrialSummary:
    config = standard_normal_solvers()[method_name]
    solver = config.make()
    particles = initial_particles(
        seed,
        n_particles,
        dimension=1,
        lower=STANDARD_NORMAL_INIT_MIN,
        upper=STANDARD_NORMAL_INIT_MAX,
    )
    start = time.perf_counter()
    result = solver.run(
        particles,
        standard_normal_score,
        n_steps=config.steps,
        step_size=config.step_size,
        seed=seed,
    )
    wall_time_s = time.perf_counter() - start
    final_particles = result.particles[:, 0]
    return TrialSummary(
        mean=float(np.mean(final_particles)),
        variance=float(np.var(final_particles)),
        wall_time_s=wall_time_s,
    )


def summarize_quality(method_name: str) -> TrialSummary:
    trials = [run_trial(method_name, STANDARD_NORMAL_PARTICLES, seed) for seed in SEEDS]
    return TrialSummary(
        mean=statistics.fmean(trial.mean for trial in trials),
        variance=statistics.fmean(trial.variance for trial in trials),
        wall_time_s=statistics.fmean(trial.wall_time_s for trial in trials),
    )


def summarize_scaling(method_name: str, n_particles: int) -> float:
    trials = [run_trial(method_name, n_particles, seed) for seed in SEEDS]
    return statistics.fmean(trial.wall_time_s for trial in trials)


def anisotropic_energy(particles: np.ndarray) -> float:
    return float(
        np.mean(np.sum(particles * (particles @ ANISOTROPIC_PRECISION), axis=1))
    )


def run_anisotropic_trial(method_name: str, seed: int) -> CurvatureTrialSummary:
    config = anisotropic_solvers()[method_name]
    solver = config.make()
    particles = initial_particles(
        seed,
        ANISOTROPIC_PARTICLES,
        dimension=2,
        lower=ANISOTROPIC_INIT_MIN,
        upper=ANISOTROPIC_INIT_MAX,
    )
    start = time.perf_counter()
    if method_name == "SteinVariationalNewton":
        result = solver.run(
            particles,
            anisotropic_gaussian_score,
            hessian_function=anisotropic_gaussian_hessian,
            n_steps=config.steps,
            step_size=config.step_size,
            seed=seed,
        )
    else:
        result = solver.run(
            particles,
            anisotropic_gaussian_score,
            n_steps=config.steps,
            step_size=config.step_size,
            seed=seed,
        )
    wall_time_s = time.perf_counter() - start
    final_particles = result.particles
    return CurvatureTrialSummary(
        energy=anisotropic_energy(final_particles),
        variance_x1=float(np.var(final_particles[:, 0])),
        variance_x2=float(np.var(final_particles[:, 1])),
        wall_time_s=wall_time_s,
    )


def summarize_anisotropic(method_name: str) -> CurvatureTrialSummary:
    trials = [run_anisotropic_trial(method_name, seed) for seed in SEEDS]
    return CurvatureTrialSummary(
        energy=statistics.fmean(trial.energy for trial in trials),
        variance_x1=statistics.fmean(trial.variance_x1 for trial in trials),
        variance_x2=statistics.fmean(trial.variance_x2 for trial in trials),
        wall_time_s=statistics.fmean(trial.wall_time_s for trial in trials),
    )


def render_markdown() -> str:
    quality_rows = {
        "SVGD": summarize_quality("SVGD"),
        "RandomBatchSVGD": summarize_quality("RandomBatchSVGD"),
        "SPOS": summarize_quality("SPOS"),
    }
    scaling_rows = {
        n_particles: {
            method_name: summarize_scaling(method_name, n_particles)
            for method_name in ("SVGD", "RandomBatchSVGD", "SPOS")
        }
        for n_particles in STANDARD_NORMAL_SCALING_PARTICLES
    }
    anisotropic_rows = {
        method_name: summarize_anisotropic(method_name)
        for method_name in (
            "SVGD",
            "RandomBatchSVGD",
            "SPOS",
            "MatrixSVGD",
            "SteinVariationalNewton",
        )
    }
    anisotropic_configs = anisotropic_solvers()

    lines = [
        "# README Benchmark",
        "",
        f"- platform: `{platform.system()} {platform.machine()}`",
        f"- python: `{platform.python_version()}`",
        f"- standard normal particles: `{STANDARD_NORMAL_PARTICLES}`",
        f"- anisotropic particles: `{ANISOTROPIC_PARTICLES}`",
        f"- seeds: `{SEEDS[0]}..{SEEDS[-1]}`",
        "",
        "## Standard Normal",
        "",
        f"- initialization: `Uniform({STANDARD_NORMAL_INIT_MIN}, {STANDARD_NORMAL_INIT_MAX})`",
        f"- steps: `{STANDARD_NORMAL_STEPS}`",
        f"- step size: `{STANDARD_NORMAL_STEP_SIZE}`",
        "",
        "| Method | Final mean | Final variance | Mean wall time (s) |",
        "| --- | ---: | ---: | ---: |",
        (
            "| `SVGD` | "
            f"`{quality_rows['SVGD'].mean:.3f}` | "
            f"`{quality_rows['SVGD'].variance:.3f}` | "
            f"`{quality_rows['SVGD'].wall_time_s:.3f}` |"
        ),
        (
            "| `RandomBatchSVGD` | "
            f"`{quality_rows['RandomBatchSVGD'].mean:.3f}` | "
            f"`{quality_rows['RandomBatchSVGD'].variance:.3f}` | "
            f"`{quality_rows['RandomBatchSVGD'].wall_time_s:.3f}` |"
        ),
        (
            "| `SPOS` | "
            f"`{quality_rows['SPOS'].mean:.3f}` | "
            f"`{quality_rows['SPOS'].variance:.3f}` | "
            f"`{quality_rows['SPOS'].wall_time_s:.3f}` |"
        ),
        "",
        "| Particles | `SVGD` time (s) | `RandomBatchSVGD` time (s) | `SPOS` time (s) |",
        "| ---: | ---: | ---: | ---: |",
    ]

    for n_particles in STANDARD_NORMAL_SCALING_PARTICLES:
        row = scaling_rows[n_particles]
        lines.append(
            f"| `{n_particles}` | "
            f"`{row['SVGD']:.3f}` | "
            f"`{row['RandomBatchSVGD']:.3f}` | "
            f"`{row['SPOS']:.3f}` |"
        )

    lines.extend(
        [
            "",
            "## Anisotropic Gaussian",
            "",
            f"- target covariance: `diag(1.0, {1.0 / ANISOTROPIC_PRECISION[1, 1]:.2f})`",
            f"- initialization: `Uniform({ANISOTROPIC_INIT_MIN}, {ANISOTROPIC_INIT_MAX})^2`",
            "- metric-aware methods are benchmarked on the same target so the stiff axis is visible",
            "- smaller Mahalanobis energy is better; the target expectation is `2.0`",
            "",
            "| Method | Steps | Var(x1) | Var(x2) | Mahalanobis energy | Mean wall time (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method_name in (
        "SVGD",
        "RandomBatchSVGD",
        "SPOS",
        "MatrixSVGD",
        "SteinVariationalNewton",
    ):
        summary = anisotropic_rows[method_name]
        lines.append(
            f"| `{method_name}` | "
            f"`{anisotropic_configs[method_name].steps}` | "
            f"`{summary.variance_x1:.3f}` | "
            f"`{summary.variance_x2:.3f}` | "
            f"`{summary.energy:.3f}` | "
            f"`{summary.wall_time_s:.3f}` |"
        )

    return "\n".join(lines)


def main() -> None:
    print(render_markdown())


if __name__ == "__main__":
    main()

from __future__ import annotations

import platform
import statistics
import time
from dataclasses import dataclass

import numpy as np

from fast_svgd import RBFKernel, RandomBatchSVGD, SPOS, SVGD

SEEDS = tuple(range(5))
QUALITY_PARTICLES = 256
SCALING_PARTICLES = (128, 256, 512)
STEP_SIZE = 0.03
N_STEPS = 250
INIT_MIN = -2.0
INIT_MAX = 2.0
RANDOM_BATCH_SIZE = 32
SPOS_TEMPERATURE = 0.01


@dataclass(frozen=True)
class TrialSummary:
    mean: float
    variance: float
    wall_time_s: float


def standard_normal_score(particles: np.ndarray) -> np.ndarray:
    return -particles


def initial_particles(seed: int, n_particles: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(INIT_MIN, INIT_MAX, size=(n_particles, 1))


def solvers() -> dict[str, callable]:
    return {
        "SVGD": lambda: SVGD(kernel=RBFKernel()),
        "RandomBatchSVGD": lambda: RandomBatchSVGD(
            kernel=RBFKernel(),
            batch_size=RANDOM_BATCH_SIZE,
        ),
        "SPOS": lambda: SPOS(
            kernel=RBFKernel(),
            temperature=SPOS_TEMPERATURE,
        ),
    }


def run_trial(method_name: str, n_particles: int, seed: int) -> TrialSummary:
    solver = solvers()[method_name]()
    particles = initial_particles(seed, n_particles)
    start = time.perf_counter()
    result = solver.run(
        particles,
        standard_normal_score,
        n_steps=N_STEPS,
        step_size=STEP_SIZE,
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
    trials = [run_trial(method_name, QUALITY_PARTICLES, seed) for seed in SEEDS]
    return TrialSummary(
        mean=statistics.fmean(trial.mean for trial in trials),
        variance=statistics.fmean(trial.variance for trial in trials),
        wall_time_s=statistics.fmean(trial.wall_time_s for trial in trials),
    )


def summarize_scaling(method_name: str, n_particles: int) -> float:
    trials = [run_trial(method_name, n_particles, seed) for seed in SEEDS]
    return statistics.fmean(trial.wall_time_s for trial in trials)


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
        for n_particles in SCALING_PARTICLES
    }

    lines = [
        "# README Benchmark",
        "",
        f"- platform: `{platform.system()} {platform.machine()}`",
        f"- python: `{platform.python_version()}`",
        f"- particles: `{QUALITY_PARTICLES}` for quality, `{SCALING_PARTICLES}` for scaling",
        f"- initialization: `Uniform({INIT_MIN}, {INIT_MAX})`",
        f"- steps: `{N_STEPS}`",
        f"- step size: `{STEP_SIZE}`",
        f"- seeds: `{SEEDS[0]}..{SEEDS[-1]}`",
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

    for n_particles in SCALING_PARTICLES:
        row = scaling_rows[n_particles]
        lines.append(
            f"| `{n_particles}` | "
            f"`{row['SVGD']:.3f}` | "
            f"`{row['RandomBatchSVGD']:.3f}` | "
            f"`{row['SPOS']:.3f}` |"
        )

    return "\n".join(lines)


def main() -> None:
    print(render_markdown())


if __name__ == "__main__":
    main()

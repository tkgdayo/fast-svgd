import numpy as np

from fast_svgd import (
    DiagonalPreconditioner,
    FastSVGD,
    GaussianNoise,
    RBFKernel,
    RandomBatchInteraction,
    RandomBatchSVGD,
    SPOS,
    SVGD,
)


def standard_normal_score(particles: np.ndarray) -> np.ndarray:
    return -particles


def test_svgd_contracts_toward_standard_normal() -> None:
    initial = np.linspace(-6.0, 6.0, 32, dtype=float).reshape(-1, 1)
    solver = SVGD(kernel=RBFKernel(bandwidth=2.0))

    result = solver.run(
        initial,
        standard_normal_score,
        n_steps=60,
        step_size=0.05,
        seed=0,
    )

    assert len(result.diagnostics) == 60
    assert np.mean(result.particles**2) < np.mean(initial**2)


def test_random_batch_svgd_contracts_with_local_interactions() -> None:
    initial = np.linspace(-4.0, 4.0, 24, dtype=float).reshape(-1, 1)
    solver = RandomBatchSVGD(batch_size=5, kernel=RBFKernel(bandwidth=1.0))

    result = solver.run(
        initial,
        standard_normal_score,
        n_steps=40,
        step_size=0.05,
        seed=1,
    )

    assert np.mean(result.particles**2) < np.mean(initial**2)


def test_spos_injects_reproducible_noise() -> None:
    initial = np.zeros((8, 2), dtype=float)
    zero_score = lambda particles: np.zeros_like(particles)
    solver = SPOS(temperature=0.25)

    result_a = solver.run(initial, zero_score, n_steps=1, step_size=0.1, seed=42)
    result_b = solver.run(initial, zero_score, n_steps=1, step_size=0.1, seed=42)

    np.testing.assert_allclose(result_a.particles, result_b.particles)
    assert np.linalg.norm(result_a.particles) > 0.0


def test_components_can_be_composed_in_fast_svgd() -> None:
    initial = np.linspace(-3.0, 3.0, 12, dtype=float).reshape(-1, 1)
    solver = FastSVGD(
        kernel=RBFKernel(bandwidth=1.0),
        interaction=RandomBatchInteraction(batch_size=4),
        preconditioner=DiagonalPreconditioner(np.array([0.5], dtype=float)),
        noise=GaussianNoise(temperature=0.0),
    )

    result = solver.run(
        initial,
        standard_normal_score,
        n_steps=20,
        step_size=0.05,
        seed=5,
        store_trajectory=True,
    )

    assert len(result.trajectory) == 21
    assert np.mean(result.particles**2) < np.mean(initial**2)


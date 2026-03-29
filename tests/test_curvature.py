import numpy as np

from fast_svgd import (
    GaussianTarget,
    MatrixRBFKernel,
    MatrixSVGD,
    RBFKernel,
    SVGD,
    SteinVariationalNewton,
)


PRECISION = np.diag([1.0, 25.0])


def anisotropic_gaussian_score(particles: np.ndarray) -> np.ndarray:
    return -(particles @ PRECISION)


def anisotropic_gaussian_hessian(particles: np.ndarray) -> np.ndarray:
    return np.repeat((-PRECISION)[None, :, :], particles.shape[0], axis=0)


def mahalanobis_energy(particles: np.ndarray) -> float:
    return float(np.mean(np.sum(particles * (particles @ PRECISION), axis=1)))


def test_matrix_svgd_improves_stiff_direction_energy() -> None:
    rng = np.random.default_rng(0)
    initial = rng.uniform(-4.0, 4.0, size=(64, 2))

    vanilla = SVGD(kernel=RBFKernel(bandwidth=5.0))
    matrix = MatrixSVGD(
        kernel=MatrixRBFKernel(metric=np.array([1.0, 5.0]), bandwidth=5.0),
    )

    vanilla_result = vanilla.run(
        initial,
        anisotropic_gaussian_score,
        n_steps=40,
        step_size=0.03,
        seed=0,
    )
    matrix_result = matrix.run(
        initial,
        anisotropic_gaussian_score,
        n_steps=40,
        step_size=0.03,
        seed=0,
    )

    assert mahalanobis_energy(matrix_result.particles) < mahalanobis_energy(
        vanilla_result.particles
    )


def test_stein_variational_newton_converges_faster_on_anisotropic_gaussian() -> None:
    rng = np.random.default_rng(0)
    initial = rng.uniform(-4.0, 4.0, size=(64, 2))

    vanilla = SVGD(kernel=RBFKernel(bandwidth=1.0))
    newton = SteinVariationalNewton(kernel=RBFKernel(bandwidth=1.0), regularization=1e-4)

    vanilla_result = vanilla.run(
        initial,
        anisotropic_gaussian_score,
        n_steps=6,
        step_size=0.08,
        seed=0,
    )
    newton_result = newton.run(
        initial,
        anisotropic_gaussian_score,
        hessian_function=anisotropic_gaussian_hessian,
        n_steps=6,
        step_size=0.5,
        seed=0,
    )

    assert mahalanobis_energy(newton_result.particles) < mahalanobis_energy(
        vanilla_result.particles
    )


def test_stein_variational_newton_accepts_bound_second_order_target() -> None:
    rng = np.random.default_rng(1)
    initial = rng.uniform(-4.0, 4.0, size=(64, 2))
    target = GaussianTarget(mean=np.zeros(2), precision=PRECISION)
    solver = SteinVariationalNewton(
        kernel=RBFKernel(bandwidth=1.0),
        regularization=1e-4,
        target=target,
    )

    result = solver.run(
        initial,
        n_steps=6,
        step_size=0.5,
        seed=1,
    )

    assert mahalanobis_energy(result.particles) < mahalanobis_energy(initial)

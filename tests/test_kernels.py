import numpy as np

from fast_svgd import IMQKernel, RBFKernel


def test_rbf_kernel_self_interaction_has_unit_diagonal() -> None:
    particles = np.array([[-1.0], [0.0], [1.5]], dtype=float)
    evaluation = RBFKernel(bandwidth=1.5).evaluate(particles, particles)

    assert evaluation.matrix.shape == (3, 3)
    assert evaluation.grad_source.shape == (3, 3, 1)
    np.testing.assert_allclose(np.diag(evaluation.matrix), np.ones(3))
    np.testing.assert_allclose(evaluation.grad_source[:, :, 0].diagonal(), np.zeros(3))


def test_imq_kernel_returns_positive_values() -> None:
    source = np.array([[0.0, 0.5], [1.0, -0.5]], dtype=float)
    target = np.array([[0.25, -0.25]], dtype=float)
    evaluation = IMQKernel(scale=2.0, beta=-0.5).evaluate(source, target)

    assert evaluation.matrix.shape == (2, 1)
    assert evaluation.grad_source.shape == (2, 1, 2)
    assert np.all(evaluation.matrix > 0.0)


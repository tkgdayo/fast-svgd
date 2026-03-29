from __future__ import annotations

import numpy as np

from fast_svgd import RandomBatchSVGD, RBFKernel


def standard_normal_score(particles: np.ndarray) -> np.ndarray:
    return -particles


def main() -> None:
    particles = np.linspace(-4.0, 4.0, 32, dtype=float).reshape(-1, 1)
    solver = RandomBatchSVGD(kernel=RBFKernel(), batch_size=8)
    result = solver.run(
        particles,
        standard_normal_score,
        n_steps=150,
        step_size=0.05,
        seed=7,
        store_trajectory=True,
    )
    print("final mean:", float(result.particles.mean()))
    print("final variance:", float(result.particles.var()))


if __name__ == "__main__":
    main()


# fast-svgd

`fast-svgd` is a small, composable Python package for Stein variational particle methods.
It starts from vanilla SVGD and adds two fast/stable variants out of the box:

- `RandomBatchSVGD`: a random-batch interaction backend inspired by the RBM-SVGD family
- `SPOS`: SVGD with Gaussian particle noise for more robust exploration

The package is organized around reusable building blocks so we can grow into matrix-valued kernels and Newton-style methods without rewriting the core solver.

## Install

```bash
pip install fast-svgd
```

For local development:

```bash
python -m pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np

from fast_svgd import RandomBatchSVGD, RBFKernel


def standard_normal_score(particles: np.ndarray) -> np.ndarray:
    return -particles


particles = np.linspace(-4.0, 4.0, 32, dtype=float).reshape(-1, 1)
solver = RandomBatchSVGD(kernel=RBFKernel(), batch_size=8)
result = solver.run(
    particles,
    standard_normal_score,
    n_steps=150,
    step_size=0.05,
    seed=7,
)

print(result.particles.mean())
print(result.particles.var())
```

## API Design

`fast-svgd` treats an update as the composition of a few interchangeable parts:

- `kernel`
- `preconditioner`
- `interaction approximator`
- `noise injector`

That means:

- vanilla `SVGD` = full interaction + no noise
- `RandomBatchSVGD` = random-batch interaction + no noise
- `SPOS` = full interaction + Gaussian noise

You can also build custom combinations with `FastSVGD`.

```python
import numpy as np

from fast_svgd import (
    DiagonalPreconditioner,
    FastSVGD,
    GaussianNoise,
    RBFKernel,
    RandomBatchInteraction,
)


solver = FastSVGD(
    kernel=RBFKernel(),
    preconditioner=DiagonalPreconditioner(np.array([0.5])),
    interaction=RandomBatchInteraction(batch_size=8),
    noise=GaussianNoise(temperature=0.1),
)
```

## Included Components

### Solvers

- `SVGD`
- `RandomBatchSVGD`
- `SPOS`
- `FastSVGD` for custom composition

### Kernels

- `RBFKernel`
- `IMQKernel`

### Interaction Backends

- `FullBatchInteraction`
- `RandomBatchInteraction`

### Preconditioners

- `IdentityPreconditioner`
- `DiagonalPreconditioner`

### Noise

- `NoNoise`
- `GaussianNoise`

## Roadmap

`v0.1` focuses on the modular core plus SVGD, random-batch interactions, and SPOS.
Planned next steps:

- matrix-valued kernels
- Newton and quasi-Newton preconditioners
- structured or subspace interactions for high-dimensional models
- experimental learned update rules

## Development

Run tests:

```bash
python -m pytest
```

Build a wheel and source distribution:

```bash
python -m build
```


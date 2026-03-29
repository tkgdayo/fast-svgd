# fast-svgd

[![GitHub stars](https://img.shields.io/github/stars/tkgdayo/fast-svgd?style=social)](https://github.com/tkgdayo/fast-svgd/stargazers)

`fast-svgd` is a small, composable Python package for Stein variational particle methods.
It starts from vanilla SVGD and adds two fast/stable variants out of the box:

- `RandomBatchSVGD`: a random-batch interaction backend inspired by the RBM-SVGD family
- `SPOS`: SVGD with Gaussian particle noise for more robust exploration

The package is organized around reusable building blocks so we can grow into matrix-valued kernels and Newton-style methods without rewriting the core solver.

## Install

Today:

```bash
pip install git+https://github.com/tkgdayo/fast-svgd.git
```

After publishing to PyPI:

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

## Benchmark

The table below is a small reproducible benchmark on a 1D standard normal target,
using the exact code in `scripts/benchmark_readme.py`.

Setup:

- target distribution: `N(0, 1)`
- particles: `256`
- initialization: `Uniform(-2, 2)`
- steps: `250`
- step size: `0.03`
- seeds: `0..4`
- environment used for the measurements below: `Darwin arm64`, `Python 3.13.11`

This is meant as a README-level sanity check, not a full scientific benchmark.
Wall-clock numbers will vary slightly by machine and current load.

| Method | Per-step cost | Final mean | Final variance | Mean wall time (s) |
| --- | --- | ---: | ---: | ---: |
| `SVGD` | `O(N^2 d)` | `0.011` | `0.909` | `0.223` |
| `RandomBatchSVGD` (`batch_size=32`) | `O(N B d)` | `0.007` | `0.724` | `0.114` |
| `SPOS` (`temperature=0.01`) | `O(N^2 d) + O(N d)` | `0.008` | `1.023` | `0.231` |

For this very simple target, all three methods remain stable. `RandomBatchSVGD`
is the cheapest at this particle count, while `SPOS` lands closest to the target
variance under the same update budget.

The same benchmark also shows the expected scaling behavior as particle count grows:

| Particles | `SVGD` time (s) | `RandomBatchSVGD` time (s) | `SPOS` time (s) |
| ---: | ---: | ---: | ---: |
| `128` | `0.060` | `0.061` | `0.066` |
| `256` | `0.228` | `0.113` | `0.213` |
| `512` | `0.909` | `0.221` | `0.914` |

At `512` particles, `RandomBatchSVGD` is about `4.1x` faster than full-batch
`SVGD` in this implementation.

To reproduce these numbers:

```bash
python scripts/benchmark_readme.py
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

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=tkgdayo/fast-svgd&type=Date)](https://star-history.com/#tkgdayo/fast-svgd&Date)

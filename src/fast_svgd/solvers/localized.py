from __future__ import annotations

from ..core.base import Kernel, Preconditioner, Target
from ..core.engine import FastSVGD
from ..interactions import KNNInteraction
from ..kernels import RBFKernel
from ..noise import NoNoise
from ..preconditioners import IdentityPreconditioner


class KNNSVGD(FastSVGD):
    def __init__(
        self,
        *,
        n_neighbors: int = 16,
        include_self: bool = True,
        backend: str = "auto",
        kernel: Kernel | None = None,
        preconditioner: Preconditioner | None = None,
        target: Target | None = None,
    ) -> None:
        super().__init__(
            kernel=kernel if kernel is not None else RBFKernel(),
            interaction=KNNInteraction(
                n_neighbors=n_neighbors,
                include_self=include_self,
                backend=backend,
            ),
            preconditioner=(
                preconditioner
                if preconditioner is not None
                else IdentityPreconditioner()
            ),
            noise=NoNoise(),
            target=target,
        )

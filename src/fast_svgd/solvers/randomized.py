from __future__ import annotations

from ..core.base import Kernel, Preconditioner
from ..core.engine import FastSVGD
from ..interactions import RandomBatchInteraction
from ..kernels import RBFKernel
from ..noise import NoNoise
from ..preconditioners import IdentityPreconditioner


class RandomBatchSVGD(FastSVGD):
    def __init__(
        self,
        *,
        batch_size: int = 16,
        kernel: Kernel | None = None,
        preconditioner: Preconditioner | None = None,
    ) -> None:
        super().__init__(
            kernel=kernel if kernel is not None else RBFKernel(),
            interaction=RandomBatchInteraction(batch_size=batch_size),
            preconditioner=(
                preconditioner
                if preconditioner is not None
                else IdentityPreconditioner()
            ),
            noise=NoNoise(),
        )


RBMSVGD = RandomBatchSVGD


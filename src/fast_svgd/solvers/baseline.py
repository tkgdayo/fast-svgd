from __future__ import annotations

from ..core.base import Kernel, Preconditioner
from ..core.engine import FastSVGD
from ..interactions import FullBatchInteraction
from ..kernels import RBFKernel
from ..noise import NoNoise
from ..preconditioners import IdentityPreconditioner


class SVGD(FastSVGD):
    def __init__(
        self,
        *,
        kernel: Kernel | None = None,
        preconditioner: Preconditioner | None = None,
    ) -> None:
        super().__init__(
            kernel=kernel if kernel is not None else RBFKernel(),
            interaction=FullBatchInteraction(),
            preconditioner=(
                preconditioner
                if preconditioner is not None
                else IdentityPreconditioner()
            ),
            noise=NoNoise(),
        )

from __future__ import annotations

from ..core.base import InteractionApproximator, Kernel, Preconditioner
from ..core.engine import FastSVGD
from ..interactions import FullBatchInteraction
from ..kernels import RBFKernel
from ..noise import GaussianNoise
from ..preconditioners import IdentityPreconditioner


class SPOS(FastSVGD):
    def __init__(
        self,
        *,
        temperature: float = 1.0,
        kernel: Kernel | None = None,
        interaction: InteractionApproximator | None = None,
        preconditioner: Preconditioner | None = None,
    ) -> None:
        super().__init__(
            kernel=kernel if kernel is not None else RBFKernel(),
            interaction=interaction if interaction is not None else FullBatchInteraction(),
            preconditioner=(
                preconditioner
                if preconditioner is not None
                else IdentityPreconditioner()
            ),
            noise=GaussianNoise(temperature=temperature),
        )

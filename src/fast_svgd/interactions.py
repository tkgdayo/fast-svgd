from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import InteractionBlock


@dataclass(frozen=True)
class FullBatchInteraction:
    """Use all particles for every interaction."""

    def blocks(
        self,
        n_particles: int,
        rng: np.random.Generator,
    ) -> tuple[InteractionBlock, ...]:
        del rng
        if n_particles <= 0:
            raise ValueError("n_particles must be positive.")
        indices = np.arange(n_particles, dtype=np.int64)
        return (InteractionBlock(active_indices=indices, source_indices=indices),)


@dataclass(frozen=True)
class RandomBatchInteraction:
    """Partition particles into random local blocks for cheaper interactions."""

    batch_size: int = 16
    shuffle: bool = True

    def __post_init__(self) -> None:
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2.")

    def blocks(
        self,
        n_particles: int,
        rng: np.random.Generator,
    ) -> tuple[InteractionBlock, ...]:
        if n_particles <= 0:
            raise ValueError("n_particles must be positive.")

        if n_particles <= self.batch_size:
            indices = np.arange(n_particles, dtype=np.int64)
            return (InteractionBlock(active_indices=indices, source_indices=indices),)

        indices = np.arange(n_particles, dtype=np.int64)
        if self.shuffle:
            rng.shuffle(indices)

        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, n_particles, self.batch_size)
        ]

        # Avoid singleton batches because they remove the repulsive interaction.
        if len(batches) > 1 and batches[-1].size == 1:
            batches[-2] = np.concatenate((batches[-2], batches[-1]))
            batches.pop()

        return tuple(
            InteractionBlock(active_indices=batch.copy(), source_indices=batch.copy())
            for batch in batches
        )


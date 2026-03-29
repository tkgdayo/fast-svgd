from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import InteractionBlock

_VALID_KNN_BACKENDS = frozenset({"auto", "dense", "ckdtree"})
_UNSET = object()
_CKDTREE_TYPE: object = _UNSET


def _as_particle_array(particles: np.ndarray) -> np.ndarray:
    array = np.asarray(particles, dtype=float)
    if array.ndim != 2:
        raise ValueError("particles must have shape (n_particles, dimension).")
    if array.shape[0] == 0:
        raise ValueError("At least one particle is required.")
    return array


def _sort_neighbor_rows(
    indices: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    order = np.argsort(distances, axis=1, kind="stable")
    return np.take_along_axis(indices, order, axis=1)


def _dense_knn_indices(
    particles: np.ndarray,
    *,
    neighbor_count: int,
    include_self: bool,
) -> np.ndarray:
    squared_distances = np.sum(
        (particles[:, None, :] - particles[None, :, :]) ** 2,
        axis=-1,
    )
    if not include_self and particles.shape[0] > 1:
        np.fill_diagonal(squared_distances, np.inf)

    partition = np.argpartition(
        squared_distances,
        kth=neighbor_count - 1,
        axis=1,
    )[:, :neighbor_count]
    partition_distances = np.take_along_axis(squared_distances, partition, axis=1)
    return _sort_neighbor_rows(partition, partition_distances).astype(np.int64, copy=False)


def _trim_tree_neighbors(
    neighbor_indices: np.ndarray,
    *,
    n_particles: int,
    neighbor_count: int,
    include_self: bool,
) -> np.ndarray:
    if include_self:
        return neighbor_indices[:, :neighbor_count]

    trimmed = np.empty((n_particles, neighbor_count), dtype=np.int64)
    for particle_index in range(n_particles):
        row = neighbor_indices[particle_index]
        filtered = row[row != particle_index]
        if filtered.size == 0:
            filtered = np.array([particle_index], dtype=np.int64)
        trimmed[particle_index] = filtered[:neighbor_count]
    return trimmed


def _ckdtree_knn_indices(
    particles: np.ndarray,
    *,
    neighbor_count: int,
    include_self: bool,
) -> np.ndarray:
    global _CKDTREE_TYPE
    if _CKDTREE_TYPE is _UNSET:
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            _CKDTREE_TYPE = None
        else:
            _CKDTREE_TYPE = cKDTree

    if _CKDTREE_TYPE is None:
        raise ImportError(
            "KNNInteraction backend='ckdtree' requires scipy. "
            "Install the optional 'neighbors' extra to enable it."
        )

    query_count = neighbor_count if include_self else min(
        particles.shape[0],
        neighbor_count + 1,
    )
    tree = _CKDTREE_TYPE(particles)
    _, neighbor_indices = tree.query(particles, k=query_count)
    array = np.asarray(neighbor_indices, dtype=np.int64)
    if array.ndim == 1:
        array = array[:, None]
    return _trim_tree_neighbors(
        array,
        n_particles=particles.shape[0],
        neighbor_count=neighbor_count,
        include_self=include_self,
    )


def _effective_neighbor_count(
    *,
    n_particles: int,
    requested_neighbors: int,
    include_self: bool,
) -> int:
    if n_particles == 1:
        return 1
    max_neighbors = n_particles if include_self else n_particles - 1
    return min(requested_neighbors, max_neighbors)


@dataclass(frozen=True)
class FullBatchInteraction:
    """Use all particles for every interaction."""

    def blocks(
        self,
        particles: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[InteractionBlock, ...]:
        del rng
        n_particles = _as_particle_array(particles).shape[0]
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
        particles: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[InteractionBlock, ...]:
        n_particles = _as_particle_array(particles).shape[0]

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


@dataclass(frozen=True)
class KNNInteraction:
    """Use only the k nearest particles for each local interaction."""

    n_neighbors: int = 16
    include_self: bool = True
    backend: str = "auto"

    def __post_init__(self) -> None:
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        if self.backend not in _VALID_KNN_BACKENDS:
            raise ValueError(
                "backend must be one of 'auto', 'dense', or 'ckdtree'."
            )

    def _neighbor_indices(self, particles: np.ndarray) -> np.ndarray:
        particles_array = _as_particle_array(particles)
        neighbor_count = _effective_neighbor_count(
            n_particles=particles_array.shape[0],
            requested_neighbors=self.n_neighbors,
            include_self=self.include_self,
        )

        if self.include_self and neighbor_count == particles_array.shape[0]:
            return np.broadcast_to(
                np.arange(particles_array.shape[0], dtype=np.int64),
                (particles_array.shape[0], particles_array.shape[0]),
            ).copy()

        if self.backend == "dense":
            return _dense_knn_indices(
                particles_array,
                neighbor_count=neighbor_count,
                include_self=self.include_self,
            )

        if self.backend == "ckdtree":
            return _ckdtree_knn_indices(
                particles_array,
                neighbor_count=neighbor_count,
                include_self=self.include_self,
            )

        try:
            return _ckdtree_knn_indices(
                particles_array,
                neighbor_count=neighbor_count,
                include_self=self.include_self,
            )
        except ImportError:
            return _dense_knn_indices(
                particles_array,
                neighbor_count=neighbor_count,
                include_self=self.include_self,
            )

    def blocks(
        self,
        particles: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[InteractionBlock, ...]:
        del rng
        particles_array = _as_particle_array(particles)
        neighbor_indices = self._neighbor_indices(particles_array)

        if self.include_self and neighbor_indices.shape[1] == particles_array.shape[0]:
            indices = np.arange(particles_array.shape[0], dtype=np.int64)
            return (InteractionBlock(active_indices=indices, source_indices=indices),)

        return tuple(
            InteractionBlock(
                active_indices=np.array([particle_index], dtype=np.int64),
                source_indices=neighbor_indices[particle_index].copy(),
            )
            for particle_index in range(particles_array.shape[0])
        )

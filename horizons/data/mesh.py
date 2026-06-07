"""HorizonSurface: in-memory representation of a triangulated horizon.

Holds vertices, faces, the PyG-style edge index, and metadata.
All arrays are torch tensors so downstream code (feature computation,
rollout) can run autograd through them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass
class HorizonSurface:
    """A single triangulated horizon.

    Attributes
    ----------
    V : torch.Tensor, shape (n_vertices, 3), dtype float32
        Vertex coordinates (x, y, z).
    F : torch.Tensor, shape (n_faces, 3), dtype int64
        Triangle indices into V (0-indexed).
    edge_index : torch.Tensor, shape (2, n_edges * 2), dtype int64
        PyG-style bidirectional edge list. Each undirected edge appears
        twice (once in each direction).
    surface_id : str
        Identifier used for split assignment and RNG seeding.
    reservoir_id : Optional[str]
        Reservoir label for stratified splits. None for synthetic fixtures.
    """
    V: torch.Tensor
    F: torch.Tensor
    edge_index: torch.Tensor
    surface_id: str
    reservoir_id: Optional[str] = None

    # ------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------
    @property
    def n_vertices(self) -> int:
        return self.V.shape[0]

    @property
    def n_faces(self) -> int:
        return self.F.shape[0]

    @property
    def n_edges(self) -> int:
        """Number of undirected edges (edge_index has 2 * n_edges columns)."""
        return self.edge_index.shape[1] // 2

    # ------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------
    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        surface_id: Optional[str] = None,
        reservoir_id: Optional[str] = None,
    ) -> "HorizonSurface":
        """Load a HorizonSurface from a .npz file containing V and F."""
        path = Path(path)
        data = np.load(path)
        V_np = data["V"]
        F_np = data["F"]

        if V_np.ndim != 2 or V_np.shape[1] != 3:
            raise ValueError(
                f"V must have shape (n, 3); got {V_np.shape} in {path}"
            )
        if F_np.ndim != 2 or F_np.shape[1] != 3:
            raise ValueError(
                f"F must have shape (n, 3); got {F_np.shape} in {path}"
            )

        V = torch.from_numpy(V_np).to(torch.float32)
        F = torch.from_numpy(F_np).to(torch.int64)
        edge_index = build_edge_index(F)

        if surface_id is None:
            surface_id = path.stem  # filename without extension

        return cls(
            V=V,
            F=F,
            edge_index=edge_index,
            surface_id=surface_id,
            reservoir_id=reservoir_id,
        )

    @classmethod
    def from_ts(
        cls,
        path: str | Path,
        surface_id: Optional[str] = None,
        reservoir_id: Optional[str] = None,
    ) -> "HorizonSurface":
        """Load a HorizonSurface from a GOCAD TSurf (.ts) file.

        Parses lines of the form:
            PVRTX <id> <x> <y> <z>
            TRGL <i> <j> <k>
        Vertex IDs in the file are 1-indexed; we convert to 0-indexed.

        Other content (headers, coordinate-system metadata, TFACE markers,
        and any other GOCAD record types) is ignored. If the file contains
        multiple TFACE blocks, all their triangles are treated as belonging
        to one mesh — they share the same vertex index space.

        Convention note: GOCAD files typically use ZPOSITIVE Depth (z
        increases downward). We preserve the original sign — the model
        treats z as opaque and doesn't care which way is "up", but the
        convention must be consistent across the dataset.
        """
        path = Path(path)

        # Parse the file
        vertices: dict[int, tuple[float, float, float]] = {}
        triangles: list[tuple[int, int, int]] = []

        with open(path, "r") as f:
            for line in f:
                tokens = line.split()
                if not tokens:
                    continue
                head = tokens[0]
                if head in ("PVRTX", "VRTX"):
                    # VRTX is sometimes used in place of PVRTX
                    # Format: <id> <x> <y> <z> [properties...]
                    if len(tokens) < 5:
                        continue
                    try:
                        vid = int(tokens[1])
                        x, y, z = float(tokens[2]), float(tokens[3]), float(tokens[4])
                    except ValueError:
                        continue
                    vertices[vid] = (x, y, z)
                elif head == "TRGL":
                    if len(tokens) < 4:
                        continue
                    try:
                        i, j, k = int(tokens[1]), int(tokens[2]), int(tokens[3])
                    except ValueError:
                        continue
                    triangles.append((i, j, k))
                # All other lines (HEADER, TFACE, COORDINATE_SYSTEM, END, etc.) ignored

        if not vertices:
            raise ValueError(f"No PVRTX records found in {path}")
        if not triangles:
            raise ValueError(f"No TRGL records found in {path}")

        # GOCAD vertex IDs are 1-indexed and may be non-contiguous. Build a
        # mapping from GOCAD ID -> 0-indexed position so triangle indices
        # are valid in our 0-indexed array.
        sorted_ids = sorted(vertices.keys())
        gocad_to_idx = {gid: i for i, gid in enumerate(sorted_ids)}

        # Validate: every triangle vertex must exist in `vertices`
        for tri in triangles:
            for vid in tri:
                if vid not in vertices:
                    raise ValueError(
                        f"Triangle references missing vertex {vid} in {path}"
                    )

        # Build V and F arrays
        V_np = np.array(
            [vertices[gid] for gid in sorted_ids], dtype=np.float64
        )
        F_np = np.array(
            [[gocad_to_idx[i], gocad_to_idx[j], gocad_to_idx[k]]
             for i, j, k in triangles],
            dtype=np.int64,
        )

        V = torch.from_numpy(V_np).to(torch.float32)
        F = torch.from_numpy(F_np).to(torch.int64)
        edge_index = build_edge_index(F)

        if surface_id is None:
            surface_id = path.stem

        return cls(
            V=V,
            F=F,
            edge_index=edge_index,
            surface_id=surface_id,
            reservoir_id=reservoir_id,
        )


def build_edge_index(F: torch.Tensor) -> torch.Tensor:
    """Build a PyG-style bidirectional edge_index from a face array.

    Each triangle (i, j, k) contributes three undirected edges:
    (i, j), (j, k), (k, i). We emit each as two directed edges
    (one each direction) and deduplicate.

    Parameters
    ----------
    F : torch.Tensor, shape (n_faces, 3), int64

    Returns
    -------
    edge_index : torch.Tensor, shape (2, n_directed_edges), int64
        Sorted, deduplicated, bidirectional.
    """
    if F.dtype != torch.int64:
        raise TypeError(f"F must be int64; got {F.dtype}")

    # Stack the three edges per triangle: shape (3 * n_faces, 2)
    e01 = F[:, [0, 1]]
    e12 = F[:, [1, 2]]
    e20 = F[:, [2, 0]]
    undirected = torch.cat([e01, e12, e20], dim=0)  # (3 * n_faces, 2)

    # Make bidirectional by adding reversed copies
    reversed_ = undirected[:, [1, 0]]
    directed = torch.cat([undirected, reversed_], dim=0)  # (6 * n_faces, 2)

    # Deduplicate (a triangle edge shared by two faces would appear twice)
    directed = torch.unique(directed, dim=0)

    # Convert to PyG convention: shape (2, n_edges)
    return directed.t().contiguous()


def compute_boundary_vertices(F: torch.Tensor) -> torch.Tensor:
    """Identify boundary vertices of a triangle mesh.

    An edge is on the boundary iff exactly one face contains it.
    A vertex is on the boundary iff at least one boundary edge touches it.

    Parameters
    ----------
    F : torch.Tensor, shape (n_faces, 3), int64
        Triangle indices.

    Returns
    -------
    boundary : torch.Tensor, shape (n_vertices,), bool
        True for boundary vertices.

    Notes
    -----
    The number of vertices is inferred from F.max() + 1. If your mesh has
    isolated vertices with no incident faces, they won't be detected here.
    """
    if F.dim() != 2 or F.shape[1] != 3:
        raise ValueError(f"F must have shape (n, 3); got {tuple(F.shape)}")
    if F.dtype != torch.int64:
        raise TypeError(f"F must be int64; got {F.dtype}")

    n_vertices = int(F.max().item()) + 1

    # All edges (i, j) per face, canonical (sorted) form so (i,j) == (j,i)
    e01 = F[:, [0, 1]]
    e12 = F[:, [1, 2]]
    e20 = F[:, [2, 0]]
    all_edges = torch.cat([e01, e12, e20], dim=0)        # (3 * n_faces, 2)
    canonical, _ = torch.sort(all_edges, dim=1)          # smaller index first

    # Count occurrences of each unique canonical edge
    unique_edges, counts = torch.unique(canonical, dim=0, return_counts=True)

    # Boundary edges appear exactly once
    boundary_edges = unique_edges[counts == 1]           # (n_boundary, 2)

    # Vertices touching any boundary edge are boundary vertices
    boundary = torch.zeros(n_vertices, dtype=torch.bool)
    boundary[boundary_edges[:, 0]] = True
    boundary[boundary_edges[:, 1]] = True
    return boundary

"""Per-vertex geometric features: normals and umbrella-Laplacian curvature.

Both functions are pure PyTorch tensor operations, fully differentiable
in their inputs. They are designed to be called every rollout iteration
(Stage 6) with autograd tracking through them.
"""
from __future__ import annotations

import torch


def compute_vertex_normals(
    V: torch.Tensor,
    F: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Area-weighted vertex normals for a triangle mesh.

    The contribution of each face to its three vertices is the raw
    (unnormalized) cross product e1 x e2, whose magnitude is 2x the
    triangle area. Summing these and normalizing per-vertex gives
    area-weighted vertex normals.

    Parameters
    ----------
    V : torch.Tensor, shape (n_vertices, 3), float
        Vertex positions.
    F : torch.Tensor, shape (n_faces, 3), int64
        Triangle indices.
    eps : float
        Small value added to the norm denominator for numerical stability
        (prevents 0/0 for isolated or degenerate vertices).

    Returns
    -------
    N : torch.Tensor, shape (n_vertices, 3), float
        Unit-length vertex normals, in the same dtype as V.
    """
    if V.dim() != 2 or V.shape[1] != 3:
        raise ValueError(f"V must have shape (n, 3); got {tuple(V.shape)}")
    if F.dim() != 2 or F.shape[1] != 3:
        raise ValueError(f"F must have shape (n, 3); got {tuple(F.shape)}")
    if F.dtype != torch.int64:
        raise TypeError(f"F must be int64; got {F.dtype}")

    # Per-face vertex positions
    v0 = V[F[:, 0]]  # (n_faces, 3)
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    # Raw face normals (magnitude = 2 * triangle area)
    face_normals = torch.linalg.cross(v1 - v0, v2 - v0, dim=1)  # (n_faces, 3)

    # Accumulate each face's contribution into its three vertices
    vertex_normals = torch.zeros_like(V)
    for k in range(3):
        vertex_normals.index_add_(0, F[:, k], face_normals)

    # Per-vertex normalization
    norm = torch.linalg.norm(vertex_normals, dim=1, keepdim=True)
    return vertex_normals / (norm + eps)


def compute_umbrella_laplacian(
    z: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Umbrella-operator discrete Laplacian, applied to a scalar field.

    For each vertex i:
        kappa_i = z_i - (1 / |N_i|) * sum_{j in N_i} z_j

    This is the simplest discrete curvature proxy: zero on planar regions,
    positive at peaks, negative at valleys (with the sign depending on
    surface orientation, but consistent across the mesh).

    The same expression is used as both:
      - an input feature to the GNN, and
      - the regularizer L_curv (its squared sum).

    Parameters
    ----------
    z : torch.Tensor, shape (n_vertices,), float
        Scalar field over vertices (e.g., the z-coordinate).
    edge_index : torch.Tensor, shape (2, n_directed_edges), int64
        PyG-style bidirectional edge list.

    Returns
    -------
    kappa : torch.Tensor, shape (n_vertices,), float
    """
    if z.dim() != 1:
        raise ValueError(f"z must be 1-D; got shape {tuple(z.shape)}")
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape (2, E); got {tuple(edge_index.shape)}"
        )

    src, dst = edge_index[0], edge_index[1]
    n = z.shape[0]

    # Sum of neighbor z-values per vertex
    z_neighbor_sum = torch.zeros_like(z)
    z_neighbor_sum.index_add_(0, src, z[dst])

    # Count of neighbors per vertex
    ones = torch.ones_like(z)
    degree = torch.zeros_like(z)
    degree.index_add_(0, src, ones[dst])

    # Avoid division by zero for isolated vertices (shouldn't happen in
    # connected meshes, but defensive)
    degree = degree.clamp(min=1.0)

    z_neighbor_mean = z_neighbor_sum / degree
    return z - z_neighbor_mean

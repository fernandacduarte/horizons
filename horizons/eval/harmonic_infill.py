"""Harmonic infill baseline.

Given a mesh with known z values on K and unknown values on U, find
z[U] such that each unknown vertex equals the mean of its neighbors:

    z[i] = (1/|N_i|) * sum(z[j] for j in N_i)  for i in U

This is equivalent to solving the discrete Laplace equation
delta(z) = 0 on U with Dirichlet boundary z[K] = z_true[K].

Linear-algebra formulation: with the graph Laplacian L = D - A
(D = degree matrix, A = adjacency matrix), partition into U and K
blocks:

    L = [ L_UU  L_UK ]
        [ L_KU  L_KK ]

We have z[K] fixed at z_true. The harmonic condition is L z = 0
restricted to U, giving:

    L_UU z[U] + L_UK z[K] = 0
    z[U] = -L_UU^{-1} L_UK z[K]

We solve via scipy.sparse.linalg.spsolve.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


def harmonic_infill(
    z_true: torch.Tensor,           # (n,) float — full ground-truth z
    edge_index: torch.Tensor,        # (2, n_directed_edges) int64
    mask: torch.Tensor,              # (n,) bool — True for K
) -> torch.Tensor:
    """Compute the harmonic infill on U given anchors on K.

    Returns z_pred of shape (n,) where z_pred[K] == z_true[K] exactly
    and z_pred[U] is the harmonic solution.
    """
    if z_true.dim() != 1:
        raise ValueError(f"z_true must be 1-D; got shape {tuple(z_true.shape)}")
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be bool; got {mask.dtype}")
    if mask.numel() != z_true.numel():
        raise ValueError(
            f"mask size {mask.numel()} != z_true size {z_true.numel()}"
        )

    n = z_true.numel()
    device = z_true.device
    dtype = z_true.dtype

    # Convert to numpy float64 for scipy
    z_true_np = z_true.detach().cpu().to(torch.float64).numpy()
    edge_np = edge_index.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    # Build the graph Laplacian L = D - A as a sparse matrix.
    # edge_index has directed edges (i, j) for every undirected edge,
    # so the adjacency matrix from row i to column j has A[i, j] = 1.
    src = edge_np[0]
    dst = edge_np[1]
    A = sp.coo_matrix(
        (np.ones(len(src)), (src, dst)),
        shape=(n, n),
        dtype=np.float64,
    ).tocsr()
    degree = np.asarray(A.sum(axis=1)).flatten()
    D = sp.diags(degree)
    L = (D - A).tocsr()

    # Partition L into UU, UK blocks.
    # We need to solve L_UU z_U = -L_UK z_K.
    U_idx = np.where(~mask_np)[0]
    K_idx = np.where(mask_np)[0]
    z_K = z_true_np[K_idx]

    L_UU = L[U_idx, :][:, U_idx]
    L_UK = L[U_idx, :][:, K_idx]

    rhs = -L_UK @ z_K

    # Solve. spsolve handles sparse LU decomposition.
    z_U = spla.spsolve(L_UU, rhs)

    # Assemble full z_pred
    z_pred_np = z_true_np.copy()
    z_pred_np[U_idx] = z_U

    return torch.from_numpy(z_pred_np).to(dtype=dtype, device=device)

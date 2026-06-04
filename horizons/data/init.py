"""Mean-plane initialization for z on unknown vertices.

Used identically in training and inference, so train/test conditions match.
Fits z = a*x + b*y + c by least squares on K, then
evaluates the fitted plane at all unknown vertices.
"""
from __future__ import annotations

import torch


def fit_mean_plane(
    xy_K: torch.Tensor, z_K: torch.Tensor,
) -> tuple[float, float, float]:
    """Least-squares fit of z ≈ a*x + b*y + c over the known vertices.

    Parameters
    ----------
    xy_K : torch.Tensor, shape (n_known, 2), float
    z_K  : torch.Tensor, shape (n_known,), float

    Returns
    -------
    (a, b, c) : tuple of Python floats
        Plane coefficients such that z(x, y) ≈ a*x + b*y + c.

    Raises
    ------
    ValueError
        If n_known < 3, or if the known points are collinear (singular fit).
    """
    if xy_K.dim() != 2 or xy_K.shape[1] != 2:
        raise ValueError(f"xy_K must have shape (n, 2); got {tuple(xy_K.shape)}")
    if z_K.dim() != 1 or z_K.shape[0] != xy_K.shape[0]:
        raise ValueError(
            f"z_K must have shape (n,) matching xy_K; got {tuple(z_K.shape)}"
        )
    if xy_K.shape[0] < 3:
        raise ValueError(f"Need at least 3 known points; got {xy_K.shape[0]}")

    # Design matrix [x, y, 1]
    ones = torch.ones(xy_K.shape[0], 1, dtype=xy_K.dtype)
    X = torch.cat([xy_K, ones], dim=1)  # (n_known, 3)

    # Solve X @ coeffs = z_K in the least-squares sense.
    # This is more numerically stable than forming inverse(X_K.T @ X_K).
    # lstsq is the numerically stable way; it handles rank-deficiency via SVD.
    solution = torch.linalg.lstsq(X, z_K.unsqueeze(1))
    coeffs = solution.solution.squeeze(1)  # (3,)

    # Sanity: check the fit isn't catastrophically singular. If the points are
    # collinear, lstsq still returns a solution but it's not meaningful.
    rank = int(torch.linalg.matrix_rank(X).item())
    if rank < 3:
        raise ValueError(
            f"Known points are collinear or insufficient (rank={rank}); "
            f"cannot fit a unique plane."
        )

    a, b, c = coeffs.tolist()
    return a, b, c


def init_z(
    V: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Build z^0 for the rollout.

    On known vertices, z^0 = z_true (the actual depth).
    On unknown vertices, z^0 = a*x + b*y + c from the mean plane fit through K.

    Parameters
    ----------
    V : torch.Tensor, shape (n_vertices, 3), float
        Vertex positions. V[:, 2] holds the true z values.
    mask : torch.Tensor, shape (n_vertices,), bool
        True for known vertices.

    Returns
    -------
    z0 : torch.Tensor, shape (n_vertices,), float
        Same dtype as V.
    """
    if V.dim() != 2 or V.shape[1] != 3:
        raise ValueError(f"V must have shape (n, 3); got {tuple(V.shape)}")
    if mask.shape != (V.shape[0],):
        raise ValueError(
            f"mask shape {tuple(mask.shape)} doesn't match V's vertex count {V.shape[0]}"
        )
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be bool; got {mask.dtype}")

    xy = V[:, :2]
    z_true = V[:, 2]
    z0 = z_true.clone()

    xy_K = xy[mask]
    z_K = z_true[mask]
    a, b, c = fit_mean_plane(xy_K, z_K)

    # Evaluate the plane at the unknown vertices and overwrite
    xy_U = xy[~mask]
    z_U_init = a * xy_U[:, 0] + b * xy_U[:, 1] + c
    z0[~mask] = z_U_init

    return z0

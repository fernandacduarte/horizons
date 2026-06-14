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

    # Promote to float64 for the linear algebra. Real-world geospatial
    # coordinates can have very large absolute values (UTM eastings ~1e5,
    # northings ~1e7); on float32 the mean computation itself loses
    # precision, so centering on float32 doesn't fully recover us. float64
    # gives us ~15 digits of precision, which is enough headroom for any
    # realistic coordinate system. The returned a, b, c are Python floats
    # so the downstream caller sees no dtype change.
    xy_K_d = xy_K.to(torch.float64)
    z_K_d = z_K.to(torch.float64)

    # Center x, y before solving (still helpful even in float64 for
    # conditioning, since the design matrix [x, y, 1] has very different
    # column norms when x ~ 1e7).
    xy_mean = xy_K_d.mean(dim=0)                                       # (2,)
    xy_centered = xy_K_d - xy_mean                                     # (n_known, 2)
    ones = torch.ones(xy_K_d.shape[0], 1, dtype=torch.float64)
    X = torch.cat([xy_centered, ones], dim=1)                          # (n_known, 3)

    # Rank check on the centered matrix is numerically reliable in float64.
    rank = int(torch.linalg.matrix_rank(X).item())
    if rank < 3:
        raise ValueError(
            f"Known points are collinear or insufficient (rank={rank}); "
            f"cannot fit a unique plane."
        )

    # Least-squares solve on centered coordinates. lstsq handles the full-rank
    # case correctly; we already checked rank > 2 above.
    solution = torch.linalg.lstsq(X, z_K_d.unsqueeze(1))
    coeffs = solution.solution.squeeze(1)                              # (3,)
    a_c, b_c, c_centered = coeffs.tolist()

    # Convert back to the original (uncentered) frame:
    #   z = a*x + b*y + c  where  c = c_centered - a*x_mean - b*y_mean
    a, b = a_c, b_c
    x_mean, y_mean = xy_mean.tolist()
    c = c_centered - a * x_mean - b * y_mean

    return a, b, c


def init_z(
    V: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Build z^0 for the rollout using mean plane.

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


def harmonic_init(
    V: torch.Tensor, mask: torch.Tensor, edge_index: torch.Tensor,
) -> torch.Tensor:
    """Build z^0 for the rollout using harmonic infill.

    On known vertices, z^0 = z_true (the actual depth).
    On unknown vertices, z^0 = harmonic infill solution given z[K].

    This is mathematically equivalent to applying harmonic_infill() to
    the true z values, then using that as the starting point for the
    learned operator. The operator only has to predict the *residual*
    from a strong baseline rather than learning extrapolation from
    scratch.

    Parameters
    ----------
    V : torch.Tensor, shape (n_vertices, 3), float
        Vertex positions. V[:, 2] holds the true z values.
    mask : torch.Tensor, shape (n_vertices,), bool
        True for known vertices.
    edge_index : torch.Tensor, shape (2, n_directed_edges), int64
        Graph connectivity for the mesh.

    Returns
    -------
    z0 : torch.Tensor, shape (n_vertices,), float
        Same dtype as V.

    Notes
    -----
    Computing harmonic infill requires solving a sparse linear system
    of size |U| × |U|. For meshes with |V| around 50k and typical
    mask sizes, this is roughly 1-2 seconds. If this becomes a
    bottleneck during training, we can consider caching per-(surface, seed)
    in the dataset.
    """
    # Local import to avoid a circular dependency if data/__init__.py
    # ever starts pulling everything in.
    from horizons.data.harmonic_infill import harmonic_infill

    if V.dim() != 2 or V.shape[1] != 3:
        raise ValueError(f"V must have shape (n, 3); got {tuple(V.shape)}")
    if mask.shape != (V.shape[0],):
        raise ValueError(
            f"mask shape {tuple(mask.shape)} doesn't match V's vertex count {V.shape[0]}"
        )
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be bool; got {mask.dtype}")

    z_true = V[:, 2]
    z0 = harmonic_infill(z_true, edge_index, mask)
    return z0.to(dtype=V.dtype)


def init_z_dispatch(
    V: torch.Tensor,
    mask: torch.Tensor,
    edge_index: torch.Tensor | None = None,
    *,
    method: str = "meanplane",
) -> torch.Tensor:
    """Build z^0 using the requested initialization method.

    Parameters
    ----------
    V : (n, 3) vertex positions.
    mask : (n,) bool, True for known vertices.
    edge_index : (2, n_directed_edges) int64, required only for harmonic init.
    method : "meanplane" or "harmonic".

    Returns
    -------
    z0 : (n,) float, same dtype as V.
    """
    if method == "meanplane":
        return init_z(V, mask)
    if method == "harmonic":
        if edge_index is None:
            raise ValueError(
                "harmonic init requires edge_index; got None."
            )
        return harmonic_init(V, mask, edge_index)
    raise ValueError(
        f"Unknown init method {method!r}; expected 'meanplane' or 'harmonic'."
    )

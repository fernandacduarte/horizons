"""Generate synthetic test-fixture surfaces.

Writes three .npz files into tests/fixtures/:
  - plane.npz      : tilted plane, no curvature
  - sphere_cap.npz : spherical cap, known outward normals
  - anticline.npz  : tilted plane + Gaussian bump (horizon-like)

Each .npz contains:
  V : (n_vertices, 3) float64 array of (x, y, z)
  F : (n_faces, 3)    int64   array of triangle vertex indices

Run from project root:
    python tests/fixtures/generate_fixtures.py
"""
import numpy as np
from pathlib import Path
from scipy.spatial import Delaunay


def _grid_with_jitter(n_side: int, extent: float, jitter: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Regular n_side x n_side grid in [-extent, extent]^2 with small jitter.
    Jitter avoids perfectly collinear points (which Delaunay dislikes)."""
    lin = np.linspace(-extent, extent, n_side)
    xs, ys = np.meshgrid(lin, lin)
    xy = np.stack([xs.ravel(), ys.ravel()], axis=1)
    xy += rng.uniform(-jitter, jitter, size=xy.shape)
    return xy


def _triangulate(xy: np.ndarray) -> np.ndarray:
    """Delaunay triangulation in (x, y). Returns int64 (n_faces, 3) array."""
    tri = Delaunay(xy)
    return tri.simplices.astype(np.int64)


def make_plane(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Tilted plane: z = 0.3 x + 0.1 y + 2.0  (geology-style depth units)."""
    rng = np.random.default_rng(seed)
    xy = _grid_with_jitter(n_side=20, extent=10.0, jitter=0.1, rng=rng)
    z = 0.3 * xy[:, 0] + 0.1 * xy[:, 1] + 2.0
    V = np.column_stack([xy, z])
    F = _triangulate(xy)
    return V, F


def make_sphere_cap(seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Cap of a sphere with R=20 centered at (0, 0, 20), evaluated over
    a disk of radius 8 in (x,y). z = 20 - sqrt(R^2 - x^2 - y^2)."""
    rng = np.random.default_rng(seed)
    R, center_z = 20.0, 20.0
    # Sample inside a disk instead of a square
    n = 400
    r = np.sqrt(rng.uniform(0, 1, size=n)) * 8.0
    theta = rng.uniform(0, 2 * np.pi, size=n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    xy = np.column_stack([x, y])
    z = center_z - np.sqrt(R**2 - x**2 - y**2)
    V = np.column_stack([xy, z])
    F = _triangulate(xy)
    return V, F


def make_anticline(seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Tilted plane + Gaussian bump. The horizon-like fixture we'll use
    for most end-to-end smoke tests."""
    rng = np.random.default_rng(seed)
    xy = _grid_with_jitter(n_side=30, extent=10.0, jitter=0.1, rng=rng)
    x, y = xy[:, 0], xy[:, 1]
    z = (
        0.2 * x + 0.1 * y + 2.0         # regional dip
        + 1.5 * np.exp(-(x**2 + y**2) / 8.0)  # central bump
    )
    V = np.column_stack([xy, z])
    F = _triangulate(xy)
    return V, F


def main() -> None:
    out_dir = Path(__file__).parent
    fixtures = {
        "plane": make_plane(),
        "sphere_cap": make_sphere_cap(),
        "anticline": make_anticline(),
    }
    for name, (V, F) in fixtures.items():
        out_path = out_dir / f"{name}.npz"
        np.savez(out_path, V=V, F=F)
        print(f"  {name}: {len(V)} vertices, {len(F)} faces -> {out_path}")
    print(f"\nWrote {len(fixtures)} fixtures.")


if __name__ == "__main__":
    main()

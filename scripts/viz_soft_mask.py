"""Visualize the soft_distance mask feature and check it is correct.

For each mask regime, shows three rows on the anticline fixture:
  (top)    binary mask K/U — what mask_mode="binary" feeds the network
  (middle) topological distance d
  (bottom) soft mask m — what mask_mode="soft_distance" feeds the network,
           computed through the ACTUAL model code path
           (LocalOperator._mask_feature), not re-derived here.

Correctness is also checked numerically before the window opens: for every
regime the script verifies
  - m = 1 exactly on known vertices,
  - m = 1 - d/(N+1) on unknown reachable vertices (N = max d),
  - m = 0 on unreachable vertices (d = -1),
  - m is strictly decreasing in d (ring means),
and prints PASS/FAIL per property.

Usage:
    python scripts/viz_soft_mask.py                  # default: seed 0
    python scripts/viz_soft_mask.py --seed 7         # different mask draw
    python scripts/viz_soft_mask.py --screenshot out.png   # headless save
"""
import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from horizons.data.mesh import HorizonSurface
from horizons.data.masking import (
    sample_half_plane_mask,
    sample_outward_rectangle_mask,
    sample_outward_rectangle_pinned_mask,
)
from horizons.data.topo_distance import compute_topological_distance, UNREACHABLE
from horizons.models.operator import LocalOperator


FIXTURES_DIR = Path("tests/fixtures")


def to_pv_mesh(surface: HorizonSurface) -> pv.PolyData:
    V_np = surface.V.numpy().astype(np.float64)
    F_np = surface.F.numpy()
    n_faces = F_np.shape[0]
    faces_pv = np.column_stack([np.full(n_faces, 3, dtype=np.int64), F_np])
    return pv.PolyData(V_np, faces_pv.ravel())


def check_soft_mask(
    title: str, mask: torch.Tensor, d: torch.Tensor, m: torch.Tensor
) -> None:
    """Verify the soft mask against its definition; print PASS/FAIL lines."""
    N = int(d.max().item())
    reachable_u = (~mask) & (d != UNREACHABLE)
    unreachable = d == UNREACHABLE

    checks: list[tuple[str, bool]] = []
    checks.append((
        "known vertices have m = 1 exactly",
        bool(torch.all(m[mask] == 1.0)),
    ))
    expected = 1.0 - d[reachable_u].float() / (N + 1)
    checks.append((
        f"unknown reachable follow m = 1 - d/(N+1), N={N}",
        bool(torch.allclose(m[reachable_u], expected)),
    ))
    checks.append((
        f"unreachable (n={int(unreachable.sum())}) have m = 0",
        bool(torch.all(m[unreachable] == 0.0)) if unreachable.any() else True,
    ))
    ring_means = [m[d == t].mean().item() for t in range(N + 1) if (d == t).any()]
    checks.append((
        "ring-mean m strictly decreases with d",
        all(a > b for a, b in zip(ring_means, ring_means[1:])),
    ))
    checks.append((
        "m within [0, 1]",
        bool((m.min() >= 0.0) and (m.max() <= 1.0)),
    ))

    print(f"  {title}:")
    for desc, ok in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {desc}")
    lo = m[reachable_u].min().item() if reachable_u.any() else float("nan")
    print(f"    m range on unknown: [{lo:.3f}, "
          f"{m[reachable_u].max().item():.3f}]  "
          f"(farthest ring should sit at 1/(N+1) = {1 / (N + 1):.3f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phi-half-plane", type=float, default=0.5)
    parser.add_argument("--phi-outward", type=float, default=0.6)
    parser.add_argument("--screenshot", type=Path, default=None,
                        help="save a PNG instead of opening a window")
    args = parser.parse_args()

    surface = HorizonSurface.from_npz(FIXTURES_DIR / "anticline.npz")

    rng_hp = torch.Generator().manual_seed(args.seed)
    mask_hp = sample_half_plane_mask(
        surface.V, phi=args.phi_half_plane, rng=rng_hp
    )
    rng_of = torch.Generator().manual_seed(args.seed + 1)
    mask_of = sample_outward_rectangle_mask(
        surface.V, phi=args.phi_outward, rng=rng_of
    )
    rng_op = torch.Generator().manual_seed(args.seed + 1)  # same seed -> nested
    mask_op = sample_outward_rectangle_pinned_mask(
        surface.V, surface.F, phi=args.phi_outward, rng=rng_op
    )

    masks = [
        (f"half_plane (phi={args.phi_half_plane})", mask_hp),
        (f"outward_free (phi={args.phi_outward})", mask_of),
        (f"outward_pinned (phi={args.phi_outward})", mask_op),
    ]

    # The model whose feature encoding we are validating — go through the
    # real code path so the plot shows exactly what the network receives.
    model = LocalOperator(mask_mode="soft_distance")

    print("=== numeric checks (soft_distance mask feature) ===")
    columns = []
    for title, mask in masks:
        d = compute_topological_distance(surface.edge_index, mask)
        m = model._mask_feature(mask, d, torch.float32).squeeze(1)
        check_soft_mask(title, mask, d, m)
        columns.append((title, mask, d, m))

    plotter = pv.Plotter(
        shape=(3, 3), window_size=(1800, 1500),
        off_screen=args.screenshot is not None,
    )

    for col, (title, mask, d, m) in enumerate(columns):
        max_d = int(d.max().item())

        # Top row: binary mask
        plotter.subplot(0, col)
        plotter.add_text(f"{title}\nbinary mask (yellow=K, purple=U)",
                         font_size=10)
        mesh = to_pv_mesh(surface)
        mesh["mask"] = mask.to(torch.int32).numpy()
        plotter.add_mesh(
            mesh, scalars="mask", cmap=["purple", "yellow"],
            clim=(0, 1), show_edges=True, edge_color="gray", line_width=0.3,
            show_scalar_bar=False,
        )

        # Middle row: topological distance d
        plotter.subplot(1, col)
        plotter.add_text(f"d (max d_i = {max_d})", font_size=10)
        mesh2 = to_pv_mesh(surface)
        mesh2["d"] = d.numpy().astype(np.float32)
        plotter.add_mesh(
            mesh2, scalars="d", cmap="viridis",
            clim=(0, max_d), show_edges=True,
            edge_color="gray", line_width=0.3,
        )

        # Bottom row: soft mask feature. Same clim for all columns so the
        # ramps are visually comparable; a correct m looks like an inverted,
        # rescaled d: exactly 1 on K, stepping down by 1/(N+1) per ring.
        plotter.subplot(2, col)
        plotter.add_text(
            f"soft mask m = 1 - d/(N+1)\n"
            f"step per ring = 1/{max_d + 1} = {1 / (max_d + 1):.3f}",
            font_size=10,
        )
        mesh3 = to_pv_mesh(surface)
        mesh3["m"] = m.numpy()
        plotter.add_mesh(
            mesh3, scalars="m", cmap="magma",
            clim=(0, 1), show_edges=True,
            edge_color="gray", line_width=0.3,
        )

    plotter.link_views()
    if args.screenshot is not None:
        plotter.screenshot(str(args.screenshot))
        print(f"\nsaved {args.screenshot}")
    else:
        plotter.show()


if __name__ == "__main__":
    main()

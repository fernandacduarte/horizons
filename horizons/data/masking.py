"""Mask sampling for masked-rollout training.

Three regimes:
  1. half_plane     : cut along a line at random orientation theta
  2. outward_free   : unknown frame around a central known rectangle
  3. outward_pinned : same as outward_free, plus a pinned outer ring

Convention
----------
A mask is a boolean tensor of shape (n_vertices,):
    mask[i] = True  <=>  vertex i is KNOWN (i in K)
    mask[i] = False <=>  vertex i is UNKNOWN (i in U)
"""
from __future__ import annotations

import math

import torch


def sample_half_plane_mask(
    V: torch.Tensor,
    phi: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """Half-plane cut: mark phi * n_vertices vertices as unknown, chosen as
    the ones with smallest signed distance to a line oriented at random theta.

    The line has normal (cos theta, sin theta) and passes through the (x,y)
    centroid of V. The rank-based selection means the actual cut sits at
    whatever offset captures exactly the target unknown fraction.

    Parameters
    ----------
    V : torch.Tensor, shape (n_vertices, 3), float
        Vertex positions; only V[:, :2] is used.
    phi : float
        Target fraction of vertices to mark as unknown, in (0, 1).
    rng : torch.Generator
        RNG used for theta. Pass a generator that's been seeded externally
        for reproducibility.

    Returns
    -------
    mask : torch.Tensor, shape (n_vertices,), bool
        True for known vertices, False for unknown.
    """
    if not (0.0 < phi < 1.0):
        raise ValueError(f"phi must be in (0, 1); got {phi}")
    if V.dim() != 2 or V.shape[1] != 3:
        raise ValueError(f"V must have shape (n, 3); got {tuple(V.shape)}")

    n = V.shape[0]
    xy = V[:, :2]
    centroid_xy = xy.mean(dim=0)

    # Sample orientation theta uniformly in [0, 2 pi)
    theta = torch.rand((), generator=rng).item() * 2.0 * math.pi
    normal = torch.tensor(
        [math.cos(theta), math.sin(theta)], dtype=xy.dtype
    )

    # Signed distance from each vertex to the line through the centroid
    # with the chosen normal: d_i = (xy_i - centroid) . normal
    signed_dist = (xy - centroid_xy) @ normal  # (n_vertices,)

    # Mark the smallest phi * n vertices as unknown.
    # "Smallest signed distance" = "most negative" = "on the negative-normal side"
    n_unknown = int(round(phi * n))
    # Guard: never produce all-known or all-unknown masks
    n_unknown = max(1, min(n - 1, n_unknown))

    sorted_idx = torch.argsort(signed_dist)        # ascending
    unknown_idx = sorted_idx[:n_unknown]

    mask = torch.ones(n, dtype=torch.bool)
    mask[unknown_idx] = False
    return mask


def sample_outward_rectangle_mask(
    V: torch.Tensor,
    phi: float,
    rng: torch.Generator,
    offset_std_frac: float = 0.05,
    aspect_range: tuple[float, float] = (0.5, 2.0),
) -> torch.Tensor:
    """Outward-from-rectangle mask (free boundary).

    The known region is a rectangle near the mesh center; the unknown
    region is everything outside it. The rectangle's size is set so that
    exactly (1 - phi) * n_vertices vertices fall inside.

    Parameters
    ----------
    V : torch.Tensor, shape (n_vertices, 3), float
        Vertex positions; only V[:, :2] is used.
    phi : float
        Fraction of vertices to mark as unknown, in (0, 1).
    rng : torch.Generator
        Seeded generator for reproducibility.
    offset_std_frac : float
        Standard deviation of the rectangle-center offset, as a fraction
        of the bounding-box half-extent. Default 0.05 keeps the rectangle
        near the centroid.
    aspect_range : tuple[float, float]
        (min, max) for the width-to-height ratio of the rectangle.
        Uniformly sampled from this range. Default (0.5, 2.0) covers a
        2:1 range in both directions.

    Returns
    -------
    mask : torch.Tensor, shape (n_vertices,), bool
        True for known (inside rectangle), False for unknown (outside).
    """
    if not (0.0 < phi < 1.0):
        raise ValueError(f"phi must be in (0, 1); got {phi}")
    if V.dim() != 2 or V.shape[1] != 3:
        raise ValueError(f"V must have shape (n, 3); got {tuple(V.shape)}")

    n = V.shape[0]
    xy = V[:, :2]
    xy_min = xy.min(dim=0).values
    xy_max = xy.max(dim=0).values
    extent = (xy_max - xy_min) / 2.0  # half-extent of bounding box
    centroid = (xy_max + xy_min) / 2.0

    # Random offset from the bounding-box center (NOT the mesh centroid).
    # Using the bbox center makes offset_std_frac comparable across meshes
    # regardless of vertex density distribution.
    offset = (
        torch.randn(2, generator=rng, dtype=xy.dtype) * offset_std_frac * extent
    )
    rect_center = centroid + offset

    # Random aspect ratio (width / height)
    a_min, a_max = aspect_range
    aspect = (
        torch.rand((), generator=rng, dtype=xy.dtype).item() * (a_max - a_min)
        + a_min
    )

    # Scaled L-infinity distance from the rectangle center:
    # d(x, y) = max(|x - cx| / aspect, |y - cy|)
    # Level sets are rectangles of width:height = aspect:1.
    centered = xy - rect_center
    d = torch.maximum(centered[:, 0].abs() / aspect, centered[:, 1].abs())

    # Mark the (1-phi)*n vertices with smallest d as known
    n_known = int(round((1.0 - phi) * n))
    n_known = max(1, min(n - 1, n_known))

    sorted_idx = torch.argsort(d)        # ascending: closest to center first
    known_idx = sorted_idx[:n_known]

    mask = torch.zeros(n, dtype=torch.bool)
    mask[known_idx] = True
    return mask


def sample_outward_rectangle_pinned_mask(
    V: torch.Tensor,
    F: torch.Tensor,
    phi: float,
    rng: torch.Generator,
    offset_std_frac: float = 0.05,
    aspect_range: tuple[float, float] = (0.5, 2.0),
) -> torch.Tensor:
    """Outward-from-rectangle mask with the mesh boundary pinned to known
    (pinned boundary).

    Produces a rectangle mask, then re-adds the mesh boundary vertices to K.
    The result: known central rectangle + known thin outer ring;
    unknown = the annular region between them.

    Note: because some originally-unknown vertices are back to known,
    the effective unknown fraction will be slightly smaller than phi.

    Parameters
    ----------
    V, phi, rng, offset_std_frac, aspect_range :
        See `sample_outward_rectangle_mask`.
    F : torch.Tensor, shape (n_faces, 3), int64
        Face indices, needed to compute boundary vertices.

    Returns
    -------
    mask : torch.Tensor, shape (n_vertices,), bool
    """
    # Import here to avoid circular imports at module level
    from horizons.data.mesh import compute_boundary_vertices

    mask = sample_outward_rectangle_mask(
        V, phi=phi, rng=rng,
        offset_std_frac=offset_std_frac,
        aspect_range=aspect_range,
    )

    # Pin boundary vertices to known
    boundary = compute_boundary_vertices(F)
    mask = mask | boundary
    return mask


# ======================================================================
# MaskSampler: the configurable entry point used by the dataset class
# ======================================================================
from dataclasses import dataclass, field
from typing import Any

from horizons.data.topo_distance import (
    compute_topological_distance,
    UNREACHABLE,
)


@dataclass
class MaskSamplerConfig:
    """Configuration for MaskSampler. Mirrors the `mask:` section of YAML."""
    # Regime mix (weights need not sum to 1; we normalize)
    regime_weights: dict[str, float] = field(
        default_factory=lambda: {
            "half_plane": 0.30,
            "outward_free": 0.40,
            "outward_pinned": 0.30,
        }
    )
    half_plane_phi: list[float] = field(default_factory=lambda: [0.30, 0.50])
    outward_phi: list[float] = field(default_factory=lambda: [0.50, 0.70])
    outward_rect_offset_std: float = 0.05
    outward_rect_aspect_range: tuple[float, float] = (0.5, 2.0)
    pinned_ring_thickness: int = 1  # currently only 1 is supported

    # Retry policy
    max_retries: int = 32

    @classmethod
    def from_dictconfig(cls, cfg: Any) -> "MaskSamplerConfig":
        """Build from a Hydra DictConfig (or any dict-like).

        Accepts the `mask:` section of the YAML config. Unknown keys are
        ignored (forward compatibility with future config additions).
        """
        # Convert OmegaConf containers to plain Python for our dataclass
        from omegaconf import OmegaConf
        if hasattr(cfg, "_content"):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        return cls(
            regime_weights=dict(cfg["regime_weights"]),
            half_plane_phi=list(cfg["half_plane_phi"]),
            outward_phi=list(cfg["outward_phi"]),
            outward_rect_offset_std=float(cfg["outward_rect_offset_std"]),
            outward_rect_aspect_range=tuple(cfg["outward_rect_aspect_range"]),
            pinned_ring_thickness=int(cfg["pinned_ring_thickness"]),
        )


class MaskSampler:
    """Samples masks from the configured mixture of regimes.

    Usage
    -----
        sampler = MaskSampler(cfg.mask)
        rng = torch.Generator().manual_seed(seed)
        mask, d, regime = sampler.sample(surface, rng)
    """

    REGIMES = ("half_plane", "outward_free", "outward_pinned")

    def __init__(self, config: MaskSamplerConfig | Any) -> None:
        if not isinstance(config, MaskSamplerConfig):
            config = MaskSamplerConfig.from_dictconfig(config)
        self._validate_config(config)
        self.cfg = config

        # Pre-normalize regime weights for sampling
        names = list(self.cfg.regime_weights.keys())
        weights = torch.tensor(
            [self.cfg.regime_weights[n] for n in names], dtype=torch.float64
        )
        self._regime_names = names
        self._regime_probs = weights / weights.sum()

    @staticmethod
    def _validate_config(cfg: MaskSamplerConfig) -> None:
        for name in cfg.regime_weights:
            if name not in MaskSampler.REGIMES:
                raise ValueError(
                    f"Unknown regime {name!r}; valid: {MaskSampler.REGIMES}"
                )
        if not all(w >= 0 for w in cfg.regime_weights.values()):
            raise ValueError("All regime weights must be non-negative")
        if sum(cfg.regime_weights.values()) <= 0:
            raise ValueError("At least one regime must have positive weight")
        if cfg.pinned_ring_thickness != 1:
            raise NotImplementedError(
                "Only pinned_ring_thickness=1 is currently supported"
            )
        if not all(0 < p < 1 for p in cfg.half_plane_phi):
            raise ValueError("All half_plane_phi values must be in (0, 1)")
        if not all(0 < p < 1 for p in cfg.outward_phi):
            raise ValueError("All outward_phi values must be in (0, 1)")

    def _sample_regime(self, rng: torch.Generator) -> str:
        idx = torch.multinomial(self._regime_probs, num_samples=1, generator=rng)
        return self._regime_names[idx.item()]

    @staticmethod
    def _sample_from_list(
        values: list[float], rng: torch.Generator
    ) -> float:
        """Uniform sample from a discrete list."""
        idx = torch.randint(0, len(values), (1,), generator=rng).item()
        return values[idx]

    def _sample_one(
        self, surface, regime: str, rng: torch.Generator,
    ) -> torch.Tensor:
        """Dispatch to the right regime sampler."""
        if regime == "half_plane":
            phi = self._sample_from_list(self.cfg.half_plane_phi, rng)
            return sample_half_plane_mask(surface.V, phi=phi, rng=rng)
        elif regime == "outward_free":
            phi = self._sample_from_list(self.cfg.outward_phi, rng)
            return sample_outward_rectangle_mask(
                surface.V, phi=phi, rng=rng,
                offset_std_frac=self.cfg.outward_rect_offset_std,
                aspect_range=self.cfg.outward_rect_aspect_range,
            )
        elif regime == "outward_pinned":
            phi = self._sample_from_list(self.cfg.outward_phi, rng)
            return sample_outward_rectangle_pinned_mask(
                surface.V, surface.F, phi=phi, rng=rng,
                offset_std_frac=self.cfg.outward_rect_offset_std,
                aspect_range=self.cfg.outward_rect_aspect_range,
            )
        else:
            raise ValueError(f"Unknown regime {regime!r}")
            
    def sample(
        self, surface, rng: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Sample a mask and its topological distance.

        Parameters
        ----------
        surface : HorizonSurface
            The mesh to mask.
        rng : torch.Generator
            Externally seeded for reproducibility.

        Returns
        -------
        mask : torch.Tensor, shape (n_vertices,), bool
            True for known, False for unknown.
        d : torch.Tensor, shape (n_vertices,), int64
            Topological distance from K. d[i] >= 0 for all i (no UNREACHABLE).
        regime : str
            Name of the regime used: "half_plane", "outward_free", or
            "outward_pinned".

        Raises
        ------
        RuntimeError
            If after `max_retries` attempts, every sampled mask had a
            disconnected unknown component. Indicates the mesh is pathological
            or the regime parameters are extreme.
        """
        for attempt in range(self.cfg.max_retries):
            regime = self._sample_regime(rng)
            mask = self._sample_one(surface, regime, rng)
            d = compute_topological_distance(surface.edge_index, mask)
            if (d != UNREACHABLE).all():
                return mask, d, regime
            # else: try again with a fresh draw

        raise RuntimeError(
            f"Failed to sample a connected mask after "
            f"{self.cfg.max_retries} attempts on surface "
            f"{surface.surface_id!r}. Last regime tried: {regime!r}."
        )


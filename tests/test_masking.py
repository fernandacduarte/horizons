"""Unit tests for horizons.data.masking."""
from pathlib import Path

import pytest
import torch

from horizons.data.mesh import HorizonSurface
from horizons.data.masking import sample_half_plane_mask


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def anticline() -> HorizonSurface:
    return HorizonSurface.from_npz(FIXTURES_DIR / "anticline.npz")


# ----------------------------------------------------------------------
# sample_half_plane_mask
# ----------------------------------------------------------------------
class TestHalfPlaneMask:
    def test_shape_and_dtype(self, anticline: HorizonSurface) -> None:
        rng = torch.Generator().manual_seed(0)
        mask = sample_half_plane_mask(anticline.V, phi=0.5, rng=rng)
        assert mask.shape == (anticline.n_vertices,)
        assert mask.dtype == torch.bool

    @pytest.mark.parametrize("phi", [0.3, 0.5, 0.7])
    def test_unknown_fraction_correct(
        self, anticline: HorizonSurface, phi: float
    ) -> None:
        """The fraction of unknown vertices should match phi (up to rounding)."""
        rng = torch.Generator().manual_seed(0)
        mask = sample_half_plane_mask(anticline.V, phi=phi, rng=rng)
        n_unknown = (~mask).sum().item()
        expected = round(phi * anticline.n_vertices)
        assert abs(n_unknown - expected) <= 1  # tolerate ±1 rounding

    def test_known_and_unknown_both_nonempty(
        self, anticline: HorizonSurface
    ) -> None:
        """Even at extreme phi, mask should have at least one of each."""
        rng = torch.Generator().manual_seed(0)
        for phi in [0.001, 0.999]:
            mask = sample_half_plane_mask(anticline.V, phi=phi, rng=rng)
            assert mask.any() and (~mask).any(), (
                f"phi={phi} produced an all-{mask.all().item()} mask"
            )

    def test_reproducible_with_seed(self, anticline: HorizonSurface) -> None:
        """Same seed -> same mask."""
        mask1 = sample_half_plane_mask(
            anticline.V, phi=0.5, rng=torch.Generator().manual_seed(42)
        )
        mask2 = sample_half_plane_mask(
            anticline.V, phi=0.5, rng=torch.Generator().manual_seed(42)
        )
        assert torch.equal(mask1, mask2)

    def test_different_seeds_different_masks(
        self, anticline: HorizonSurface
    ) -> None:
        """Different seeds should generally give different masks (probabilistic
        but vanishingly unlikely to coincide for 900 vertices)."""
        mask1 = sample_half_plane_mask(
            anticline.V, phi=0.5, rng=torch.Generator().manual_seed(1)
        )
        mask2 = sample_half_plane_mask(
            anticline.V, phi=0.5, rng=torch.Generator().manual_seed(2)
        )
        assert not torch.equal(mask1, mask2)

    def test_unknown_region_is_a_half_plane(
        self, anticline: HorizonSurface
    ) -> None:
        """The sampler partitions vertices by rank along *some* direction.
        We verify this by searching for a direction that perfectly separates
        K and U via signed-distance rank. If the sampler is correct, the
        true theta it used must be among (or very close to) the candidates,
        so at least one candidate must achieve perfect separation.

        We use a fine angular grid (1-degree resolution) — much finer than
        needed since the sampler samples theta continuously, but we
        independently picked a discrete grid here for the test."""
        rng = torch.Generator().manual_seed(7)
        mask = sample_half_plane_mask(anticline.V, phi=0.5, rng=rng)
        xy = anticline.V[:, :2]
        centroid = xy.mean(dim=0)
        n_unknown = (~mask).sum().item()

        # Try directions at 1-degree increments
        n_angles = 360
        angles = torch.linspace(0, 2 * 3.14159265358979, n_angles + 1)[:-1]
        best_match = 0
        for theta in angles:
            normal = torch.tensor([theta.cos(), theta.sin()], dtype=xy.dtype)
            sd = (xy - centroid) @ normal
            # The n_unknown vertices with smallest sd should match ~mask
            sorted_idx = torch.argsort(sd)
            predicted_unknown = torch.zeros_like(mask)
            predicted_unknown[sorted_idx[:n_unknown]] = True
            # Compare to actual unknown set
            agreement = (predicted_unknown == ~mask).sum().item()
            if agreement > best_match:
                best_match = agreement

        # At least one direction in our grid should reproduce the mask
        # perfectly (or nearly so — within rounding due to discrete grid).
        n = anticline.n_vertices
        assert best_match >= n - 2, (
            f"No half-plane direction reproduces the mask: best match "
            f"{best_match}/{n}. Sampler is not producing a half-plane cut."
        )

    def test_phi_validation(self, anticline: HorizonSurface) -> None:
        rng = torch.Generator().manual_seed(0)
        with pytest.raises(ValueError, match="phi must be"):
            sample_half_plane_mask(anticline.V, phi=0.0, rng=rng)
        with pytest.raises(ValueError, match="phi must be"):
            sample_half_plane_mask(anticline.V, phi=1.0, rng=rng)
        with pytest.raises(ValueError, match="phi must be"):
            sample_half_plane_mask(anticline.V, phi=-0.1, rng=rng)

    def test_V_shape_validation(self) -> None:
        rng = torch.Generator().manual_seed(0)
        with pytest.raises(ValueError, match="V must have shape"):
            sample_half_plane_mask(torch.zeros(10), phi=0.5, rng=rng)
        with pytest.raises(ValueError, match="V must have shape"):
            sample_half_plane_mask(torch.zeros(10, 2), phi=0.5, rng=rng)


# ----------------------------------------------------------------------
# sample_outward_rectangle_mask
# ----------------------------------------------------------------------
from horizons.data.masking import sample_outward_rectangle_mask


class TestOutwardRectangleMask:
    def test_shape_and_dtype(self, anticline: HorizonSurface) -> None:
        rng = torch.Generator().manual_seed(0)
        mask = sample_outward_rectangle_mask(anticline.V, phi=0.5, rng=rng)
        assert mask.shape == (anticline.n_vertices,)
        assert mask.dtype == torch.bool

    @pytest.mark.parametrize("phi", [0.5, 0.7])
    def test_unknown_fraction_correct(
        self, anticline: HorizonSurface, phi: float
    ) -> None:
        rng = torch.Generator().manual_seed(0)
        mask = sample_outward_rectangle_mask(anticline.V, phi=phi, rng=rng)
        n_unknown = (~mask).sum().item()
        expected = round(phi * anticline.n_vertices)
        assert abs(n_unknown - expected) <= 1

    def test_known_region_is_rectangle(
        self, anticline: HorizonSurface
    ) -> None:
        """The known region's bounding box (in xy) should not contain any
        unknown vertices — verifying that the partition is by axis-aligned
        rectangle membership (in the aspect-scaled metric).

        Specifically: there must exist some axis-aligned rectangle such that
        all known vertices are inside it and all unknown vertices are outside.
        We find the candidate rectangle as the axis-aligned bounding box of
        the known set, then verify that no unknown vertex lies strictly inside.
        """
        rng = torch.Generator().manual_seed(5)
        mask = sample_outward_rectangle_mask(anticline.V, phi=0.6, rng=rng)
        xy = anticline.V[:, :2]
        xy_K = xy[mask]
        xy_U = xy[~mask]

        # Bounding box of known vertices
        K_min = xy_K.min(dim=0).values
        K_max = xy_K.max(dim=0).values

        # No unknown vertex should be strictly inside the K bounding box.
        # "Strictly inside" allows boundary touches because the sampler's
        # rank-based selection may place a vertex exactly at the boundary.
        # We tolerate at most a small number of boundary cases.
        inside = (
            (xy_U[:, 0] > K_min[0]) & (xy_U[:, 0] < K_max[0]) &
            (xy_U[:, 1] > K_min[1]) & (xy_U[:, 1] < K_max[1])
        )
        # Strict containment should be empty (or near-empty for tie cases)
        n_violations = inside.sum().item()
        assert n_violations <= 2, (
            f"Found {n_violations} unknown vertices strictly inside the "
            f"known region's bounding box; expected 0-2 (boundary ties only)."
        )

    def test_known_region_centered_ish(
        self, anticline: HorizonSurface
    ) -> None:
        """The center of the known region's bbox should be near the mesh
        bbox center. With default offset_std_frac=0.05, the offset is
        typically a few percent of the half-extent."""
        rng = torch.Generator().manual_seed(0)
        mask = sample_outward_rectangle_mask(anticline.V, phi=0.5, rng=rng)
        xy = anticline.V[:, :2]
        xy_K = xy[mask]

        K_center = (xy_K.max(dim=0).values + xy_K.min(dim=0).values) / 2.0
        mesh_center = (xy.max(dim=0).values + xy.min(dim=0).values) / 2.0
        half_extent = (xy.max(dim=0).values - xy.min(dim=0).values) / 2.0

        offset_frac = ((K_center - mesh_center).abs() / half_extent).max()
        # Generous bound: well within 30% of half-extent for any reasonable seed
        assert offset_frac < 0.30, (
            f"Known region is too off-center: offset = {offset_frac:.2%} "
            f"of half-extent"
        )

    def test_reproducible_with_seed(self, anticline: HorizonSurface) -> None:
        mask1 = sample_outward_rectangle_mask(
            anticline.V, phi=0.5, rng=torch.Generator().manual_seed(42)
        )
        mask2 = sample_outward_rectangle_mask(
            anticline.V, phi=0.5, rng=torch.Generator().manual_seed(42)
        )
        assert torch.equal(mask1, mask2)

    def test_different_seeds_different_masks(
        self, anticline: HorizonSurface
    ) -> None:
        mask1 = sample_outward_rectangle_mask(
            anticline.V, phi=0.5, rng=torch.Generator().manual_seed(1)
        )
        mask2 = sample_outward_rectangle_mask(
            anticline.V, phi=0.5, rng=torch.Generator().manual_seed(2)
        )
        assert not torch.equal(mask1, mask2)

    def test_phi_validation(self, anticline: HorizonSurface) -> None:
        rng = torch.Generator().manual_seed(0)
        with pytest.raises(ValueError, match="phi must be"):
            sample_outward_rectangle_mask(anticline.V, phi=0.0, rng=rng)
        with pytest.raises(ValueError, match="phi must be"):
            sample_outward_rectangle_mask(anticline.V, phi=1.5, rng=rng)


# ----------------------------------------------------------------------
# sample_outward_rectangle_pinned_mask
# ----------------------------------------------------------------------
from horizons.data.masking import sample_outward_rectangle_pinned_mask
from horizons.data.mesh import compute_boundary_vertices


class TestOutwardRectanglePinnedMask:
    def test_shape_and_dtype(self, anticline: HorizonSurface) -> None:
        rng = torch.Generator().manual_seed(0)
        mask = sample_outward_rectangle_pinned_mask(
            anticline.V, anticline.F, phi=0.5, rng=rng
        )
        assert mask.shape == (anticline.n_vertices,)
        assert mask.dtype == torch.bool

    def test_all_boundary_vertices_are_known(
        self, anticline: HorizonSurface
    ) -> None:
        """Defining property: every boundary vertex is in K."""
        rng = torch.Generator().manual_seed(3)
        mask = sample_outward_rectangle_pinned_mask(
            anticline.V, anticline.F, phi=0.6, rng=rng
        )
        boundary = compute_boundary_vertices(anticline.F)
        # All boundary vertices must be known (mask=True)
        assert mask[boundary].all(), (
            "Some boundary vertices are not marked as known"
        )

    def test_more_known_than_unpinned_regime(
        self, anticline: HorizonSurface
    ) -> None:
        """The pinned mask should have strictly more known vertices than
        the corresponding unpinned mask with the same seed and phi
        (because we pin boundary vertices that were previously unknown)."""
        rng_a = torch.Generator().manual_seed(11)
        rng_b = torch.Generator().manual_seed(11)
        mask_unpinned = sample_outward_rectangle_mask(
            anticline.V, phi=0.6, rng=rng_a
        )
        mask_pinned = sample_outward_rectangle_pinned_mask(
            anticline.V, anticline.F, phi=0.6, rng=rng_b
        )
        n_known_unpinned = mask_unpinned.sum().item()
        n_known_pinned = mask_pinned.sum().item()
        assert n_known_pinned > n_known_unpinned, (
            f"Pinned mask should have more known vertices: "
            f"unpinned={n_known_unpinned}, pinned={n_known_pinned}"
        )

    def test_pinned_mask_is_superset_of_unpinned(
        self, anticline: HorizonSurface
    ) -> None:
        """Every vertex known in the unpinned mask should remain known in
        the pinned mask (we only add to K, never remove)."""
        rng_a = torch.Generator().manual_seed(11)
        rng_b = torch.Generator().manual_seed(11)
        mask_unpinned = sample_outward_rectangle_mask(
            anticline.V, phi=0.6, rng=rng_a
        )
        mask_pinned = sample_outward_rectangle_pinned_mask(
            anticline.V, anticline.F, phi=0.6, rng=rng_b
        )
        # mask_unpinned True ==> mask_pinned True (pinned ⊇ unpinned)
        assert (mask_pinned[mask_unpinned]).all()

    def test_reproducible_with_seed(self, anticline: HorizonSurface) -> None:
        mask1 = sample_outward_rectangle_pinned_mask(
            anticline.V, anticline.F, phi=0.5,
            rng=torch.Generator().manual_seed(42),
        )
        mask2 = sample_outward_rectangle_pinned_mask(
            anticline.V, anticline.F, phi=0.5,
            rng=torch.Generator().manual_seed(42),
        )
        assert torch.equal(mask1, mask2)

    def test_unknown_region_in_annulus(
        self, anticline: HorizonSurface
    ) -> None:
        """The unknown region should sit between the central rectangle
        and the outer boundary. Check by partition:
        - All boundary vertices: known
        - Some interior vertices: known (the rectangle)
        - Some interior vertices: unknown (the annulus)
        And there should be at least some unknown vertices."""
        rng = torch.Generator().manual_seed(7)
        mask = sample_outward_rectangle_pinned_mask(
            anticline.V, anticline.F, phi=0.6, rng=rng
        )
        boundary = compute_boundary_vertices(anticline.F)
        n_unknown = (~mask).sum().item()
        assert n_unknown > 0, "No unknown vertices in pinned mask"
        # Sanity: no unknown vertex is on the boundary
        assert not (~mask & boundary).any()


# ======================================================================
# MaskSampler
# ======================================================================
from horizons.data.masking import MaskSampler, MaskSamplerConfig
from horizons.data.topo_distance import UNREACHABLE


class TestMaskSamplerConfig:
    def test_defaults(self) -> None:
        cfg = MaskSamplerConfig()
        # The three regimes should be present and sum to ~1
        assert set(cfg.regime_weights.keys()) == {
            "half_plane", "outward_free", "outward_pinned",
        }
        total = sum(cfg.regime_weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_from_dictconfig(self) -> None:
        """Loading from a Hydra DictConfig works end-to-end."""
        from omegaconf import OmegaConf
        yaml_like = {
            "regime_weights": {
                "half_plane": 0.5, "outward_free": 0.3, "outward_pinned": 0.2,
            },
            "half_plane_phi": [0.3, 0.5],
            "outward_phi": [0.5, 0.7],
            "outward_rect_offset_std": 0.05,
            "outward_rect_aspect_range": [0.5, 2.0],
            "pinned_ring_thickness": 1,
        }
        oc = OmegaConf.create(yaml_like)
        cfg = MaskSamplerConfig.from_dictconfig(oc)
        assert cfg.regime_weights["half_plane"] == 0.5
        assert cfg.outward_rect_aspect_range == (0.5, 2.0)


class TestMaskSampler:
    def test_basic_sample(self, anticline: HorizonSurface) -> None:
        sampler = MaskSampler(MaskSamplerConfig())
        rng = torch.Generator().manual_seed(0)
        mask, d, regime = sampler.sample(anticline, rng)
        assert mask.shape == (anticline.n_vertices,)
        assert d.shape == (anticline.n_vertices,)
        assert mask.dtype == torch.bool
        assert d.dtype == torch.int64
        assert regime in MaskSampler.REGIMES
        # Connected: no UNREACHABLE
        assert (d != UNREACHABLE).all()
        # Known vertices have d=0
        assert (d[mask] == 0).all()
        # Unknown vertices have d > 0
        assert (d[~mask] > 0).all()

    def test_reproducibility(self, anticline: HorizonSurface) -> None:
        sampler = MaskSampler(MaskSamplerConfig())
        m1, d1, r1 = sampler.sample(
            anticline, torch.Generator().manual_seed(42)
        )
        m2, d2, r2 = sampler.sample(
            anticline, torch.Generator().manual_seed(42)
        )
        assert torch.equal(m1, m2)
        assert torch.equal(d1, d2)
        assert r1 == r2

    def test_regime_mix(self, anticline: HorizonSurface) -> None:
        """Over many samples, observed regime frequencies should match
        the configured weights (within statistical noise)."""
        cfg = MaskSamplerConfig(
            regime_weights={
                "half_plane": 0.5, "outward_free": 0.3, "outward_pinned": 0.2,
            }
        )
        sampler = MaskSampler(cfg)
        rng = torch.Generator().manual_seed(0)
        n_trials = 400
        counts = {"half_plane": 0, "outward_free": 0, "outward_pinned": 0}
        for _ in range(n_trials):
            _, _, regime = sampler.sample(anticline, rng)
            counts[regime] += 1
        # Each frequency should be within +/- 6% of its target
        # (binomial std for p=0.5, n=400 is ~0.025, so 6% is ~2.4 sigma)
        assert abs(counts["half_plane"] / n_trials - 0.5) < 0.06
        assert abs(counts["outward_free"] / n_trials - 0.3) < 0.06
        assert abs(counts["outward_pinned"] / n_trials - 0.2) < 0.06

    def test_invalid_regime_name_rejected(self) -> None:
        cfg = MaskSamplerConfig(regime_weights={"bogus": 1.0})
        with pytest.raises(ValueError, match="Unknown regime"):
            MaskSampler(cfg)

    def test_negative_weight_rejected(self) -> None:
        cfg = MaskSamplerConfig(
            regime_weights={"half_plane": -0.1, "outward_free": 1.1}
        )
        with pytest.raises(ValueError, match="non-negative"):
            MaskSampler(cfg)

    def test_unsupported_ring_thickness_rejected(self) -> None:
        cfg = MaskSamplerConfig(pinned_ring_thickness=2)
        with pytest.raises(NotImplementedError):
            MaskSampler(cfg)

    def test_known_vertices_consistent_across_regimes(
        self, anticline: HorizonSurface
    ) -> None:
        """For every regime sampled, d[i] == 0 iff mask[i] is True.
        This is the central invariant tying mask and d together."""
        sampler = MaskSampler(MaskSamplerConfig())
        rng = torch.Generator().manual_seed(0)
        for _ in range(20):
            mask, d, _ = sampler.sample(anticline, rng)
            assert torch.equal(mask, d == 0)

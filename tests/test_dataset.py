"""Unit tests for horizons.data.dataset."""
from pathlib import Path

import pytest
import torch

from horizons.data.dataset import HorizonDataset, load_fixture_dataset
from horizons.data.mesh import HorizonSurface
from horizons.data.masking import MaskSampler, MaskSamplerConfig


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def train_dataset() -> HorizonDataset:
    return load_fixture_dataset(
        ["plane", "sphere_cap", "anticline"],
        fixtures_dir=FIXTURES_DIR,
        split="train",
    )


@pytest.fixture
def val_dataset() -> HorizonDataset:
    return load_fixture_dataset(
        ["plane", "sphere_cap", "anticline"],
        fixtures_dir=FIXTURES_DIR,
        split="val",
    )


# ----------------------------------------------------------------------
# Basic dataset behavior
# ----------------------------------------------------------------------
class TestBasic:
    def test_length(self, train_dataset: HorizonDataset) -> None:
        assert len(train_dataset) == 3

    def test_empty_dataset_rejected(self) -> None:
        from horizons.data.masking import MaskSampler
        with pytest.raises(ValueError, match="non-empty"):
            HorizonDataset(
                surfaces=[],
                mask_sampler=MaskSampler(MaskSamplerConfig()),
                split="train",
            )

    def test_item_keys(self, train_dataset: HorizonDataset) -> None:
        item = train_dataset[0]
        expected_keys = {
            "V", "F", "edge_index",
            "mask", "d", "N",
            "z0", "z_true",
            "xy_mean", "z_mean",
            "xy_scale", "z_scale",
            "surface_id", "reservoir_id", "regime",
        }
        assert set(item.keys()) == expected_keys

    def test_item_shapes_and_dtypes(self, train_dataset: HorizonDataset) -> None:
        item = train_dataset[0]
        n = item["V"].shape[0]
        assert item["V"].shape == (n, 3) and item["V"].dtype == torch.float32
        assert item["F"].ndim == 2 and item["F"].shape[1] == 3
        assert item["F"].dtype == torch.int64
        assert item["edge_index"].ndim == 2 and item["edge_index"].shape[0] == 2
        assert item["mask"].shape == (n,) and item["mask"].dtype == torch.bool
        assert item["d"].shape == (n,) and item["d"].dtype == torch.int64
        assert item["z0"].shape == (n,) and item["z0"].dtype == torch.float32
        assert item["z_true"].shape == (n,) and item["z_true"].dtype == torch.float32
        assert isinstance(item["N"], int)
        assert isinstance(item["surface_id"], str)
        assert item["regime"] in {"half_plane", "outward_free", "outward_pinned"}


# ----------------------------------------------------------------------
# Mask consistency: the relationships between mask, d, z0 must hold
# ----------------------------------------------------------------------
class TestMaskConsistency:
    def test_known_vertices_have_d_zero(
        self, train_dataset: HorizonDataset
    ) -> None:
        for i in range(len(train_dataset)):
            item = train_dataset[i]
            mask, d = item["mask"], item["d"]
            assert torch.equal(mask, d == 0), (
                f"mask and d disagree on item {i}: surface {item['surface_id']}"
            )

    def test_z0_equals_z_true_on_K(
        self, train_dataset: HorizonDataset
    ) -> None:
        for i in range(len(train_dataset)):
            item = train_dataset[i]
            assert torch.equal(
                item["z0"][item["mask"]], item["z_true"][item["mask"]]
            )

    def test_N_equals_max_d(self, train_dataset: HorizonDataset) -> None:
        for i in range(len(train_dataset)):
            item = train_dataset[i]
            assert item["N"] == int(item["d"].max().item())

    def test_no_unreachable_vertices(
        self, train_dataset: HorizonDataset
    ) -> None:
        from horizons.data.topo_distance import UNREACHABLE
        for i in range(len(train_dataset)):
            item = train_dataset[i]
            assert (item["d"] != UNREACHABLE).all()


# ----------------------------------------------------------------------
# Epoch-based mask resampling
# ----------------------------------------------------------------------
class TestEpochResampling:
    def test_train_masks_vary_with_epoch(
        self, train_dataset: HorizonDataset
    ) -> None:
        """Train masks should change when the epoch changes."""
        train_dataset.set_epoch(0)
        m0 = train_dataset[0]["mask"].clone()
        train_dataset.set_epoch(1)
        m1 = train_dataset[0]["mask"].clone()
        train_dataset.set_epoch(2)
        m2 = train_dataset[0]["mask"].clone()
        # At least one of (m0, m1, m2) must differ from another
        # (probabilistic but virtually certain for 900-vertex meshes)
        assert not (torch.equal(m0, m1) and torch.equal(m1, m2))

    def test_train_masks_reproducible_within_epoch(
        self, train_dataset: HorizonDataset
    ) -> None:
        """Same epoch -> same mask, regardless of access order."""
        train_dataset.set_epoch(5)
        m_a = train_dataset[0]["mask"].clone()
        m_b = train_dataset[0]["mask"].clone()
        assert torch.equal(m_a, m_b)

    def test_val_masks_stable_across_epochs(
        self, val_dataset: HorizonDataset
    ) -> None:
        """Val masks must be deterministic across epochs."""
        val_dataset.set_epoch(0)
        m0 = val_dataset[0]["mask"].clone()
        val_dataset.set_epoch(7)
        m7 = val_dataset[0]["mask"].clone()
        val_dataset.set_epoch(42)
        m42 = val_dataset[0]["mask"].clone()
        assert torch.equal(m0, m7)
        assert torch.equal(m0, m42)

    def test_train_and_val_get_different_masks(
        self,
        train_dataset: HorizonDataset,
        val_dataset: HorizonDataset,
    ) -> None:
        """Same surface, same epoch, different split -> different mask
        (because the split string is part of the seed)."""
        train_dataset.set_epoch(0)
        val_dataset.set_epoch(0)
        # Find the index of the same surface in each
        train_ids = [train_dataset[i]["surface_id"] for i in range(len(train_dataset))]
        val_ids = [val_dataset[i]["surface_id"] for i in range(len(val_dataset))]
        common = set(train_ids) & set(val_ids)
        assert common, "Test setup expects overlapping surface_ids"
        sid = next(iter(common))
        t_item = train_dataset[train_ids.index(sid)]
        v_item = val_dataset[val_ids.index(sid)]
        # Not deeply equal because the seed strings differ
        assert not torch.equal(t_item["mask"], v_item["mask"])



class TestCentering:
    """Tests for per-surface (x, y, z) centering (D4.6)."""

    @pytest.fixture
    def utm_like_surfaces(self) -> list[HorizonSurface]:
        """Load real surfaces from data/surfaces/ to test centering.

        Earlier we tried synthesizing fake UTM-scale surfaces, but the
        mask sampler requires a properly-triangulated mesh (which our
        synthetic random triangles aren't), so it failed with
        connectivity errors. The real surfaces are exactly what we
        need: UTM-scale coords + valid topology. Skip if not present.
        """
        from horizons.data.mesh import HorizonSurface
        surfaces_dir = Path("data/surfaces")
        if not surfaces_dir.exists():
            pytest.skip(
                "data/surfaces/ not found — run scripts/build_dataset.py "
                "to populate."
            )
        # Pick two arbitrary real surfaces — small enough to be fast
        candidates = sorted(surfaces_dir.glob("0*.npz"))[:2]
        if len(candidates) < 2:
            pytest.skip(f"Need 2+ real surfaces; found {len(candidates)}")
        return [HorizonSurface.from_npz(p, surface_id=p.stem) for p in candidates]

    def test_centered_dataset_returns_centered_coords(
        self, utm_like_surfaces: list[HorizonSurface],
    ) -> None:
        """With center_per_surface=True, returned V should be small."""
        sampler = MaskSampler(MaskSamplerConfig())
        ds = HorizonDataset(
            utm_like_surfaces, sampler, split="train",
            center_per_surface=True,
        )
        item = ds[0]
        V = item["V"]
        # Centered x, y, z should all be small relative to original
        # (UTM coords are ~1e5-1e7; centered should be at most ~5e4 m,
        # i.e., the half-extent of a typical horizon footprint).
        # The critical property is "much smaller than absolute UTM
        # magnitudes", not "< 10 km".
        original_xy_max = utm_like_surfaces[0].V[:, :2].abs().max().item()
        centered_xy_max = V[:, :2].abs().max().item()
        assert centered_xy_max < original_xy_max / 10, (
            f"x/y not centered; centered max={centered_xy_max:.0f}, "
            f"original max={original_xy_max:.0f}"
        )
        # Z is a smaller-scale variable but still benefits from centering.
        original_z_max = utm_like_surfaces[0].V[:, 2].abs().max().item()
        centered_z_max = V[:, 2].abs().max().item()
        assert centered_z_max < original_z_max, (
            f"z not centered; centered max={centered_z_max:.2f}, "
            f"original max={original_z_max:.2f}"
        )

    def test_centering_offsets_returned_in_dict(
        self, utm_like_surfaces: list[HorizonSurface],
    ) -> None:
        """xy_mean and z_mean must be in the dict and match the actual offsets."""
        sampler = MaskSampler(MaskSamplerConfig())
        ds = HorizonDataset(
            utm_like_surfaces, sampler, split="train",
            center_per_surface=True,
        )
        item = ds[0]
        surface = utm_like_surfaces[0]
        # xy_mean should equal the all-vertex mean
        assert torch.allclose(item["xy_mean"], surface.V[:, :2].mean(dim=0))
        # z_mean should equal the mean over the known vertices in this item
        mask = item["mask"]
        expected_z_mean = surface.V[mask, 2].mean()
        assert torch.allclose(item["z_mean"], expected_z_mean)

    def test_uncentered_dataset_returns_raw_coords(
        self, utm_like_surfaces: list[HorizonSurface],
    ) -> None:
        """With center_per_surface=False, returned V should equal the input."""
        sampler = MaskSampler(MaskSamplerConfig())
        ds = HorizonDataset(
            utm_like_surfaces, sampler, split="train",
            center_per_surface=False,
        )
        item = ds[0]
        assert torch.allclose(item["V"], utm_like_surfaces[0].V)
        # xy_mean and z_mean should be zero
        assert torch.allclose(item["xy_mean"], torch.zeros(2))
        assert item["z_mean"].item() == 0.0

    def test_z_centering_uses_only_known_vertices(
        self, utm_like_surfaces: list[HorizonSurface],
    ) -> None:
        """z_mean must be computed from z[K], not from all z values.

        This is the critical no-leakage property: we cannot use z[U] for
        anything the model can see, because the model is supposed to
        predict z[U].
        """
        sampler = MaskSampler(MaskSamplerConfig())
        ds = HorizonDataset(
            utm_like_surfaces, sampler, split="train",
            center_per_surface=True,
        )
        item = ds[0]
        mask = item["mask"]
        surface = utm_like_surfaces[0]

        z_mean_from_K = surface.V[mask, 2].mean()
        z_mean_from_all = surface.V[:, 2].mean()
        # These should differ in general (otherwise the test is trivial)
        assert not torch.allclose(z_mean_from_K, z_mean_from_all), (
            "Test fixture too symmetric: z_mean over K equals z_mean over all. "
            "Try a different RNG seed."
        )
        # The returned z_mean must match z[K]'s mean, not z's full mean
        assert torch.allclose(item["z_mean"], z_mean_from_K), (
            "z centering used the wrong vertices (possible information leakage)"
        )

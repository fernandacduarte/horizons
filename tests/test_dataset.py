"""Unit tests for horizons.data.dataset."""
from pathlib import Path

import pytest
import torch

from horizons.data.dataset import HorizonDataset, load_fixture_dataset
from horizons.data.masking import MaskSamplerConfig


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

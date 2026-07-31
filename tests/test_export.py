"""Tests for the ParaView (.vtp) export.

The point of these is that the exported file must be a faithful copy of the
data: the geometry as it is in the source mesh, and depths as the models
produced them. An export that quietly shifts or rescales anything would be
worse than no export at all, since the numbers read off in ParaView would
look plausible and be wrong.
"""
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
import torch

from horizons.data.masking import MaskSampler, MaskSamplerConfig
from horizons.data.mesh import HorizonSurface
from horizons.eval.predict import (
    load_split_cases,
    predict_surface,
    split_seed,
)
from horizons.models.operator import LocalOperator
from horizons.viz.export import prediction_to_polydata, save_prediction


FIXTURES_DIR = Path(__file__).parent / "fixtures"
SEED = 3


@pytest.fixture
def anticline() -> HorizonSurface:
    return HorizonSurface.from_npz(FIXTURES_DIR / "anticline.npz")


@pytest.fixture
def prediction(anticline: HorizonSurface):
    torch.manual_seed(0)
    model = LocalOperator(hidden_dim=16, n_message_passing=2)
    sampler = MaskSampler(MaskSamplerConfig())
    return predict_surface(model, anticline, sampler, SEED)


class TestPredictionToPolydata:
    def test_geometry_matches_the_source_mesh(
        self, prediction, anticline
    ) -> None:
        mesh = prediction_to_polydata(prediction)
        V = anticline.V.numpy().astype(np.float64)
        assert np.allclose(mesh.points, V, atol=1e-4)
        assert np.array_equal(
            mesh.faces.reshape(-1, 4)[:, 1:], anticline.F.numpy()
        )

    def test_carries_every_depth_and_error_field(self, prediction) -> None:
        mesh = prediction_to_polydata(prediction)
        expected = {
            "z_true", "z_init", "z_harmonic", "z_model",
            "err_init", "err_harmonic", "err_model",
            "abs_err_init", "abs_err_harmonic", "abs_err_model",
            "known", "unknown", "d",
        }
        assert expected <= set(mesh.point_data.keys())

    @pytest.mark.parametrize("method", ["init", "harmonic", "model"])
    def test_warp_by_error_lands_on_the_method_surface(
        self, prediction, method: str
    ) -> None:
        """This is how ParaView displays a prediction: warping the geometry
        by err_<method> along Z must reproduce z_<method> exactly."""
        mesh = prediction_to_polydata(prediction)
        warped = mesh.points[:, 2] + mesh[f"err_{method}"]
        assert np.allclose(warped, mesh[f"z_{method}"], atol=1e-9)

    def test_absolute_error_is_the_magnitude(self, prediction) -> None:
        mesh = prediction_to_polydata(prediction)
        assert np.allclose(
            mesh["abs_err_model"], np.abs(mesh["err_model"])
        )

    def test_rmse_recomputed_from_the_file_matches(self, prediction) -> None:
        mesh = prediction_to_polydata(prediction)
        unknown = mesh["unknown"].astype(bool)
        for method in ("harmonic", "model"):
            rmse = np.sqrt((mesh[f"err_{method}"][unknown] ** 2).mean())
            assert rmse == pytest.approx(
                getattr(prediction, f"rmse_{method}"), rel=1e-4
            )

    def test_mask_arrays_are_complementary(self, prediction) -> None:
        mesh = prediction_to_polydata(prediction)
        assert np.array_equal(mesh["known"] + mesh["unknown"],
                              np.ones(mesh.n_points, np.uint8))
        assert mesh["known"].sum() == prediction.n_K

    def test_depths_are_not_rescaled(self, prediction, anticline) -> None:
        """z_true must be the surface's own depths, not a centered or
        exaggerated version of them."""
        mesh = prediction_to_polydata(prediction)
        assert np.allclose(
            mesh["z_true"], anticline.V[:, 2].numpy(), atol=1e-4
        )

    def test_centered_frame_differs_only_by_the_offsets(
        self, prediction
    ) -> None:
        world = prediction_to_polydata(prediction, world_coordinates=True)
        centered = prediction_to_polydata(prediction, world_coordinates=False)
        offset = np.append(
            prediction.xy_offset.numpy(), prediction.z_offset
        )
        assert np.allclose(world.points - centered.points, offset, atol=1e-3)
        assert np.allclose(world["err_model"], centered["err_model"])

    def test_records_the_case_metadata(self, prediction) -> None:
        mesh = prediction_to_polydata(prediction)
        assert mesh.field_data["surface_id"][0] == prediction.surface_id
        assert mesh.field_data["regime"][0] == prediction.regime
        assert mesh.field_data["N"][0] == prediction.N
        assert mesh.field_data["z_offset"][0] == pytest.approx(
            prediction.z_offset
        )


class TestSavePrediction:
    def test_round_trips_through_disk(self, prediction, tmp_path) -> None:
        path = save_prediction(prediction, tmp_path / "sub" / "s.vtp")
        assert path.exists()

        mesh = pv.read(path)
        expected = prediction_to_polydata(prediction)
        assert mesh.n_points == expected.n_points
        assert mesh.n_cells == expected.n_cells
        assert np.allclose(mesh["z_model"], expected["z_model"])
        assert np.allclose(mesh.points, expected.points)


class TestSplitCases:
    def test_seed_follows_the_driver_scheme(self) -> None:
        assert split_seed(1000, 0, 0) == 1000
        assert split_seed(1000, 3, 2) == 1302

    def test_filtering_keeps_the_full_split_index(self) -> None:
        """Exporting one surface must reuse the seed it would have had in a
        whole-split run, or its mask (and metrics) would silently change."""
        split_file = "data/splits/split_v2.json"
        every = load_split_cases("test_ood", split_file=split_file)
        target = every[2][1].surface_id

        picked = load_split_cases(
            "test_ood", split_file=split_file, surface_ids=[target]
        )
        assert picked[0][0] == 2

    def test_unknown_surface_raises(self) -> None:
        with pytest.raises(KeyError, match="nope"):
            load_split_cases(
                "test_ood",
                split_file="data/splits/split_v2.json",
                surface_ids=["nope"],
            )

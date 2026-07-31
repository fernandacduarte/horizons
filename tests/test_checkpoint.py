"""Tests for checkpoint loading across input-projection variants.

The input projection changed from a single Linear to a 2-layer MLP in July
2026. Checkpoints from before that must keep loading, so the loader reads the
depth off the saved weights instead of assuming the current default.
"""
from pathlib import Path

import pytest
import torch

from horizons.eval.checkpoint import infer_input_proj_layers, load_checkpoint
from horizons.models.operator import LocalOperator


def save_fake_checkpoint(path: Path, model: LocalOperator) -> Path:
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": 1,
            "step": 10,
            "best_val_loss": 0.5,
        },
        path,
    )
    return path


class TestInferInputProjLayers:
    @pytest.mark.parametrize("layers", [1, 2])
    def test_reads_the_depth_off_the_state_dict(self, layers: int) -> None:
        model = LocalOperator(hidden_dim=8, input_proj_layers=layers)
        assert infer_input_proj_layers(model.state_dict()) == layers

    def test_unrecognizable_state_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="input_proj"):
            infer_input_proj_layers({"head.0.weight": torch.zeros(1)})


class TestLoadCheckpoint:
    @pytest.mark.parametrize("layers", [1, 2])
    def test_round_trip_preserves_predictions(
        self, layers: int, tmp_path: Path
    ) -> None:
        torch.manual_seed(0)
        model = LocalOperator(hidden_dim=8, input_proj_layers=layers)
        path = save_fake_checkpoint(tmp_path / "best.pt", model)

        loaded = load_checkpoint(path, hidden_dim=8).model
        assert isinstance(loaded, LocalOperator)

        # A plane with a couple of triangles is enough to exercise the forward.
        V_xy = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        z = torch.tensor([0.0, 0.1, 0.2, 0.3])
        F = torch.tensor([[0, 1, 2], [1, 3, 2]])
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 0, 2, 1, 3], [1, 0, 2, 1, 3, 2, 2, 0, 3, 1]]
        )
        mask = torch.tensor([True, True, False, False])
        d = torch.tensor([0, 0, 1, 1])

        args = (z, V_xy, edge_index, F, mask, d)
        assert torch.allclose(loaded(*args), model(*args))

    def test_explicit_layers_override_inference(self, tmp_path: Path) -> None:
        model = LocalOperator(hidden_dim=8, input_proj_layers=1)
        path = save_fake_checkpoint(tmp_path / "best.pt", model)
        with pytest.raises(RuntimeError, match="architecture"):
            load_checkpoint(path, hidden_dim=8, input_proj_layers=2)

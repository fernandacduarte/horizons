"""Unit tests for the LR schedule (compute_lr)."""
import math

import pytest

from horizons.training.loop import compute_lr


class TestConstantSchedule:
    def test_constant_returns_lr_max(self) -> None:
        for e in [0, 5, 50, 99]:
            lr = compute_lr(
                e, lr_max=1e-3, lr_min=1e-5,
                warmup_epochs=5, n_epochs=100, schedule="constant",
            )
            assert lr == 1e-3


class TestCosineSchedule:
    def test_warmup_starts_at_lr_min(self) -> None:
        """At the very first warmup step, we get lr just above lr_min."""
        # Note: our formula uses (epoch + 1) / warmup_epochs, so at
        # epoch=0 we already get lr_min + (lr_max - lr_min) / warmup_epochs.
        # That's deliberate: epoch=0 should already have started moving.
        lr = compute_lr(
            0, lr_max=1.0, lr_min=0.0,
            warmup_epochs=5, n_epochs=100, schedule="cosine",
        )
        assert lr == pytest.approx(0.2)  # 1/5

    def test_warmup_ramps_linearly(self) -> None:
        for epoch, expected_frac in [(0, 0.2), (1, 0.4), (2, 0.6), (3, 0.8), (4, 1.0)]:
            lr = compute_lr(
                epoch, lr_max=1.0, lr_min=0.0,
                warmup_epochs=5, n_epochs=100, schedule="cosine",
            )
            assert lr == pytest.approx(expected_frac), (
                f"epoch={epoch}: expected lr={expected_frac}, got {lr}"
            )

    def test_peak_at_end_of_warmup(self) -> None:
        """LR equals lr_max at the start of cosine decay (epoch == warmup_epochs)."""
        lr = compute_lr(
            5, lr_max=1.0, lr_min=0.01,
            warmup_epochs=5, n_epochs=100, schedule="cosine",
        )
        assert lr == pytest.approx(1.0)

    def test_floor_at_last_epoch(self) -> None:
        """LR equals lr_min at the last epoch."""
        lr = compute_lr(
            99, lr_max=1.0, lr_min=0.01,
            warmup_epochs=5, n_epochs=100, schedule="cosine",
        )
        assert lr == pytest.approx(0.01)

    def test_midpoint_decay(self) -> None:
        """At the halfway point of decay, cosine gives (lr_max + lr_min) / 2."""
        # Decay range: epochs [5, 99] (95 epochs). Midpoint: epoch ~52.
        # cos(pi * 0.5) = 0, so LR = lr_min + 0.5 * (lr_max - lr_min)
        mid_epoch = 5 + (99 - 5) // 2  # epoch 52
        lr = compute_lr(
            mid_epoch, lr_max=1.0, lr_min=0.0,
            warmup_epochs=5, n_epochs=100, schedule="cosine",
        )
        assert lr == pytest.approx(0.5, abs=0.05)

    def test_monotonic_during_decay(self) -> None:
        """During the cosine decay phase, LR is monotonically decreasing."""
        warmup_epochs = 5
        n_epochs = 100
        prev_lr = float("inf")
        for epoch in range(warmup_epochs, n_epochs):
            lr = compute_lr(
                epoch, lr_max=1.0, lr_min=0.0,
                warmup_epochs=warmup_epochs, n_epochs=n_epochs,
                schedule="cosine",
            )
            assert lr <= prev_lr, (
                f"LR went UP at epoch {epoch}: prev={prev_lr}, now={lr}"
            )
            prev_lr = lr


class TestEdgeCases:
    def test_zero_warmup(self) -> None:
        """warmup_epochs=0 should mean 'start at lr_max'."""
        lr = compute_lr(
            0, lr_max=1.0, lr_min=0.01,
            warmup_epochs=0, n_epochs=100, schedule="cosine",
        )
        assert lr == pytest.approx(1.0)

    def test_unknown_schedule_rejected(self) -> None:
        with pytest.raises(ValueError, match="schedule"):
            compute_lr(
                0, lr_max=1.0, lr_min=0.0,
                warmup_epochs=5, n_epochs=100, schedule="linear",
            )

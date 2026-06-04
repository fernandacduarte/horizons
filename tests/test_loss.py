"""Unit tests for horizons.training.loss."""
import pytest
import torch

from horizons.training.loss import per_iteration_data_loss, rollout_data_loss


def _make_setup(n: int = 10, n_K: int = 3) -> dict:
    """Build a tiny synthetic setup with known structure.

    Vertices 0..n_K-1 are known (d=0); vertices n_K..n have increasing d.
    """
    z_true = torch.arange(n, dtype=torch.float32)
    d = torch.zeros(n, dtype=torch.int64)
    # Assign d = 1, 2, 3, ... to the unknown vertices in groups
    for i, idx in enumerate(range(n_K, n)):
        d[idx] = (i % 3) + 1  # values cycle through 1, 2, 3
    return {"z_true": z_true, "d": d}


# ----------------------------------------------------------------------
# per_iteration_data_loss
# ----------------------------------------------------------------------
class TestPerIterationDataLoss:
    def test_zero_when_zt_equals_ztrue(self) -> None:
        """If z^t exactly equals z_true, the loss is zero."""
        setup = _make_setup()
        z_t = setup["z_true"].clone()
        L = per_iteration_data_loss(z_t, setup["z_true"], setup["d"], t=2)
        assert L.item() == 0.0

    def test_supervises_only_frontier_and_filled(self) -> None:
        """At iteration t=2, only F_2={d=2} and P_2={d=1} contribute;
        vertices with d=0 (K) or d=3 are ignored."""
        setup = _make_setup()
        d = setup["d"]
        z_true = setup["d"].to(torch.float32) * 10.0  # arbitrary
        z_t = z_true.clone()
        # Inject error only on d=3 vertices — these are NOT in F_2 or P_2
        # at t=2, so the loss should be exactly 0.
        z_t[d == 3] += 5.0
        L = per_iteration_data_loss(z_t, z_true, d, t=2)
        assert L.item() == 0.0

    def test_lambda_f_weights_frontier(self) -> None:
        """Doubling lambda_f should double the contribution from F_t,
        leaving P_t unaffected."""
        d = torch.tensor([0, 0, 1, 1, 2, 2])
        z_true = torch.zeros(6)
        # Errors of magnitude 1 on F_2 ({d=2}), zero on P_2 ({d=1})
        z_t = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

        L1 = per_iteration_data_loss(z_t, z_true, d, t=2,
                                     lambda_f=1.0, lambda_p=0.1)
        L2 = per_iteration_data_loss(z_t, z_true, d, t=2,
                                     lambda_f=2.0, lambda_p=0.1)
        assert L2.item() == pytest.approx(2 * L1.item())

    def test_lambda_p_weights_filled(self) -> None:
        d = torch.tensor([0, 0, 1, 1, 2, 2])
        z_true = torch.zeros(6)
        # Errors of magnitude 1 on P_2 ({d=1}), zero on F_2 ({d=2})
        z_t = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])

        L1 = per_iteration_data_loss(z_t, z_true, d, t=2,
                                     lambda_f=1.0, lambda_p=0.1)
        L2 = per_iteration_data_loss(z_t, z_true, d, t=2,
                                     lambda_f=1.0, lambda_p=0.2)
        assert L2.item() == pytest.approx(2 * L1.item())

    def test_t_zero_rejected(self) -> None:
        setup = _make_setup()
        with pytest.raises(ValueError, match="t must be"):
            per_iteration_data_loss(
                setup["z_true"], setup["z_true"], setup["d"], t=0
            )

    def test_differentiable_w_r_t_z_t(self) -> None:
        """Gradients should flow from L back to z^t (on the supervised vertices)."""
        d = torch.tensor([0, 1, 2])
        z_true = torch.tensor([0.0, 0.0, 0.0])
        z_t = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
        L = per_iteration_data_loss(z_t, z_true, d, t=2,
                                    lambda_f=1.0, lambda_p=0.1)
        L.backward()
        # z_t[0] is known (d=0), should have no gradient contribution
        # (we don't supervise K; it's structurally zero).
        # z_t[1] is in P_2 (d=1), gets weight lambda_p
        # z_t[2] is in F_2 (d=2), gets weight lambda_f
        assert z_t.grad[0].item() == 0.0
        assert z_t.grad[1].item() != 0.0
        assert z_t.grad[2].item() != 0.0
        # Frontier gradient should be larger in magnitude (lambda_f > lambda_p
        # and z_t[2] has larger error)
        assert z_t.grad[2].abs() > z_t.grad[1].abs()


# ----------------------------------------------------------------------
# rollout_data_loss
# ----------------------------------------------------------------------
class TestRolloutDataLoss:
    def test_zero_when_trajectory_is_truth(self) -> None:
        """If every z^t exactly equals z_true, total loss is zero."""
        n = 10
        z_true = torch.arange(n, dtype=torch.float32)
        d = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 3, 3])
        # Build a trajectory of length N+1 = 4 (N=3)
        trajectory = [z_true.clone() for _ in range(4)]
        L = rollout_data_loss(trajectory, z_true, d)
        assert L.item() == 0.0

    def test_sums_over_iterations(self) -> None:
        """L should equal the sum of per-iteration losses."""
        n = 6
        z_true = torch.zeros(n)
        d = torch.tensor([0, 0, 1, 1, 2, 2])
        # z^0 (not supervised), z^1, z^2
        z0 = torch.zeros(n)
        z1 = torch.tensor([0.0, 0.0, 0.5, 0.5, 0.0, 0.0])
        z2 = torch.tensor([0.0, 0.0, 0.3, 0.3, 0.7, 0.7])
        trajectory = [z0, z1, z2]

        L = rollout_data_loss(trajectory, z_true, d).item()
        L1 = per_iteration_data_loss(z1, z_true, d, t=1).item()
        L2 = per_iteration_data_loss(z2, z_true, d, t=2).item()
        assert L == pytest.approx(L1 + L2)

    def test_rollout_weights_applied(self) -> None:
        """Uniform weights should give plain sum; non-uniform weights scale."""
        n = 6
        z_true = torch.zeros(n)
        d = torch.tensor([0, 0, 1, 1, 2, 2])
        z0 = torch.zeros(n)
        z1 = torch.ones(n)
        z2 = torch.ones(n)
        trajectory = [z0, z1, z2]

        L_uniform = rollout_data_loss(trajectory, z_true, d).item()
        L_weighted = rollout_data_loss(
            trajectory, z_true, d, rollout_weights=[1.0, 2.0]
        ).item()
        # The second iteration's L should now contribute twice
        L1 = per_iteration_data_loss(z1, z_true, d, t=1).item()
        L2 = per_iteration_data_loss(z2, z_true, d, t=2).item()
        assert L_uniform == pytest.approx(L1 + L2)
        assert L_weighted == pytest.approx(L1 + 2 * L2)

    def test_rollout_weights_length_validation(self) -> None:
        z_true = torch.zeros(3)
        d = torch.tensor([0, 1, 2])
        trajectory = [torch.zeros(3), torch.zeros(3), torch.zeros(3)]
        with pytest.raises(ValueError, match="rollout_weights"):
            rollout_data_loss(trajectory, z_true, d, rollout_weights=[1.0])

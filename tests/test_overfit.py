"""Single-example overfit test — the keystone sanity check.

If this test fails, the rollout / loss / model pipeline is broken in some
fundamental way.

What it does: load the anticline fixture, sample one mask, train TinySAGE
on that single (surface, mask) pair for a fixed number of steps, and
assert that the loss reduced by at least the threshold factor.
"""
from pathlib import Path

import pytest
import torch

from horizons.data.mesh import HorizonSurface
from horizons.data.masking import MaskSampler, MaskSamplerConfig
from horizons.data.init import init_z
from horizons.models.placeholder import TinySAGE
from horizons.training.rollout import rollout
from horizons.training.loss import rollout_data_loss


FIXTURES_DIR = Path(__file__).parent / "fixtures"


# Test parameters chosen to:
#   - run in well under 10 seconds (no slowdown to the test suite),
#   - clearly demonstrate learning (much more than statistical noise),
#   - stay well under the 61x reduction we saw experimentally with the overfit_one.py script (safety margin).
N_STEPS = 300
MIN_REDUCTION_FACTOR = 10.0


def test_overfit_one_example_with_placeholder() -> None:
    """The placeholder model + rollout + data loss must drive loss down
    by at least MIN_REDUCTION_FACTOR on a single (surface, mask) pair.

    This is the keystone sanity check for the entire pipeline.
    A failure here means one of:
      - rollout machinery is broken (anchoring, BPTT, or autograd flow)
      - loss is wrong (per-ring supervision, weight application, etc.)
      - the placeholder cannot learn (init scale too large, no gradient flow)
      - the dataset returns inconsistent (mask, d, z0) tuples
    """
    # ------------------------------------------------------------------
    # Reproducible setup
    # ------------------------------------------------------------------
    torch.manual_seed(42)

    surface = HorizonSurface.from_npz(FIXTURES_DIR / "anticline.npz")

    sampler = MaskSampler(MaskSamplerConfig())
    mask_rng = torch.Generator().manual_seed(42)
    mask, d, _regime = sampler.sample(surface, mask_rng)
    z0 = init_z(surface.V, mask)
    N = int(d.max().item())

    V_xy = surface.V[:, :2]
    F = surface.F
    edge_index = surface.edge_index
    z_true = surface.V[:, 2]

    # ------------------------------------------------------------------
    # Model + optimizer
    # ------------------------------------------------------------------
    model = TinySAGE(hidden_dim=32, output_init_scale=0.01)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    initial_loss = None
    for step in range(N_STEPS):
        optimizer.zero_grad()

        result = rollout(
            model,
            z0=z0, z_true=z_true,
            V_xy=V_xy, F=F, edge_index=edge_index,
            mask=mask, d=d, N=N,
        )
        loss = rollout_data_loss(
            result.z_trajectory, z_true, d,
            lambda_f=1.0, lambda_p=0.1,
        )

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    final_loss = loss.item()

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    # 1. No NaN/Inf in the loss
    assert torch.isfinite(torch.tensor(final_loss)), (
        f"Final loss is not finite: {final_loss}"
    )

    # 2. Significant reduction
    reduction = initial_loss / max(final_loss, 1e-12)
    assert reduction >= MIN_REDUCTION_FACTOR, (
        f"Loss did not reduce enough: initial={initial_loss:.6f}, "
        f"final={final_loss:.6f}, reduction={reduction:.1f}x "
        f"(required: >= {MIN_REDUCTION_FACTOR}x)"
    )

    # 3. Anchoring still intact: even after training, z^N on K must equal z_true.
    #    A bug that lets the optimizer modify K vertices would only show up
    #    here, not in earlier tests (which used an untrained model).
    with torch.no_grad():
        final_result = rollout(
            model,
            z0=z0, z_true=z_true,
            V_xy=V_xy, F=F, edge_index=edge_index,
            mask=mask, d=d, N=N,
        )
    z_N_K = final_result.z_trajectory[-1][mask]
    assert torch.equal(z_N_K, z_true[mask]), (
        "Anchoring broken: z^N on K does not equal z_true on K after training"
    )

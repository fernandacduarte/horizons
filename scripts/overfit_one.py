"""Single-example overfit smoke test.

Loads the anticline fixture, samples one mask, and runs the placeholder
model + rollout + data loss + AdamW for a fixed number of optimizer
steps on the same (surface, mask) pair. Plots the loss curve at the end.

Purpose: verify the rollout machinery is correct. If we can't drive
the loss down on a single example, something fundamental is broken.

Usage:
    python scripts/overfit_one.py                          # defaults
    python scripts/overfit_one.py n_steps=2000 optim.lr=5e-3
"""
from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig

from horizons.data.mesh import HorizonSurface
from horizons.data.masking import MaskSampler, MaskSamplerConfig
from horizons.data.init import init_z
from horizons.models.placeholder import TinySAGE
from horizons.models.operator import LocalOperator
from horizons.training.rollout import rollout
from horizons.training.loss import rollout_data_loss, per_iteration_data_loss


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    # For overfit testing we want CPU regardless of config — small mesh,
    # no batching, MPS overhead would dominate.
    device = torch.device("cpu")
    torch.manual_seed(cfg.seed)

    # Load the anticline fixture
    fixtures_dir = Path("tests/fixtures")
    surface = HorizonSurface.from_npz(fixtures_dir / "anticline.npz")

    # Sample one mask (held fixed for the duration of training)
    sampler = MaskSampler(MaskSamplerConfig.from_dictconfig(cfg.mask))
    mask_rng = torch.Generator().manual_seed(cfg.seed)
    mask, d, regime = sampler.sample(surface, mask_rng)
    z0 = init_z(surface.V, mask)
    N = int(d.max().item())

    # Move everything to device
    V_xy = surface.V[:, :2].to(device)
    F = surface.F.to(device)
    edge_index = surface.edge_index.to(device)
    z_true = surface.V[:, 2].to(device)
    z0 = z0.to(device)
    mask = mask.to(device)
    d = d.to(device)

    print(f"Surface: {surface.surface_id} | regime: {regime}")
    print(f"|V|={surface.n_vertices}, |K|={mask.sum().item()}, "
          f"|U|={(~mask).sum().item()}, N={N}")
    print()

    # ------------------------------------------------------------------
    # Model + optimizer
    # ------------------------------------------------------------------
    # Use a slightly larger placeholder for this overfit test to give it
    # enough capacity to make visible progress.
    # Choose model based on config: "placeholder" (TinySAGE) or
    # "operator" (real LocalOperator).
    model_kind = cfg.model_kind
    if model_kind == "placeholder":
        model = TinySAGE(hidden_dim=32, output_init_scale=0.01).to(device)
    elif model_kind == "operator":
        model = LocalOperator(
            hidden_dim=cfg.model.hidden_dim,
            n_message_passing=cfg.model.n_layers,
            output_init_scale=cfg.model.output_init_scale,
        ).to(device)
    else:
        raise ValueError(
            f"Unknown model_kind {model_kind!r}; expected 'placeholder' or 'operator'"
        )
    print(f"Model: {model_kind} | params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    n_steps = cfg.get("n_steps", 1000)
    log_every = max(1, n_steps // 50)  # ~50 log points

    history = {"step": [], "loss": []}
    per_iter_history: dict[int, list[float]] = {t: [] for t in range(1, N + 1)}

    for step in range(n_steps):
        optimizer.zero_grad()

        result = rollout(
            model,
            z0=z0, z_true=z_true,
            V_xy=V_xy, F=F, edge_index=edge_index,
            mask=mask, d=d, N=N,
        )

        loss = rollout_data_loss(
            result.z_trajectory, z_true, d,
            lambda_f=cfg.loss.lambda_f, lambda_p=cfg.loss.lambda_p,
        )

        loss.backward()

        # Gradient clipping (configured but useful sanity)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=cfg.optim.grad_clip_norm,
        )

        optimizer.step()

        # Logging
        if step % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                per_iter = [
                    per_iteration_data_loss(
                        result.z_trajectory[t], z_true, d, t=t,
                        lambda_f=cfg.loss.lambda_f, lambda_p=cfg.loss.lambda_p,
                    ).item()
                    for t in range(1, N + 1)
                ]
            history["step"].append(step)
            history["loss"].append(loss.item())
            for t in range(1, N + 1):
                per_iter_history[t].append(per_iter[t - 1])

            if step % (log_every * 5) == 0 or step == n_steps - 1:
                print(f"step {step:5d}  loss {loss.item():.6f}  "
                      f"per-iter min/max [{min(per_iter):.4f}, {max(per_iter):.4f}]")

    print()
    print(f"Initial loss: {history['loss'][0]:.6f}")
    print(f"Final loss:   {history['loss'][-1]:.6f}")
    print(f"Reduction:    {history['loss'][0] / max(history['loss'][-1], 1e-12):.1f}x")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["step"], history["loss"])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("rollout data loss")
    axes[0].set_title(f"Overfit on one (surface, mask)\n"
                      f"{surface.surface_id} | {regime} | N={N}")
    axes[0].grid(True, which="both", linewidth=0.3)

    cmap = plt.cm.viridis
    for t in range(1, N + 1):
        color = cmap(t / N)
        axes[1].plot(history["step"], per_iter_history[t],
                     color=color, label=f"t={t}", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("per-iteration loss L_t")
    axes[1].set_title("Per-iteration loss (color = depth t)")
    axes[1].grid(True, which="both", linewidth=0.3)
    axes[1].legend(fontsize=7, ncol=2)

    fig.tight_layout()
    out_path = Path(cfg.train.log_dir) / "overfit_one.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

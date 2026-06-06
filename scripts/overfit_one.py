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
from horizons.training.loss import rollout_loss, per_iteration_data_loss


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

    history = {"step": [], "loss": [], "data": [], "curv": [], "res": []}
    per_iter_history: dict[int, list[float]] = {t: [] for t in range(1, N + 1)}

    for step in range(n_steps):
        optimizer.zero_grad()

        result = rollout(
            model,
            z0=z0, z_true=z_true,
            V_xy=V_xy, F=F, edge_index=edge_index,
            mask=mask, d=d, N=N,
        )

        loss_dict = rollout_loss(
            z_trajectory=result.z_trajectory,
            dz_trajectory=result.dz_trajectory,
            z_true=z_true, d=d, edge_index=edge_index, mask=mask,
            lambda_f=cfg.loss.lambda_f, lambda_p=cfg.loss.lambda_p,
            lambda_c=cfg.loss.lambda_c, lambda_r=cfg.loss.lambda_r,
        )
        loss = loss_dict["total"]

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
            history["data"].append(loss_dict["data"].item())
            history["curv"].append(loss_dict["curv"].item())
            history["res"].append(loss_dict["res"].item())
            for t in range(1, N + 1):
                per_iter_history[t].append(per_iter[t - 1])

            if step % (log_every * 5) == 0 or step == n_steps - 1:
                print(
                    f"step {step:5d}  total {loss.item():.6f}  "
                    f"(data {loss_dict['data'].item():.4f}  "
                    f"curv {loss_dict['curv'].item():.4f}  "
                    f"res {loss_dict['res'].item():.4f})"
                )

    print()
    print(f"Initial loss: {history['loss'][0]:.6f}")
    print(f"Final loss:   {history['loss'][-1]:.6f}")
    print(f"Reduction:    {history['loss'][0] / max(history['loss'][-1], 1e-12):.1f}x")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Panel 1: total loss
    axes[0].plot(history["step"], history["loss"])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("rollout total loss")
    axes[0].set_title(f"Overfit on one (surface, mask)\n"
                      f"{surface.surface_id} | {regime} | N={N}")
    axes[0].grid(True, which="both", linewidth=0.3)

    # Panel 2: per-iteration L_t (data only, for comparability with Stage 6)
    cmap = plt.cm.viridis
    for t in range(1, N + 1):
        color = cmap(t / N)
        axes[1].plot(history["step"], per_iter_history[t],
                     color=color, label=f"t={t}", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("per-iteration data loss L_data_t")
    axes[1].set_title("Per-iteration data loss (color = depth t)")
    axes[1].grid(True, which="both", linewidth=0.3)
    axes[1].legend(fontsize=7, ncol=2)

    # Panel 3: three loss components on a shared axis
    axes[2].plot(history["step"], history["data"], label="data", color="C0")
    axes[2].plot(history["step"], history["curv"], label="curv (unscaled)", color="C1")
    axes[2].plot(history["step"], history["res"], label="res (unscaled)", color="C2")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("optimizer step")
    axes[2].set_ylabel("loss component (unscaled)")
    axes[2].set_title("Components (unscaled — λ_c, λ_r not applied here)")
    axes[2].grid(True, which="both", linewidth=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    out_path = Path(cfg.train.log_dir) / "overfit_one.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

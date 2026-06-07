"""Overfit one (surface, mask) pair using a real horizon.

Same machinery as scripts/overfit_one.py, but loads from data/surfaces/.
Use this to confirm that the full pipeline works end-to-end on the actual
dataset before launching Stage 8's training loop.

Usage:
    python scripts/overfit_real.py                          # default surface
    python scripts/overfit_real.py surface_id=Horizonte3
    python scripts/overfit_real.py surface_id=01_FMar n_steps=2000
"""
from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig

from horizons.data.loaders import load_split
from horizons.data.masking import MaskSampler, MaskSamplerConfig
from horizons.data.init import init_z
from horizons.models.placeholder import TinySAGE
from horizons.models.operator import LocalOperator
from horizons.training.rollout import rollout
from horizons.training.loss import rollout_loss, per_iteration_data_loss


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    device = torch.device("cpu")
    torch.manual_seed(cfg.seed)

    # Pick a surface from the train split; user can override on CLI
    surface_id = cfg.get("surface_id", None)
    train_surfaces = load_split("train")
    if surface_id is None:
        surface = train_surfaces[0]
    else:
        matches = [s for s in train_surfaces if s.surface_id == surface_id]
        if not matches:
            available = [s.surface_id for s in train_surfaces]
            raise SystemExit(
                f"surface_id={surface_id!r} not in train split. "
                f"Available: {available}"
            )
        surface = matches[0]

    sampler = MaskSampler(MaskSamplerConfig.from_dictconfig(cfg.mask))
    mask_rng = torch.Generator().manual_seed(cfg.seed)
    mask, d, regime = sampler.sample(surface, mask_rng)
    z0 = init_z(surface.V, mask)
    N = int(d.max().item())

    # Per-surface (x, y) centering. Real-world horizons can sit at UTM
    # coordinates (x ~ 1e5, y ~ 1e7). Feeding those directly to the GNN's
    # input projection causes float32 precision loss and unstable
    # training. Centering by the dataset mean keeps everything O(1e3 m)
    # without losing any geometric information.
    V_full = surface.V.to(device)
    xy_mean = V_full[:, :2].mean(dim=0)
    V_xy = V_full[:, :2] - xy_mean

    F = surface.F.to(device)
    edge_index = surface.edge_index.to(device)
    z_true = V_full[:, 2]
    z0 = z0.to(device)
    mask = mask.to(device)
    d = d.to(device)

    # Per-surface z normalization (D4.6): center z by the mean of z[K]
    # so the model sees relative depths instead of absolute geological depths.
    z_mean = z_true[mask].mean()
    z_true_norm = z_true - z_mean
    z0_norm = z0 - z_mean

    print(f"Surface: {surface.surface_id} ({surface.reservoir_id})")
    print(f"|V|={surface.n_vertices}, |K|={mask.sum().item()}, "
          f"|U|={(~mask).sum().item()}, N={N}, regime={regime}")
    print(f"xy_mean (subtracted): ({xy_mean[0]:.1f}, {xy_mean[1]:.1f})")
    print(f"z range (true): [{z_true.min():.1f}, {z_true.max():.1f}]")
    print(f"z_mean (subtracted): {z_mean:.1f}")
    print()

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
        raise ValueError(f"Unknown model_kind {model_kind!r}")
    print(f"Model: {model_kind} | params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )

    n_steps = cfg.get("n_steps", 1000)
    log_every = max(1, n_steps // 50)

    history = {"step": [], "loss": [], "data": [], "curv": [], "res": []}
    per_iter_history: dict[int, list[float]] = {t: [] for t in range(1, N + 1)}

    for step in range(n_steps):
        optimizer.zero_grad()
        result = rollout(
            model,
            z0=z0_norm, z_true=z_true_norm,
            V_xy=V_xy, F=F, edge_index=edge_index,
            mask=mask, d=d, N=N,
        )
        loss_dict = rollout_loss(
            z_trajectory=result.z_trajectory,
            dz_trajectory=result.dz_trajectory,
            z_true=z_true_norm, d=d, edge_index=edge_index, mask=mask,
            lambda_f=cfg.loss.lambda_f, lambda_p=cfg.loss.lambda_p,
            lambda_c=cfg.loss.lambda_c, lambda_r=cfg.loss.lambda_r,
        )
        loss = loss_dict["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=cfg.optim.grad_clip_norm,
        )
        optimizer.step()

        if step % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                per_iter = [
                    per_iteration_data_loss(
                        result.z_trajectory[t], z_true_norm, d, t=t,
                        lambda_f=cfg.loss.lambda_f, lambda_p=cfg.loss.lambda_p,
                    ).item() for t in range(1, N + 1)
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

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(history["step"], history["loss"])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("rollout total loss")
    axes[0].set_title(f"Overfit on real horizon\n"
                      f"{surface.surface_id} | {regime} | N={N}")
    axes[0].grid(True, which="both", linewidth=0.3)

    cmap = plt.cm.viridis
    for t in range(1, N + 1):
        color = cmap(t / N)
        axes[1].plot(history["step"], per_iter_history[t],
                     color=color, label=f"t={t}", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("per-iteration data loss")
    axes[1].set_title("Per-iteration data loss (color = depth t)")
    axes[1].grid(True, which="both", linewidth=0.3)
    if N <= 12:
        axes[1].legend(fontsize=7, ncol=2)

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
    out_path = Path(cfg.train.log_dir) / f"overfit_real_{surface.surface_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

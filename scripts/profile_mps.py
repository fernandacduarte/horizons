"""Quick benchmark: compare CPU vs MPS for one training epoch.

Runs 5 surfaces forward + backward on each device, reports wall time.
"""
import time
import torch

from horizons.data.loaders import load_split
from horizons.data.dataset import HorizonDataset
from horizons.data.masking import MaskSampler, MaskSamplerConfig
from horizons.models.operator import LocalOperator
from horizons.training.rollout import rollout
from horizons.training.loss import rollout_loss


def time_one_epoch(device: str, n_surfaces: int = 5):
    print(f"\n=== Benchmarking on {device} ===")

    surfaces = load_split("train")[:n_surfaces]  # subset to keep it short
    sampler = MaskSampler(MaskSamplerConfig())
    ds = HorizonDataset(
        surfaces, sampler, split="train",
        normalize_per_surface=True,
        n_masks_per_epoch=1,
    )

    model = LocalOperator(hidden_dim=64, n_message_passing=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Warmup (first call may be slow due to JIT)
    item = ds[0]
    z0 = item["z0"].to(device)
    z_true = item["z_true"].to(device)
    V_xy = item["V"][:, :2].to(device)
    F = item["F"].to(device)
    edge_index = item["edge_index"].to(device)
    mask = item["mask"].to(device)
    d = item["d"].to(device)
    N = item["N"]

    result = rollout(model, z0=z0, z_true=z_true,
                     V_xy=V_xy, F=F, edge_index=edge_index,
                     mask=mask, d=d, N=N)
    loss_dict = rollout_loss(
        z_trajectory=result.z_trajectory,
        dz_trajectory=result.dz_trajectory,
        z_true=z_true, d=d, edge_index=edge_index, mask=mask,
    )
    loss_dict["total"].backward()
    optimizer.step()
    optimizer.zero_grad()

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    # Timed pass
    t0 = time.time()
    for i in range(n_surfaces):
        item = ds[i]
        z0 = item["z0"].to(device)
        z_true = item["z_true"].to(device)
        V_xy = item["V"][:, :2].to(device)
        F = item["F"].to(device)
        edge_index = item["edge_index"].to(device)
        mask = item["mask"].to(device)
        d = item["d"].to(device)
        N = item["N"]

        result = rollout(model, z0=z0, z_true=z_true,
                         V_xy=V_xy, F=F, edge_index=edge_index,
                         mask=mask, d=d, N=N)
        loss_dict = rollout_loss(
            z_trajectory=result.z_trajectory,
            dz_trajectory=result.dz_trajectory,
            z_true=z_true, d=d, edge_index=edge_index, mask=mask,
        )
        loss = loss_dict["total"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  surface {i}: {item['surface_id']:<22} N={N:<3} loss={loss.item():.4f}")

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"  total: {elapsed:.1f}s ({elapsed/n_surfaces:.1f}s per surface)")
    return elapsed


if __name__ == "__main__":
    cpu_time = time_one_epoch("cpu")
    if torch.backends.mps.is_available():
        mps_time = time_one_epoch("mps")
        speedup = cpu_time / mps_time
        print(f"\nSpeedup: {speedup:.2f}x (mps vs cpu)")
    else:
        print("\nMPS not available on this system.")

"""Topological (graph) distance from a known set to all vertices via BFS.

Used to:
  - Determine the rollout depth N = max d_i.
  - Provide d_i as a per-vertex feature to the GNN.
  - Define the per-iteration loss schedule: the frontier F_t = {i : d_i = t}.
"""
from __future__ import annotations

import torch


UNREACHABLE = -1  # sentinel distance for vertices with no path to K


def compute_topological_distance(
    edge_index: torch.Tensor,
    known_mask: torch.Tensor,
) -> torch.Tensor:
    """Multi-source BFS distance from the known set to every vertex.

    Parameters
    ----------
    edge_index : torch.Tensor, shape (2, n_directed_edges), int64
        PyG-style bidirectional edge index. Each undirected edge appears
        in both directions.
    known_mask : torch.Tensor, shape (n_vertices,), bool
        True for known vertices (sources of the BFS).

    Returns
    -------
    d : torch.Tensor, shape (n_vertices,), int64
        d[i] = shortest-path distance (in edges) from vertex i to the
        nearest known vertex.
          - d[i] = 0 iff known_mask[i] = True.
          - d[i] = UNREACHABLE (-1) iff i has no path to any known vertex.
    """
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape (2, E); got {tuple(edge_index.shape)}"
        )
    if known_mask.dim() != 1:
        raise ValueError(
            f"known_mask must be 1-D; got shape {tuple(known_mask.shape)}"
        )
    if known_mask.dtype != torch.bool:
        raise TypeError(f"known_mask must be bool; got {known_mask.dtype}")

    n = known_mask.shape[0]
    src, dst = edge_index[0], edge_index[1]

    # Initialize distances: 0 for known, UNREACHABLE for everyone else
    d = torch.full((n,), UNREACHABLE, dtype=torch.int64)
    d[known_mask] = 0

    # Frontier = vertices assigned a distance on the most recent iteration
    frontier = known_mask.clone()
    current_dist = 0

    # Safety cap: BFS terminates in at most n iterations. We add a generous
    # margin and assert termination, so an algorithmic bug can't loop forever.
    max_iters = n + 1
    for _ in range(max_iters):
        if not frontier.any():
            break  # nothing left to expand

        # Find edges leaving the frontier
        frontier_edges = frontier[src]  # bool, shape (n_directed_edges,)
        # Destinations of those edges that haven't been assigned yet
        candidate_dst = dst[frontier_edges]
        unassigned = d[candidate_dst] == UNREACHABLE
        new_dst = candidate_dst[unassigned]
        if new_dst.numel() == 0:
            break  # no new vertices found; BFS done

        # Assign next distance. unique() because the same vertex may be
        # reached by multiple edges from this frontier — all get the same
        # distance, which is correct.
        new_dst_unique = torch.unique(new_dst)
        current_dist += 1
        d[new_dst_unique] = current_dist

        # The new frontier is exactly the vertices just assigned
        frontier = torch.zeros(n, dtype=torch.bool)
        frontier[new_dst_unique] = True
    else:
        # Loop did not break (executed max_iters times) — this should never
        # happen in a finite graph; indicates a bug.
        raise RuntimeError("BFS did not terminate; this indicates a bug.")

    return d

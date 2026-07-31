"""Export a prediction as a VTK PolyData file for ParaView.

One file holds everything about one (surface, mask) case: the triangulation
once, plus every depth field and error field as a point array. Nothing is
rescaled or exaggerated — depths are the models' own values, in the survey's
coordinates.

The geometry sits at the ground-truth depth. To *see* another method's
surface in ParaView, apply "Warp By Scalar" with its error array along
(0, 0, 1) and scale 1: since err = z_method - z_true, that displaces each
vertex to exactly z_method. Vertical exaggeration is then a separate
Transform filter, so the stored depths stay untouched either way.

    from horizons.viz.export import prediction_to_polydata
    prediction_to_polydata(pred).save("06TopoCretaceoSuperior.vtp")
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

from horizons.eval.predict import SurfacePrediction
from horizons.viz.mesh import triangulated_polydata

#: Depth fields written to every file, in display order.
DEPTH_FIELDS = ("z_true", "z_init", "z_harmonic", "z_model")

#: Methods that get an error array (signed and absolute) against z_true.
ERROR_METHODS = ("init", "harmonic", "model")


def prediction_to_polydata(
    pred: SurfacePrediction,
    *,
    world_coordinates: bool = True,
) -> pv.PolyData:
    """Bundle a prediction into a single PolyData with all fields attached.

    Parameters
    ----------
    pred : SurfacePrediction
    world_coordinates : bool
        True (default) undoes the per-surface centering, so coordinates match
        the source data. That is an exact additive shift, but it puts UTM
        eastings/northings on the points, and their magnitude (~1e6) costs
        precision in the float32 GPU pipeline, which can show up as jitter
        when you zoom far in. Pass False to keep the centered frame, which
        renders cleanly; the offsets are stored either way.

    Returns
    -------
    pv.PolyData
        Points at the ground-truth depth. Point arrays:
        z_true, z_init, z_harmonic, z_model, err_* / abs_err_* for each
        method, `known` (1 on K), `unknown` (1 on U) and `d`.
        Field data records the surface metadata, RMSEs and the offsets.
    """
    xy = pred.world_xy() if world_coordinates else pred.xy
    depths = {
        name: (
            pred.to_world(getattr(pred, name))
            if world_coordinates
            else getattr(pred, name)
        )
        for name in DEPTH_FIELDS
    }

    mesh = triangulated_polydata(xy, pred.faces, depths["z_true"])

    for name, z in depths.items():
        mesh.point_data[name] = z.detach().cpu().numpy().astype(np.float64)

    z_true = mesh.point_data["z_true"]
    for method in ERROR_METHODS:
        err = mesh.point_data[f"z_{method}"] - z_true
        mesh.point_data[f"err_{method}"] = err
        mesh.point_data[f"abs_err_{method}"] = np.abs(err)

    known = pred.mask.detach().cpu().numpy()
    mesh.point_data["known"] = known.astype(np.uint8)
    mesh.point_data["unknown"] = (~known).astype(np.uint8)
    mesh.point_data["d"] = pred.d.detach().cpu().numpy().astype(np.int32)

    mesh.field_data["surface_id"] = [pred.surface_id]
    mesh.field_data["regime"] = [pred.regime]
    mesh.field_data["reservoir_id"] = [pred.reservoir_id or ""]
    mesh.field_data["N"] = [pred.N]
    mesh.field_data["n_K"] = [pred.n_K]
    mesh.field_data["n_U"] = [pred.n_U]
    mesh.field_data["rmse_harmonic_m"] = [pred.rmse_harmonic]
    mesh.field_data["rmse_model_m"] = [pred.rmse_model]
    mesh.field_data["xy_offset"] = pred.xy_offset.detach().cpu().numpy()
    mesh.field_data["z_offset"] = [pred.z_offset]
    mesh.field_data["world_coordinates"] = [int(world_coordinates)]

    return mesh


def save_prediction(
    pred: SurfacePrediction,
    path: str | Path,
    *,
    world_coordinates: bool = True,
) -> Path:
    """Write one prediction to `path` (.vtp recommended). Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction_to_polydata(
        pred, world_coordinates=world_coordinates
    ).save(path)
    return path

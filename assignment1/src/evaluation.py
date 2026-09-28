from pathlib import Path

import torch
from kaolin.io.obj import import_mesh
from kaolin.ops.conversions import voxelgrids_to_trianglemeshes
from kaolin.ops.mesh import check_sign, sample_points

from metrics import (
    chamfer_distance,
    iou,
    lfd_cameras,
    light_field_distance,
    render_silhouettes,
)
from models import IMNet

DEVICE = "cuda"
EVAL_SEED = 1
GRID_RESOLUTION = 128
EVAL_POINTS = 32768
SURFACE_POINTS = 4096


def predict_field(
    model: IMNet,
    points: torch.Tensor,
    z: torch.Tensor | None,
    target: str,
    batch_size: int,
) -> torch.Tensor:
    """Query points [N, 3] in batches and return a field [N, 1].

    target is "occ" or "sdf". All outputs use positive inside and zero on
    the surface, so evaluation can use the same threshold for every head.
    """
    values = torch.cat([model(batch.unsqueeze(0), z).squeeze(0) for batch in points.split(batch_size)])
    if target == "sdf":
        return -values
    if model.output_format == "imnet":
        return values - 0.5
    return values


@torch.no_grad()
def evaluate(
    model: IMNet,
    z: torch.Tensor | None,
    mesh_path: Path,
    target: str,
    batch_size: int,
    output_dir: Path,
) -> dict[str, float | str | None]:
    """Evaluate one fitted shape and write its reconstructed OBJ to output_dir.

    Return IoU, Chamfer, LFD and a status string. An empty reconstruction
    produces an empty OBJ with unavailable surface metrics set to None;
    an empty rendered view leaves LFD as None.
    """
    model.eval()
    mesh = import_mesh(str(mesh_path))
    target_vertices = mesh.vertices.to(device=DEVICE, dtype=torch.float32)
    target_faces = mesh.faces.to(device=DEVICE)

    # IoU uses uniform queries independent of the training points.
    with torch.random.fork_rng():
        torch.manual_seed(EVAL_SEED)
        points = torch.rand(EVAL_POINTS, 3, device=DEVICE) - 0.5
        target_inside = check_sign(target_vertices.unsqueeze(0), target_faces, points.unsqueeze(0))
        field = predict_field(model, points, z, target, batch_size)
        overlap = iou((field > 0).unsqueeze(0), target_inside.unsqueeze(-1))
        target_surface = sample_points(target_vertices.unsqueeze(0), target_faces, SURFACE_POINTS)[0]

    # This regular grid is for surface extraction; IoU uses the samples above.
    axis = torch.linspace(-0.5, 0.5, GRID_RESOLUTION, device=DEVICE)
    grid_points = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    field = predict_field(model, grid_points, z, target, batch_size)
    metrics = {"iou": overlap.item(), "chamfer": None, "lfd": None, "status": "ok"}
    # Kaolin's mesh kernel cannot process an empty reconstruction.
    if not (field > 0).any():
        (output_dir / f"{mesh_path.stem}.obj").write_text("# Empty reconstruction: no extracted surface.\n")
        metrics["status"] = "empty_reconstruction"
        return metrics

    grid = field.reshape(1, GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION) + 0.5
    vertices, faces = voxelgrids_to_trianglemeshes(grid, iso_value=0.5)
    # Kaolin adds a one-voxel border; undo that offset before mapping to xyz.
    vertices = (vertices[0] - 1) / (GRID_RESOLUTION - 1) - 0.5
    faces = faces[0]

    with (output_dir / f"{mesh_path.stem}.obj").open("w", encoding="utf-8") as obj_file:
        obj_file.write(f"# {mesh_path.stem}: {vertices.shape[0]} vertices, {faces.shape[0]} triangles\n")
        for x, y, z_coordinate in vertices.cpu().tolist():
            obj_file.write(f"v {x:.9g} {y:.9g} {z_coordinate:.9g}\n")
        # OBJ face indices start at one.
        for first, second, third in (faces + 1).cpu().tolist():
            obj_file.write(f"f {first} {second} {third}\n")

    # Chamfer compares samples on the extracted and ground-truth mesh surfaces.
    with torch.random.fork_rng():
        torch.manual_seed(EVAL_SEED)
        predicted_surface = sample_points(vertices.unsqueeze(0), faces, SURFACE_POINTS)[0]
    metrics["chamfer"] = chamfer_distance(predicted_surface, target_surface).item()

    cameras, permutations = lfd_cameras()
    predicted_masks = render_silhouettes(vertices, faces, cameras)
    target_masks = render_silhouettes(target_vertices, target_faces, cameras)
    if not predicted_masks.flatten(-2).any(-1).all():
        metrics["status"] = "empty_silhouette"
        return metrics
    metrics["lfd"] = light_field_distance(
        predicted_masks,
        target_masks,
        permutations,
    ).item()
    return metrics

import torch
from kaolin.io.obj import import_mesh
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.ops.mesh import check_sign, sample_points
from torch.utils.data import Dataset


class ShapeDataset(Dataset):
    """Sample fixed query points and labels from a watertight OBJ in [-0.4, 0.4]^3.

    Uniform sampling covers [-0.5, 0.5]^3. Near-surface sampling replaces
    half the points with surface samples plus Gaussian noise. sigma is the
    per-coordinate standard deviation in mesh units; noise is not clipped.

    Each item is a float32 point [3] and target [1], stored on device.
    Occupancy is 1 inside and 0 outside; SDF is negative inside.
    Points and labels are sampled once and remain fixed across epochs.
    """

    def __init__(
        self,
        mesh_path: str,
        num_points: int,
        target: str = "occ",
        sampling: str = "uniform",
        sigma: float = 0.02,
        seed: int = 0,
        device: str = "cuda",
    ) -> None:
        mesh = import_mesh(mesh_path)
        vertices = mesh.vertices.to(device=device, dtype=torch.float32).unsqueeze(0)
        faces = mesh.faces.to(device=device)

        # Isolate the data seed from model initialization and loader shuffling.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            points = torch.rand(1, num_points, 3, device=device, dtype=torch.float32) - 0.5
            if sampling == "near_surface":
                num_surface = num_points // 2
                surface_points = sample_points(vertices, faces, num_surface)[0]
                points[:, :num_surface] = surface_points + sigma * torch.randn_like(surface_points)

        inside = check_sign(vertices, faces, points)
        values = inside.float()
        if target == "sdf":
            # Kaolin takes triangle coordinates [1, F, 3, 3] and returns squared distances.
            face_vertices = vertices[:, faces]
            squared_distances, _, _ = point_to_mesh_distance(points, face_vertices)
            distances = squared_distances.sqrt()
            values = torch.where(inside, -distances, distances)

        self.points = points.squeeze(0)
        self.targets = values.reshape(num_points, 1)

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.points[index], self.targets[index]

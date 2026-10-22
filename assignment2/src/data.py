"""Mesh loading, signed distance grid and OBJ export (TA code).

Grid conventions: see the docstring of dual_contouring.py.
"""

from pathlib import Path

import kaolin
import numpy as np
import torch

G = 64
REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "data"

TRAIN_SHAPES = [
    "00000067",
    "00000119",
    "00000123",
    "00000157",
    "00000180",
    "00000280",
    "00000474",
    "00000610",
    "00000654",
    "00000849",
    "00001107",
    "00001190",
    "00001353",
    "00001375",
    "00001433",
    "00001491",
    "00001712",
    "00001796",
    "00001972",
    "00002495",
    "00002531",
    "00002556",
    "00002660",
    "00002665",
    "00002944",
    "00003308",
    "00003401",
    "00003527",
    "00003588",
    "00003782",
    "00004240",
    "00004408",
    "00004418",
    "00004518",
    "00004794",
    "00005044",
    "00005197",
    "00005377",
    "00005912",
    "00006048",
    "00006051",
    "00006437",
    "00006542",
    "00006587",
    "00006690",
    "00006737",
    "00006823",
    "00006882",
    "00007001",
    "00007563",
    "00008008",
    "00008016",
    "00008165",
    "00008210",
    "00008525",
    "00008642",
    "00008667",
    "00008875",
    "00008948",
    "00009041",
    "00009094",
    "00009411",
    "00009559",
    "00009747",
]

EVAL_SHAPES = ["00003173", "00004857", "00006462", "00007496", "00008714", "00009943"]

# keeps mesh symmetry planes off the grid lines (kaolin's ray test fails there); the offset must stay
# well above float32 rounding at this magnitude (about 1e-5), so do not shrink it
CENTRE = G / 2 + 0.01
SPHERE_RADIUS = 0.45 * G

# smallest scale factor per axis in random_transform (training poses only); at most 1, so shapes never grow
SCALE_MIN = 0.6

# value of the signed distance away from the surface band (the network clips its input to this anyway)
SDF_CLIP = 2.0


def load_mesh(name):
    """Load a training or evaluation mesh in grid units: centred at CENTRE, bounding-sphere radius 0.45 G.

    name      str            shape name (from TRAIN_SHAPES or EVAL_SHAPES)
    returns
    vertices  (V, 3)         vertex positions in grid units
    faces     (F, 3) long    triangle vertex ids
    """
    folder = "eval" if name in EVAL_SHAPES else "train"
    mesh_file = np.load(DATA_DIR / folder / f"{name}.npz")
    vertices = torch.from_numpy(mesh_file["vertices"]).float().cuda()
    faces = torch.from_numpy(mesh_file["faces"]).long().cuda()

    scale = SPHERE_RADIUS / vertices.norm(dim=1).max()
    vertices = vertices * scale + CENTRE
    return vertices, faces


def random_transform(vertices):
    """Random pose: stretch each axis by its own factor in [SCALE_MIN, 1], rotate, and shift by up to 0.5 per axis.

    The stretch is applied in the mesh's own frame, before the rotation, so the stretched directions end up pointing
    anywhere. All three moves are about CENTRE. The factors are at most 1, so the bounding-sphere radius stays at
    most 0.45 G and the shape stays more than one cell away from the grid border.

    vertices  (V, 3)   vertex positions in grid units
    returns
    vertices  (V, 3)   moved vertex positions
    """
    scale = SCALE_MIN + (1 - SCALE_MIN) * torch.rand(3, device="cuda")

    # a uniformly random unit quaternion gives a uniformly random rotation
    w, x, y, z = torch.nn.functional.normalize(torch.randn(4, device="cuda"), dim=0)
    rotation = torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
        ]
    )

    translation = torch.rand(3, device="cuda") - 0.5

    stretched = (vertices - CENTRE) * scale
    vertices = stretched @ rotation.T + CENTRE + translation
    return vertices


def sdf_grid(vertices, faces, has_hermite):
    """Signed distance to the mesh at every grid point: exact near the surface, +-SDF_CLIP elsewhere.

    The sign comes from kaolin's inside test at every grid point. Exact distances are computed only in a band of
    grid points within 3 steps (per axis) of an edge the mesh crosses.

    vertices     (V, 3)                    mesh vertex positions in grid units
    faces        (F, 3) long               triangle vertex ids
    has_hermite  (3, G+1, G+1, G+1) bool   the mesh crosses grid edge [a, i, j, k] (from exact_hermite)
    returns
    sdf          (G+1, G+1, G+1)           signed distance, negative inside
    """
    coordinates = torch.arange(G + 1, dtype=torch.float32, device="cuda")
    # all grid points as one batch of shape (1, (G+1)^3, 3), the shape kaolin expects
    points = torch.stack(torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij"), dim=-1).reshape(1, -1, 3)

    inside = kaolin.ops.mesh.check_sign(vertices[None], faces, points).reshape(G + 1, G + 1, G + 1)
    sdf = torch.where(inside, -SDF_CLIP, SDF_CLIP)

    # grid points at either end of an edge the mesh crosses
    near_surface = has_hermite.any(dim=0)
    for a in range(3):
        near_surface = near_surface | torch.roll(has_hermite[a], 1, a)

    # band: all grid points within 3 steps (per axis) of an end point of a crossed edge
    band = torch.nn.functional.max_pool3d(near_surface[None, None].float(), 7, stride=1, padding=3)[0, 0] > 0

    face_vertices = kaolin.ops.mesh.index_vertices_by_faces(vertices[None], faces)
    band_points = points[:, band.reshape(-1)]
    squared_distance, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(band_points, face_vertices)
    distance = squared_distance[0].sqrt()
    sdf[band] = torch.where(inside[band], -distance, distance)
    return sdf


def write_obj(path, vertices, faces):
    """Write a mesh as an OBJ file; grid units are mapped back to the unit cube centred at the origin.

    path      Path or str     output file
    vertices  (V, 3)          vertex positions in grid units
    faces     (F, 3) long     triangle vertex ids
    """
    points = ((vertices - CENTRE) / G).tolist()
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in points]
    lines += [f"f {first + 1} {second + 1} {third + 1}" for first, second, third in faces.tolist()]
    Path(path).write_text("\n".join(lines) + "\n")

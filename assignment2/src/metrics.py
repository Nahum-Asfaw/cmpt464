"""Surface and sharp-edge sampling and the Chamfer metrics (TA code).

Grid conventions: see the docstring of dual_contouring.py.
"""

import math

import kaolin
import torch

SHARP_ANGLE = 30
SURFACE_SAMPLES = 100000
EDGE_SAMPLES = 10000


def sample_surface(vertices, faces, count):
    """Uniform random points on the mesh surface.

    vertices  (V, 3)          vertex positions
    faces     (F, 3) long     triangle vertex ids
    count     int             number of points
    returns
    points    (count, 3)      points on the surface
    """
    samples, _ = kaolin.ops.mesh.sample_points(vertices[None], faces, count)
    points = samples[0]
    return points


def sample_edges(vertices, faces, count):
    """Uniform random points on the sharp edges of a mesh (where the faces meet at more than SHARP_ANGLE degrees).

    vertices  (V, 3)          vertex positions
    faces     (F, 3) long     triangle vertex ids
    count     int             number of points
    returns
    points    (count, 3)      points on sharp edges, or (0, 3) if the mesh has none
    """
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    sorted_edges = edges.sort(dim=1).values
    unique_edges, unique_edge_index = torch.unique(sorted_edges, dim=0, return_inverse=True)

    # after sorting by edge, neighbouring copies of one edge come from the faces that share it
    order = torch.argsort(unique_edge_index)
    shared_with_next = unique_edge_index[order[1:]] == unique_edge_index[order[:-1]]
    # entry k of the flattened edge list belongs to face k // 3
    first_face = order[:-1][shared_with_next] // 3
    second_face = order[1:][shared_with_next] // 3
    pair_edge = unique_edge_index[order[:-1][shared_with_next]]

    corners = vertices[faces]
    face_normal = torch.linalg.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    face_normal = torch.nn.functional.normalize(face_normal, dim=1)

    cosine = (face_normal[first_face] * face_normal[second_face]).sum(dim=1)
    is_sharp_pair = cosine < math.cos(math.radians(SHARP_ANGLE))
    sharp = torch.unique(pair_edge[is_sharp_pair])
    if sharp.numel() == 0:
        return torch.zeros(0, 3, device="cuda")

    start = vertices[unique_edges[sharp, 0]]
    end = vertices[unique_edges[sharp, 1]]
    length = (end - start).norm(dim=1)
    chosen_edge = torch.multinomial(length, count, replacement=True)
    fraction = torch.rand(count, 1, device="cuda")
    points = start[chosen_edge] + fraction * (end[chosen_edge] - start[chosen_edge])
    return points


def mesh_distance(points, vertices, faces):
    """Exact distance from every point to the nearest point of the mesh surface.

    points    (P, 3)          query points
    vertices  (V, 3)          vertex positions
    faces     (F, 3) long     triangle vertex ids
    returns
    distance  (P,)            distance to the surface
    """
    face_vertices = kaolin.ops.mesh.index_vertices_by_faces(vertices[None], faces)
    squared_distance, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(points[None], face_vertices)
    distance = squared_distance[0].sqrt()
    return distance


def point_distance(points_a, points_b):
    """Distance from every point of points_a to its nearest point in points_b.

    points_a  (A, 3)   query points
    points_b  (B, 3)   target points
    returns
    distance  (A,)     distance to the nearest target point
    """
    squared_distance, _ = kaolin.metrics.pointcloud.sided_distance(points_a[None], points_b[None])
    distance = squared_distance[0].sqrt()
    return distance


def chamfer_distance(a_to_b, b_to_a):
    """Chamfer distance: the mean distance of each direction, averaged over the two directions.

    a_to_b   (A,)     distances from the points of a to b
    b_to_a   (B,)     distances from the points of b to a
    returns
    chamfer  float
    """
    chamfer = ((a_to_b.mean() + b_to_a.mean()) / 2).item()
    return chamfer


def evaluate(vertices, faces, reference_vertices, reference_faces):
    """Surface and sharp-edge Chamfer distance against the reference mesh, plus the mesh size.

    vertices            (V, 3)        mesh to evaluate, grid units
    faces               (F, 3) long
    reference_vertices  (V', 3)       reference mesh, grid units
    reference_faces     (F', 3) long
    returns
    result              dict          chamfer, edge_chamfer (None without sharp edges), num_vertices, num_faces
    """
    surface = sample_surface(vertices, faces, SURFACE_SAMPLES)
    reference_surface = sample_surface(reference_vertices, reference_faces, SURFACE_SAMPLES)
    edges = sample_edges(vertices, faces, EDGE_SAMPLES)
    reference_edges = sample_edges(reference_vertices, reference_faces, EDGE_SAMPLES)

    surface_to_reference = mesh_distance(surface, reference_vertices, reference_faces)
    reference_to_surface = mesh_distance(reference_surface, vertices, faces)
    result = {
        "chamfer": chamfer_distance(surface_to_reference, reference_to_surface),
        "edge_chamfer": None,
        "num_vertices": vertices.shape[0],
        "num_faces": faces.shape[0],
    }

    # smooth shapes have no sharp edges; their edge metrics stay None
    if edges.shape[0] > 0 and reference_edges.shape[0] > 0:
        edges_to_reference = point_distance(edges, reference_edges)
        reference_to_edges = point_distance(reference_edges, edges)
        result["edge_chamfer"] = chamfer_distance(edges_to_reference, reference_to_edges)

    return result

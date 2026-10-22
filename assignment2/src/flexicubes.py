"""FlexiCubes: Dual Marching Cubes with learned weights, and fitting a mesh to a shape from a random SDF.

Grid conventions: see the docstring of dual_contouring.py; flat layouts and entries: see the docstring of dmc.py.
"""

import time

import dmc
import kaolin
import torch
from data import G, load_mesh, write_obj
from metrics import evaluate, sample_surface
from torch.nn import functional

WEIGHT_SCALE = 0.99
STEPS = 1000
LEARNING_RATE = 0.01

# fit works in the official units: the grid is the box [-1, 1]^3, so one unit is G / 2 cells
UNIT = G / 2
DEFORM_LIMIT = (2 - 1e-8) / (G * 2)  # grid points move just under half a cell

POINTS_UNIFORM = 2000
POINTS_SURFACE = 2000
SURFACE_NOISE = 0.05

SDF_LOSS_WEIGHT = 200000  # chosen from a sweep over 2e3, 2e4, 2e5
SIGN_LOSS_START = 0.2  # sign loss weight, decays linearly to SIGN_LOSS_END over the first quarter of the steps
SIGN_LOSS_END = 0.01
DEV_LOSS_WEIGHT = 0.5
L1_LOSS_WEIGHT = 0.1

LOG_EVERY = 100


def bound_weights(raw_weights):
    """Map the free per-cube parameters into the ranges of alpha, beta and gamma.

    raw_weights  (G^3, 21)   free parameters, any real values: columns 0-11 for beta, 12-19 for alpha, 20 for gamma
                             (the layout of the official code)
    returns
    alpha        (G^3, 8)    one weight per cube corner, in (0.01, 1.99)
    beta         (G^3, 12)   one weight per cube edge, in (0.01, 1.99)
    gamma        (G^3,)      one weight per cube, in (0.005, 0.995)
    """
    alpha = torch.tanh(raw_weights[:, 12:20]) * WEIGHT_SCALE + 1
    beta = torch.tanh(raw_weights[:, :12]) * WEIGHT_SCALE + 1
    gamma = torch.sigmoid(raw_weights[:, 20]) * WEIGHT_SCALE + (1 - WEIGHT_SCALE) / 2
    return alpha, beta, gamma


def edge_crossings(sdf_flat, positions, edges, alpha):
    """Zero crossing on each edge after scaling its two end values by their corner weights alpha (eq. 5).

    sdf_flat   ((G+1)^3,)      SDF value of every grid point
    positions  ((G+1)^3, 3)    position of every grid point
    edges      (K, 2) long     grid point ids of the two ends of each edge
    alpha      (K, 2)          alpha of the corner at each end
    returns
    crossings  (K, 3)          crossing point on each edge
    """
    ##### your code starts here ##### Problem 3.2
    # NOTE: use batched torch tensor operations that process all grid points, edges and cells at once, and avoid
    # Python for loops over them (a loop over the three axes is fine). Look through the torch documentation for
    # the functions that, combined, do this.

    # SDF value at the two ends of each edge, each scaled by the alpha of its corner, (K,) each
    sdf_start = ...
    sdf_end = ...

    # positions of the two ends of each edge, (K, 3) each
    position_start = positions[edges[:, 0]]
    position_end = positions[edges[:, 1]]

    crossings = ...
    ##### your code ends here #####
    return crossings


def dual_vertices(crossings, entry_vertex, vertex_count, beta):
    """Each vertex is the beta-weighted mean of the crossings on its edges (eq. 6).

    crossings     (K, 3)       alpha-weighted crossing of each entry
    entry_vertex  (K,) long    vertex of each entry
    vertex_count  int          number of vertices V
    beta          (K,)         beta of each entry's edge
    returns
    vertices      (V, 3)       vertex positions
    """
    ##### your code starts here ##### Problem 3.2
    # NOTE: use batched torch tensor operations that process all grid points, edges and cells at once, and avoid
    # Python for loops over them (a loop over the three axes is fine). Look through the torch documentation for
    # the functions that, combined, do this.

    # beta_sum (V,): the sum of the betas of each vertex's entries
    beta_sum = torch.zeros(vertex_count, device="cuda").index_add(0, entry_vertex, beta)

    # weighted_crossings (K, 3): each entry's crossing times its beta;
    # weighted_crossing_sum (V, 3): their sum over each vertex's entries
    weighted_crossings = ...
    weighted_crossing_sum = torch.zeros(vertex_count, 3, device="cuda").index_add(0, entry_vertex, weighted_crossings)

    vertices = ...
    ##### your code ends here #####
    return vertices


def quads_to_triangles(quads, vertices, vertex_gamma, training):
    """Split every quad into triangles using gamma: around a centre vertex when training (eq. 7), else along one
    diagonal.

    quads         (Q, 4) long   vertex ids of each quad, in order around the quad
    vertices      (V, 3)        vertex positions
    vertex_gamma  (V,)          gamma of each vertex (the gamma of its cube)
    training      bool          True: 4 triangles around a new centre vertex; False: 2 triangles
    returns
    vertices      (V + Q, 3) when training (the centre vertices appended), else (V, 3)
    faces         (4Q, 3) long when training, else (2Q, 3) long
    """
    ##### your code starts here ##### Problem 3.3
    # NOTE: use batched torch tensor operations that process all grid points, edges and cells at once, and avoid
    # Python for loops over them (a loop over the three axes is fine). Look through the torch documentation for
    # the functions that, combined, do this.

    # gamma_02, gamma_13 (Q,): the product of the gammas at the two ends of diagonal 0-2 and of diagonal 1-3
    gamma_02 = ...
    gamma_13 = ...

    if training:
        # differentiable split: a centre vertex between the two diagonal midpoints, weighted by gamma, and 4 triangles;
        # mid_02, mid_13, centre (Q, 3); centre_id (Q,) long: the ids the centre vertices get when appended
        mid_02 = (vertices[quads[:, 0]] + vertices[quads[:, 2]]) / 2
        mid_13 = (vertices[quads[:, 1]] + vertices[quads[:, 3]]) / 2
        centre = ...

        centre_id = torch.arange(quads.shape[0], device="cuda") + vertices.shape[0]
        vertices = torch.cat([vertices, centre])

        faces = []
        for corner in range(4):
            faces.append(torch.stack([quads[:, corner], quads[:, (corner + 1) % 4], centre_id], dim=1))
        faces = torch.cat(faces)
    else:
        # split_02 (Q, 1) bool: the quad is split along diagonal 0-2;
        # diagonal 0-2 gives triangles (0, 1, 2), (0, 2, 3); diagonal 1-3 gives (0, 1, 3), (1, 2, 3)
        split_02 = ...
        triangle_pairs = torch.where(split_02, quads[:, [0, 1, 2, 0, 2, 3]], quads[:, [0, 1, 3, 1, 2, 3]])
        faces = triangle_pairs.reshape(-1, 3)
    ##### your code ends here #####
    return vertices, faces


def deviation_loss(plain_crossings, vertices, entry_vertex):
    """L_dev (eq. 8): how far each entry's distance to its vertex is from the mean distance of that vertex's entries.

    The distances are measured to the unweighted crossings, as in the official code.

    plain_crossings  (K, 3)       unweighted crossing of each entry (from dmc.edge_crossings)
    vertices         (V, 3)       vertex positions (from dual_vertices)
    entry_vertex     (K,) long    vertex of each entry
    returns
    L_dev            (K,)         deviation term of each entry
    """
    ##### your code starts here ##### Problem 3.4
    # NOTE: use batched torch tensor operations that process all grid points, edges and cells at once, and avoid
    # Python for loops over them (a loop over the three axes is fine). Look through the torch documentation for
    # the functions that, combined, do this.

    vertex_count = vertices.shape[0]

    # distance (K,): distance from each entry's plain crossing to its vertex
    distance = ...

    # entry_count, distance_sum (V,): the number of each vertex's entries and the sum of their distances;
    # mean_distance (V,): the mean distance of each vertex's entries
    one_per_entry = torch.ones_like(distance)
    entry_count = torch.zeros(vertex_count, device="cuda").index_add(0, entry_vertex, one_per_entry)
    distance_sum = torch.zeros(vertex_count, device="cuda").index_add(0, entry_vertex, distance)
    mean_distance = ...

    L_dev = ...
    ##### your code ends here #####
    return L_dev


def flexicubes(sdf, positions, raw_weights, training):
    """FlexiCubes mesh of the SDF on the (deformed) grid, plus the regulariser L_dev per entry.

    sdf          (G+1, G+1, G+1)   SDF value of every grid point
    positions    ((G+1)^3, 3)      position of every grid point, in any unit
    raw_weights  (G^3, 21)         free per-cube parameters (see bound_weights)
    training     bool              split of the quads (see quads_to_triangles)
    returns
    vertices     (V, 3)            mesh vertices, in the units of positions
    faces        (F, 3) long       vertex ids of each triangle
    L_dev        (K,)              deviation term of each entry
    """
    # only the corner ids: the grid point positions come in as an argument (fit deforms them)
    cubes = dmc.cube_grid()[1]
    sdf_flat = sdf.reshape(-1)
    alpha, beta, gamma = bound_weights(raw_weights)

    cases = dmc.cube_cases(sdf_flat, cubes)

    # see dmc.dual_topology for what each list holds
    entry_edges, entry_cube, entry_local_edge, entry_vertex, vertex_cube, quads = dmc.dual_topology(sdf_flat, cases, cubes)

    # alpha at the two corners of each entry's edge, in the same order as the edge's two grid points
    entry_alpha = alpha[entry_cube[:, None], dmc.CUBE_EDGES[entry_local_edge]]
    crossings = edge_crossings(sdf_flat, positions, entry_edges, entry_alpha)

    entry_beta = beta[entry_cube, entry_local_edge]
    vertices = dual_vertices(crossings, entry_vertex, vertex_cube.shape[0], entry_beta)

    vertex_gamma = gamma[vertex_cube]
    mesh_vertices, faces = quads_to_triangles(quads, vertices, vertex_gamma, training)

    plain_crossings = dmc.edge_crossings(sdf_flat, positions, entry_edges)
    L_dev = deviation_loss(plain_crossings, vertices, entry_vertex)
    return mesh_vertices, faces, L_dev


def signed_distance(points, vertices, faces):
    """Distance from each point to the mesh, negative inside; the sign carries no gradient.

    points    (P, 3)          query points
    vertices  (V, 3)          mesh vertex positions
    faces     (F, 3) long     triangle vertex ids
    returns
    signed    (P,)            signed distance
    """
    # a triangle collapsed to a point makes kaolin return NaN for every point, so collapsed triangles are left out
    corners = vertices[faces]
    edge_cross_product = torch.linalg.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    not_collapsed = edge_cross_product.ne(0).any(dim=1)
    faces = faces[not_collapsed]

    face_vertices = kaolin.ops.mesh.index_vertices_by_faces(vertices[None], faces)
    squared_distance, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(points[None], face_vertices)
    # + 1e-10 keeps the derivative of sqrt finite at distance 0 (the official depth loss adds 1e-8 the same way)
    distance = (squared_distance[0] + 1e-10).sqrt()

    with torch.no_grad():
        inside = kaolin.ops.mesh.check_sign(vertices[None], faces, points[None])[0]
    signed = torch.where(inside, -distance, distance)
    return signed


def fit(shape, output_dir):
    """Fit FlexiCubes to a shape, starting from a random SDF.

    The fit runs in the official FlexiCubes units, where the grid is the box [-1, 1]^3. Each step extracts the mesh,
    compares its signed distance with the reference's at random points (in the box and near the surface), and adds
    the official regularisers. It optimises the SDF, the grid deformation and the weights. The final mesh is moved
    back to grid units, written and evaluated.

    shape       str    name of the shape
    output_dir  Path   where fit_log.txt and the OBJ files are written
    returns
    result      dict   evaluate()'s metrics, the time in seconds and the last step's unweighted losses
    """
    reference_vertices, reference_faces = load_mesh(shape)
    target_vertices = (reference_vertices - G / 2) / UNIT
    grid_point_positions = dmc.cube_grid()[0]
    grid_positions = (grid_point_positions - G / 2) / UNIT

    sdf = (torch.rand(G + 1, G + 1, G + 1, device="cuda") - 0.1).requires_grad_()
    raw_deform = torch.zeros((G + 1) ** 3, 3, device="cuda", requires_grad=True)
    raw_weights = torch.zeros(G**3, 21, device="cuda", requires_grad=True)
    optimizer = torch.optim.Adam([sdf, raw_deform, raw_weights], lr=LEARNING_RATE)

    # every grid edge as a pair of grid point ids, for the sign loss
    grid_point_id = torch.arange((G + 1) ** 3, device="cuda").reshape(G + 1, G + 1, G + 1)
    grid_edges = []
    for a in range(3):
        edge_start = grid_point_id.narrow(a, 0, G).reshape(-1)
        edge_end = grid_point_id.narrow(a, 1, G).reshape(-1)
        grid_edges.append(torch.stack([edge_start, edge_end], dim=1))
    grid_edges = torch.cat(grid_edges)

    log_path = output_dir / "fit_log.txt"
    start_time = time.perf_counter()
    for step in range(STEPS):
        positions = grid_positions + DEFORM_LIMIT * torch.tanh(raw_deform)
        vertices, faces, L_dev = flexicubes(sdf, positions, raw_weights, True)

        # SDF loss on random points (official sample_random_points): uniform in the box, and near the reference surface
        surface_points = sample_surface(target_vertices, reference_faces, POINTS_SURFACE)
        uniform_points = torch.rand(POINTS_UNIFORM, 3, device="cuda") * 2 - 1
        noise = torch.randn_like(surface_points) * SURFACE_NOISE
        points = torch.cat([uniform_points, surface_points + noise])

        with torch.no_grad():
            target_sdf = signed_distance(points, target_vertices, reference_faces)
        sdf_loss = functional.mse_loss(signed_distance(points, vertices, faces), target_sdf)

        # official sdf_reg_loss: on every grid edge with a sign change, push each end towards the other end's sign
        edge_ends = sdf.reshape(-1)[grid_edges]
        sign_differs = torch.sign(edge_ends[:, 0]) != torch.sign(edge_ends[:, 1])
        edge_ends = edge_ends[sign_differs]
        sign_loss = functional.binary_cross_entropy_with_logits(edge_ends[:, 0], (edge_ends[:, 1] > 0).float()) + functional.binary_cross_entropy_with_logits(edge_ends[:, 1], (edge_ends[:, 0] > 0).float())
        sign_weight = SIGN_LOSS_START - (SIGN_LOSS_START - SIGN_LOSS_END) * min(1, 4 * step / STEPS)

        dev_loss = L_dev.mean()
        l1_loss = raw_weights[:, :20].abs().mean()

        loss = SDF_LOSS_WEIGHT * sdf_loss + sign_weight * sign_loss + DEV_LOSS_WEIGHT * dev_loss + L1_LOSS_WEIGHT * l1_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % LOG_EVERY == 0:
            log_line = f"{shape} flexicubes step {step} sdf {sdf_loss.item():.3e} sign {sign_loss.item():.4f} " f"dev {dev_loss.item():.4f} l1 {l1_loss.item():.4f}"
            print(log_line)
            with open(log_path, "a") as log_file:
                log_file.write(log_line + "\n")

    with torch.no_grad():
        positions = grid_positions + DEFORM_LIMIT * torch.tanh(raw_deform)
        vertices, faces, L_dev = flexicubes(sdf, positions, raw_weights, False)

    vertices = vertices * UNIT + G / 2

    torch.cuda.synchronize()
    seconds = time.perf_counter() - start_time

    write_obj(output_dir / f"{shape}_input.obj", reference_vertices, reference_faces)
    write_obj(output_dir / f"{shape}_flexicubes.obj", vertices, faces)

    result = evaluate(vertices, faces, reference_vertices, reference_faces)
    result["seconds"] = seconds
    result["sdf_loss"] = sdf_loss.item()
    result["sign_loss"] = sign_loss.item()
    result["dev_loss"] = dev_loss.item()
    result["l1_loss"] = l1_loss.item()
    return result

"""Dual contouring (student code, plain torch): exact Hermite data from the mesh, one QEF vertex per cell, the faces,
and Hermite data estimated from the SDF alone.

Grid conventions, used in every file:
  grid point (i, j, k), 0 <= i, j, k <= G, sits at (i, j, k) in grid units.
  cell (i, j, k), 0 <= i, j, k < G, is the unit cube whose lowest corner is grid point (i, j, k).
  grid edge [a, i, j, k] goes from grid point (i, j, k) one step along axis a (0 = x, 1 = y, 2 = z).
  Per-edge tensors have shape (3, G+1, G+1, G+1); the last slice along a would leave the grid and is never used.
  A crossing on an edge sits at  edge start + t * (one step along a),  0 <= t < 1.
  sdf is negative inside; a grid point is inside when sdf <= 0.
  A cell's vertex is stored as vertex_in_cell, its position inside the cell, in [0, 1]^3.
  Cell ids are (i * G + j) * G + k, the order of a (G, G, G) tensor read row by row.
  The shapes stay at least one cell away from the grid border, so every crossed edge has four cells.
"""

import torch
from data import G

# The four cells around a grid edge along axis a, relative to the edge's start point, so that, seen from the +a end,
# they go round counter-clockwise; dual_faces relies on this order.
CELL_OFFSETS = torch.tensor(
    [
        [[0, -1, -1], [0, 0, -1], [0, 0, 0], [0, -1, 0]],
        [[-1, 0, -1], [-1, 0, 0], [0, 0, 0], [0, 0, -1]],
        [[-1, -1, 0], [0, -1, 0], [0, 0, 0], [-1, 0, 0]],
    ],
    device="cuda",
)

# The 8 corners of a cell (order does not matter here: only used for counting).
CORNERS = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]

LAMBDA_START = 1e-3
LAMBDA_GROWTH = 10
LAMBDA_ROUNDS = 8


def edge_cells(edge_start, a):
    """Ids of the four cells around each grid edge along axis a, in CELL_OFFSETS order.

    edge_start  (E, 3) long   start grid point of each edge
    a           int           axis of the edges
    returns
    cell_id     (E, 4) long   cell ids of the four cells, counter-clockwise seen from the +a end
    """
    cells = edge_start[:, None, :] + CELL_OFFSETS[a]
    cell_id = (cells[..., 0] * G + cells[..., 1]) * G + cells[..., 2]
    return cell_id


def crossed_edges(sdf):
    """Which grid points are inside, which grid edges change sign, and which cells need a vertex.

    sdf          (G+1, G+1, G+1)          value at every grid point; only its sign is used
    returns
    inside       (G+1, G+1, G+1) bool     sdf <= 0
    sign_change  (3, G+1, G+1, G+1) bool  the two ends of edge [a, i, j, k] are on different sides
    has_vertex   (G, G, G) bool           the cell has corners on both sides
    """
    ##### your code starts here ##### Problem 1.1
    # NOTE: use batched torch tensor operations that process all grid points, edges and cells at once, and avoid
    # Python for loops over them (a loop over the three axes is fine). Look through the torch documentation for
    # the functions that, combined, do this.

    # Inside: points in the sdf that are <= 0
    # some sort of selection
    inside = sdf <= 0
    #print(f'sdf: {sdf}\ninside: {inside}')
    #print(f'inside[:, :, 0][0]: {inside[:, :, 0][0]}')
    #print(f'inside[:, :, 0][0][0]: {inside[:, :, 0][0][0]}')


    # compare every grid point with its next neighbour along each axis;
    # the last slice along that axis has no edge and stays False
    sign_change = torch.zeros(3, G + 1, G + 1, G + 1, device="cuda", dtype=torch.bool)
    #print(f'sign_change: {sign_change}')
    #sign_change[0, :G, :, :] = ... # true if the two ends of the edge are on diff sides... how do I check that?
                                   # this is in the x dir
                                   # idea: if inside[:, x][0][0] != inside[:, x][0][1] then
                                   # sign_change[:, x][0][0] = true else false

    # AI:
    sign_change[0, :G, :, :] = inside[:-1, :, :] != inside[1:, :, :]
    sign_change[1, :, :G, :] = inside[:, :-1, :] != inside[:, 1:, :]
    sign_change[2, :, :, :G] = inside[:, :, :-1] != inside[:, :, 1:]
    #print(f"sign_change[0, :G, :, :]: {sign_change[0, :G, :, :]}")

    # My initial implemntation created a bug where ~ half of the triangles weren't shaded...
    # I did: sign_change[0, :G, :, :] = torch.ne(inside[:G, :, :][:, :], inside[:G, :, :][:, : + 1])
    # It also scored NaN on the majority of chamfer distance calculations

    # inside_corner_count (G, G, G) int: how many of the cell's 8 corners are inside;
    # a cell needs a vertex when its 8 corners are not all on one side
    # "for each offset, take a (G, G, G)-shaped view of inside starting at that offset"
    # 
    # if its (0, 1, 1) -> inside[:, :G, :G]
    # if its (1, 0, 1) -> inside[:G, :, :G] ??
    inside_corner_count = torch.zeros(G, G, G, device="cuda")

    # AI:
    for dx, dy, dz in CORNERS:
        corner_inside = inside[
            dx:dx + G,
            dy:dy + G,
            dz:dz + G,
        ]
        inside_corner_count += corner_inside.to(torch.int64)

    has_vertex = (inside_corner_count > 0) & (inside_corner_count < 8)
    ##### your code ends here #####
    return inside, sign_change, has_vertex


def dual_faces(inside, sign_change, has_vertex):
    """Two triangles per sign-changing edge, joining the vertices of its four cells, facing outward.

    inside       (G+1, G+1, G+1) bool      grid point is inside
    sign_change  (3, G+1, G+1, G+1) bool   edge changes sign
    has_vertex   (G, G, G) bool            cell has a vertex
    returns
    faces        (F, 3) long               vertex ids of each triangle, in has_vertex.nonzero() order
    """
    ##### your code starts here ##### Problem 1.2
    # NOTE: use batched torch tensor operations that process all grid points, edges and cells at once, and avoid
    # Python for loops over them (a loop over the three axes is fine). Look through the torch documentation for
    # the functions that, combined, do this.

    # vertex_id (G^3,) long: number the cells that have a vertex 0, 1, 2, ... in has_vertex.nonzero() order, the
    # order in which dual_contouring lists the vertices (entries of cells without a vertex are never used)
    vertex_id = torch.cumsum(has_vertex.reshape(-1), dim=0) - 1

    faces = []
    for a in range(3):
        # edge_start (E, 3) long: start grid point of every edge along axis a that changes sign;
        # quad (E, 4) long: vertex ids of the four cells around each of these edges, in CELL_OFFSETS order
        edge_start = sign_change[a].nonzero() # similar line in qef_verticies
                                              # im not sure what nonzero() does
        #print(f"edge_start:", edge_start)

        # for edge in edge_start:
        #  quads.append(vertex id from cell_offset of edge)
        quad = vertex_id[edge_cells(edge_start, a)]
        #print("quad: ", quad)

        # start_inside (E,) bool: whether the start grid point of each edge is inside the shape
        # true if edge_start is in inside, else false
        #print("inside:", inside)
        start_inside = inside[edge_start[:, 0], edge_start[:, 1], edge_start[:, 2]]
        #print("start_inside:", start_inside)
        # flip the winding when the edge starts outside, so every face points outward
        # can't seperate them cause then it would be out of order

        # AI:
        quad = torch.where(start_inside[:, None], quad, quad[:, [0, 3, 2, 1]])
        # again I'm not familiar with pytorch ops, so I didn't know how to index start_inside
        # and I didn't know what torch.where() was

        faces.append(quad[:, [0, 1, 2]])
        faces.append(quad[:, [0, 2, 3]])

    faces = torch.cat(faces)
    ##### your code ends here #####
    return faces


def exact_hermite(vertices, faces):
    """Exact Hermite data of the mesh on every grid edge: does the mesh cross it, where, and with which normal.

    This is the ground truth that estimate_hermite (Problem 1.5) estimates from the SDF alone. It is used for the
    exact run of dual contouring (Problem 1), for the NDC training targets (Problem 2), and by sdf_grid to decide
    where exact distances are needed. For each axis, every grid line along that axis is tested against the triangles
    whose bounding box it passes through, seen along the axis as a point inside a flat triangle; each hit gives the
    edge it lands on a crossing position t and the normal of the triangle it hit. If several triangles cross one
    edge (a thin part of the shape), the crossing nearest the edge start is kept.

    vertices     (V, 3)                    mesh vertex positions in grid units
    faces        (F, 3) long               triangle vertex ids
    returns
    has_hermite  (3, G+1, G+1, G+1) bool   the mesh crosses grid edge [a, i, j, k]
    t            (3, G+1, G+1, G+1)        crossing position on the edge, as its distance from the edge start
    normal       (3, G+1, G+1, G+1, 3)     unit normal of the crossing triangle
    """
    ##### your code starts here ##### Problem 1.3
    # NOTE: use batched torch tensor operations that process all grid points, edges and cells at once, and avoid
    # Python for loops over them (a loop over the three axes is fine). Look through the torch documentation for
    # the functions that, combined, do this.

    has_hermite = torch.zeros(3, G + 1, G + 1, G + 1, dtype=torch.bool, device="cuda")
    t = torch.zeros(3, G + 1, G + 1, G + 1, device="cuda")
    normal = torch.zeros(3, G + 1, G + 1, G + 1, 3, device="cuda")

    # the corners of every triangle and two of its edges, (F, 3) each
    corner0, corner1, corner2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    edge1 = corner1 - corner0
    edge2 = corner2 - corner0

    # unit normal of every triangle, (F, 3), pointing to the side from which its corners go round counter-clockwise
    face_normal = torch.nn.functional.normalize(torch.linalg.cross(edge1, edge2)) # how do i normalize this???
    #print(f'face_normal: {face_normal}')

    # the integer grid coordinates inside each triangle's bounding box, (F, 3) long
    corners = torch.stack([corner0, corner1, corner2], dim=1)
    box_low = corners.min(dim=1).values.ceil().long()
    box_high = corners.max(dim=1).values.floor().long()

    # every grid line along axis a is fixed by its two other coordinates b and c
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        direction = torch.zeros(3, device="cuda")
        direction[a] = 1

        # candidates: each triangle paired with every grid line inside its bounding box in the (b, c) plane;
        # count_b, count_c, line_count (F,) long: how many integer b, c, and (b, c) pairs the triangle's box holds
        count_b = (box_high[:, b] - box_low[:, b] + 1).clamp(min=0)
        count_c = (box_high[:, c] - box_low[:, c] + 1).clamp(min=0)
        line_count = count_b * count_c

        # candidate_triangle (P,) long: the triangle id of every candidate pair, each triangle repeated line_count
        # times; P = line_count.sum() is the number of candidate pairs
        candidate_triangle = torch.repeat_interleave(torch.arange(faces.shape[0], device="cuda"), line_count)

        # index_in_triangle (P,) long: number of each candidate among its own triangle's candidates,
        # 0, 1, ..., line_count - 1; first_candidate (F,) long is where each triangle's candidates begin
        first_candidate = torch.cumsum(line_count, dim=0) - line_count
        candidate_id = torch.arange(candidate_triangle.shape[0], device="cuda")
        index_in_triangle = candidate_id - first_candidate[candidate_triangle]

        # P candidate pairs: pair p tests the grid line  line_origin[p] + s * direction,  0 <= s <= G,
        # against triangle candidate_triangle[p]; line_origin (P, 3) is 0 in coordinate a and an integer in b and c
        line_origin = torch.zeros(candidate_triangle.shape[0], 3, device="cuda")
        line_origin[:, b] = box_low[candidate_triangle, b] + index_in_triangle // count_c[candidate_triangle]
        line_origin[:, c] = box_low[candidate_triangle, c] + index_in_triangle % count_c[candidate_triangle]

        # the corners of each pair's triangle, (P, 3) each
        pair_corner0 = corner0[candidate_triangle]
        pair_corner1 = corner1[candidate_triangle]
        pair_corner2 = corner2[candidate_triangle]

        # A simplified ray-triangle test. Projected along axis a, the grid line becomes a point q in the (b, c) plane,
        # and the line hits the triangle exactly when q lies on the same side of all three edges of the projected
        # triangle. The sign of a 2D cross product tells the side: w0 = (corner1 - q) x (corner2 - q) is positive on
        # one side of edge 1-2 and negative on the other, and w1, w2 do the same for edges 2-0 and 0-1.

        # vectors from q to the three corners in the (b, c) plane, (P, 2) each
        to_corner0 = pair_corner0[:, [b, c]] - line_origin[:, [b, c]]
        to_corner1 = pair_corner1[:, [b, c]] - line_origin[:, [b, c]]
        to_corner2 = pair_corner2[:, [b, c]] - line_origin[:, [b, c]]

        # w0, w1, w2: the side of each edge that q lies on, and their sum, (P,) each
        w0 = to_corner1[:, 0] * to_corner2[:, 1] - to_corner1[:, 1] * to_corner2[:, 0]
        w1 = to_corner2[:, 0] * to_corner0[:, 1] - to_corner2[:, 1] * to_corner0[:, 0]
        w2 = to_corner0[:, 0] * to_corner1[:, 1] - to_corner0[:, 1] * to_corner1[:, 0]
        area = w0 + w1 + w2

        # same_sign (P,) bool: w0, w1, w2 share one sign (zeros count, so a line through an edge shared by two
        # triangles hits both); hit (P,) bool: same sign and a nonzero sum (a zero sum means the triangle is
        # parallel to the line)
        #same_sign = torch.zeros(1, G + 1, device="cuda")
        #for i in range(len(w0)):
        #    same_sign[i] = True if ((w0[i] >= 0 and w1[i] >= 0 and w2[i] >= 0) or (w0[i] <= 0 and w1[0] <= 0 and w2[0] <= 0)) else False
         
        # AI:
        same_sign = (((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0)))

        #print(f'same_sign:{same_sign}')
        hit = (area != 0) & same_sign

        # divided by their sum, w0, w1, w2 weight the three corners to rebuild q, and the same weights give the
        # coordinate a of the hit point, (P,); line_origin is 0 in coordinate a, so this is also s along the line
        line_position = (w0 * pair_corner0[:, a] + w1 * pair_corner1[:, a] + w2 * pair_corner2[:, a]) / area

        # H hits: each lies on the edge that starts at the integer part of its position along the line, at t = the
        # fractional part; hit_triangle (H,) long, edge_start (H, 3) long, hit_t (H,)
        hit_position = line_position[hit]
        hit_triangle = candidate_triangle[hit]
        edge_start = line_origin[hit].long()
        #print(f'edge_start: {edge_start}\nhit_triangle: {hit_triangle}')
        # what does edge_start mean?
        # the edge_start => int component of hit_position
        # hit_t => decimal component of hit_position
        edge_start[:, a] = hit_position.int()
        hit_t = hit_position - hit_position.int() # smth - smth => [0, 1]
        #print(f"edge_start: {edge_start}\nhit_triangle: {hit_triangle}\n" \
        #      f"hit_position: {hit_position}\nhit_t: {hit_t}")

        # flat id of each hit's edge in the (G+1)^3 slice of axis a, (H,) long
        edge_id = (edge_start[:, 0] * (G + 1) + edge_start[:, 1]) * (G + 1) + edge_start[:, 2]

        # when several triangles hit the same edge (a part thinner than a cell, or a line through a shared triangle
        # edge), we keep the hit nearest the edge start; the edge gets its t and the hit triangle's normal;
        # smallest_t ((G+1)^3,): the smallest t of the hits on each edge (inf where there is none),
        # a per-edge minimum that torch.Tensor.scatter_reduce computes in one call;
        # nearest (H,) bool: the hit has the smallest t on its edge
        smallest_t = torch.full(((G + 1) ** 3,), float("inf"), device="cuda")

        # AI:
        smallest_t = torch.Tensor.scatter_reduce_(smallest_t, 0, edge_id, hit_t, reduce="amin", include_self=False)
        nearest = (smallest_t[edge_id] == hit_t)
        #print(f"nearest: {nearest}\nsmallest_t: {smallest_t}")

        has_hermite[a].view(-1)[edge_id[nearest]] = True
        t[a].view(-1)[edge_id[nearest]] = hit_t[nearest]
        #print(f'face_normal[:, a]: {face_normal[:, a]}')
        #print(f'torch.masked_select(hit_triangle, nearest)]: {torch.masked_select(hit_triangle, nearest)}')
        # Basically using nearest to select the triangles in hit_triangles to get the normal from.
        normal[a].view(-1, 3)[edge_id[nearest]] = face_normal[(torch.masked_select(hit_triangle, nearest))]

    ##### your code ends here #####
    return has_hermite, t, normal


def qef_vertices(qef_edges, t, normal):
    """Vertex of every cell from the QEF of the crossings around it, pulled slightly towards their mass point.

    The QEF of a cell sums, over the crossings on its edges, the squared distance to the plane through the crossing
    with the crossing's normal. A solution that leaves its cell is solved again with a 10 times stronger pull
    (LAMBDA_ROUNDS attempts); cells that never land inside keep the cell centre.

    qef_edges       (3, G+1, G+1, G+1) bool   edges whose Hermite data go into the QEF
    t               (3, G+1, G+1, G+1)        crossing position on each edge
    normal          (3, G+1, G+1, G+1, 3)     unit surface normal at each crossing
    returns
    vertex_in_cell  (G, G, G, 3)              vertex position inside each cell, in [0, 1]^3
    """
    ##### your code starts here ##### Problem 1.4
    # NOTE: use batched torch tensor operations that process all grid points, edges and cells at once, and avoid
    # Python for loops over them (a loop over the three axes is fine). Look through the torch documentation for
    # the functions that, combined, do this.

    # The vertex x of a cell minimizes E(x) = sum_i (n_i . (x - p_i))^2 over the crossings (p_i, n_i) on the cell's
    # edges, the squared distances from x to their tangent planes. Setting the gradient to 0 gives A x = b with
    # A = sum_i n_i n_i^T and b = sum_i n_i (n_i . p_i). A is singular where the surface is flat, so we add
    # lambda |x - c|^2, which pulls x towards the mean crossing point c: (A + lambda I) x = b + lambda c.

    # every crossing counts once in each of the four cells around its edge, so each crossing is repeated 4 times to
    # match the 4 cell ids from edge_cells; with C crossings in total, crossing_cell is (4C,) long and
    # crossing_point and crossing_normal are (4C, 3)
    cells, points, normals = [], [], []
    for a in range(3):
        # edge_start (E, 3) long: start of every edge along axis a that goes into the QEF;
        # point (E, 3): the crossing point on each of these edges, its start plus t along axis a
        edge_start = qef_edges[a].nonzero()
        point = edge_start.float()
        #print("edge_start:", edge_start)
        #print(f"point: {point}\nt[a, qef_edges[a]]: {t[a, qef_edges[a]]}")
        point[:, a] += t[a][qef_edges[a]] # "its start plus t along axis a"
                                          # like the normals statement below

        cells.append(edge_cells(edge_start, a).reshape(-1))
        points.append(point.repeat_interleave(4, dim=0))
        normals.append(normal[a][qef_edges[a]].repeat_interleave(4, dim=0))

    crossing_cell = torch.cat(cells)
    crossing_point = torch.cat(points)
    crossing_normal = torch.cat(normals)

    # each entry's share of A and b, (4C, 3, 3) and (4C, 3)
    # matrix_term = sum_i(n_i*n_i^T), i.e, = A
    #print(f"crossing_normal: {crossing_normal}")

    # AI:
    matrix_term = torch.matmul(crossing_normal.unsqueeze(-1), crossing_normal.unsqueeze(-2))
    # I understand that it gives A, but I'm not familiar with pytorch so I didn't know
    # how to use torch.unsqueeze(). Also, how is the transpose handled?
    #print(f"matrix_term: {matrix_term}")

    # AI:
    rhs_term = crossing_normal * (crossing_normal * crossing_point).sum(dim=1, keepdim=True)
    # I don't understand this term. IK it's b, but thats it.
    #print(f"rhs_term: {rhs_term}")

    # for every cell: A, matrix (G^3, 3, 3); b, rhs (G^3, 3); the number of crossings, crossing_count (G^3,);
    # their sum, point_sum (G^3, 3); and the mean crossing point c, mass_point (G^3, 3)
    matrix = torch.zeros(G ** 3, 3, 3, device="cuda").index_add(0, crossing_cell, matrix_term)
    rhs = torch.zeros(G ** 3, 3, device="cuda").index_add(0, crossing_cell, rhs_term)

    one_per_crossing = torch.ones_like(crossing_point[:, 0])
    crossing_count = torch.zeros(G ** 3, device="cuda").index_add(0, crossing_cell, one_per_crossing)
    point_sum = torch.zeros(G ** 3, 3, device="cuda").index_add(0, crossing_cell, crossing_point)
    # is mean c
    mass_point = torch.zeros(G ** 3, 3, device="cuda").index_reduce_(0, crossing_cell, crossing_point, "mean") 

    # low corner of every cell, (G^3, 3)
    coordinates = torch.arange(G, device="cuda")
    cell_corner = torch.stack(torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij"), dim=-1)
    cell_corner = cell_corner.reshape(-1, 3).float()

    #print(f"cell_corner {cell_corner}")
    # solve every cell at once; a cell whose solution leaves the cell is solved again with a 10 times stronger pull,
    # and a cell never solved inside keeps its centre; vertex_position (G^3, 3), unsolved (G^3,) bool
    vertex_position = cell_corner + 0.5
    unsolved = torch.ones(G ** 3, dtype=torch.bool, device="cuda")
    identity = torch.eye(3, device="cuda")
    regularization = LAMBDA_START
    for attempt in range(LAMBDA_ROUNDS):
        # solution (G^3, 3): the minimizer of every cell's regularized QEF, all cells solved at once
        # (see torch.linalg.solve); solution_in_cell (G^3,) bool

        #(A + lambda I) x = b + lambda c.
        # => A + regularization*Identity = b + regularization*mass_point
        # so A is (matrix + regularization*Identity) and b = (rhs + regularization*mass_point) ?
        solution = torch.linalg.solve((matrix + regularization*identity), (rhs + regularization*mass_point))
        #print(f"solution: {solution}")

        # idk
        solution_in_cell = torch.zeros(G ** 3, dtype=torch.bool, device="cuda")
        #print(f"cell_corner[:G][0]: {cell_corner[:G][0]}")
        # cell_corner < solution < cell_corner + 1
        solution_in_cell = torch.all((torch.lt(cell_corner, solution) & torch.lt(solution, torch.add(cell_corner, 1))), 1)
        #print(f"solution_in_cell: {solution_in_cell}")
        #print(f"len(solution_in_cell): {len(solution_in_cell)}")
        #print(f"len(unsolved): {len(unsolved)}")

        newly_solved = unsolved & solution_in_cell
        vertex_position[newly_solved] = solution[newly_solved]
        unsolved &= ~solution_in_cell

        regularization *= LAMBDA_GROWTH

    vertex_in_cell = (vertex_position - cell_corner).reshape(G, G, G, 3)
    ##### your code ends here #####
    return vertex_in_cell


def estimate_hermite(sdf):
    """Hermite data estimated from the SDF alone: crossing position t and unit normal on every grid edge.

    sdf     (G+1, G+1, G+1)         signed distance at every grid point
    returns
    t       (3, G+1, G+1, G+1)      crossing position on each edge (not finite where the sign does not change)
    normal  (3, G+1, G+1, G+1, 3)   unit normal at each crossing
    """
    ##### your code starts here ##### Problem 1.5
    # NOTE: use batched torch tensor operations that process all grid points, edges and cells at once, and avoid
    # Python for loops over them (a loop over the three axes is fine). Look through the torch documentation for
    # the functions that, combined, do this.

    gradient = torch.stack(torch.gradient(sdf), dim=-1)

    # SDF value and gradient at the start and at the end of every edge along x, y and z; sdf_start and sdf_end are
    # (3, G+1, G+1, G+1), gradient_start and gradient_end (3, G+1, G+1, G+1, 3); the last slice along each axis has
    # no edge and stays 0
    sdf_start = torch.zeros(3, G + 1, G + 1, G + 1, device="cuda")
    sdf_start[0, :G, :, :] = sdf[:-1, :, :]
    sdf_start[1, :, :G, :] = sdf[:, :-1, :]
    sdf_start[2, :, :, :G] = sdf[:, :, :-1]

    sdf_end = torch.zeros(3, G + 1, G + 1, G + 1, device="cuda")
    sdf_end[0, :G, :, :] = sdf[1:, :, :]
    sdf_end[1, :, :G, :] = sdf[:, 1:, :]
    sdf_end[2, :, :, :G] = sdf[:, :, 1:]

    gradient_start = torch.zeros(3, G + 1, G + 1, G + 1, 3, device="cuda")
    gradient_start[0, :G, :, :] = gradient[:-1, :, :]
    gradient_start[1, :, :G, :] = gradient[:, :-1, :]
    gradient_start[2, :, :, :G] = gradient[:, :, :-1]

    gradient_end = torch.zeros(3, G + 1, G + 1, G + 1, 3, device="cuda")
    gradient_end[0, :G, :, :] = gradient[1:, :, :]
    gradient_end[1, :, :G, :] = gradient[:, 1:, :]
    gradient_end[2, :, :, :G] = gradient[:, :, 1:]

    # not finite on edges without a sign change; those edges are masked out later
    # should be s0/s0-s1
    # -> sdf_start / sdf_start - sdf_end
    t = torch.div(sdf_start, torch.sub(sdf_start, sdf_end))
    #print("sdf_start:", sdf_start)
    #print("t:", t)
    
    # gradient_at_crossing (3, G+1, G+1, G+1, 3): the two end gradients interpolated at the crossing
    # g_t(i, j, k) = (s(i+1, j, k) - s(i-1, j, k))/2, where i, j, k define the point in t
    # how to write this?

    gradient_at_crossing = ... # needed?

    # n = ((1-t)g_0 + t*g_1)/||((1-t)g_0 + t*g_1)||
    # which is the normalization of ((1-t)*g_0 + t*g1)
    normal = torch.nn.functional.normalize(((1-t[..., None])*gradient_end + t[..., None]*gradient_start))
    ##### your code ends here #####
    return t, normal


def dual_contouring(sdf, has_hermite, t, normal):
    """Dual contouring mesh: the SDF signs decide the topology, the Hermite data the vertex positions.

    sdf          (G+1, G+1, G+1)           signed distance at every grid point
    has_hermite  (3, G+1, G+1, G+1) bool   edge has Hermite data
    t            (3, G+1, G+1, G+1)        crossing position on each edge
    normal       (3, G+1, G+1, G+1, 3)     unit normal at each crossing
    returns
    vertices     (V, 3)                    mesh vertices in grid units
    faces        (F, 3) long               vertex ids of each triangle
    """
    inside, sign_change, has_vertex = crossed_edges(sdf)

    # an edge goes into the QEF only if the signs change on it and it has Hermite data
    # (they can disagree on thin parts of a shape)
    vertex_in_cell = qef_vertices(sign_change & has_hermite, t, normal)

    faces = dual_faces(inside, sign_change, has_vertex)

    # vertex = low corner of its cell + its position inside the cell
    vertices = vertex_in_cell[has_vertex] + has_vertex.nonzero().float()
    return vertices, faces

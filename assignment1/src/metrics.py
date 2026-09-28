"""Reconstruction metrics and CUDA silhouette processing for LFD."""

from itertools import product

import torch
from kaolin.render.mesh import rasterize
from torch.nn import functional as F

DEVICE = "cuda"
CAMERA_SEED = 2
IMAGE_RESOLUTION = 256
CONTOUR_POINTS = 256


@torch.no_grad()
def iou(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return occupancy IoU averaged over the batch, as a scalar tensor.

    Inputs are boolean labels [B, N, 1] at the same query points.
    Each shape's prediction and target have a nonempty union.
    """
    # Labels are already thresholded; return a scalar tensor on their device.
    ##### your code starts here ##### Problem 1.3: IoU
    raise NotImplementedError("Problem 1.3: IoU")
    ##### your code ends here #####


@torch.no_grad()
def chamfer_distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return symmetric squared Chamfer distance averaged over the batch.

    Inputs are nonempty point sets [B, N, 3] and [B, M, 3]. Return a scalar
    using the sum of the two directional means, without a factor of 1/2.
    """
    # N and M may differ. Compare each shape's own point sets and keep the
    # returned scalar tensor on the input device.
    ##### your code starts here ##### Problem 1.3: Chamfer distance
    raise NotImplementedError("Problem 1.3: Chamfer distance")
    ##### your code ends here #####


def lfd_cameras() -> tuple[torch.Tensor, torch.Tensor]:
    """Return camera frames [100, 3, 3] and view permutations [60, 10].

    Frames are ordered by ten rig orientations, each containing ten views.
    Frame columns are the camera's right, up and viewing axes.
    Each permutation maps the rig's ten view indices under one rotation.
    """
    # A dodecahedron has eight cube corners and twelve rectangle vertices.
    golden_ratio = (1 + 5**0.5) / 2
    corners = torch.tensor(list(product([-1.0, 1.0], repeat=3)), device=DEVICE)
    rectangle = torch.tensor(
        [[0, a / golden_ratio, b * golden_ratio] for a, b in product([-1.0, 1.0], repeat=2)],
        device=DEVICE,
    )
    vertices = torch.cat([corners, rectangle, rectangle.roll(1, 1), rectangle.roll(2, 1)])
    vertices = F.normalize(vertices, dim=-1)
    # Keep one viewing direction from each antipodal vertex pair.
    positive_half = (vertices[:, 0] > 0) | ((vertices[:, 0] == 0) & (vertices[:, 1] > 0))
    views = vertices[positive_half]

    # Map a reference edge frame onto each directed edge to obtain 60 rotations.
    neighbors = torch.cdist(vertices, vertices).topk(4, largest=False).indices[:, 1:]
    edge_start = vertices.repeat_interleave(3, dim=0)
    edge_end = vertices[neighbors].reshape(-1, 3)
    radial_component = (edge_end * edge_start).sum(-1, keepdim=True) * edge_start
    edge_right = F.normalize(edge_end - radial_component, dim=-1)
    edge_up = torch.linalg.cross(edge_start, edge_right)
    frames = torch.stack([edge_right, edge_up, edge_start], dim=-1)
    rotations = frames @ frames[0].T
    rotated_views = torch.einsum("rij,vj->rvi", rotations, views)
    # Absolute dot products match view axes regardless of their sign.
    permutations = (rotated_views @ views.T).abs().argmax(dim=-1)

    # Identity plus nine seeded rotations gives ten orientations of the same rig.
    generator = torch.Generator(device=DEVICE).manual_seed(CAMERA_SEED)
    quaternions = torch.randn(10, 4, device=DEVICE, generator=generator)
    quaternions[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE)
    w, x, y, z = F.normalize(quaternions, dim=-1).unbind(-1)
    orientations = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(10, 3, 3)
    directions = torch.einsum("oij,vj->ovi", orientations, views).reshape(-1, 3)
    world_up = torch.tensor([0.0, 1.0, 0.0], device=DEVICE).expand_as(directions)
    right = F.normalize(torch.linalg.cross(world_up, directions), dim=-1)
    up = torch.linalg.cross(directions, right)
    return torch.stack([right, up, directions], dim=-1), permutations


def render_silhouettes(vertices: torch.Tensor, faces: torch.Tensor, cameras: torch.Tensor) -> torch.Tensor:
    """Render one mesh into boolean masks [1, 10 orientations, 10 views, H, W].

    vertices [V, 3], triangle faces [F, 3] and camera frames [100, 3, 3]
    are CUDA tensors. Cameras follow the ordering returned by lfd_cameras.
    """
    masks = []
    for camera_batch in cameras.split(10):
        camera_vertices = torch.einsum("vj,bjk->bvk", vertices, camera_batch)
        camera_vertices[..., 2] -= 2.0
        triangles = camera_vertices[:, faces]
        # Orthographic projection uses x/y directly; z resolves visibility.
        # Rasterization requires features, but only face coverage is used here.
        features = torch.ones((*triangles.shape[:-1], 1), device=vertices.device)
        _, face_indices = rasterize(
            IMAGE_RESOLUTION,
            IMAGE_RESOLUTION,
            triangles[..., 2].contiguous(),
            triangles[..., :2].contiguous(),
            features,
            backend="cuda",
        )
        masks.append(face_indices >= 0)
    return torch.cat(masks).reshape(1, 10, 10, IMAGE_RESOLUTION, IMAGE_RESOLUTION)


def trace_boundaries(
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trace closed pixel-boundary loops in a nonempty binary mask [H, W].

    For E boundary edges, return:
      starts [E, 2]: edge start coordinates (x, y).
      roots [E]: the smallest edge index in each edge's loop.
      areas [E]: twice the signed loop area at root indices; zero elsewhere.
      steps_to_root [E]: successor steps from each edge back to its root.
    """
    height, width = mask.shape
    start_offsets = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], device=mask.device)
    directions = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=mask.device)
    # Inspect neighbors in top/right/bottom/left order. A foreground pixel
    # contributes a boundary edge wherever its neighbor is background.
    padded = F.pad(mask, (1, 1, 1, 1))
    neighbors = torch.stack(
        [
            padded[:-2, 1:-1],
            padded[1:-1, 2:],
            padded[2:, 1:-1],
            padded[1:-1, :-2],
        ],
        dim=-1,
    )
    # Image rows increase downward; contour coordinates are (column, row).
    row, column, direction = (mask[..., None] & ~neighbors).nonzero(as_tuple=True)
    starts = torch.stack([column, row], dim=-1) + start_offsets[direction]
    ends = starts + directions[direction]
    edge_ids = torch.arange(starts.shape[0], device=mask.device)

    # Turn right at diagonal contacts to keep the two loops separate.
    # Each pixel corner has up to four outgoing edge directions.
    num_pixel_corners = (height + 1) * (width + 1)
    outgoing = torch.full((num_pixel_corners, 4), -1, device=mask.device, dtype=torch.long)
    outgoing[starts[:, 1] * (width + 1) + starts[:, 0], direction] = edge_ids
    turn_order = (direction[:, None] + torch.tensor([1, 0, 3, 2], device=mask.device)) % 4
    end_corner_ids = ends[:, 1] * (width + 1) + ends[:, 0]
    candidates = outgoing[end_corner_ids[:, None], turn_order]
    first_valid = (candidates >= 0).long().argmax(dim=1, keepdim=True)
    successor = candidates.gather(1, first_valid).squeeze(1)

    # Each jump doubles the reachable distance. Propagating the smallest ID
    # groups all edges of a closed loop without tracing them one at a time.
    roots = edge_ids.clone()
    jump = successor.clone()
    doubling_steps = (starts.shape[0] - 1).bit_length()
    for _ in range(doubling_steps):
        roots = torch.minimum(roots, roots[jump])
        jump = jump[jump]
    # Shoelace terms accumulate at each loop's root; orientation sets the sign.
    signed_area = starts[:, 0] * ends[:, 1] - ends[:, 0] * starts[:, 1]
    areas = torch.zeros_like(edge_ids).scatter_add_(0, roots, signed_area)

    # Break each cycle at its root, then count successor steps to that root.
    # These counts let join_boundaries recover the ordered contour.
    jump = successor.clone()
    steps_to_root = torch.ones_like(edge_ids)
    root_edges = edge_ids == roots
    jump[root_edges] = edge_ids[root_edges]
    steps_to_root[root_edges] = 0
    for _ in range(doubling_steps):
        steps_to_root = steps_to_root + steps_to_root[jump]
        jump = jump[jump]
    return starts, roots, areas, steps_to_root


def join_boundaries(
    starts: torch.Tensor,
    roots: torch.Tensor,
    areas: torch.Tensor,
    steps_to_root: torch.Tensor,
) -> torch.Tensor:
    """Join the exterior loops from trace_boundaries into one contour [P, 2].

    Start with the largest loop and attach the nearest remaining loop.
    Bridges are traversed both ways. Hole boundaries are excluded here;
    holes remain in the masks used for Zernike moments.
    """
    # Exterior loops have positive signed area in image coordinates.
    exterior = areas[roots] > 0
    components = roots[exterior].unique()
    component = components[areas[components].argmax()]
    selected = (roots == component).nonzero().squeeze(1)
    boundary_order = steps_to_root[selected].argsort(descending=True)
    contour = starts[selected[boundary_order]].float()
    remaining = exterior & (roots != component)

    for _ in range(components.numel() - 1):
        candidates = remaining.nonzero().squeeze(1)
        pairwise_distances = torch.cdist(contour, starts[candidates].float())
        closest_pair = pairwise_distances.argmin()
        contour_index = closest_pair // candidates.numel()
        next_edge = candidates[closest_pair % candidates.numel()]
        component = roots[next_edge]
        selected = (roots == component).nonzero().squeeze(1)
        boundary_order = steps_to_root[selected].argsort(descending=True)
        selected = selected[boundary_order]
        next_index = (selected == next_edge).long().argmax()
        next_order = (torch.arange(selected.numel(), device=starts.device) + next_index) % selected.numel()
        next_contour = starts[selected[next_order]].float()
        # Rotate both loops to start at the chosen bridge endpoints.
        order = (torch.arange(contour.shape[0], device=starts.device) + contour_index) % contour.shape[0]
        contour = contour[order]
        # Repeated endpoints close each loop; the two connecting segments
        # traverse the bridge in opposite directions when the contour closes.
        contour = torch.cat([contour, contour[:1], next_contour, next_contour[:1]])
        remaining &= roots != component
    return contour


def resample_contour(contour: torch.Tensor) -> torch.Tensor:
    """Resample a closed contour [P, 2] into [CONTOUR_POINTS, 2].

    Samples have equal arc-length spacing, including the closing segment.
    The first sample is not repeated at the end.
    """
    segments = contour.roll(-1, dims=0) - contour
    lengths = segments.norm(dim=-1)
    cumulative = F.pad(lengths.cumsum(0), (1, 0))
    spacing = cumulative[-1] / CONTOUR_POINTS
    positions = torch.arange(CONTOUR_POINTS, device=contour.device) * spacing
    # Find the segment containing each arc-length position, then interpolate.
    indices = torch.searchsorted(cumulative[1:], positions, right=True)
    fractions = (positions - cumulative[indices]) / lengths[indices]
    return contour[indices] + fractions[:, None] * segments[indices]


def silhouette_contours(masks: torch.Tensor) -> torch.Tensor:
    """Convert nonempty masks [..., H, W] to contours [..., CONTOUR_POINTS, 2].

    Each contour contains all exterior components, joined and sampled in order.
    """
    height, width = masks.shape[-2:]
    contours = []
    for mask in masks.reshape(-1, height, width):
        starts, roots, areas, steps_to_root = trace_boundaries(mask)
        contour = join_boundaries(starts, roots, areas, steps_to_root)
        contours.append(resample_contour(contour))
    return torch.stack(contours).reshape(*masks.shape[:-2], CONTOUR_POINTS, 2)


def zernike_moments(masks: torch.Tensor) -> torch.Tensor:
    """Describe masks [..., H, W] with 35 normalized Zernike magnitudes [..., 35].

    Use degrees 1 through 10 with their admissible nonnegative angular orders.
    Each mask must have nonzero foreground area and radius.
    """
    weights = masks.float()
    height, width = masks.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=masks.device, dtype=weights.dtype),
        torch.arange(width, device=masks.device, dtype=weights.dtype),
        indexing="ij",
    )
    # Express pixel positions relative to the foreground's area centroid.
    area = weights.sum(dim=(-2, -1), keepdim=True)
    center_x = (weights * x).sum(dim=(-2, -1), keepdim=True) / area
    center_y = (weights * y).sum(dim=(-2, -1), keepdim=True) / area
    radii = ((x - center_x).square() + (y - center_y).square()).sqrt()
    angles = torch.atan2(y - center_y, x - center_x)

    # The farthest foreground pixel defines the unit disk; background has no weight.
    radii = radii * weights
    radii = radii / radii.amax(dim=(-2, -1), keepdim=True)
    weights = weights / area

    # Starting the cumulative product with 1 includes 0! in the table.
    factorials = torch.arange(11, device=masks.device, dtype=weights.dtype).clamp_min(1)
    factorials = factorials.cumprod(dim=0)

    moments = []
    for degree in range(1, 11):
        # Angular order m has the same parity as degree n; build R_n^m(r).
        for order in range(degree % 2, degree + 1, 2):
            radial_polynomial = torch.zeros_like(radii)
            for term in range((degree - order) // 2 + 1):
                numerator = factorials[degree - term]
                denominator = (
                    factorials[term]
                    * factorials[(degree + order) // 2 - term]
                    * factorials[(degree - order) // 2 - term]
                )
                coefficient = numerator / denominator
                radial_polynomial += (-1) ** term * coefficient * radii.pow(degree - 2 * term)

            # Project onto R_n^m(r) * exp(-i*m*theta), retaining the magnitude.
            weighted_radial = weights * radial_polynomial
            real = (weighted_radial * torch.cos(order * angles)).sum(dim=(-2, -1))
            imaginary = -(weighted_radial * torch.sin(order * angles)).sum(dim=(-2, -1))
            magnitude = (real.square() + imaginary.square()).sqrt()
            moments.append(magnitude * (degree + 1) / torch.pi)

    return torch.stack(moments, dim=-1)


def fourier_descriptors(contours: torch.Tensor) -> torch.Tensor:
    """Describe ordered, equally spaced contours [..., P, 2] by [..., 10] features.

    Use the first ten nonconstant Fourier magnitudes of the radial signal,
    normalized by its zero-frequency (DC) magnitude. The closing endpoint is
    not duplicated.
    """
    center = contours.mean(dim=-2, keepdim=True)
    radii = (contours - center).norm(dim=-1)
    # The signal records distance to the contour mean at uniform arc-length steps.
    coefficients = torch.fft.rfft(radii, dim=-1)
    return coefficients[..., 1:11].abs() / coefficients[..., :1].abs()


@torch.no_grad()
def light_field_distance(
    predicted_masks: torch.Tensor,
    target_masks: torch.Tensor,
    view_permutations: torch.Tensor,
) -> torch.Tensor:
    """Return a scalar LFD averaged over the batch.

    Inputs are nonempty silhouette masks [B, O, 10, H, W] on CUDA, where O
    counts rig orientations, and view_permutations [60, 10] from lfd_cameras.
    Each row maps the ten views together. Descriptor coefficients remain
    floating point; this implementation does not quantize them.
    """
    predicted_contours = silhouette_contours(predicted_masks)
    target_contours = silhouette_contours(target_masks)
    # Each view contributes 35 region coefficients and 10 contour coefficients.
    predicted_features = torch.cat(
        [
            zernike_moments(predicted_masks),
            fourier_descriptors(predicted_contours),
        ],
        dim=-1,
    )
    target_features = torch.cat(
        [
            zernike_moments(target_masks),
            fourier_descriptors(target_contours),
        ],
        dim=-1,
    )

    # A permutation pairs all ten views together, preserving their rig layout.
    # Matched targets: [B, target orientation, rotation, view, coefficient].
    matched_targets = target_features[:, :, view_permutations]

    # Broadcast all pairs of predicted and target rig orientations.
    predicted_pairs = predicted_features[:, :, None, None, :, :]
    target_pairs = matched_targets[:, None, :, :, :, :]
    differences = predicted_pairs - target_pairs
    # Sum over views and coefficients for each candidate rig alignment.
    # Distances: [B, predicted orientation, target orientation, rig rotation].
    distances = differences.abs().sum(dim=(-1, -2))
    # Select an alignment per shape before averaging across the batch.
    best_matches = distances.amin(dim=(1, 2, 3))
    return best_matches.mean()

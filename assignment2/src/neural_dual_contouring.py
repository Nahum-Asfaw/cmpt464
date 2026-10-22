"""Neural dual contouring: a small 3D CNN predicts the vertex position inside every cell (student code, plain torch).

Grid conventions: see the docstring of dual_contouring.py.
"""

import torch
from data import SDF_CLIP, G, load_mesh, random_transform, sdf_grid
from dual_contouring import crossed_edges, dual_faces, exact_hermite, qef_vertices
from torch import nn
from torch.nn import functional

CHANNELS = 64
LEARNING_RATE = 1e-4
TRAIN_STEPS = 10000
LOG_EVERY = 100


class NDC(nn.Module):
    """The network of the NDC paper for SDF input; see the paper for its layers.

    The convolutions have no padding of their own: forward pads the input once, so that the network gives one output
    per grid point.
    """

    def __init__(self):
        """Create the layers of the network."""
        super().__init__()
        ##### your code starts here ##### Problem 2.1
        # the layers that forward uses, in order: conv0 to conv4, then the output layer conv_out
        self.conv0 = ...
        self.conv1 = ...
        self.conv2 = ...
        self.conv3 = ...
        self.conv4 = ...
        self.conv_out = ...
        ##### your code ends here #####

    def forward(self, sdf):
        """Predict the vertex position inside every cell from the SDF.

        sdf             (B, 1, G+1, G+1, G+1)   signed distance grid
        returns
        vertex_in_cell  (B, G, G, G, 3)         predicted vertex position inside each cell
        """
        features = sdf.clamp(-SDF_CLIP, SDF_CLIP)
        # three 3x3x3 convolutions shrink each side by 3; padding with "far outside" keeps G+1 outputs
        features = functional.pad(features, (3, 3, 3, 3, 3, 3), value=SDF_CLIP)

        for layer in (self.conv0, self.conv1, self.conv2, self.conv3, self.conv4):
            features = functional.leaky_relu(layer(features), 0.01)

        features = self.conv_out(features)

        # the output at grid point (i, j, k) belongs to the cell whose low corner it is
        vertex_in_cell = features[:, :, :G, :G, :G]

        # channels last, like every other per-cell tensor
        vertex_in_cell = vertex_in_cell.permute(0, 2, 3, 4, 1)
        return vertex_in_cell


def masked_mse(prediction, target, mask):
    """Mean squared error of the vertex position over the cells in mask (the paper's loss).

    prediction  (G, G, G, 3)    predicted vertex position inside each cell
    target      (G, G, G, 3)    QEF vertex position inside each cell
    mask        (G, G, G) bool  cells that have a vertex
    returns
    loss        scalar          sum of squared differences over the masked cells and 3 components, / (3 * mask count)
    """
    ##### your code starts here ##### Problem 2.2
    # NOTE: use batched torch tensor operations that process all grid points, edges and cells at once, and avoid
    # Python for loops over them (a loop over the three axes is fine). Look through the torch documentation for
    # the functions that, combined, do this.

    # squared_error (G, G, G): squared distance between the predicted and the target vertex of each cell
    squared_error = ...
    loss = ...
    ##### your code ends here #####
    return loss


def random_symmetry(sdf, vertex_in_cell, mask):
    """Apply one random symmetry of the grid to all three tensors: axis permutation, axis flips, inside/outside swap.

    sdf             (G+1, G+1, G+1)   signed distance grid
    vertex_in_cell  (G, G, G, 3)      vertex position inside each cell
    mask            (G, G, G) bool    cells that have a vertex
    returns         the same three tensors, transformed by the same symmetry
    """
    ##### your code starts here ##### Problem 2.3
    # NOTE: use batched torch tensor operations that process all grid points, edges and cells at once, and avoid
    # Python for loops over them (a loop over the three axes is fine). Look through the torch documentation for
    # the functions that, combined, do this.

    # permutation: the new order of the three axes
    permutation = torch.randperm(3).tolist()
    sdf = ...
    mask = ...
    # the grid axes and the three coordinates of the vertex follow the same permutation
    vertex_in_cell = ...

    for a in range(3):
        if torch.rand(1).item() < 0.5:
            # flip axis a
            sdf = ...
            mask = ...
            vertex_in_cell = ...
            vertex_in_cell[..., a] = ...

    # swap inside and outside
    if torch.rand(1).item() < 0.5:
        sdf = ...
    ##### your code ends here #####
    return sdf, vertex_in_cell, mask


def train(shapes, output_dir):
    """Train NDC on random poses of the given shapes, one shape per step; save and return the model.

    Each step puts one shape in a random pose, computes its SDF and the exact QEF vertex of every cell (the target),
    applies one random grid symmetry, and takes one optimiser step on the mean squared error over the cells that
    have a vertex.

    shapes      list of str   training shape names
    output_dir  Path          where train_log.txt and ndc.pt are written
    returns
    model       NDC           the trained network
    """
    model = NDC().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    log_path = output_dir / "train_log.txt"
    loss_sum = 0.0
    for step in range(1, TRAIN_STEPS + 1):
        name = shapes[torch.randint(len(shapes), (1,)).item()]
        vertices, faces = load_mesh(name)
        vertices = random_transform(vertices)

        with torch.no_grad():
            has_hermite, t, normal = exact_hermite(vertices, faces)
            sdf = sdf_grid(vertices, faces, has_hermite)
            inside, sign_change, has_vertex = crossed_edges(sdf)
            target = qef_vertices(sign_change & has_hermite, t, normal)

            sdf, target, mask = random_symmetry(sdf, target, has_vertex)

        prediction = model(sdf[None, None])[0]
        loss = masked_mse(prediction, target, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        if step % LOG_EVERY == 0:
            log_line = f"step {step} loss {loss_sum / LOG_EVERY:.6f}"
            print(log_line)
            with open(log_path, "a") as log_file:
                log_file.write(log_line + "\n")
            loss_sum = 0.0

    torch.save(model.state_dict(), output_dir / "ndc.pt")
    return model


def predict_mesh(model, sdf):
    """Mesh from the SDF alone: topology from the signs, vertex positions from the network.

    model     NDC                trained network
    sdf       (G+1, G+1, G+1)    signed distance grid
    returns
    vertices  (V, 3)             mesh vertices in grid units
    faces     (F, 3) long        vertex ids of each triangle
    """
    inside, sign_change, has_vertex = crossed_edges(sdf)

    with torch.no_grad():
        vertex_in_cell = model(sdf[None, None])[0]

    faces = dual_faces(inside, sign_change, has_vertex)

    # vertex = low corner of its cell + its position inside the cell
    vertices = vertex_in_cell[has_vertex] + has_vertex.nonzero().float()
    return vertices, faces

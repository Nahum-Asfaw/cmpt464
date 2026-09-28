import torch
from torch.nn import functional as F


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error between predictions and targets.

    Both inputs have shape [B, N, 1]. Return a scalar tensor.
    """
    # Keep the scalar on the input device and connected to the prediction's graph.
    ##### your code starts here ##### Problem 1.2
    raise NotImplementedError("Problem 1.2: MSE loss")
    ##### your code ends here #####


def bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean binary cross-entropy for occupancy supervision.

    Inputs are raw logits and float 0/1 labels, both [B, N, 1].
    Return a scalar tensor.
    """
    # Use the logits-based PyTorch loss from Problem 2.1, without a prior sigmoid.
    ##### your code starts here ##### Problem 2.1
    raise NotImplementedError("Problem 2.1: BCE loss")
    ##### your code ends here #####


def eikonal_loss(predicted_sdf: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Compute the Eikonal penalty as a scalar tensor.

    predicted_sdf [B, N, 1] must be computed from points [B, N, 3]
    with gradients enabled for those coordinates.
    """
    # Differentiate with respect to the coordinates. The returned penalty must
    # still support backpropagation into model parameters; see the hint in 2.3.
    ##### your code starts here ##### Problem 2.3 (Bonus)
    raise NotImplementedError("Problem 2.3: Eikonal loss")
    ##### your code ends here #####

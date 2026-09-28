import torch
from torch import nn


class IMNet(nn.Module):
    """Coordinate MLP with an optional shape code.

    hidden_dims lists the hidden-layer widths; z_dim=0 disables conditioning.
    output_format="imnet" uses the IM-NET occupancy activation.
    output_format="logit" returns raw values for BCE or SDF regression.
    """

    def __init__(
        self,
        hidden_dims: list[int],
        z_dim: int = 0,
        output_format: str = "imnet",
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.output_format = output_format

        layers = []
        # Build a flat list of modules in layers; do not nest Sequential modules.
        # The input contains three coordinates and z_dim latent features.
        # For each width in hidden_dims, use a Linear layer with bias followed
        # by LeakyReLU with negative_slope=0.02. Do not hard-code the widths.
        # Finish with a biased Linear layer producing one scalar.
        # Add no other layers; forward handles the output activation.
        ##### your code starts here ##### Problem 1.1
        raise NotImplementedError("Problem 1.1: network architecture")
        ##### your code ends here #####
        self.mlp = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """Map points [B, N, 3] to values [B, N, 1].

        Conditional models take one z [B, z_dim] per shape, shared by its N points.
        """
        inputs = points
        if self.z_dim > 0 and z is not None:
            inputs = torch.cat([points, z.unsqueeze(1).expand(-1, points.shape[1], -1)], dim=-1)

        values = self.mlp(inputs)

        if self.output_format == "imnet":
            upper_clipped = torch.minimum(values, 0.01 * values + 0.99)
            values = torch.maximum(upper_clipped, 0.01 * values)

        return values

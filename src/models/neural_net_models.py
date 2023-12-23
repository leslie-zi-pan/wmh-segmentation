import torch
from monai.networks.nets import UNet
import torch.nn.functional as F

class UNet(UNet):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        #TODO: double check dims values
        """

        x = super().forward(x)
        x = F.softmax(x)
        return x

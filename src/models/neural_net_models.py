import torch
from monai.networks.nets import UNet
import torch.nn.functional as F

class UNet(UNet):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        x = F.softmax(x)
        return x
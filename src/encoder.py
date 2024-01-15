import torch
from torch import nn
from torch.nn import functional as F
from vae_utils import VAE_Attention_Block, VAE_Residual_Block


class Encoder(nn.Sequential):
    """
    Encoder for the VAE.
    Args:
        x: (images) torch.Tensor of shape (batch_size, channels, height, width)
        noise: torch.Tensor of shape (batch_size, 4, height/8, width/8)
    Returns:
        y: (latent) torch.Tensor of shape (batch_size, 4, height/8, width/8)
    """
    def __init__(self):
        super().__init__(
            # (batch_size, channels=3, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # (batch_size, 128, height, width)
            VAE_Residual_Block(128, 128),
            # (batch_size, 128, height, width)
            VAE_Residual_Block(128, 128),
            # (batch_size, 128, height, width)

            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            # (batch_size, 128, height/2, width/2)
            VAE_Residual_Block(128, 256),
            # (batch_size, 256, height/2, width/2)
            VAE_Residual_Block(256, 256),
            # (batch_size, 256, height/2, width/2)

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # (batch_size, 256, height/4, width/4)
            VAE_Residual_Block(256, 512),
            # (batch_size, 512, height/4, width/4)
            VAE_Residual_Block(512, 512),
            # (batch_size, 512, height/4, width/4)

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            # (batch_size, 512, height/8, width/8)
            VAE_Residual_Block(256, 512),
            # (batch_size, 512, height/8, width/8)
            VAE_Residual_Block(512, 512),
            # (batch_size, 512, height/8, width/8)
            VAE_Residual_Block(512, 512),
            # (batch_size, 512, height/8, width/8)

            VAE_Attention_Block(512),
            # (batch_size, 512, height/8, width/8)

            VAE_Residual_Block(512, 512),
            # (batch_size, 512, height/8, width/8)

            nn.GroupNorm(32, 512),
            # (batch_size, 512, height/8, width/8)

            nn.SiLU(),
            # (batch_size, 512, height/8, width/8)

            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (batch_size, 8, height/8, width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
            # (batch_size, 8, height/8, width/8)
        )
    
    def forward(self, x, noise):

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))

            x = module(x)
        # (batch_size, 8, height/8, width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # (batch_size, 4, height/8, width/8)
        log_variance = torch.clamp(log_variance, min=-30., max=20.)
        # (batch_size, 4, height/8, width/8)
        variance = torch.exp(log_variance)
        # (batch_size, 4, height/8, width/8)
        stdev = variance.sqrt()
        # (batch_size, 4, height/8, width/8)

        x = mean + stdev * noise
        # (batch_size, 4, height/8, width/8)

        x *= 0.18215

        return x 
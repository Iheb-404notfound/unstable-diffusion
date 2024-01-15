import torch
from torch import nn 
from torch.nn import functional as F
from vae_utils import VAE_Attention_Block, VAE_Residual_Block

class Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (batch_size, 4, height/8, width/8)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            # (batch_size, 4, height/8, width/8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            # (batch_size, 512, height/8, width/8)
            VAE_Residual_Block(512, 512),
            # (batch_size, 512, height/8, width/8)
            VAE_Attention_Block(512),
            # (batch_size, 512, height/8, width/8)
            VAE_Residual_Block(512, 512),
            # (batch_size, 512, height/8, width/8)
            VAE_Residual_Block(512, 512),
            # (batch_size, 512, height/8, width/8)
            VAE_Residual_Block(512, 512),
            # (batch_size, 512, height/8, width/8)
            VAE_Residual_Block(512, 512),
            # (batch_size, 512, height/8, width/8)
            nn.Upsample(scale_factor=2),
            # (batch_size, 512, height/4, width/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # (batch_size, 512, height/4, width/4)
            VAE_Residual_Block(512, 512),
            # (batch_size, 512, height/4, width/4)
            VAE_Residual_Block(512, 512),
            # (batch_size, 512, height/4, width/4)
            VAE_Residual_Block(512, 512),
            # (batch_size, 512, height/4, width/4)
            nn.Upsample(scale_factor=2),
            # (batch_size, 512, height/2, width/2)
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            # (batch_size, 256, height/2, width/2)
            VAE_Residual_Block(512, 256),
            # (batch_size, 256, height/2, width/2)
            VAE_Residual_Block(256, 256),
            # (batch_size, 256, height/2, width/2)
            VAE_Residual_Block(256, 256),
            # (batch_size, 256, height/2, width/2)
            nn.Upsample(scale_factor=2),
            # (batch_size, 256, height, width)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # (batch_size, 256, height, width)
            VAE_Residual_Block(256, 128),
            # (batch_size, 128, height, width)
            VAE_Residual_Block(128, 128),
            # (batch_size, 128, height, width)
            VAE_Residual_Block(128, 128),
            # (batch_size, 128, height, width)
            nn.GroupNorm(32, 128),
            # (batch_size, 128, height, width)
            nn.SiLU(),
            # (batch_size, 128, height, width)
            nn.Conv2d(128, 3, kernel_size=1, padding=1)
            # (batch_size, 3, height, width)
        )
    
    def forward(self, x):
        # (batch_size, 4, height/8, width/8)
        x /= 0.18215

        for module in self:
            x = module(x)
        
        # (batch_size, 3, height, width)
        return x
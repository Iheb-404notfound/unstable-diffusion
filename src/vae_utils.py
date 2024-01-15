import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_Attention_Block(nn.Module):
    """
    VAE Attention Block
    Args:
        x: (convolved images) torch.Tensor of shape (batch_size, features, height, width)
        features=channels=512: int
    Returns:
        y: (image) torch.Tensor of shape (batch_size, features, height, width)
    """
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        residual = x

        # (batch_size, features, height, width)
        x = self.groupnorm(x)
        # (batch_size, features, height, width)
        n, f, h, w = x.shape
        x = x.view((n, f, h*w))
        # (batch_size, features, height*width) ~ (batch_size, sequence_length, d_model)
        x = x.transpose(-1, -2)
        # (batch_size, height*width, features)
        x = self.attention(x)
        # (batch_size, height*width, features)
        x = x.transpose(-1, -2)
        # (batch_size, features, height*width)
        x = x.view((n, f, h, w))
        # (batch_size, features, height, width)
        x = x + residual
        # (batch_size, features, height, width)
        return x



class VAE_Residual_Block(nn.Module):
    """
    VAE Residual Block
    Args:
        x: (convolved images) torch.Tensor of shape (batch_size, channels, height, width)
    Returns:
        y: (convolved images) torch.Tensor of shape (batch_size, channels, height, width)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residual = x
        # (batch_size, in_channels, height, width)
        x = self.groupnorm1(x)
        # (batch_size, in_channels, height, width)
        x = F.silu(x)
        # (batch_size, in_channels, height, width)
        x = self.conv1(x)
        # (batch_size, out_channels, height, width)
        x = self.groupnorm2(x)
        # (batch_size, out_channels, height, width)
        x = F.silu(x)
        # (batch_size, out_channels, height, width)
        x = self.conv2(x)
        # (batch_size, out_channels, height, width)
        x = x + self.residual_layer(residual)
        # (batch_size, out_channels, height, width)
        return x

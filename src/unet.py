import torch
from torch import nn 
from torch.nn import functional as F 
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.lin1 = nn.Linear(n_embed, 4 * n_embed)
        self.lin2 = nn.Linear(4 * n_embed, 4 * n_embed)
    
    def forward(self, x):
        # (1, 320)
        x = self.lin1(x)
        # (1, 1280)
        x = F.silu(x)
        # (1, 1280)
        x = self.lin2(x)
        # (1, 1280)
        return x

class UNET_Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x, time):
        # (batch_size, in_channels, height, width)
        residual = x
        # (batch_size, in_channels, height, width)
        x = self.groupnorm1(x)
        # (batch_size, in_channels, height, width)
        x = F.silu(x)
        # (batch_size, in_channels, height, width)
        x = self.conv1(x)
        # (batch_size, out_channels, height, width)

        # (1, 1280)
        time = F.silu(time)
        # (1, 1280)
        time = self.linear_time(time)
        # (1, out_channels)

        merged = x + time.unsqueeze(-1).unsqueeze(-1)
        # (batch_size, out_channels, height, width)
        merged = self.groupnorm2(merged)
        # (batch_size, out_channels, height, width)
        merged = F.silu(merged)
        # (batch_size, out_channels, height, width)
        merged = self.conv2(merged)
        # (batch_size, out_channels, height, width)
        merged = merged + self.residual_layer(residual)
        # (batch_size, out_channels, height, width)
        return merged

# TODO Continue coding the UNET
class UNET_Attention_Block(nn.Module):
    def __init__(self, n_heads, embed_dim, d_model=768):
        super().__init__()
        channels = n_heads * embed_dim

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_in = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm1 = nn.LayerNorm(channels)
        self.attention1 = SelfAttention(n_heads, channels, in_bias=False)
        self.layernorm2 = nn.LayerNorm(channels)
        self.attention2 = CrossAttention(n_heads, channels, d_model, in_bias=False)
        self.layernorm3 = nn.LayerNorm(channels)
        self.geglu1 = nn.Linear(channels, 4 * 2 * channels)
        self.geglu2 = nn.Linear(4 * channels, channels)

        self.conv_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        # (batch_size, channels, height, width)
        residual_long = x
        # (batch_size, channels, height, width)
        x = self.groupnorm(x)
        # (batch_size, channels, height, width)
        x = self.conv_in(x)
        # (batch_size, channels, height, width)

        batch_size, channels, height, width = x.shape
        # (batch_size, channels, height, width)
        x = x.view((batch_size, channels, height * width))
        # (batch_size, channels, height * width)
        x = x.transpose(-1, -2)
        # (batch_size, height * width, channels)

        residual = x
        # (batch_size, height * width, channels)
        x = self.layernorm1(x)
        # (batch_size, height * width, channels)
        x = self.attention1(x)
        # (batch_size, height * width, channels)
        x += residual
        # (batch_size, height * width, channels)
        residual = x
        # (batch_size, height * width, channels)

        x = self.layernorm2(x)
        # (batch_size, height * width, channels)
        x = self.attention2(x, context)
        # (batch_size, height * width, channels)
        x += residual
        # (batch_size, height * width, channels)
        residual = x

        x = self.layernorm3(x)
        # (batch_size, height * width, channels)
        x, gate = self.geglu1(x).chunk(2, dim=-1)
        # (batch_size, height * width, channels * 4) (batch_size, height * width, channels * 4)
        x = x * F.gelu(gate)
        # (batch_size, height * width, channels * 4)
        x = self.geglu2(x)
        # (batch_size, height * width, channels)
        x += residual
        # (batch_size, height * width, channels)
        x = x.transpose(-1, -2)
        # (batch_size, channels, height * width)
        x = x.view((batch_size, channels, height, width))
        # (batch_size, channels, height, width)
        x = self.conv_out(x) + residual_long
        # (batch_size, channels, height, width)
        return x
    
class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # (batch_size, channels, height, width)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # (batch_size, channels, height * 2, width * 2)
        x = self.conv(x)
        # (batch_size, channels, height * 2, width * 2)
        return x

class VariateSequential(nn.Sequential):
    def forward(self, x, context, time):
        for module in self:
            if isinstance(module, UNET_Residual_Block):
                x = module(x, time)
            elif isinstance(module, UNET_Attention_Block):
                x = module(x, context)
            else:
                x = module(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            # (batch_size, 4, height/8, width/8)
            VariateSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            # (batch_size, 320, height/8, width/8)
            VariateSequential(UNET_Residual_Block(320, 320), UNET_Attention_Block(8, 40)),
            # (batch_size, 320, height/8, width/8)
            VariateSequential(UNET_Residual_Block(320, 320), UNET_Attention_Block(8, 40)),
            # (batch_size, 320, height/8, width/8)
            VariateSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            # (batch_size, 320, height/16, width/16)
            VariateSequential(UNET_Residual_Block(320, 640), UNET_Attention_Block(8, 80)),
            # (batch_size, 640, height/16, width/16)
            VariateSequential(UNET_Residual_Block(640, 640), UNET_Attention_Block(8, 80)),
            # (batch_size, 640, height/16, width/16)
            VariateSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            # (batch_size, 640, height/32, width/32)
            VariateSequential(UNET_Residual_Block(640, 1280), UNET_Attention_Block(8, 160)),
            # (batch_size, 1280, height/32, width/32)
            VariateSequential(UNET_Residual_Block(1280, 1280), UNET_Attention_Block(8, 160)),
            # (batch_size, 1280, height/32, width/32)
            VariateSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            # (batch_size, 1280, height/64, width/64)
            VariateSequential(UNET_Residual_Block(1280, 1280)),
            # (batch_size, 1280, height/64, width/64)
            VariateSequential(UNET_Residual_Block(1280, 1280))
            # (batch_size, 1280, height/64, width/64)
        ])

        self.bottleneck = VariateSequential(
            # (batch_size, 1280, height/64, width/64)
            UNET_Residual_Block(1280, 1280),
            # (batch_size, 1280, height/64, width/64)
            UNET_Attention_Block(8, 160),
            # (batch_size, 1280, height/64, width/64)
            UNET_Residual_Block(1280, 1280)
            # (batch_size, 1280, height/64, width/64)
        )

        self.decoders = nn.ModuleList([
            # (batch_size, 2560, height/64, width/64)
            VariateSequential(UNET_Residual_Block(2560, 1280)),
            # (batch_size, 1280, height/64, width/64)
            VariateSequential(UNET_Residual_Block(2560, 1280)),
            # (batch_size, 1280, height/64, width/64)
            VariateSequential(UNET_Residual_Block(2560, 1280), Upsample(1280)),
            # (batch_size, 1280, height/32, width/32)
            VariateSequential(UNET_Residual_Block(2560, 1280), UNET_Attention_Block(8, 160)),
            # (batch_size, 1280, height/32, width/32)
            VariateSequential(UNET_Residual_Block(2560, 1280), UNET_Attention_Block(8, 160)),
            # (batch_size, 1280, height/32, width/32)
            VariateSequential(UNET_Residual_Block(1920, 640), UNET_Attention_Block(8, 160), Upsample(1280)),
            # (batch_size, 640, height/16, width/16)
            VariateSequential(UNET_Residual_Block(1920, 640), UNET_Attention_Block(8, 80)),
            # (batch_size, 640, height/16, width/16)
            VariateSequential(UNET_Residual_Block(1280, 640), UNET_Attention_Block(8, 80)),
            # (batch_size, 640, height/16, width/16)
            VariateSequential(UNET_Residual_Block(960, 320), UNET_Attention_Block(8, 80), Upsample(640)),
            # (batch_size, 320, height/8, width/8)
            VariateSequential(UNET_Residual_Block(960, 320), UNET_Attention_Block(8, 40)),
            # (batch_size, 320, height/8, width/8)
            VariateSequential(UNET_Residual_Block(640, 320), UNET_Attention_Block(8, 40)),
            # (batch_size, 320, height/8, width/8)
            VariateSequential(UNET_Residual_Block(640, 320), UNET_Attention_Block(8, 40))
            # (batch_size, 320, height/8, width/8)
        ])
    
    def forward(self, x, context, time):
        """
        x: (image) torch.Tensor (batch_size, 4, height/8, width/8)
        context: (prompt) torch.Tensor (batch_size, seq_len, d_model)
        time: torch.Tensor (1, 1280)
        """
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x, context, time)
            skip_connections.append(x)
        
        x = self.bottleneck(x, context, time)

        for decoder in self.decoders:
            skip = skip_connections.pop()
            x = torch.cat((x, skip), dim=1)
            x = decoder(x, context, time)
        
        return x

class UNET_Ouput_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # (batch_size, in_channels, height/8, width/8)
        x = self.groupnorm(x)
        # (batch_size, in_channels, height/8, width/8)
        x = F.silu(x)
        # (batch_size, in_channels, height/8, width/8)
        x = self.conv(x)
        # (batch_size, out_channels, height/8, width/8)
        return x


import torch
from torch import nn
from torch.nn import functional as F
from unet import UNET, UNET_Ouput_Layer, TimeEmbedding

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = TimeEmbedding(320)
        self.unet = UNET()
        self.output_layer = UNET_Ouput_Layer(320, 4)
    
    def forward(self, x, context, time):
        """
        Args:
            x: (batch_size, 4, height/4, width/4)
            context: (batch_size, seq_len, d_model)
            time: (1, 320)
        Returns:
            output: (batch_size, 4, height/8, width/8)
        """
        time = self.time_embed(time)
        # (1, 1280)
        output = self.unet(x, context, time)
        # (batch_size, 320, height/8, width/8)
        output = self.output_layer(output)
        # (batch_size, 4, height/8, width/8)
        return output

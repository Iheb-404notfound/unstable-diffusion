from clip import CLIP
from encoder import Encoder
from decoder import Decoder
from diffusion import Diffusion

import model_converting

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converting.load_from_standard_weights(ckpt_path, device)
    
    encoder = Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
        'clip': clip
    }

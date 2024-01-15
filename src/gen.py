import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LWIDTH = WIDTH // 8 # Latent width
LHEIGHT = HEIGHT // 8 # Latent height

def generate(
        prompt, # prompt to generate from
        negative_prompt=None, # stuff to avoid
        input_image=None, # image to condition on
        models = {}, # models to use (encoder, decoder...)
        tokenizer=None, # tokenizer to use
        inference_steps=50, # number of steps to run
        strength=0.8, # how much to put attention on the prompt
        classifier_free_guidance=True, # whether to use classifier-free guidance or not (with or without making and combining conditioned and unconditioned samples)
        cfg_importance=7.5, # importance of the classifier-free guidance
        seed=None, # seed to use
        device=None, # device to use
        idle_device=None # idle device to use
):
    with torch.no_grad():
        
        if idle_device is None:
            to_idle = lambda x: x
        else:
            to_idle = lambda x: x.to(idle_device)
        
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()
        
        clip = models['clip']
        clip.to(device)
        
        if classifier_free_guidance:
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # (batch_size, seq_len)
            tokens = torch.tensor(tokens, device=device, dtype=torch.long)
            # (batch_size, seq_len)
            pos_context = clip(tokens)
            # (batch_size, seq_len, embed_dim)

            neg_tokens = tokenizer.batch_encode_plus([negative_prompt], padding="max_length", max_length=77).input_ids
            # (batch_size, seq_len)
            neg_tokens = torch.tensor(neg_tokens, device=device, dtype=torch.long)
            # (batch_size, seq_len)
            neg_context = clip(neg_tokens)
            # (batch_size, seq_len, embed_dim)
            context = torch.cat([pos_context, neg_context], dim=0)
            # (2 * batch_size, seq_len, embed_dim)
        else:
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # (batch_size, seq_len)
            tokens = torch.tensor(tokens, device=device, dtype=torch.long)
            # (batch_size, seq_len)
            context = clip(tokens)
            # (batch_size, seq_len, embed_dim)
        to_idle(clip)

        sampler = DDPMSampler(generator)
        sampler.set_inference_timesteps(inference_steps)
        lshape = (1, 4, LHEIGHT, LWIDTH)

        if input_image is not None:
            encoder = models['encoder']
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            # (height, width, channels)
            input_image_tensor = np.array(input_image_tensor)
            # (height, width, channels)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            # (height, width, channels)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (height, width, channels)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (batch_size, height, width, channels)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
            # (batch_size, channels, height, width)
            encoder_noise = torch.randn(lshape, generator=generator, device=device)
            # (batch_size, channels, height, width)
            latents = encoder(input_image_tensor, encoder_noise)
            # (batch_size, channels, LHEIGHT, LWIDTH)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            latents = torch.randn(lshape, generator=generator, device=device)

        diffusion = models['diffusion']
        diffusion.to(device)
        
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)
            # (1, 320)
            model_input = latents
            # (batch_size, channels, LHEIGHT, LWIDTH)
            if classifier_free_guidance:
                model_input = model_input.repeat(2, 1, 1, 1)
                # (2 * batch_size, channels, LHEIGHT, LWIDTH)
            model_output = diffusion(model_input, context, time_embedding)
            # (batch_size, channels, LHEIGHT, LWIDTH)
            latents = sampler.step(timestep, latents, model_output)
            # (batch_size, channels, LHEIGHT, LWIDTH)

        to_idle(diffusion)

        decoder = models['decoder']
        decoder.to(device)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (batch_size, channels, height, width)
        images = images.permute(0, 2, 3, 1)
        # (batch_size, height, width, channels)
        images = images.to('cpu', torch.uint8).numpy()
        return images[0]
    

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x*=(new_max - new_min)/(old_max - old_min)
    x += new_min
    if clamp:
        x = torch.clamp(x, new_min, new_max)
    return x

def get_time_embedding(timestep):
    freqs = torch.pow(1000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (160, )
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 160)
    x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    # (1, 320)
    return x



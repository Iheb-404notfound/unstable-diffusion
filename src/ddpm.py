import torch
import numpy as np 

# TODO: Implement the DDPM Sampler
class DDPMSampler:
    def __init__(self, generator, train_steps=1000, beta_start = 85e-5, beta_end = 12e-3):
        self.betas = np.linspace(beta_start**0.5, beta_end**0.5, train_steps, dtype=np.float32)**2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator

        self.train_timesteps = train_steps

        self.timesteps = torch.arange(start=0, end=train_steps, dtype=torch.float32)[::-1]
    
    def set_inference_timesteps(self, inference_steps):
        self.inference_steps = inference_steps
        ratio = self.train_timesteps // inference_steps
        self.timesteps = torch.round(torch.arange(start=0, end=inference_steps) * ratio)[::-1].to(torch.int64)

    def _get_previous_timestep(self, timestep):
        return timestep - self.train_timesteps // self.inference_steps

    def _get_variance(self, timestep):
        prev = self._get_previous_timestep(timestep)

        alpha_prod = self.alphas_cumprod[timestep]
        alpha_prod_prev = self.alphas_cumprod[prev] if prev >= 0 else self.one
        current_beta = 1 - alpha_prod / alpha_prod_prev


        var = (1 - alpha_prod_prev) / (1 - alpha_prod) * current_beta
        var = torch.clamp(var, min=1e-20)
        return var

    def set_strength(self, strength=1):
        start_step = self.inference_steps - int(self.inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep, latents, model_output):
        t = timestep
        prev = self._get_previous_timestep(t)

        alpha_prod = self.alphas_cumprod[t]
        alpha_prod_prev = self.alphas_cumprod[prev] if prev >= 0 else self.one
        beta_prod = 1 - alpha_prod
        beta_prod_prev = 1 - alpha_prod_prev
        current_alpha = alpha_prod / alpha_prod_prev
        current_beta = 1 - current_alpha

        original = (latents - beta_prod ** (0.5) * model_output) / alpha_prod ** (0.5)

        pred_coeff = alpha_prod_prev ** (0.5) * current_beta / beta_prod
        current_coeff = current_alpha ** (0.5) * beta_prod_prev / beta_prod

        pred_prev = pred_coeff * original + current_coeff * latents

        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise
        
        pred_prev = pred_prev + variance
        return pred_prev
    
    def add_noise(self, latents, timestep):
        alphas_cumprod = self.alphas_cumprod.to(latents.device, dtype=latents.dtype)
        timesteps = timesteps.to(latents.device)

        sqrt_alphas_prod = (alphas_cumprod[timesteps] ** 0.5).flatten()
        while len(sqrt_alphas_prod.shape) < len(latents.shape):
            sqrt_alphas_prod = sqrt_alphas_prod.unsqueeze(-1)
        
        sqrt_compliment_alpha_prod = ((1 - alphas_cumprod[timesteps]) ** 0.5).flatten()
        while len(sqrt_compliment_alpha_prod.shape) < len(latents.shape):
            sqrt_compliment_alpha_prod = sqrt_compliment_alpha_prod.unsqueeze(-1)
        
        noise = torch.randn(latents.shape, generator=self.generator, device=latents.device, dtype=latents.dtype)
        noisy_latents = sqrt_alphas_prod * latents + sqrt_compliment_alpha_prod * noise
        return noisy_latents




from typing import Union, Tuple, Optional

import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from torchvision import transforms as tvt


class DDIMInversionPipeline:

    def __init__(self, model_path='stabilityai/stable-diffusion-2-1', num_steps: int = 2,
                 device='cuda' if torch.cuda.is_available() else 'cpu', batch_size: int = 8):

        self.num_steps = num_steps
        self.device = device
        self.dtype = torch.float16
        self.batch_size = batch_size
        self.inverse_scheduler = DDIMInverseScheduler.from_pretrained(model_path, subfolder='scheduler')
        self.scheduler = DDIMScheduler.from_pretrained(model_path, subfolder='scheduler')
        self.pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None,
                                                            torch_dtype=self.dtype).to(self.device)

    @torch.no_grad()
    def encode_and_diffuse(self, x: torch.Tensor):

        self.pipe.scheduler = self.inverse_scheduler

        inv_latents = []
        for i in range(x.shape[0] // self.batch_size + 1):

            x_batch = x[i*self.batch_size: (i+1)*self.batch_size]

            ### Encode ###
            x_batch = 2 * x_batch - 1  # x is now in range [-1, 1]
            latents = self.pipe.vae.encode(x_batch).latent_dist.mean

            ### Diffuse ###
            prompt = [""] * x_batch.shape[0]
            inv_latents_batch, _ = self.pipe(prompt=prompt, negative_prompt=prompt, guidance_scale=1.,
                                             width=x_batch.shape[-1], height=x_batch.shape[-2],
                                             output_type='latent', return_dict=False,
                                             num_inference_steps=self.num_steps, latents=latents)
            inv_latents.append(inv_latents_batch)

        inv_latents = torch.cat(inv_latents, dim=0)
        return inv_latents

    @torch.no_grad()
    def denoise_and_decode(self, inv_latents: torch.Tensor):

        self.pipe.scheduler = self.scheduler

        images = []
        for i in range(inv_latents.shape[0] // self.batch_size + 1):

            inv_latents_batch = inv_latents[i*self.batch_size: (i+1)*self.batch_size]

            ### Denoise and Decode ###
            prompt = [""] * inv_latents_batch.shape[0]
            images_batch = self.pipe(prompt=prompt, negative_prompt=prompt, guidance_scale=1.,
                          num_inference_steps=self.num_steps, latents=inv_latents_batch).images

            images += images_batch

        return images


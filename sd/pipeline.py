import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8


def generate(prompt: str, uncond_prompt: str, input_image=None, strength=0.8, do_cfg=True, cfg_scale=7.5, sampler_name='ddpm', n_inference_steps=50,
             models={},
             seed=None,
             device=None,
             idle_device=None,
             tokenizer=None):

    with torch.no_grad():

        if not (0 < strength <= 1):
            raise ValueError('Strength must be between 0 and 1')

        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
        
        generator = torch.Generator(device=device)

        if seed is None:
            generate.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models['clip']
        clip.to(device)

        if do_cfg:

            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids

            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)

            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

            uncond_context = clip(uncond_tokens)

            context = torch.cat([cond_context, uncond_context])

        else:

            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids

            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

            context = clip(tokens)
        
        to_idle(clip)

        if sampler_name == 'ddpm':
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)

        else:
            raise ValueError(f'Unknown sampler: {sampler_name}')

        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image is not None:
            encoder = models['encoder']
            encoder.to(device)

            input_image_tensor = input_image.reseize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            latents = encoder(input_image_tensor, encoder_noise)
            
            sampler.set_strength(strength=strength)

            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)

        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)
        
        diffusion = models['diffusion']
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)

        for t in timesteps:
            time_embedding = get_time_embedding(t).to(device)
            model_input = latents

            if do_cfg:
                model_input = model_input.reshape(2, 1, 1, 1)
            
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)

                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(t, latents, model_output)

        to_idle(diffusion)

        decoder = models['decoder']
        decoder.to(device)

        image = decoder(latents)

        to_idle(decoder)

        image = rescale(image, (-1, 1), (0, 255))
        image = image.squeeze(0)
        image = image.permute(1, 2, 0)
        image = image.cpu().numpy()

        return image

def rescale(x, src_range, dst_range, clamp=False):
    src_min, src_max = src_range
    dst_min, dst_max = dst_range
    out = dst_min + (x - src_min) * (dst_max - dst_min) / (src_max - src_min)
    if clamp:
        out = out.clamp(dst_min, dst_max)
    return out

def get_time_embedding(t):
    freq = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor(t, dtype=torch.float32)[:, None] * freq
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

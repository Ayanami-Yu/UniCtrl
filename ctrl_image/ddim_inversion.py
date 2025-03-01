import torch
import numpy as np

from tqdm import tqdm
from typing import List, Optional, Union


def encode_text(model, prompts):
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return text_encoding


def get_noise_pred(model, latent, t, context, cfg_scale):
    latents_input = torch.cat([latent] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + cfg_scale * (
        noise_prediction_text - noise_pred_uncond
    )
    return noise_pred


def next_step(
    model,
    model_output: Union[torch.FloatTensor, np.ndarray],
    timestep: int,
    sample: Union[torch.FloatTensor, np.ndarray],
):
    timestep, next_timestep = (
        min(
            timestep
            - model.scheduler.config.num_train_timesteps
            // model.scheduler.num_inference_steps,
            999,
        ),
        timestep,
    )
    alpha_prod_t = (
        model.scheduler.alphas_cumprod[timestep]
        if timestep >= 0
        else model.scheduler.final_alpha_cumprod
    )
    alpha_prod_t_next = model.scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (
        sample - beta_prod_t**0.5 * model_output
    ) / alpha_prod_t**0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
    return next_sample


@torch.no_grad()
def ddim_inversion(model, w0, prompt, cfg_scale):
    text_embedding = encode_text(model, prompt)
    uncond_embedding = encode_text(model, "")
    context = torch.cat([uncond_embedding, text_embedding])
    latent = w0.clone().detach()
    for i in tqdm(range(model.scheduler.num_inference_steps)):
        t = model.scheduler.timesteps[len(model.scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred(model, latent, t, context, cfg_scale)
        latent = next_step(model, noise_pred, t, latent)
    return latent


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(
        batch_size, model.unet.in_channels, height // 8, width // 8
    ).to(model.device)
    return latent, latents


def diffusion_step(model, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])[
            "sample"
        ]
        noise_prediction_text = model.unet(
            latents, t, encoder_hidden_states=context[1]
        )["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)[
            "sample"
        ]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    cfg_scales_tensor = torch.Tensor(guidance_scale).view(-1, 1, 1, 1).to(model.device)
    noise_pred = noise_pred_uncond + cfg_scales_tensor * (
        noise_prediction_text - noise_pred_uncond
    )
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)

    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(
            model, latents, context, t, guidance_scale, low_resource
        )

    return latents, latent

import torch
from torchvision.utils import save_image


@torch.no_grad()
def pred_clean_image(
    model, latents, noise_pred, t, generator, save=False, path=None, **extra_step_kwargs
):
    # compute the previous noisy sample x_t -> x_t-1
    intermediate_latents = model.scheduler.step(
        noise_pred, t, latents, **extra_step_kwargs, return_dict=False
    )[0]
    intermediate_image = model.vae.decode(
        intermediate_latents / model.vae.config.scaling_factor,
        return_dict=False,
        generator=generator,
    )[0]

    if save:
        save_image(intermediate_image, path)
    return intermediate_image

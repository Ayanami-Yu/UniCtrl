import torch
from tqdm import tqdm
from ctrl_image.ddim_inversion import encode_text


def sample_xts_from_x0(model, x0, num_inference_steps=50):
    """
    Sample from P(x_1:T|x_0)
    """
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5

    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros(
        (
            num_inference_steps + 1,
            model.unet.in_channels,
            model.unet.sample_size,
            model.unet.sample_size,
        )
    ).to(x0.device)
    xts[0] = x0
    for t in reversed(timesteps):
        idx = num_inference_steps - t_to_idx[int(t)]
        xts[idx] = (
            x0 * (alpha_bar[t] ** 0.5)
            + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
        )

    return xts


def forward_step(model, model_output, timestep, sample):
    next_timestep = min(
        model.scheduler.config.num_train_timesteps - 2,
        timestep
        + model.scheduler.config.num_train_timesteps
        // model.scheduler.num_inference_steps,
    )
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    beta_prod_t = 1 - alpha_prod_t

    # Compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (
        sample - beta_prod_t ** 0.5 * model_output
    ) / alpha_prod_t ** 0.5

    next_sample = model.scheduler.add_noise(
        pred_original_sample, model_output, torch.LongTensor([next_timestep])
    )
    return next_sample


def get_variance(model, timestep):
    prev_timestep = (
        timestep
        - model.scheduler.config.num_train_timesteps
        // model.scheduler.num_inference_steps
    )
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        model.scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else model.scheduler.final_alpha_cumprod
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance


def reverse_step(model, model_output, timestep, sample, eta=0, variance_noise=None):
    prev_timestep = (
        timestep
        - model.scheduler.config.num_train_timesteps
        // model.scheduler.num_inference_steps
    )
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        model.scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else model.scheduler.final_alpha_cumprod
    )
    beta_prod_t = 1 - alpha_prod_t

    # Compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (
        sample - beta_prod_t ** 0.5 * model_output
    ) / alpha_prod_t ** 0.5

    # Compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = get_variance(model, timestep)

    # Take care of asymetric reverse process (asyrp)
    model_output_direction = model_output

    # Direction pointing to x_t
    pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** 0.5 * model_output_direction

    # x_t without random noise
    prev_sample = (
        alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
    )
    # Add noise if eta > 0
    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape, device=model.device)
        sigma_z = eta * variance ** 0.5 * variance_noise
        prev_sample = prev_sample + sigma_z

    return prev_sample


def inversion_forward_process(
    model,
    x0,
    etas=None,
    prog_bar=False,
    prompt="",
    cfg_scale=3.5,
    num_inference_steps=50,
):

    if not prompt == "":
        text_embeddings = encode_text(model, prompt)
    uncond_embedding = encode_text(model, "")
    timesteps = model.scheduler.timesteps.to(model.device)
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels,
        model.unet.sample_size,
        model.unet.sample_size,
    )
    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]:
            etas = [etas] * model.scheduler.num_inference_steps
        xts = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps)
        alpha_bar = model.scheduler.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape, device=model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0

    op = tqdm(timesteps) if prog_bar else timesteps
    for t in op:
        idx = num_inference_steps - t_to_idx[int(t)] - 1
        # Predict noise residual
        if not eta_is_zero:
            xt = xts[idx + 1].unsqueeze(0)

        with torch.no_grad():
            out = model.unet.forward(
                xt, timestep=t, encoder_hidden_states=uncond_embedding
            )
            if not prompt == "":
                cond_out = model.unet.forward(
                    xt, timestep=t, encoder_hidden_states=text_embeddings
                )

        if not prompt == "":
            noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
        else:
            noise_pred = out.sample

        # Compute noisier image and set x_t -> x_t+1
        if eta_is_zero:
            xt = forward_step(model, noise_pred, t, xt)
        else:
            xtm1 = xts[idx].unsqueeze(0)
            pred_original_sample = (  # Predicted x0
                xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred
            ) / alpha_bar[t] ** 0.5

            prev_timestep = (  # Direction to xt
                t
                - model.scheduler.config.num_train_timesteps
                // model.scheduler.num_inference_steps
            )
            alpha_prod_t_prev = (
                model.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else model.scheduler.final_alpha_cumprod
            )

            variance = get_variance(model, t)
            pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance) ** (
                0.5
            ) * noise_pred

            mu_xt = (
                alpha_prod_t_prev ** 0.5 * pred_original_sample
                + pred_sample_direction
            )

            z = (xtm1 - mu_xt) / (etas[idx] * variance ** 0.5)
            zs[idx] = z

            # Correction to avoid error accumulation
            xtm1 = mu_xt + (etas[idx] * variance ** 0.5) * z
            xts[idx] = xtm1

    if not zs is None:
        zs[0] = torch.zeros_like(zs[0])

    return xt, zs, xts


def inversion_reverse_process(
    model,
    xT,
    etas=0,
    prompts="",
    cfg_scales=None,
    prog_bar=False,
    zs=None,
):

    batch_size = len(prompts)
    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1, 1, 1, 1).to(model.device)

    text_embeddings = encode_text(model, prompts)
    uncond_embedding = encode_text(model, [""] * batch_size)

    if etas is None:
        etas = 0
    if type(etas) in [int, float]:
        etas = [etas] * model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps
    timesteps = model.scheduler.timesteps.to(model.device)

    xt = xT.expand(batch_size, -1, -1, -1)
    op = tqdm(timesteps[-zs.shape[0] :]) if prog_bar else timesteps[-zs.shape[0] :]

    t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0] :])}

    for t in op:
        idx = (
            model.scheduler.num_inference_steps
            - t_to_idx[int(t)]
            - (model.scheduler.num_inference_steps - zs.shape[0] + 1)
        )
        # Unconditional embedding
        with torch.no_grad():
            uncond_out = model.unet.forward(
                xt, timestep=t, encoder_hidden_states=uncond_embedding
            )

        # Conditional embedding
        if prompts:
            with torch.no_grad():
                cond_out = model.unet.forward(
                    xt, timestep=t, encoder_hidden_states=text_embeddings
                )

        z = zs[idx] if not zs is None else None
        z = z.expand(batch_size, -1, -1, -1)
        if prompts:
            noise_pred = uncond_out.sample + cfg_scales_tensor * (
                cond_out.sample - uncond_out.sample
            )
        else:
            noise_pred = uncond_out.sample
        # Compute less noisy image and set x_t -> x_t-1
        xt = reverse_step(model, noise_pred, t, xt, eta=etas[idx], variance_noise=z)

    return xt, zs

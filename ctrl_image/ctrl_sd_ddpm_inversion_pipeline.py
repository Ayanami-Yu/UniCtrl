import torch
from typing import Any, Dict, List, Optional, Union
from diffusers import DDIMScheduler
from torch import autocast, inference_mode

from ctrl_utils.ctrl_utils import aggregate_noise_pred
from ctrl_utils.image_utils import load_512
from .ddpm_inversion import inversion_forward_process, inversion_reverse_process
from .ddim_inversion import text2image_ldm_stable
from .pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
    retrieve_timesteps,
)


class CtrlSDDDPMInversionPipeline(StableDiffusionPipeline):

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 100,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        use_plain_cfg=False,
        w_src=1.0,
        w_tgt=1.0,
        w_src_ctrl_type: str = "static",
        w_tgt_ctrl_type: str = "static",
        ctrl_mode: str = "add",
        image_path: str = None,
        do_ddpm_inversion: bool = True,
        cfg_scale_src: float = 3.5,  # TODO
        cfg_scale_tgt: float = 15.0,
        ddpm_inversion_skip: int = 36,
        **kwargs,
    ):
        assert image_path is not None, "Provide the path to the image to be edited"
        if not use_plain_cfg:
            assert (
                isinstance(prompt, list) and len(prompt) == 2
            ), "Two prompts only, one as source and one as target"

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        if do_ddpm_inversion:
            self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
        else:
            self.scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            )
        # Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # Encode image with VAE
        x0 = load_512(image_path)
        x0 = torch.from_numpy(x0).float() / 127.5 - 1
        x0 = x0.permute(2, 0, 1).unsqueeze(0).to(device)
        with autocast("cuda"), inference_mode():
            w0 = (self.vae.encode(x0).latent_dist.mode() * self.vae.config.scaling_factor).float()

        # Perform inversion
        # Find Zs and wts - forward process
        prompt_src, prompt_tgt = prompt[0], prompt[1]
        if do_ddpm_inversion:
            _, zs, wts = inversion_forward_process(
                self,
                w0,
                etas=eta,
                prompt=prompt_src,
                cfg_scale=cfg_scale_src,
                prog_bar=True,
                num_inference_steps=num_inference_steps,
            )
        else:
            wT = self.ddim_inversion(w0, prompt_src, cfg_scale_src)

        if do_ddpm_inversion:  # Reverse process (via Zs and wT)
            w0, _ = inversion_reverse_process(
                self,
                xT=wts[num_inference_steps - ddpm_inversion_skip],
                etas=eta,
                prompts=[prompt_tgt],
                cfg_scales=[cfg_scale_tgt],
                prog_bar=True,
                zs=zs[: (num_inference_steps - ddpm_inversion_skip)],
            )
        else:  # Perform DDIM Inversion
            prompts = [prompt_src, prompt_tgt]
            cfg_scale_list = [cfg_scale_src, cfg_scale_tgt]
            w0, latent = text2image_ldm_stable(
                self,
                prompts,
                num_inference_steps,
                cfg_scale_list,  # FIXME list or just one float
                latent=wT,
            )
            w0 = w0[1:2]

        # Decode image with VAE
        with autocast("cuda"), inference_mode():
            x0_dec = self.vae.decode(1 / self.vae.config.scaling_factor * w0).sample

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            None,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        # TODO Remove redundant prompt encoding
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # Predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # Perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

                    if use_plain_cfg:
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_cond - noise_pred_uncond
                        )
                    else:
                        aggregated_noise = aggregate_noise_pred(
                            noise_pred_uncond,
                            noise_pred_cond,
                            t,
                            w_src,
                            w_tgt,
                            w_src_ctrl_type,
                            w_tgt_ctrl_type,
                            ctrl_mode,
                        )

                        noise_pred = (
                            noise_pred_uncond + self.guidance_scale * aggregated_noise
                        )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_cond,
                        guidance_rescale=self.guidance_rescale,
                    )

                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]


                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(  # TODO
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload all models
        self.maybe_free_model_hooks()

        # TODO when do_direct_inversion is False return only one image
        return image

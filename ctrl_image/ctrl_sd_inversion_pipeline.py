import torch
from typing import Any, Dict, List, Optional, Union
from diffusers import DDIMScheduler

from ctrl_utils.ctrl_utils import aggregate_noise_pred
from ctrl_utils.image_utils import *
from .pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
    retrieve_timesteps,
)
from .inversion import DirectInversion


class CtrlSDInversionPipeline(StableDiffusionPipeline):

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
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
        do_direct_inversion: bool = True,  # TODO
        **kwargs,
    ):
        assert image_path is not None, "Provide the path to the image to be edited"
        if not use_plain_cfg:
            assert (
                isinstance(prompt, list) and len(prompt) == 2
            ), "Two prompts only, one as source and one as target"

        # Load image and perform Direct Inversion
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        image_gt = load_512(image_path)

        # Prepare timesteps
        # NOTE Direct Inversion only applies to DDIM scheduler
        self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # Perform inversion
        inversion = DirectInversion(model=self, num_ddim_steps=num_inference_steps)
        x_stars, noise_loss_list = inversion.invert(
            image_gt=image_gt, prompt=prompt, guidance_scale=guidance_scale
        )
        x_t = x_stars[-1]
        latents = x_t.expand(len(prompt), -1, -1, -1)

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

        self._guidance_scale = guidance_scale
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
                        # Keep the first half of noise_pred conditioned on
                        # source prompt for Direct Inversion
                        if do_direct_inversion:  # TODO
                            aggregated_noise = torch.cat(
                                (
                                    noise_pred_cond[:1] - noise_pred_uncond[:1],
                                    aggregated_noise.unsqueeze(0),
                                ),
                                dim=0,
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
                # Add loss back to source branch
                if do_direct_inversion:  # TODO
                    latents = torch.cat(
                        (latents[:1] + noise_loss_list[i][:1], latents[1:]), dim=0
                    )

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

        images = self.image_processor.postprocess(  # TODO
            image, output_type=output_type, do_denormalize=do_denormalize
        )
        image_rec, image_edit = images[0], images[1]

        # Offload all models
        self.maybe_free_model_hooks()

        return image_rec, image_edit

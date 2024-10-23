from typing import Callable, List, Optional, Union

import numpy as np
import torch
from diffusers.utils import logging

from ctrl_3d.LGM.mvdream.mv_unet import get_camera
from ctrl_3d.LGM.mvdream.pipeline_mvdream import MVDreamPipeline
from ctrl_utils.ctrl_utils import *

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CtrlMVDreamPipeline(MVDreamPipeline):

    @torch.no_grad()
    def __call__(
        self,
        prompt: str = "",
        image: Optional[np.ndarray] = None,
        height: int = 256,
        width: int = 256,
        elevation: float = 0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "numpy",  # pil, numpy, latents
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        num_frames: int = 4,
        device=torch.device("cuda:0"),
        latents: Optional[torch.Tensor] = None,
        use_plain_cfg=False,
        guidance_type: str = "static",
        w_src=1.0,
        w_tgt=1.0,
        w_src_ctrl_type: str = "static",
        w_tgt_ctrl_type: str = "static",
        t_ctrl_start: Optional[int] = None,
    ):
        if not use_plain_cfg:
            assert (
                isinstance(prompt, list) and len(prompt) == 2
            ), "Two prompts are required, one as source and one as target"

        self.unet = self.unet.to(device=device)
        self.vae = self.vae.to(device=device)
        self.text_encoder = self.text_encoder.to(device=device)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # imagedream variant
        if image is not None:
            assert isinstance(image, np.ndarray) and image.dtype == np.float32
            self.image_encoder = self.image_encoder.to(device=device)
            image_embeds_neg, image_embeds_pos = self.encode_image(
                image, device, num_images_per_prompt
            )
            image_latents_neg, image_latents_pos = self.encode_image_latents(
                image, device, num_images_per_prompt
            )

        _prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )  # type: ignore
        prompt_embeds_neg, prompt_embeds_pos = _prompt_embeds.chunk(2)

        # Prepare latent variables
        actual_num_frames = num_frames if image is None else num_frames + 1
        latents: torch.Tensor = self.prepare_latents(
            actual_num_frames * num_images_per_prompt,
            4,
            height,
            width,
            prompt_embeds_pos.dtype,
            device,
            generator,
            # None,
            latents=latents,
        )  # (4, 4, 32, 32)

        if image is not None:
            camera = get_camera(num_frames, elevation=elevation, extra_view=True).to(
                dtype=latents.dtype, device=device
            )
        else:
            camera = get_camera(num_frames, elevation=elevation, extra_view=False).to(
                dtype=latents.dtype, device=device
            )
        camera = camera.repeat_interleave(num_images_per_prompt, dim=0)

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                multiplier = 2 if do_classifier_free_guidance else 1

                # expand the latents if using prompt ctrl
                multiplier = 2 * multiplier if not use_plain_cfg else multiplier
                latent_model_input = torch.cat([latents] * multiplier)
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                unet_inputs = {
                    "x": latent_model_input,
                    "timesteps": torch.tensor(
                        [t] * actual_num_frames * multiplier,
                        dtype=latent_model_input.dtype,
                        device=device,
                    ),
                    "num_frames": actual_num_frames,
                    "camera": torch.cat([camera] * multiplier),
                }

                if image is not None:
                    unet_inputs["ip"] = torch.cat(
                        [image_embeds_neg] * actual_num_frames
                        + [image_embeds_pos] * actual_num_frames
                    )
                    unet_inputs["ip_img"] = torch.cat(
                        [image_latents_neg] + [image_latents_pos]
                    )  # no repeat

                if use_plain_cfg:
                    unet_inputs["context"] = torch.cat(
                        [prompt_embeds_neg] * actual_num_frames
                        + [prompt_embeds_pos] * actual_num_frames
                    )
                else:
                    unet_inputs["context"] = torch.cat(
                        (prompt_embeds_neg, prompt_embeds_pos), dim=0
                    ).repeat_interleave(actual_num_frames, dim=0)

                # predict the noise residual
                noise_pred = self.unet.forward(**unet_inputs)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    if use_plain_cfg:
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                    else:
                        noise_pred_uncond_src, noise_pred_uncond_tgt = (
                            noise_pred_uncond.chunk(2)
                        )
                        noise_pred_text_src, noise_pred_text_tgt = (
                            noise_pred_text.chunk(2)
                        )
                        delta_noise_pred_src = (
                            noise_pred_text_src - noise_pred_uncond_src
                        )
                        delta_noise_pred_tgt = (
                            noise_pred_text_tgt - noise_pred_uncond_tgt
                        )

                        if t_ctrl_start is not None and t > t_ctrl_start:
                            noise_pred = (
                                noise_pred_uncond_src
                                + guidance_weight(t, guidance_scale, guidance_type)
                                * delta_noise_pred_src
                            )
                        else:  # aggregate noise
                            w_src_cur = ctrl_weight(t, w_src, w_src_ctrl_type)
                            w_tgt_cur = ctrl_weight(t, w_tgt, w_tgt_ctrl_type)

                            noise_pred = noise_pred_uncond_src + guidance_weight(
                                t, guidance_scale, guidance_type
                            ) * add_aggregator_v1(
                                delta_noise_pred_src,
                                w_src_cur,
                                delta_noise_pred_tgt,
                                w_tgt_cur,
                                mode="latent",
                            )
                # compute the previous noisy sample x_t -> x_t-1
                latents: torch.Tensor = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)  # type: ignore

        # Post-processing
        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            image = self.decode_latents(latents)
            image = self.numpy_to_pil(image)
        else:  # numpy
            image = self.decode_latents(latents)  # (4, 256, 256, 3)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        return image

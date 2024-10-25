import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from torchvision.utils import save_image

from ctrl_3d.LucidDreamer.guidance.perpneg_utils import (
    weighted_perpendicular_aggregator,
)
from ctrl_3d.LucidDreamer.guidance.sd_step import pred_original
from ctrl_3d.LucidDreamer.guidance.sd_utils import (
    SpecifyGradient,
    StableDiffusion,
    rgb2sat,
)
from ctrl_utils.ctrl_utils import *


class StableDiffusionCtrl(StableDiffusion):

    # NOTE text_inverse in ISM is empty string, so no use to apply prompt ctrl here
    def add_noise_with_cfg(
        self,
        latents,
        noise,
        ind_t,
        ind_prev_t,
        text_embeddings=None,
        cfg=1.0,
        delta_t=1,
        inv_steps=1,
        is_noisy_latent=False,
        eta=0.0,
    ):

        text_embeddings = text_embeddings.to(self.precision_t)
        if cfg <= 1.0:
            uncond_text_embedding = text_embeddings.reshape(
                2, -1, text_embeddings.shape[-2], text_embeddings.shape[-1]
            )[1]

        unet = self.unet

        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            prev_noisy_lat = self.scheduler.add_noise(
                latents, noise, self.timesteps[ind_prev_t]
            )

        cur_ind_t = ind_prev_t
        cur_noisy_lat = prev_noisy_lat

        pred_scores = []

        for _ in range(inv_steps):
            # pred noise
            cur_noisy_lat_ = self.scheduler.scale_model_input(
                cur_noisy_lat, self.timesteps[cur_ind_t]
            ).to(self.precision_t)

            if cfg > 1.0:
                latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_])
                timestep_model_input = (
                    self.timesteps[cur_ind_t]
                    .reshape(1, 1)
                    .repeat(latent_model_input.shape[0], 1)
                    .reshape(-1)
                )
                unet_output = unet(
                    latent_model_input,
                    timestep_model_input,
                    encoder_hidden_states=text_embeddings,
                ).sample

                uncond, cond = torch.chunk(unet_output, chunks=2)

                unet_output = cond + cfg * (
                    uncond - cond
                )  # reverse cfg to enhance the distillation
            else:
                timestep_model_input = (
                    self.timesteps[cur_ind_t]
                    .reshape(1, 1)
                    .repeat(cur_noisy_lat_.shape[0], 1)
                    .reshape(-1)
                )
                unet_output = unet(
                    cur_noisy_lat_,
                    timestep_model_input,
                    encoder_hidden_states=uncond_text_embedding,
                ).sample

            pred_scores.append((cur_ind_t, unet_output))

            next_ind_t = min(cur_ind_t + delta_t, ind_t)
            cur_t, next_t = self.timesteps[cur_ind_t], self.timesteps[next_ind_t]
            delta_t_ = (
                next_t - cur_t
                if isinstance(self.scheduler, DDIMScheduler)
                else next_ind_t - cur_ind_t
            )

            cur_noisy_lat = self.sche_func(
                self.scheduler, unet_output, cur_t, cur_noisy_lat, -delta_t_, eta
            ).prev_sample
            cur_ind_t = next_ind_t

            del unet_output
            torch.cuda.empty_cache()

            if cur_ind_t == ind_t:
                break

        return prev_noisy_lat, cur_noisy_lat, pred_scores[::-1]

    def train_step(
        self,
        text_embeddings,
        pred_rgb,
        pred_depth=None,
        pred_alpha=None,
        grad_scale=1,
        use_control_net=False,
        save_folder: Path = None,
        iteration=0,
        warm_up_rate=0,
        resolution=(512, 512),
        guidance_opt=None,
        as_latent=False,
        embedding_inverse=None,
        weights=None,
        use_plain_cfg=False,
        guidance_type: str = "static",
        w_src=1.0,
        w_tgt=1.0,
        w_src_ctrl_type: str = "static",
        w_tgt_ctrl_type: str = "static",
        t_ctrl_start: Optional[int] = None,
        ctrl_mode: str = "add",
        removal_version: int = 1,
    ):
        """
        Params:
            text_embeddings:
                Tensor of shape (2, 2B, 77, 1024), where text_embeddings[0, ...] are uncond, text_embeddings[1, :B, ...] are src, text_embeddings[1, B:2B, ...] are tgt
            embedding_inverse:
                Tensor of shape (2, 77, 1024), used in add_noise_with_cfg
        """

        pred_rgb, pred_depth, pred_alpha = self.augmentation(
            pred_rgb, pred_depth, pred_alpha
        )

        if guidance_opt.perpneg:
            weights = torch.stack(weights, dim=1)

        B = pred_rgb.shape[0]
        if as_latent:
            latents, _ = self.encode_imgs(
                pred_depth.repeat(1, 3, 1, 1).to(self.precision_t)
            )
        else:
            latents, _ = self.encode_imgs(pred_rgb.to(self.precision_t))
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level

        if self.noise_temp is None:
            self.noise_temp = torch.randn(
                (
                    latents.shape[0],  # batch size
                    4,
                    resolution[0] // 8,
                    resolution[1] // 8,
                ),
                dtype=latents.dtype,
                device=latents.device,
                generator=self.noise_gen,
            ) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(
                latents.shape[0], 1, 1, 1
            )

        if guidance_opt.fix_noise:
            noise = self.noise_temp
        else:
            noise = torch.randn(
                (
                    latents.shape[0],
                    4,
                    resolution[0] // 8,
                    resolution[1] // 8,
                ),
                dtype=latents.dtype,
                device=latents.device,
                generator=self.noise_gen,
            ) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(
                latents.shape[0], 1, 1, 1
            )

        text_embeddings = text_embeddings.reshape(
            -1, text_embeddings.shape[-2], text_embeddings.shape[-1]
        )

        inverse_text_embeddings = (
            embedding_inverse.unsqueeze(1)
            .repeat(1, B, 1, 1)
            .reshape(-1, embedding_inverse.shape[-2], embedding_inverse.shape[-1])
        )

        if guidance_opt.annealing_intervals:
            current_delta_t = int(
                guidance_opt.delta_t
                + (warm_up_rate) * (guidance_opt.delta_t_start - guidance_opt.delta_t)
            )
        else:
            current_delta_t = guidance_opt.delta_t

        ind_t = torch.randint(
            self.min_step,
            self.max_step + int(self.warmup_step * warm_up_rate),
            (1,),
            dtype=torch.long,
            generator=self.noise_gen,
            device=self.device,
        )[0]
        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)

        t = self.timesteps[ind_t]
        prev_t = self.timesteps[ind_prev_t]

        with torch.no_grad():
            # step unroll via ddim inversion
            if not self.ism:
                prev_latents_noisy = self.scheduler.add_noise(latents, noise, prev_t)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                target = noise
            else:
                # Step 1: sample x_s with larger steps
                xs_delta_t = (
                    guidance_opt.xs_delta_t
                    if guidance_opt.xs_delta_t is not None
                    else current_delta_t
                )
                xs_inv_steps = (
                    guidance_opt.xs_inv_steps
                    if guidance_opt.xs_inv_steps is not None
                    else int(np.ceil(ind_prev_t / xs_delta_t))
                )
                starting_ind = max(
                    # ind_prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0
                    ind_prev_t - xs_delta_t * xs_inv_steps,
                    torch.zeros_like(ind_t),
                )

                _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(
                    latents,
                    noise,
                    ind_prev_t,
                    starting_ind,
                    inverse_text_embeddings,
                    guidance_opt.denoise_guidance_scale,
                    xs_delta_t,
                    xs_inv_steps,
                    eta=guidance_opt.xs_eta,
                )
                # Step 2: sample x_t
                _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(
                    prev_latents_noisy,
                    noise,
                    ind_t,
                    ind_prev_t,
                    inverse_text_embeddings,
                    guidance_opt.denoise_guidance_scale,
                    current_delta_t,
                    1,
                    is_noisy_latent=True,
                )

                pred_scores = pred_scores_xt + pred_scores_xs
                target = pred_scores[0][1]

        with torch.no_grad():
            latent_model_input = latents_noisy.repeat(
                4 if not use_plain_cfg else 2, 1, 1, 1
            )
            if guidance_opt.perpneg:
                latent_model_input = latent_model_input.repeat(2, 1, 1, 1)
            tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, tt[0]
            )
            if use_control_net:
                raise NotImplementedError("ControlNet not supported")
            else:
                unet_output = self.unet(
                    latent_model_input.to(self.precision_t),
                    tt.to(self.precision_t),
                    encoder_hidden_states=text_embeddings.to(self.precision_t),
                ).sample

            if guidance_opt.perpneg:
                noise_pred_chunks = unet_output.chunk(4)
                noise_pred_uncond, noise_pred_text = (
                    noise_pred_chunks[0],
                    noise_pred_chunks[1:],
                )
            else:
                noise_pred_uncond, noise_pred_text = unet_output.chunk(2)

            noise_pred_uncond_src, noise_pred_uncond_tgt = noise_pred_uncond.chunk(2)
            if guidance_opt.perpneg:
                noise_pred_text_src = torch.cat(
                    [x.chunk(2)[0] for x in noise_pred_text], dim=0
                )
                noise_pred_text_tgt = torch.cat(
                    [x.chunk(2)[1] for x in noise_pred_text], dim=0
                )
            else:
                noise_pred_text_src, noise_pred_text_tgt = noise_pred_text.chunk(2)

            if guidance_opt.perpneg:
                weights_src = torch.cat([w.chunk(2)[0] for w in weights], dim=0)
                weights_tgt = torch.cat([w.chunk(2)[1] for w in weights], dim=0)

                delta_noise_pred_src = weighted_perpendicular_aggregator(
                    noise_pred_text_src - noise_pred_uncond_src.repeat(3, 1, 1, 1),
                    weights_src,
                    B,
                )
                delta_noise_pred_tgt = weighted_perpendicular_aggregator(
                    noise_pred_text_tgt - noise_pred_uncond_tgt.repeat(3, 1, 1, 1),
                    weights_tgt,
                    B,
                )
            else:
                delta_noise_pred_src = noise_pred_text_src - noise_pred_uncond_src
                delta_noise_pred_tgt = noise_pred_text_tgt - noise_pred_uncond_tgt

        if use_plain_cfg:
            noise_pred = noise_pred_uncond + guidance_opt.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        elif t_ctrl_start is not None and t > t_ctrl_start:
            # use the source prompt only
            noise_pred = (
                noise_pred_uncond_src
                + guidance_weight(t, guidance_opt.guidance_scale, guidance_type)
                * delta_noise_pred_src
            )
        else:  # aggregate noise
            w_src_cur = ctrl_weight(t, w_src, w_src_ctrl_type)
            w_tgt_cur = ctrl_weight(t, w_tgt, w_tgt_ctrl_type)

            # TODO test
            if ctrl_mode == "add":
                aggregated_noise = add_aggregator_v1(
                    delta_noise_pred_src,
                    w_src_cur,
                    delta_noise_pred_tgt,
                    w_tgt_cur,
                    mode="latent",
                )
            elif ctrl_mode == "remove":
                remove_aggregator = (
                    remove_aggregator_v1
                    if removal_version == 1
                    else remove_aggregator_v2
                )
                aggregated_noise = remove_aggregator(
                    delta_noise_pred_src,
                    w_src_cur,
                    delta_noise_pred_tgt,
                    w_tgt_cur,
                    mode="latent",
                )
            else:
                raise ValueError("Unrecognized prompt ctrl mode")

            # NOTE noise_pred_uncond_src should be the same as noise_pred_uncond_tgt
            noise_pred = (
                noise_pred_uncond_src
                + guidance_weight(t, guidance_opt.guidance_scale, guidance_type)
                * aggregated_noise
            )

        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)

        grad = w(self.alphas[t]) * (noise_pred - target)

        grad = torch.nan_to_num(grad_scale * grad)
        loss = SpecifyGradient.apply(latents, grad)

        if iteration % guidance_opt.vis_interval == 0:
            # noise_pred_post = noise_pred_uncond + 7.5 * delta_DSD
            noise_pred_post = noise_pred
            lat2rgb = lambda x: torch.clip(
                (x.permute(0, 2, 3, 1) @ self.rgb_latent_factors.to(x.dtype)).permute(
                    0, 3, 1, 2
                ),
                0.0,
                1.0,
            )
            save_path_iter = os.path.join(
                save_folder, "iter_{}_step_{}.jpg".format(iteration, prev_t.item())
            )
            with torch.no_grad():
                pred_x0_latent_sp = pred_original(
                    # self.scheduler, noise_pred_uncond, prev_t, prev_latents_noisy
                    self.scheduler,
                    noise_pred_uncond_src,
                    prev_t,
                    prev_latents_noisy,
                )
                pred_x0_latent_pos = pred_original(
                    self.scheduler, noise_pred_post, prev_t, prev_latents_noisy
                )
                pred_x0_pos = self.decode_latents(
                    pred_x0_latent_pos.type(self.precision_t)
                )
                pred_x0_sp = self.decode_latents(
                    pred_x0_latent_sp.type(self.precision_t)
                )

                grad_abs = torch.abs(grad.detach())
                norm_grad = F.interpolate(
                    (grad_abs / grad_abs.max()).mean(dim=1, keepdim=True),
                    (resolution[0], resolution[1]),
                    mode="bilinear",
                    align_corners=False,
                ).repeat(1, 3, 1, 1)

                latents_rgb = F.interpolate(
                    lat2rgb(latents),
                    (resolution[0], resolution[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                latents_sp_rgb = F.interpolate(
                    lat2rgb(pred_x0_latent_sp),
                    (resolution[0], resolution[1]),
                    mode="bilinear",
                    align_corners=False,
                )

                viz_images = torch.cat(
                    [
                        pred_rgb,
                        pred_depth.repeat(1, 3, 1, 1),
                        pred_alpha.repeat(1, 3, 1, 1),
                        rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                        latents_rgb,
                        latents_sp_rgb,
                        norm_grad,
                        pred_x0_sp,
                        pred_x0_pos,
                    ],
                    dim=0,
                )
                save_image(viz_images, save_path_iter)

        return loss

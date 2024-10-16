import torch
import numpy as np

from typing import Optional
from tqdm import tqdm
from diffusers.schedulers import PNDMScheduler
from ctrl_image.base_pipeline import BasePipeline


def cfg_aggregator(noise_pred_con, noise_pred_uncon):
    return noise_pred_con - noise_pred_uncon  # (B, 4, 64, 64)


def get_perpendicular_component(x, y):
    """Get the component of x that is perpendicular to y"""
    assert x.shape == y.shape
    return x - ((torch.mul(x, y).sum()) / (torch.norm(y) ** 2)) * y


def add_aggregator_v1(delta_noise_pred_src, w_src, delta_noise_pred_tgt, w_tgt):
    return w_src * delta_noise_pred_src + w_tgt * get_perpendicular_component(
        delta_noise_pred_tgt, delta_noise_pred_src
    )


def add_aggregator_v2(delta_noise_pred_src, w_src, delta_noise_pred_tgt, w_tgt):
    return w_src * delta_noise_pred_src + w_tgt * (
        get_perpendicular_component(delta_noise_pred_tgt, delta_noise_pred_src)
        - delta_noise_pred_src
    )


def guidance_weight(t, w0, guidance_type: str, t_total=1000, clamp=4):
    if guidance_type == "static":
        w = w0
    elif guidance_type == "linear":
        w = w0 * 2 * (1 - t / t_total)
    elif guidance_type == "cosine":
        w = w0 * (np.cos(np.pi * t / t_total) + 1)
    else:
        raise ValueError("Unrecognized guidance type")
    return max(clamp, w) if clamp else w


def ctrl_weight(t, w0, ctrl_type: str, t_total=1000, clamp=None):
    if ctrl_type == "static":
        w = w0
    elif ctrl_type == "linear":
        w = w0 * 2 * (1 - t / t_total)
    else:
        raise ValueError("Unrecognized guidance type")
    return max(clamp, w) if clamp else w


class CustomPipeline(BasePipeline):

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        use_plain_cfg=False,
        guidance_type: str = "static",
        w_src=1.0,
        w_tgt=1.0,
        w_src_ctrl_type: str = "static",
        w_tgt_ctrl_type: str = "static",
        t_ctrl_start: Optional[int] = None,
        **kwds,
    ):
        DEVICE = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size
        
        if not use_plain_cfg:
            assert batch_size == 2, "Two prompts only, one as source and one as target"

        # text embeddings
        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=77, return_tensors="pt"
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        if kwds.get("dir"):
            dir = text_embeddings[-2] - text_embeddings[-1]
            u, s, v = torch.pca_lowrank(dir.transpose(-1, -2), q=1, center=True)
            text_embeddings[-1] = text_embeddings[-1] + kwds.get("dir") * v
            print(u.shape)
            print(v.shape)

        # define initial latents
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(latents_shape, device=DEVICE)
        else:
            assert (
                latents.shape == latents_shape
            ), f"The shape of input latent tensor {latents.shape} should equal to predefined one."

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.0:
            max_length = text_input.input_ids.shape[-1]
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            # uc_text = "ugly, tiling, poorly drawn hands, poorly drawn feet, body out of frame, cut off, low contrast, underexposed, distorted face"
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt",
            )
            # unconditional_input.input_ids = unconditional_input.input_ids[:, 1:]
            unconditional_embeddings = self.text_encoder(
                unconditional_input.input_ids.to(DEVICE)
            )[0]
            text_embeddings = torch.cat(  # (2B, 77, 1024)
                [unconditional_embeddings, text_embeddings], dim=0
            )

        print("latents shape: ", latents.shape)  # (B, 4, 64, 64)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if ref_intermediate_latents is not None:
                # note that the batch_size >= 2
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])

            if guidance_scale > 1.0:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat(
                    [unconditioning[i].expand(*text_embeddings.shape), text_embeddings]
                )
            # predict noise
            noise_pred = self.unet(
                model_inputs, t, encoder_hidden_states=text_embeddings
            ).sample

            # aggregate noise
            if guidance_scale > 1.0:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)

                if use_plain_cfg:
                    noise_pred = noise_pred_uncon + guidance_scale * cfg_aggregator(
                        noise_pred_con, noise_pred_uncon
                    )
                elif t_ctrl_start is not None and t > t_ctrl_start:
                    noise_pred = noise_pred_uncon + guidance_weight(t, guidance_scale, guidance_type) * cfg_aggregator(noise_pred_con[0], noise_pred_uncon[0])
                else:
                    delta_noise_pred_src = noise_pred_con[0] - noise_pred_uncon[0]
                    delta_noise_pred_tgt = noise_pred_con[1] - noise_pred_uncon[1]

                    w_src_cur = ctrl_weight(t, w_src, w_src_ctrl_type)
                    w_tgt_cur = ctrl_weight(t, w_tgt, w_tgt_ctrl_type)

                    noise_pred = noise_pred_uncon + guidance_weight(
                        t, guidance_scale, guidance_type
                    ) * add_aggregator_v2(
                        delta_noise_pred_src, w_src_cur, delta_noise_pred_tgt, w_tgt_cur
                    )

            # compute the previous noise sample x_t -> x_t-1
            if isinstance(self.scheduler, PNDMScheduler):
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            else:
                latents, _ = self.step(noise_pred, t, latents)
            latents_list.append(latents)

        # if noises have been aggregated then they are the same
        image = self.latent2image(latents, return_type="pt")
        image = image[0] if torch.all(image[0] == image[1]) else image
        if return_intermediates:
            latents_list = [
                self.latent2image(img, return_type="pt") for img in latents_list
            ]
            return image, latents_list

        return image

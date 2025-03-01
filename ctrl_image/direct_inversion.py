import torch
from ctrl_utils.image_utils import image2latent, latent2image


class DirectInversion:

    def __init__(self, model, num_ddim_steps):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.prompt = None
        self.context = None
        self.num_ddim_steps = num_ddim_steps

    @property
    def scheduler(self):
        return self.model.scheduler

    def prev_step(self, model_output, timestep: int, sample):
        prev_timestep = (
            timestep
            - self.scheduler.config.num_train_timesteps
            // self.scheduler.num_inference_steps
        )
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (
            sample - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = (
            alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        )

        difference_scale_pred_original_sample = -(beta_prod_t**0.5) / alpha_prod_t**0.5
        difference_scale_pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5
        difference_scale = (
            alpha_prod_t_prev**0.5 * difference_scale_pred_original_sample
            + difference_scale_pred_sample_direction
        )

        return prev_sample, difference_scale

    def next_step(self, model_output, timestep: int, sample):
        timestep, next_timestep = (
            min(
                timestep
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps,
                999,
            ),
            timestep,
        )
        alpha_prod_t = (
            self.scheduler.alphas_cumprod[timestep]
            if timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (
            sample - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = (
            alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
        )
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)[
            "sample"
        ]
        return noise_pred

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""] * len(prompt),
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.model.text_encoder(
            uncond_input.input_ids.to(self.model.device)
        )[0]
        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(
            text_input.input_ids.to(self.model.device)
        )[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        _, cond_embeddings = self.context.chunk(2)
        cond_embeddings = cond_embeddings[[0]]  # Source prompt
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):  # Timestep increases
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @torch.no_grad()
    def ddim_inversion(self, image):
        """
        Perform DDIM Inversion using only source prompt
        """
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def offset_calculate(self, latents, guidance_scale):
        noise_loss_list = []
        latent_cur = torch.cat([latents[-1]] * (self.context.shape[0] // 2))
        # As i increases latents[i] gets noisier
        for i in range(self.num_ddim_steps):
            latent_prev = torch.cat(
                [latents[len(latents) - i - 2]] * latent_cur.shape[0]
            )
            t = self.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(
                    torch.cat([latent_cur] * 2), t, self.context
                )
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                latents_prev_rec, _ = self.prev_step(
                    noise_pred_w_guidance, t, latent_cur
                )
                # NOTE Only loss[:1] which corresponds to source prompt will be used
                loss = latent_prev - latents_prev_rec

            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss

        return noise_loss_list

    def invert(
        self,
        image_gt,
        prompt,
        guidance_scale,
    ):
        self.init_prompt(prompt)
        _, ddim_latents = self.ddim_inversion(image_gt)

        noise_loss_list = self.offset_calculate(
            ddim_latents,
            guidance_scale,
        )
        return ddim_latents, noise_loss_list

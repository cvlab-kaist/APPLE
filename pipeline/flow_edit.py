from typing import Any, Callable

import numpy as np
import torch
from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
    calculate_shift,
    retrieve_timesteps,
)
from PIL import Image

from pipeline.common import RFEditingFluxPipeline


class FlowEditFluxPipeline(RFEditingFluxPipeline):
    @torch.inference_mode()
    def __call__(
        self,
        source_img: str | Image.Image,
        source_prompt: str | list[str],
        target_prompt: str | list[str],
        source_prompt_2: str | list[str] | None = None,
        target_prompt_2: str | list[str] | None = None,
        num_inference_steps: int = 28,
        num_average_steps: int = 1,
        source_guidance_scale: float = 1.5,
        target_guidance_scale: float = 3.5,
        interpolate_start_step: int = 0,
        interpolate_end_step: int = 24,
        num_images_per_prompt: int | None = 1,
        latents: torch.FloatTensor | None = None,
        source_prompt_embeds: torch.FloatTensor | None = None,
        source_pooled_prompt_embeds: torch.FloatTensor | None = None,
        target_prompt_embeds: torch.FloatTensor | None = None,
        target_pooled_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 512,
        joint_attention_kwargs: dict[str, Any] | None = None,
        height: int | None = None,
        width: int | None = None,
        generator: torch.Generator | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
        if joint_attention_kwargs is None:
            joint_attention_kwargs = {}

        device = self._execution_device
        (
            source_prompt_embeds,
            source_pooled_prompt_embeds,
            source_text_ids,
        ) = self.encode_prompt(
            prompt=source_prompt,
            prompt_2=source_prompt_2,
            prompt_embeds=source_prompt_embeds,
            pooled_prompt_embeds=source_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        target_size = (width, height) if height is not None and width is not None else None
        source_img_latents, source_latent_image_ids, height, width, ori_height, ori_width = self.encode_img(
            source_img, source_prompt_embeds.dtype, target_size=target_size
        )

        (
            target_prompt_embeds,
            target_pooled_prompt_embeds,
            target_text_ids,
        ) = self.encode_prompt(
            prompt=target_prompt,
            prompt_2=target_prompt_2,
            prompt_embeds=target_prompt_embeds,
            pooled_prompt_embeds=target_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        image_seq_len = source_img_latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        dtype = source_img_latents.dtype

        if self.transformer.config.guidance_embeds:
            source_guidance = torch.full([1], source_guidance_scale, device=device, dtype=torch.float32)
            target_guidance = torch.full([1], target_guidance_scale, device=device, dtype=torch.float32)
            source_guidance = source_guidance.expand(source_img_latents.shape[0])
            target_guidance = target_guidance.expand(source_img_latents.shape[0])
        else:
            source_guidance = None
            target_guidance = None

        source_img_latents_edit = source_img_latents.clone()
        with self.progress_bar(total=interpolate_end_step) as progress_bar:
            for step_idx, t_curr in enumerate(timesteps):
                if num_inference_steps - step_idx > interpolate_end_step:
                    continue
                sigma_curr = t_curr / self.scheduler.config.num_train_timesteps

                if num_inference_steps - step_idx > interpolate_start_step:
                    delta_avg = torch.zeros_like(source_img_latents)
                    for _ in range(num_average_steps):
                        forward_noise = torch.randn(
                            source_img_latents.shape,
                            dtype=dtype,
                            device=device,
                            layout=source_img_latents.layout,
                            generator=generator,
                        )
                        source_img_latents_noisy = (
                            1 - sigma_curr
                        ) * source_img_latents + sigma_curr * forward_noise
                        target_img_latents_noisy = (
                            source_img_latents_edit + source_img_latents_noisy - source_img_latents
                        )
                        noise_pred_source = self.transformer(
                            hidden_states=source_img_latents_noisy,
                            timestep=sigma_curr.expand(source_img_latents.shape[0]).to(dtype),
                            guidance=source_guidance,
                            pooled_projections=source_pooled_prompt_embeds,
                            encoder_hidden_states=source_prompt_embeds,
                            txt_ids=source_text_ids,
                            img_ids=source_latent_image_ids,
                            joint_attention_kwargs=joint_attention_kwargs,
                            return_dict=False,
                        )[0].float()

                        noise_pred_target = self.transformer(
                            hidden_states=target_img_latents_noisy,
                            timestep=sigma_curr.expand(target_img_latents_noisy.shape[0]).to(dtype),
                            guidance=target_guidance,
                            pooled_projections=target_pooled_prompt_embeds,
                            encoder_hidden_states=target_prompt_embeds,
                            txt_ids=target_text_ids,
                            img_ids=source_latent_image_ids,
                            joint_attention_kwargs=joint_attention_kwargs,
                            return_dict=False,
                        )[0].float()

                        delta_avg += (1 / num_average_steps) * (noise_pred_target - noise_pred_source)

                    source_img_latents_edit = self.scheduler.step(
                        delta_avg, t_curr, source_img_latents_edit, return_dict=False
                    )[0]
                    source_img_latents_edit = source_img_latents_edit.to(dtype)
                else:
                    if step_idx == num_inference_steps - interpolate_start_step:
                        forward_noise = torch.randn(
                            source_img_latents.shape,
                            dtype=dtype,
                            device=device,
                            layout=source_img_latents.layout,
                            generator=generator,
                        )
                        source_img_latents_noisy = (
                            1 - sigma_curr
                        ) * source_img_latents + sigma_curr * forward_noise
                        target_img_latents_noisy = (
                            source_img_latents_edit + source_img_latents_noisy - source_img_latents
                        )

                    noise_pred_target = self.transformer(
                        hidden_states=target_img_latents_noisy,
                        timestep=sigma_curr.expand(target_img_latents_noisy.shape[0]).to(dtype),
                        guidance=target_guidance,
                        pooled_projections=target_pooled_prompt_embeds,
                        encoder_hidden_states=target_prompt_embeds,
                        txt_ids=target_text_ids,
                        img_ids=source_latent_image_ids,
                        joint_attention_kwargs=joint_attention_kwargs,
                        return_dict=False,
                    )[0].float()
                    target_img_latents_noisy = self.scheduler.step(
                        noise_pred_target, t_curr, target_img_latents_noisy, return_dict=False
                    )[0]
                    target_img_latents_noisy = target_img_latents_noisy.to(dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    local_vars = locals()
                    for k in callback_on_step_end_tensor_inputs:
                        if k not in local_vars:
                            continue
                        callback_kwargs[k] = local_vars[k]
                    callback_on_step_end(self, step_idx, t_curr, callback_kwargs)

                # call the callback, if provided
                if step_idx == len(timesteps) - 2 or (
                    (step_idx + 1) > num_warmup_steps and (step_idx + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        latents = source_img_latents_edit if interpolate_start_step == 0 else target_img_latents_noisy
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
            if target_size is not None:
                image = [
                    self.image_processor.resize(img, height=ori_height, width=ori_width) for img in image
                ]

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

    def multiturn(
        self,
        source_img: np.ndarray | torch.Tensor,
        source_prompt: str | list[str],
        prompt_sequence: list[str | list[str]],
        num_inference_steps: int = 8,
        source_guidance_scale: float = 1.5,
        target_guidance_scale: float = 3.5,
        interpolate_start_step: int = 0,
        interpolate_end_step: int = 24,
        joint_attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int = 512,
        num_images_per_prompt: int | None = 1,
        height: int | None = None,
        width: int | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
        generate_images = []
        for target_prompt in prompt_sequence:
            image = self(
                source_img,
                source_prompt,
                target_prompt,
                num_inference_steps=num_inference_steps,
                source_guidance_scale=source_guidance_scale,
                target_guidance_scale=target_guidance_scale,
                joint_attention_kwargs=joint_attention_kwargs,
                max_sequence_length=max_sequence_length,
                interpolate_start_step=interpolate_start_step,
                interpolate_end_step=interpolate_end_step,
                num_images_per_prompt=num_images_per_prompt,
                height=height,
                width=width,
                output_type=output_type,
                return_dict=return_dict,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            ).images[0]
            source_prompt = target_prompt
            source_img = image
            generate_images.append(image)
        return generate_images

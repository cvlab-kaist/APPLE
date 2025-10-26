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


class DNAEditFluxPipeline(RFEditingFluxPipeline):
    def denoise(
        self,
        latents: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        guidance_scale: float,
        num_inference_steps: int,
        start_timestep: int,
        device: torch.device,
        joint_attention_kwargs: dict[str, Any] | None = None,
        mvg: float = 0.85,
        dx_list: list[torch.Tensor] | None = None,
        velocity_list: list[torch.Tensor] | None = None,
        original_latents: torch.Tensor | None = None,
        inverse: bool = False,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
        if joint_attention_kwargs is None:
            joint_attention_kwargs = {}
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            num_inference_steps + 1,
            device,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - 1 - num_inference_steps * self.scheduler.order, 0)

        if inverse:
            timesteps = torch.flip(timesteps, [0])
        dtype = latents.dtype
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if inverse:
            dx_list = []
            velocity_list = []

        random_noise = torch.randn_like(latents)
        with self.progress_bar(
            total=num_inference_steps - start_timestep if inverse else num_inference_steps
        ) as progress_bar:
            for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:], strict=True)):
                if inverse and num_inference_steps - step_idx == start_timestep:
                    break
                elif not inverse and step_idx < start_timestep:
                    progress_bar.update()
                    continue

                sigma_curr = t_curr / self.scheduler.config.num_train_timesteps
                sigma_prev = t_prev / self.scheduler.config.num_train_timesteps

                if inverse:
                    input_latents = (sigma_prev - sigma_curr) / (1 - sigma_curr) * (
                        random_noise - latents
                    ) + latents
                else:
                    input_latents = latents + dx_list[step_idx - start_timestep].to(dtype)

                noise_pred = self.transformer(
                    hidden_states=input_latents,
                    timestep=(sigma_prev if inverse else sigma_curr).expand(input_latents.shape[0]).to(dtype),
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0].float()

                if inverse:
                    # input_latents: latents_prev, latents: latents_curr
                    delta_v = (input_latents - latents) / (sigma_prev - sigma_curr) - noise_pred
                    latents = input_latents - delta_v * (sigma_prev - sigma_curr)
                    dx = delta_v * (sigma_prev - sigma_curr)
                    dx_list.append(dx)
                    velocity_list.append(noise_pred)
                else:
                    original_latents = original_latents + (sigma_prev - sigma_curr) * (
                        noise_pred - velocity_list[step_idx - start_timestep]
                    )
                    velocity = noise_pred * mvg + (latents - original_latents) / sigma_curr * (1 - mvg)
                    latents = latents + velocity * (sigma_prev - sigma_curr)
                latents = latents.to(dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    local_vars = locals()
                    for k in callback_on_step_end_tensor_inputs:
                        if k not in local_vars:
                            continue
                        callback_kwargs[k] = local_vars[k]
                    callback_on_step_end(self, step_idx, t_prev, callback_kwargs)

                # call the callback, if provided
                if step_idx == len(timesteps) - 2 or (
                    (step_idx + 1) > num_warmup_steps and (step_idx + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        if inverse:
            dx_list = dx_list[::-1]
            velocity_list = velocity_list[::-1]
            return latents, dx_list, velocity_list

        return latents

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
        source_guidance_scale: float = 1,
        target_guidance_scale: float = 2.5,
        start_timestep: int = 4,
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

        inverse_latents, dx_list, velocity_list = self.denoise(
            latents=source_img_latents,
            pooled_prompt_embeds=source_pooled_prompt_embeds,
            prompt_embeds=source_prompt_embeds,
            text_ids=source_text_ids,
            latent_image_ids=source_latent_image_ids,
            guidance_scale=source_guidance_scale,
            num_inference_steps=num_inference_steps,
            start_timestep=start_timestep,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=True,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        latents = self.denoise(
            latents=inverse_latents,
            pooled_prompt_embeds=target_pooled_prompt_embeds,
            prompt_embeds=target_prompt_embeds,
            text_ids=target_text_ids,
            latent_image_ids=source_latent_image_ids,
            guidance_scale=target_guidance_scale,
            num_inference_steps=num_inference_steps,
            start_timestep=start_timestep,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            dx_list=dx_list,
            velocity_list=velocity_list,
            original_latents=source_img_latents,
            inverse=False,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        del dx_list, velocity_list

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

    @torch.inference_mode()
    def reconstruction(
        self,
        source_img: str | Image.Image,
        source_prompt: str | list[str],
        source_prompt_2: str | list[str] | None = None,
        source_prompt_embeds: torch.FloatTensor | None = None,
        source_pooled_prompt_embeds: torch.FloatTensor | None = None,
        guidance_scale: float = 2,
        num_inference_steps: int = 28,
        start_timestep: int = 4,
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
        inverse_latents, dx_list, velocity_list = self.denoise(
            latents=source_img_latents,
            pooled_prompt_embeds=source_pooled_prompt_embeds,
            prompt_embeds=source_prompt_embeds,
            text_ids=source_text_ids,
            latent_image_ids=source_latent_image_ids,
            guidance_scale=1,
            start_timestep=start_timestep,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=True,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        latents = self.denoise(
            latents=inverse_latents,
            pooled_prompt_embeds=source_pooled_prompt_embeds,
            prompt_embeds=source_prompt_embeds,
            text_ids=source_text_ids,
            latent_image_ids=source_latent_image_ids,
            guidance_scale=guidance_scale,
            start_timestep=start_timestep,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            dx_list=dx_list,
            velocity_list=velocity_list,
            original_latents=source_img_latents,
            inverse=False,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        del dx_list, velocity_list

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
        num_inference_steps: int = 28,
        source_guidance_scale: float = 1,
        target_guidance_scale: float = 2.5,
        start_timestep: int = 4,
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
                start_timestep=start_timestep,
                joint_attention_kwargs=joint_attention_kwargs,
                max_sequence_length=max_sequence_length,
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

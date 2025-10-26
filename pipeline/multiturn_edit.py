import os
from typing import Any, Callable

import numpy as np
import torch
from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
    calculate_shift,
    retrieve_timesteps,
)

from pipeline.common import RFEditingFluxPipeline


class MultiTurnEditFluxPipeline(RFEditingFluxPipeline):
    def denoise(
        self,
        latents: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        guidance_scale: float,
        inject_step: int,
        num_inference_steps: int,
        device: torch.device,
        original_latents: torch.Tensor | None = None,
        prev_latents: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        with_second_order: bool = False,
        inverse: bool = False,
        attn_guidance_start_block: int = -1,
        stop_timestep: float = 0.25,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
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
        using_inject_list = [True] * inject_step + [False] * (len(timesteps[:-1]) - inject_step)
        gamma_steps = int(stop_timestep * len(timesteps[:-1]))
        gamma = [0.9] * gamma_steps + [0] * (len(timesteps[:-1]) - gamma_steps)
        if inverse:
            timesteps = torch.flip(timesteps, [0])
            using_inject_list = using_inject_list[::-1]
            gamma = [0.5] * len(timesteps[:-1])

        dtype = latents.dtype

        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        random_noise = torch.randn_like(latents)
        next_step_velocity = None
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:], strict=True)):
                sigma_curr = t_curr / self.scheduler.config.num_train_timesteps
                sigma_prev = t_prev / self.scheduler.config.num_train_timesteps

                joint_attention_kwargs["inverse"] = inverse
                joint_attention_kwargs["editing_strategy"] = "attn_guidance"
                joint_attention_kwargs["attn_guidance_start_block"] = attn_guidance_start_block
                joint_attention_kwargs["second_order"] = False
                joint_attention_kwargs["inject"] = using_inject_list[step_idx]
                joint_attention_kwargs["timestep"] = (
                    step_idx  # we directly borrow the `timestep` for control the current index
                )

                if next_step_velocity is None:
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=sigma_curr.expand(latents.shape[0]).to(dtype),
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=joint_attention_kwargs,
                        return_dict=False,
                    )[0].float()
                else:
                    noise_pred = next_step_velocity

                if not with_second_order:
                    latents = latents + (sigma_prev - sigma_curr) * noise_pred
                    latents = latents.to(dtype)
                else:
                    mid_sample = latents + (sigma_prev - sigma_curr) / 2 * noise_pred
                    mid_sample = mid_sample.to(dtype)

                    sigma_mid = torch.full(
                        (mid_sample.shape[0],),
                        (sigma_curr + (sigma_prev - sigma_curr) / 2),
                        dtype=mid_sample.dtype,
                        device=mid_sample.device,
                    )
                    joint_attention_kwargs["second_order"] = True
                    mid_noise_pred = self.transformer(
                        hidden_states=mid_sample,
                        timestep=sigma_mid.expand(latents.shape[0]).to(latents.dtype),
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=joint_attention_kwargs,
                        return_dict=False,
                    )[0].float()
                    next_step_velocity = mid_noise_pred

                    uncond_vector_field = mid_noise_pred if inverse else -mid_noise_pred
                    if inverse:
                        cond_vector_field = (random_noise - latents) / (1 - sigma_curr)
                    else:
                        time_norm = step_idx / len(timesteps[:-1])
                        if prev_latents is None:
                            cond_vector_field = (original_latents - latents) / (1 - time_norm)
                        else:
                            cond_vector_field = (original_latents - latents) / (1 - time_norm) + 0.7 * (
                                (prev_latents - latents) / (1 - time_norm)
                                - (original_latents - latents) / (1 - time_norm)
                            )

                    cond_vector_field = uncond_vector_field + gamma[step_idx] * (
                        cond_vector_field - uncond_vector_field
                    )
                    latents = latents + torch.abs(sigma_prev - sigma_curr) * cond_vector_field
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

        return latents

    @torch.inference_mode()
    def __call__(
        self,
        source_img: np.ndarray | torch.Tensor,
        source_prompt: str | list[str],
        target_prompt: str | list[str],
        original_img: np.ndarray | torch.Tensor | None = None,
        source_as_prev_img: bool = False,
        source_prompt_2: str | list[str] | None = None,
        target_prompt_2: str | list[str] | None = None,
        inject_step: int = 0,
        attn_guidance_start_block: int = 11,
        num_inference_steps: int = 15,
        guidance_scale: float = 3.5,
        stop_timestep: float = 0.25,
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
        with_second_order: bool = True,
        clear_memory: bool = True,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
        if not self.initialize_processor:
            raise ValueError("Please call `add_processor` before running the pipeline.")

        os.makedirs("heatmap/average_heatmaps", exist_ok=True)

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
        original_latents, *_ = self.encode_img(
            original_img, source_prompt_embeds.dtype, target_size=target_size
        )
        prev_img_latents = source_img_latents.clone() if source_as_prev_img else None

        inverse_latents = self.denoise(
            latents=source_img_latents,
            pooled_prompt_embeds=source_pooled_prompt_embeds,
            prompt_embeds=source_prompt_embeds,
            text_ids=source_text_ids,
            latent_image_ids=source_latent_image_ids,
            guidance_scale=1,
            inject_step=inject_step,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=True,
            with_second_order=with_second_order,
            stop_timestep=stop_timestep,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
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

        latents = self.denoise(
            latents=inverse_latents,
            pooled_prompt_embeds=target_pooled_prompt_embeds,
            prompt_embeds=target_prompt_embeds,
            text_ids=target_text_ids,
            latent_image_ids=source_latent_image_ids,  # reuse the same image ids
            guidance_scale=guidance_scale,
            inject_step=inject_step,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=False,
            attn_guidance_start_block=attn_guidance_start_block,
            with_second_order=with_second_order,
            stop_timestep=stop_timestep,
            original_latents=original_latents,
            prev_latents=prev_img_latents,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )
        if clear_memory:
            for processor in self.processors.values():
                if hasattr(processor, "clear_memory"):
                    processor.clear_memory()

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
        inject_step: int = 0,
        attn_guidance_start_block: int = 11,
        num_inference_steps: int = 15,
        guidance_scale: float = 3.5,
        stop_timestep: float = 0.25,
        output_type: str | None = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 512,
        joint_attention_kwargs: dict[str, Any] | None = None,
        height: int | None = None,
        width: int | None = None,
        with_second_order: bool = True,
        clear_memory: bool = True,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ):
        generate_images = []
        original_img = source_img
        for prompt_idx, target_prompt in enumerate(prompt_sequence):
            image = self(
                source_img,
                source_prompt,
                target_prompt,
                inject_step=inject_step,
                guidance_scale=guidance_scale,
                attn_guidance_start_block=attn_guidance_start_block,
                num_inference_steps=num_inference_steps,
                stop_timestep=stop_timestep,
                with_second_order=with_second_order,
                height=height,
                width=width,
                output_type=output_type,
                return_dict=return_dict,
                max_sequence_length=max_sequence_length,
                joint_attention_kwargs=joint_attention_kwargs,
                original_img=original_img,
                source_as_prev_img=prompt_idx != 0,
                clear_memory=clear_memory,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            ).images[0]
            source_prompt = target_prompt
            source_img = image
            generate_images.append(image)
        return generate_images

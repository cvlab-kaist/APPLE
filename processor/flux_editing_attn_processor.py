import math
from typing import Literal

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from PIL import Image
from torchvision.utils import save_image


def auto_mask(
    load_list: list[str],
    mask_accumulator: torch.FloatTensor,
    thresh: float,
    attn_guidance_start_block: int,
    mask_num: int = 4,
):
    mask_list = []
    for img_path in load_list:
        load_mask_img = Image.open(img_path).convert("L")
        transform = transforms.PILToTensor()
        mask_tensor = transform(load_mask_img)
        mask_tensor = mask_tensor.to(device=mask_accumulator.device, dtype=mask_accumulator.dtype)
        mask_tensor /= 255.0
        mask_list.append(mask_tensor)

    # Sort masks based on their activation levels
    mask_list.sort(key=lambda x: x.sum().item(), reverse=True)
    # Select the 5 medium activated masks
    num_masks = len(mask_list)
    if num_masks > mask_num:
        # selected_masks = mask_list[num_masks//2 - mask_num : num_masks//2]
        attn_guidance_end_block = attn_guidance_start_block + mask_num
        if attn_guidance_end_block > num_masks - 1:
            selected_masks = mask_list[-mask_num:]
        else:
            selected_masks = mask_list[attn_guidance_start_block:attn_guidance_end_block]
    else:
        selected_masks = mask_list

    # Accumulate the selected masks
    for mask in selected_masks:
        mask_accumulator += mask

    mask_tensor = (mask_accumulator / len(selected_masks)).to(
        dtype=mask_accumulator.dtype
    )  # Average the masks and convert back to original dtype
    mask_tensor[mask_tensor >= thresh] = 1
    mask_tensor[mask_tensor < thresh] = 0

    return mask_tensor


def adaptive_attention(
    query: torch.FloatTensor,
    key: torch.FloatTensor,
    value: torch.FloatTensor,
    txt_shape: int,
    img_shape: int,
    cur_step: int,
    cur_block: int,
    attn_guidance_start_block: int,
    layer: list[int] = list(range(19)),  # noqa: B008 B006
    is_causal: bool = False,
    token_index: int = 2,
    attn_mask: torch.FloatTensor | None = None,
    scale: float | None = None,
    coefficient: float = 10.0,
    mask_num: int = 4,
    thresh: float = 0.3,
    highlight_factor: float = 2.0,  # Factor to increase weights in the masked area
    reduce_factor: float = 0.8,  # Factor to decrease weights in the unmasked area
) -> torch.FloatTensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).cuda()
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    txt_img_cross = attn_weight[:, :, -img_shape:, :txt_shape]  # lower left part
    # each column maps to a token's heatmap
    token_heatmap = txt_img_cross[:, :, :, token_index]  # Shape: [1, 24, 1024]
    token_heatmap = token_heatmap.mean(dim=1)[0]  # Shape: [1024]
    min_val, max_val = token_heatmap.min(), token_heatmap.max()
    norm_heatmap = (token_heatmap - min_val) / (max_val - min_val)

    mask_img = torch.sigmoid(coefficient * (norm_heatmap - 0.5))

    H = W = int(math.sqrt(mask_img.size(0)))
    mask_img = mask_img.reshape(H, W)

    save_path = f"heatmap/step_{cur_step}_layer_{cur_block}_token{token_index}.png"
    load_path = [f"heatmap/step_{cur_step - 1}_layer_{i}_token{token_index}.png" for i in layer]
    save_image(mask_img.unsqueeze(0), save_path)

    mask_img[mask_img >= thresh] = 1
    mask_img[mask_img < thresh] = 0

    mask_tensor = torch.zeros_like(mask_img)  # Set mask_tensor as a zero tensor
    if cur_step >= mask_num:
        mask_accumulator = torch.zeros_like(
            mask_tensor.unsqueeze(0), dtype=mask_img.dtype
        )  # Accumulator for averaging masks
        mask_tensor = auto_mask(
            load_path, mask_accumulator, thresh, attn_guidance_start_block, mask_num=mask_num
        )
        if cur_block == 1:
            save_image(
                mask_tensor,
                f"heatmap/average_heatmaps/step_{cur_step}_layer_{cur_block}_token{token_index}.png",
            )

    if not torch.all(mask_tensor == 0):
        mask_tensor = mask_tensor.reshape(1, H * W)
        mask_tensor = mask_tensor.unsqueeze(1).unsqueeze(-1)
        # Create a multiplier tensor: 2.0 where mask is active, 0.5 where mask is inactive.
        multiplier = torch.where(
            mask_tensor.bool(), torch.tensor(highlight_factor), torch.tensor(reduce_factor)
        )
        attn_weight[:, :, -img_shape:, :15] *= multiplier

    return attn_weight @ value


class FluxAttnProcessorWithMemory:
    def __init__(self, block_idx: int = -1):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FluxAttnProcessorWithMemory requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.memory = {}
        self.block_idx = block_idx

    def clear_memory(self):
        self.memory.clear()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        second_order: bool = False,
        timestep: int = -1,
        inject: bool = False,
        editing_strategy: Literal["reuse_v", "replace_v", "add_v", "replace_k", "add_k", "replace_q", "add_q"]
        | None = None,
        inverse: bool = False,
        attn_guidance_start_block: int = -1,
        qkv_ratio: list[float] = [1.0, 1.0, 1.0],  # noqa: B006
    ) -> torch.FloatTensor:
        batch_size, _, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # For sharing value to insert similar features
        if encoder_hidden_states is None and inject:  # single-stream block
            q_feature_name = f"{timestep}_{second_order}_q"
            k_feature_name = f"{timestep}_{second_order}_k"
            v_feature_name = f"{timestep}_{second_order}_v"
            if inverse:
                if "replace_v" in editing_strategy:
                    self.memory[v_feature_name] = value.cpu()
                else:
                    if "q" in editing_strategy:
                        self.memory[q_feature_name] = (query * qkv_ratio[0]).cpu()
                    if "k" in editing_strategy:
                        self.memory[k_feature_name] = (key * qkv_ratio[1]).cpu()
                    if "v" in editing_strategy:
                        self.memory[v_feature_name] = (value * qkv_ratio[2]).cpu()
            else:
                if "replace_q" in editing_strategy and q_feature_name in self.memory:
                    query = self.memory[q_feature_name].to(hidden_states.device)
                elif "add_q" in editing_strategy and q_feature_name in self.memory:
                    query += self.memory[q_feature_name].to(hidden_states.device)
                if "replace_k" in editing_strategy and k_feature_name in self.memory:
                    key = self.memory[k_feature_name].to(hidden_states.device)
                elif "add_k" in editing_strategy and k_feature_name in self.memory:
                    key += self.memory[k_feature_name].to(hidden_states.device)
                if "replace_v" in editing_strategy and v_feature_name in self.memory:
                    value = self.memory[v_feature_name].to(hidden_states.device)
                elif "add_v" in editing_strategy and v_feature_name in self.memory:
                    value += self.memory[v_feature_name].to(hidden_states.device)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if (
            encoder_hidden_states is not None and not inverse and "attn_guidance" in editing_strategy
        ):  # dual-stream block
            hidden_states = adaptive_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attention_mask,
                txt_shape=encoder_hidden_states.shape[1],
                img_shape=hidden_states.shape[1],
                cur_step=timestep,
                cur_block=self.block_idx,
                attn_guidance_start_block=attn_guidance_start_block,
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

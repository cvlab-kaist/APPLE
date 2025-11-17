from diffusers import FluxTransformer2DModel

import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    apply_rotary_emb,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle

from eva_clip import create_model_and_transforms
from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

from pulid.utils import img2tensor, tensor2img
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack

from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from huggingface_hub import hf_hub_download, snapshot_download
from insightface.app import FaceAnalysis
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize
import torchvision.transforms.functional as TFF
from Face_models.encoders.model_irse import Backbone
from omini.pipeline.flux_omini import transformer_forward_ca


import insightface

import cv2
import gc



logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _get_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_fused_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)

    encoder_query = encoder_key = encoder_value = (None,)
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_qkv_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)


class FluxAttnProcessor:
    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query, key, value, attn_mask=attention_mask, backend=self._attention_backend
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class FluxIPAdapterAttnProcessor(torch.nn.Module):
    """Flux Attention processor for IP-Adapter."""

    _attention_backend = None

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        num_tokens=(4,),
        scale=1.0,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]

        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError("`scale` should be a list of integers with the same length as `num_tokens`.")
        self.scale = scale

        self.to_k_ip = nn.ModuleList(
            [
                nn.Linear(cross_attention_dim, hidden_size, bias=True, device=device, dtype=dtype)
                for _ in range(len(num_tokens))
            ]
        )
        self.to_v_ip = nn.ModuleList(
            [
                nn.Linear(cross_attention_dim, hidden_size, bias=True, device=device, dtype=dtype)
                for _ in range(len(num_tokens))
            ]
        )

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        ip_hidden_states: Optional[List[torch.Tensor]] = None,
        ip_adapter_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)
        ip_query = query

        if encoder_hidden_states is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            # IP-adapter
            ip_attn_output = torch.zeros_like(hidden_states)

            for current_ip_hidden_states, scale, to_k_ip, to_v_ip in zip(
                ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip
            ):
                ip_key = to_k_ip(current_ip_hidden_states)
                ip_value = to_v_ip(current_ip_hidden_states)

                ip_key = ip_key.view(batch_size, -1, attn.heads, attn.head_dim)
                ip_value = ip_value.view(batch_size, -1, attn.heads, attn.head_dim)

                current_ip_hidden_states = dispatch_attention_fn(
                    ip_query,
                    ip_key,
                    ip_value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    backend=self._attention_backend,
                )
                current_ip_hidden_states = current_ip_hidden_states.reshape(batch_size, -1, attn.heads * attn.head_dim)
                current_ip_hidden_states = current_ip_hidden_states.to(ip_query.dtype)
                ip_attn_output += scale * current_ip_hidden_states

            return hidden_states, encoder_hidden_states, ip_attn_output
        else:
            return hidden_states


class FluxAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = FluxAttnProcessor
    _available_processors = [
        FluxAttnProcessor,
        FluxIPAdapterAttnProcessor,
    ]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        context_pre_only: Optional[bool] = None,
        pre_only: bool = False,
        elementwise_affine: bool = True,
        processor=None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.pre_only:
            self.to_out = torch.nn.ModuleList([])
            self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
            self.to_out.append(torch.nn.Dropout(dropout))

        if added_kv_proj_dim is not None:
            self.norm_added_q = torch.nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = torch.nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"joint_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)


@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=FluxAttnProcessor(),
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states


@maybe_allow_in_graph
class FluxTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
    ):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=FluxAttnProcessor(),
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}

        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxPosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class FluxTransformer2DModelCA(
    FluxTransformer2DModel
):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        patch_size (`int`, defaults to `1`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `19`):
            The number of layers of dual stream DiT blocks to use.
        num_single_layers (`int`, defaults to `38`):
            The number of layers of single stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `4096`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        pooled_projection_dim (`int`, defaults to `768`):
            The number of dimensions to use for the pooled projection.
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        onnx_provider: str = 'cuda',
        local_rank: int = 0,
        # ID Loss training
        use_netarc: bool = False, # For ID loss training
        netarc_path: str = None,
        use_irse50: bool = False, # For ID loss training
        # Gaze conditioning
        use_gaze: bool = False, # For gaze conditioning
        gaze_type: str = 'unigaze', # 'gaze' for GazeTR or 'unigaze'
        gaze_conditioning_type: str = 'CA', # 'CA' or 'temb'
        # Target CLIP conditioning
        use_target_clip: bool = False, # For target CLIP conditioning
    ):
        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
        )

        if gaze_type == 'gaze':
            gaze_embed_dim = 32
        elif gaze_type == 'unigaze':
            gaze_embed_dim = 1280
        else:
            raise ValueError(f'Unknown gaze type: {gaze_type}')
        if use_gaze:
            print(f'[INFO] Gaze type: {gaze_type}, Gaze embedding dim: {gaze_embed_dim}, Conditioning type: {gaze_conditioning_type}')
            
        if use_target_clip:
            print('[INFO] Using Target CLIP Conditioning')

        double_interval = 2
        single_interval = 4
        self.double_interval = double_interval
        self.single_interval = single_interval
        self.use_netarc = use_netarc
        self.use_irse50 = use_irse50
        self.use_gaze = use_gaze
        self.use_target_clip = use_target_clip
        self.onnx_provider = onnx_provider

        # init ID loss model
        self.netarc = None
        if use_netarc:
            if netarc_path is None:
                raise ValueError('netarc_path must be provided when use_netarc is True')
            self.netarc = torch.load(netarc_path, weights_only=False)
            self.netarc.eval()
            print('[INFO] Using ArcFace ResNet18 model for ID loss')
            
        if use_irse50:
            assert self.netarc is None, "Choose either irse50 or netarc"
            self.netarc = IDLoss()
            self.netarc.eval()
            print('[INFO] Using IRSE50 model for ID loss')

        # init encoder
        self.pulid_encoder = IDFormer(use_target_clip=self.use_target_clip).to(self.device, self.dtype)
        # initialize pulid encoder weights
        for name, param in self.pulid_encoder.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)

            elif 'weight' in name and param.dim() == 1:
                # LayerNorm, BatchNorm 등
                nn.init.ones_(param)

            elif param.dim() > 1:
                # Linear / Conv weights
                nn.init.xavier_normal_(param, gain=1e-4)

        num_ca = 19 // double_interval + 38 // single_interval
        if 19 % double_interval != 0:
            num_ca += 1
        if 38 % single_interval != 0:
            num_ca += 1
        self.pulid_ca = nn.ModuleList([
            PerceiverAttentionCA().to(self.device, self.dtype) for _ in range(num_ca)
        ])
        if self.use_gaze:
            print('[INFO] Using Gaze Conditioning')
            if gaze_conditioning_type == 'CA':
                self.gaze_ca = nn.ModuleList([
                    PerceiverAttentionCA(kv_dim=gaze_embed_dim, bias=True).to(self.device, self.dtype) for _ in range(num_ca)
                ])
                # # Initialize gaze projection layers with xavier uniform
                # for name, param in self.gaze_ca.named_parameters():
                #     if param.dim() > 1:
                #         print (f'[INFO] Initializing gaze CA param: {name}')
                #         nn.init.constant_(param, 1e-2)
            elif gaze_conditioning_type == 'temb' or gaze_conditioning_type == 'omini':
                # TODO: implement temb conditioning
                from diffusers.models.embeddings import TimestepEmbedding
                self.gaze_temb_proj = TimestepEmbedding(in_channels=gaze_embed_dim, time_embed_dim=self.inner_dim).to(self.dtype)
            else :
                raise ValueError(f'Unknown gaze conditioning type: {gaze_conditioning_type}')


        # preprocessors
        # face align and parsing
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.onnx_provider,
        )
        self.face_helper.face_parse = None
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device=self.onnx_provider)
        # clip-vit backbone
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
        model = model.visual
        self.clip_vision_model = model.to(self.device, dtype=self.dtype)
        eva_transform_mean = getattr(self.clip_vision_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(self.clip_vision_model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        self.eva_transform_mean = eva_transform_mean
        self.eva_transform_std = eva_transform_std
        
        # antelopev2
        snapshot_download('DIAMONIK7777/antelopev2', local_dir='./antelopev2')
        providers = ['CPUExecutionProvider'] if onnx_provider == 'cpu' \
            else [('CUDAExecutionProvider', {'device_id': local_rank})]
        self.app = FaceAnalysis(name='antelopev2', root='.', providers=providers)
        self.app.prepare(ctx_id=local_rank, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model('./antelopev2/glintr100.onnx',
                                                            providers=providers)
        self.handler_ante.prepare(ctx_id=local_rank)

        gc.collect()
        torch.cuda.empty_cache()
        
        # Load pretrained pulid weights
        self.load_pretrain() 
        
        # other configs
        self.debug_img_list = []

    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     encoder_hidden_states: torch.Tensor = None,
    #     pooled_projections: torch.Tensor = None,
    #     timestep: torch.LongTensor = None,
    #     img_ids: torch.Tensor = None,
    #     txt_ids: torch.Tensor = None,
    #     guidance: torch.Tensor = None,
    #     joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    #     controlnet_block_samples=None,
    #     controlnet_single_block_samples=None,
    #     return_dict: bool = True,
    #     controlnet_blocks_repeat: bool = False,
    #     # ID conditioning
    #     id_embed: torch.Tensor = None,
    #     id_weight: float = 1.0,
    #     # Gaze conditioning
    #     gaze_embed: torch.Tensor = None,
    #     gaze_weight: float = 1.0,
    # ) -> Union[torch.Tensor, Transformer2DModelOutput]:
    #     """
    #     The [`FluxTransformer2DModel`] forward method.

    #     Args:
    #         hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
    #             Input `hidden_states`.
    #         encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
    #             Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
    #         pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
    #             from the embeddings of input conditions.
    #         timestep ( `torch.LongTensor`):
    #             Used to indicate denoising step.
    #         block_controlnet_hidden_states: (`list` of `torch.Tensor`):
    #             A list of tensors that if specified are added to the residuals of transformer blocks.
    #         joint_attention_kwargs (`dict`, *optional*):
    #             A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
    #             `self.processor` in
    #             [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
    #         return_dict (`bool`, *optional*, defaults to `True`):
    #             Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
    #             tuple.

    #     Returns:
    #         If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
    #         `tuple` where the first element is the sample tensor.
    #     """
    #     if joint_attention_kwargs is not None:
    #         joint_attention_kwargs = joint_attention_kwargs.copy()
    #         lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    #     else:
    #         lora_scale = 1.0

    #     if USE_PEFT_BACKEND:
    #         # weight the lora layers by setting `lora_scale` for each PEFT layer
    #         scale_lora_layers(self, lora_scale)
    #     else:
    #         if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
    #             logger.warning(
    #                 "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
    #             )

    #     hidden_states = self.x_embedder(hidden_states)

    #     timestep = timestep.to(hidden_states.dtype) * 1000
    #     if guidance is not None:
    #         guidance = guidance.to(hidden_states.dtype) * 1000

    #     temb = (
    #         self.time_text_embed(timestep, pooled_projections)
    #         if guidance is None
    #         else self.time_text_embed(timestep, guidance, pooled_projections)
    #     )
    #     encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    #     if txt_ids.ndim == 3:
    #         logger.warning(
    #             "Passing `txt_ids` 3d torch.Tensor is deprecated."
    #             "Please remove the batch dimension and pass it as a 2d torch Tensor"
    #         )
    #         txt_ids = txt_ids[0]
    #     if img_ids.ndim == 3:
    #         logger.warning(
    #             "Passing `img_ids` 3d torch.Tensor is deprecated."
    #             "Please remove the batch dimension and pass it as a 2d torch Tensor"
    #         )
    #         img_ids = img_ids[0]

    #     ids = torch.cat((txt_ids, img_ids), dim=0)
    #     image_rotary_emb = self.pos_embed(ids)

    #     if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
    #         ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
    #         ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
    #         joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})


    #     ca_idx = 0
    #     gaze_ca_idx = 0
    #     for index_block, block in enumerate(self.transformer_blocks):
    #         if torch.is_grad_enabled() and self.gradient_checkpointing:
    #             encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
    #                 block,
    #                 hidden_states,
    #                 encoder_hidden_states,
    #                 temb,
    #                 image_rotary_emb,
    #                 joint_attention_kwargs,
    #             )
    #             if index_block % self.double_interval == 0 and id_embed is not None:
    #                 hidden_states_id = self._gradient_checkpointing_func(
    #                     self.pulid_ca[ca_idx],
    #                     id_embed,
    #                     hidden_states,
    #                 )
    #                 hidden_states = hidden_states + id_weight * hidden_states_id
    #                 ca_idx += 1
    #             if self.use_gaze and (index_block % self.double_interval == 0) and (gaze_embed is not None):
    #                 hidden_states_gaze = self._gradient_checkpointing_func(
    #                     self.gaze_ca[gaze_ca_idx],
    #                     gaze_embed,
    #                     hidden_states,
    #                 )
    #                 hidden_states = hidden_states + gaze_weight * hidden_states_gaze
    #                 gaze_ca_idx += 1

    #         else:
    #             encoder_hidden_states, hidden_states = block(
    #                 hidden_states=hidden_states,
    #                 encoder_hidden_states=encoder_hidden_states,
    #                 temb=temb,
    #                 image_rotary_emb=image_rotary_emb,
    #                 joint_attention_kwargs=joint_attention_kwargs,
    #             )
    #             if index_block % self.double_interval == 0 and id_embed is not None:
    #                 # print("add id cross attention for block", index_block, "ca idx", ca_idx)
    #                 hidden_states = hidden_states + id_weight * self.pulid_ca[ca_idx](id_embed, hidden_states)
    #                 ca_idx += 1
    #             if self.use_gaze and (index_block % self.double_interval == 0) and (gaze_embed is not None):
    #                 hidden_states = hidden_states + gaze_weight * self.gaze_ca[gaze_ca_idx](gaze_embed, hidden_states)
    #                 gaze_ca_idx += 1

    #         # controlnet residual
    #         if controlnet_block_samples is not None:
    #             interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
    #             interval_control = int(np.ceil(interval_control))
    #             # For Xlabs ControlNet.
    #             if controlnet_blocks_repeat:
    #                 hidden_states = (
    #                     hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
    #                 )
    #             else:
    #                 hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

    #     for index_block, block in enumerate(self.single_transformer_blocks):
    #         if torch.is_grad_enabled() and self.gradient_checkpointing:
    #             encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
    #                 block,
    #                 hidden_states,
    #                 encoder_hidden_states,
    #                 temb,
    #                 image_rotary_emb,
    #                 joint_attention_kwargs,
    #             )
    #             if index_block % self.single_interval == 0 and id_embed is not None:
    #                 hidden_states_id = self._gradient_checkpointing_func(
    #                     self.pulid_ca[ca_idx],
    #                     id_embed,
    #                     hidden_states,
    #                 )
    #                 hidden_states = hidden_states + id_weight * hidden_states_id
    #                 ca_idx += 1
    #             if self.use_gaze and (index_block % self.single_interval == 0) and (gaze_embed is not None):
    #                 hidden_states_gaze = self._gradient_checkpointing_func(
    #                     self.gaze_ca[gaze_ca_idx],
    #                     gaze_embed,
    #                     hidden_states,
    #                 )
    #                 hidden_states = hidden_states + gaze_weight * hidden_states_gaze
    #                 gaze_ca_idx += 1

    #         else:
    #             encoder_hidden_states, hidden_states = block(
    #                 hidden_states=hidden_states,
    #                 encoder_hidden_states=encoder_hidden_states,
    #                 temb=temb,
    #                 image_rotary_emb=image_rotary_emb,
    #                 joint_attention_kwargs=joint_attention_kwargs,
    #             )
    #             if index_block % self.single_interval == 0 and id_embed is not None:
    #                 hidden_states = hidden_states + id_weight * self.pulid_ca[ca_idx](id_embed, hidden_states)
    #                 ca_idx += 1
    #             if self.use_gaze and (index_block % self.single_interval == 0) and (gaze_embed is not None):
    #                 hidden_states = hidden_states + gaze_weight * self.gaze_ca[gaze_ca_idx](gaze_embed, hidden_states)
    #                 gaze_ca_idx += 1

    #         # controlnet residual
    #         if controlnet_single_block_samples is not None:
    #             interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
    #             interval_control = int(np.ceil(interval_control))
    #             hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

    #     hidden_states = self.norm_out(hidden_states, temb)
    #     output = self.proj_out(hidden_states)

    #     if USE_PEFT_BACKEND:
    #         # remove `lora_scale` from each PEFT layer
    #         unscale_lora_layers(self, lora_scale)

    #     if not return_dict:
    #         return (output,)

    #     return Transformer2DModelOutput(sample=output)


    #### Offloading 위한 forward 함수 ####
    # def forward(
    #     self,
    #     image_features: List[torch.Tensor],
    #     text_features: List[torch.Tensor] = None,
    #     img_ids: List[torch.Tensor] = None,
    #     txt_ids: List[torch.Tensor] = None,
    #     pooled_projections: List[torch.Tensor] = None,
    #     timesteps: List[torch.LongTensor] = None,
    #     guidances: List[torch.Tensor] = None,
    #     adapters: List[str] = None, #  [None, None, "default"]
    #     # Assign the function to be used for the forward pass
    #     single_block_forward=None,
    #     block_forward=None,
    #     attn_forward=None,
    #     # ID embedding parameters
    #     id_embed: torch.Tensor = None,
    #     id_weight: float = 1.0,
    #     # Gaze embedding parameters
    #     gaze_embed: torch.Tensor = None,
    #     gaze_weight: float = 1.0,
    #     **kwargs: dict,
    # ):
    #     return transformer_forward_ca(
    #         transformer=self,
    #         image_features=image_features,
    #         text_features=text_features,
    #         img_ids=img_ids,
    #         txt_ids=txt_ids,
    #         pooled_projections=pooled_projections,
    #         timesteps=timesteps,
    #         guidances=guidances,
    #         adapters=adapters,
    #         single_block_forward=single_block_forward,
    #         block_forward=block_forward,
    #         attn_forward=attn_forward,
    #         id_embed=id_embed,
    #         id_weight=id_weight,
    #         gaze_embed=gaze_embed,
    #         gaze_weight=gaze_weight,
    #         **kwargs,
    #     )

    
    def components_to_device(self, device, dtype):
        # everything but pulid_ca
        del self.face_helper
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.onnx_provider,
        )
        self.face_helper.face_parse = None
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device=self.onnx_provider)


        self.clip_vision_model = self.clip_vision_model.to(device, dtype)
        self.pulid_encoder = self.pulid_encoder.to(device, dtype)
        self.pulid_ca = self.pulid_ca.to(device, dtype)
        if self.netarc is not None:
            self.netarc = self.netarc.to(device, dtype)
        if self.use_gaze and self.gaze_conditioning_type == 'CA':
            self.gaze_ca = self.gaze_ca.to(device, dtype)

    def to(self, *args):
        super().to(*args)
        self.components_to_device(self.device, self.dtype)

        return self


    def components_requires_grad(self, grad_option:bool):
        self.face_helper.face_parse.requires_grad_(grad_option)
        self.clip_vision_model.requires_grad_(grad_option)
        self.pulid_encoder.requires_grad_(grad_option)
        self.pulid_ca.requires_grad_(grad_option)
        if self.use_gaze and self.gaze_conditioning_type == 'CA':
            self.gaze_ca.requires_grad_(grad_option)
    
    def load_pretrain(self, pretrain_path=None, version='v0.9.1'):
        hf_hub_download('guozinan/PuLID', f'pulid_flux_{version}.safetensors', local_dir='models')
        ckpt_path = f'models/pulid_flux_{version}.safetensors'
        if pretrain_path is not None:
            ckpt_path = pretrain_path
        state_dict = load_file(ckpt_path)
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_dict.setdefault(module, {})
            new_k = k[len(module) + 1:]
            state_dict_dict[module][new_k] = v

        for module in state_dict_dict:
            print(f'loading from {module}')
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=False)

        del state_dict
        del state_dict_dict

    def to_gray(self, img):
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        x = x.repeat(1, 3, 1, 1)
        return x

    @torch.no_grad()
    def get_id_embedding(self, image, cal_uncond=False, trg_image=None):
        """
        Args:
            image: numpy rgb image, range [0, 255]
            trg_image: numpy rgb image, range [0, 255]
        """
        self.face_helper.clean_all()
        self.debug_img_list = []
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # get antelopev2 embedding
        face_info = self.app.get(image_bgr)
        if len(face_info) > 0:
            face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
                -1
            ]  # only use the maximum face
            id_ante_embedding = face_info['embedding']
            self.debug_img_list.append(
                image[
                    int(face_info['bbox'][1]) : int(face_info['bbox'][3]),
                    int(face_info['bbox'][0]) : int(face_info['bbox'][2]),
                ]
            )
        else:
            id_ante_embedding = None

        # NaN check
        if id_ante_embedding is not None:
            if np.isnan(id_ante_embedding).any():
                raise RuntimeError('antelopev2 embedding is nan')

        # using facexlib to detect and align face
        self.face_helper.read_image(image_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            raise RuntimeError('facexlib align face fail')
        align_face = self.face_helper.cropped_faces[0]

        # incase insightface didn't detect face
        if id_ante_embedding is None:
            print('fail to detect face using insightface, extract embedding on align face')
            id_ante_embedding = self.handler_ante.get_feat(align_face)
            if np.isnan(id_ante_embedding).any():
                raise RuntimeError('antelopev2 embedding from align face is nan')

        id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device, self.dtype)
        if id_ante_embedding.ndim == 1:
            id_ante_embedding = id_ante_embedding.unsqueeze(0)

        # NaN check
        if torch.isnan(id_ante_embedding).any():
            raise RuntimeError('antelopev2 embedding is nan')
        
        # parsing
        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        input = input.to(self.device)
        parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        if torch.isnan(parsing_out).any():
            raise RuntimeError('parsing_out is nan')
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)
        # only keep the face features
        face_features_image = torch.where(bg, white_image, self.to_gray(input))
        self.debug_img_list.append(tensor2img(face_features_image, rgb2bgr=False))

        # transform img before sending to eva-clip-vit
        face_features_image = resize(face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC)
        face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
        id_cond_vit, id_vit_hidden = self.clip_vision_model(
            face_features_image.to(self.dtype), return_all_features=False, return_hidden=True, shuffle=False
        )
        if torch.isnan(id_cond_vit).any():
            raise RuntimeError('id_cond_vit is nan')
        id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
        id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)
        if torch.isnan(id_cond_vit).any():
            raise RuntimeError('id_cond_vit after norm is nan')

        # id_ante_embedding : torch.Size([1, 512])
        # id_cond_vit : torch.Size([1, 768])
        # id_cond : torch.Size([1, 1280])
        id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)

        # (Pdb) [ x.shape for x in id_vit_hidden]
        # [torch.Size([1, 577, 1024]), torch.Size([1, 577, 1024]), torch.Size([1, 577, 1024]), torch.Size([1, 577, 1024]), torch.Size([1, 577, 1024])]
        if self.use_target_clip and trg_image is not None:
            # Do not detect / align, directly send to clip-vit
            print('[INFO] transformer.get_id_embedding : using target clip image for target id embedding')
            trg_image_bgr = cv2.cvtColor(trg_image, cv2.COLOR_RGB2BGR)
            input = img2tensor(trg_image_bgr, bgr2rgb=True).unsqueeze(0) / 255.0
            input = input.to(self.device)
            trg_image_resized = resize(input, self.clip_vision_model.image_size, InterpolationMode.BICUBIC)
            trg_face_features_image = normalize(trg_image_resized, self.eva_transform_mean, self.eva_transform_std)
            trg_id_cond_vit, trg_id_vit_hidden = self.clip_vision_model(
                trg_face_features_image.to(self.dtype), return_all_features=False, return_hidden=True, shuffle=False
            )
            id_embedding = self.pulid_encoder(id_cond, id_vit_hidden, trg_id_cond_vit)
        else:
            id_embedding = self.pulid_encoder(id_cond, id_vit_hidden)
        if torch.isnan(id_embedding).any():
            raise RuntimeError('id_embedding is nan')

        if not cal_uncond:
            return id_embedding, None

        id_uncond = torch.zeros_like(id_cond) # zero embedding for uncond
        id_vit_hidden_uncond = []
        for layer_idx in range(0, len(id_vit_hidden)):
            id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[layer_idx])) # zero hidden for uncond
        uncond_id_embedding = self.pulid_encoder(id_uncond, id_vit_hidden_uncond)
        if torch.isnan(uncond_id_embedding).any():
            raise RuntimeError('uncond_id_embedding is nan')

        return id_embedding, uncond_id_embedding
    
    @torch.no_grad()
    def get_id_embedding_(self, image, cal_uncond=False, trg_image=None):
        """
        Args:
            image: numpy rgb image, range [0, 255]
            trg_image: numpy rgb image, range [0, 255]
        """
        self.face_helper.clean_all()
        self.debug_img_list = []
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # get antelopev2 embedding
        face_info = self.app.get(image_bgr)
        if len(face_info) > 0:
            face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
                -1
            ]  # only use the maximum face
            id_ante_embedding = face_info['embedding']
            self.debug_img_list.append(
                image[
                    int(face_info['bbox'][1]) : int(face_info['bbox'][3]),
                    int(face_info['bbox'][0]) : int(face_info['bbox'][2]),
                ]
            )
        else:
            id_ante_embedding = None

        # NaN check
        if id_ante_embedding is not None:
            if np.isnan(id_ante_embedding).any():
                raise RuntimeError('antelopev2 embedding is nan')

        # using facexlib to detect and align face
        self.face_helper.read_image(image_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            raise RuntimeError('facexlib align face fail')
        align_face = self.face_helper.cropped_faces[0]

        # incase insightface didn't detect face
        if id_ante_embedding is None:
            # print('fail to detect face using insightface, extract embedding on align face')
            id_ante_embedding = self.handler_ante.get_feat(align_face)
            if np.isnan(id_ante_embedding).any():
                raise RuntimeError('antelopev2 embedding from align face is nan')

        id_ante_embedding = torch.from_numpy(id_ante_embedding)
        return id_ante_embedding
        # id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device, self.dtype)
        # if id_ante_embedding.ndim == 1:
        #     id_ante_embedding = id_ante_embedding.unsqueeze(0)

        # # NaN check
        # if torch.isnan(id_ante_embedding).any():
        #     raise RuntimeError('antelopev2 embedding is nan')
        
        # # parsing
        # input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        # input = input.to(self.device)
        # parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        # if torch.isnan(parsing_out).any():
        #     raise RuntimeError('parsing_out is nan')
        # parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        # bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        # bg = sum(parsing_out == i for i in bg_label).bool()
        # white_image = torch.ones_like(input)
        # # only keep the face features
        # face_features_image = torch.where(bg, white_image, self.to_gray(input))
        # self.debug_img_list.append(tensor2img(face_features_image, rgb2bgr=False))

        # # transform img before sending to eva-clip-vit
        # face_features_image = resize(face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC)
        # face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
        # id_cond_vit, id_vit_hidden = self.clip_vision_model(
        #     face_features_image.to(self.dtype), return_all_features=False, return_hidden=True, shuffle=False
        # )
        # if torch.isnan(id_cond_vit).any():
        #     raise RuntimeError('id_cond_vit is nan')
        # id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
        # id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)
        # if torch.isnan(id_cond_vit).any():
        #     raise RuntimeError('id_cond_vit after norm is nan')

        # # id_ante_embedding : torch.Size([1, 512])
        # # id_cond_vit : torch.Size([1, 768])
        # # id_cond : torch.Size([1, 1280])
        # id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)

        # # (Pdb) [ x.shape for x in id_vit_hidden]
        # # [torch.Size([1, 577, 1024]), torch.Size([1, 577, 1024]), torch.Size([1, 577, 1024]), torch.Size([1, 577, 1024]), torch.Size([1, 577, 1024])]
        # if self.use_target_clip and trg_image is not None:
        #     # Do not detect / align, directly send to clip-vit
        #     print('[INFO] transformer.get_id_embedding : using target clip image for target id embedding')
        #     trg_image_bgr = cv2.cvtColor(trg_image, cv2.COLOR_RGB2BGR)
        #     input = img2tensor(trg_image_bgr, bgr2rgb=True).unsqueeze(0) / 255.0
        #     input = input.to(self.device)
        #     trg_image_resized = resize(input, self.clip_vision_model.image_size, InterpolationMode.BICUBIC)
        #     trg_face_features_image = normalize(trg_image_resized, self.eva_transform_mean, self.eva_transform_std)
        #     trg_id_cond_vit, trg_id_vit_hidden = self.clip_vision_model(
        #         trg_face_features_image.to(self.dtype), return_all_features=False, return_hidden=True, shuffle=False
        #     )
        #     id_embedding = self.pulid_encoder(id_cond, id_vit_hidden, trg_id_cond_vit)
        # else:
        #     id_embedding = self.pulid_encoder(id_cond, id_vit_hidden)
        # if torch.isnan(id_embedding).any():
        #     raise RuntimeError('id_embedding is nan')

        # if not cal_uncond:
        #     return id_embedding, None

        # id_uncond = torch.zeros_like(id_cond) # zero embedding for uncond
        # id_vit_hidden_uncond = []
        # for layer_idx in range(0, len(id_vit_hidden)):
        #     id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[layer_idx])) # zero hidden for uncond
        # uncond_id_embedding = self.pulid_encoder(id_uncond, id_vit_hidden_uncond)
        # if torch.isnan(uncond_id_embedding).any():
        #     raise RuntimeError('uncond_id_embedding is nan')

        # return id_embedding, uncond_id_embedding
    
    @torch.no_grad()
    def get_id_embedding_from_id_and_clip(self, id_img, clip_img):
        """
        Args:
            image: numpy rgb image, range [0, 255]
        """

        self.face_helper.clean_all()
        self.debug_img_list = []
        image_bgr = cv2.cvtColor(id_img, cv2.COLOR_RGB2BGR)
        # get antelopev2 embedding
        face_info = self.app.get(image_bgr)
        if len(face_info) > 0:
            face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
                -1
            ]  # only use the maximum face
            id_ante_embedding = face_info['embedding']
            self.debug_img_list.append(
                id_img[
                    int(face_info['bbox'][1]) : int(face_info['bbox'][3]),
                    int(face_info['bbox'][0]) : int(face_info['bbox'][2]),
                ]
            )
        else:
            id_ante_embedding = None

        # NaN check
        if id_ante_embedding is not None:
            if np.isnan(id_ante_embedding).any():
                raise RuntimeError('antelopev2 embedding is nan')

        # using facexlib to detect and align face
        self.face_helper.read_image(image_bgr) # id_img로부터 face align
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            raise RuntimeError('facexlib align face fail')
        align_face = self.face_helper.cropped_faces[0]

        # incase insightface didn't detect face
        if id_ante_embedding is None:
            print('fail to detect face using insightface, extract embedding on align face')
            id_ante_embedding = self.handler_ante.get_feat(align_face)
            if np.isnan(id_ante_embedding).any():
                raise RuntimeError('antelopev2 embedding from align face is nan')

        id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device, self.dtype)
        if id_ante_embedding.ndim == 1:
            id_ante_embedding = id_ante_embedding.unsqueeze(0)

        # NaN check
        if torch.isnan(id_ante_embedding).any():
            raise RuntimeError('antelopev2 embedding is nan')
        
        ########## parsing for clip feature

        image_bgr = cv2.cvtColor(clip_img, cv2.COLOR_RGB2BGR)

        # using facexlib to detect and align face
        self.face_helper.clean_all()
        self.face_helper.read_image(image_bgr) # clip_img로부터 face align
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            raise RuntimeError('facexlib align face fail')
        align_face = self.face_helper.cropped_faces[0]


        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        input = input.to(self.device)
        parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        if torch.isnan(parsing_out).any():
            raise RuntimeError('parsing_out is nan')
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)
        # only keep the face features
        face_features_image = torch.where(bg, white_image, self.to_gray(input))
        self.debug_img_list.append(tensor2img(face_features_image, rgb2bgr=False))

        # transform img before sending to eva-clip-vit
        face_features_image = resize(face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC)
        face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
        id_cond_vit, id_vit_hidden = self.clip_vision_model( # CLS 토큰 및 중간레이어
            face_features_image.to(self.dtype), return_all_features=False, return_hidden=True, shuffle=False
        )
        if torch.isnan(id_cond_vit).any():
            raise RuntimeError('id_cond_vit is nan')
        id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
        id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)
        if torch.isnan(id_cond_vit).any():
            raise RuntimeError('id_cond_vit after norm is nan')

        id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1) # antelopev2 + clip-vit, channel 방향 concat

        id_embedding = self.pulid_encoder(id_cond, id_vit_hidden)
        if torch.isnan(id_embedding).any():
            raise RuntimeError('id_embedding is nan')

        # if not cal_uncond:
        return id_embedding

        # id_uncond = torch.zeros_like(id_cond) # zero embedding for uncond
        # id_vit_hidden_uncond = []
        # for layer_idx in range(0, len(id_vit_hidden)):
        #     id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[layer_idx])) # zero hidden for uncond
        # uncond_id_embedding = self.pulid_encoder(id_uncond, id_vit_hidden_uncond)
        # if torch.isnan(uncond_id_embedding).any():
        #     raise RuntimeError('uncond_id_embedding is nan')

        # return id_embedding, uncond_id_embedding

################## ArcFace IRSE50 model ##################
def un_norm_clip(x1):
    x = x1*1.0 # to avoid changing the original tensor or clone() can be used
    reduce=False
    if len(x.shape)==3:
        x = x.unsqueeze(0)
        reduce=True
    x[:,0,:,:] = x[:,0,:,:] * 0.26862954 + 0.48145466
    x[:,1,:,:] = x[:,1,:,:] * 0.26130258 + 0.4578275
    x[:,2,:,:] = x[:,2,:,:] * 0.27577711 + 0.40821073
    
    if reduce:
        x = x.squeeze(0)
    return x

class IDLoss(nn.Module):
    def __init__(self,multiscale=True):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        # self.opts = opts 
        self.multiscale = multiscale
        self.face_pool_1 = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
    
        self.facenet.load_state_dict(torch.load("Other_dependencies/arcface/model_ir_se50.pth"))
        
        self.face_pool_2 = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        
        self.set_requires_grad(False)
            
    def set_requires_grad(self, flag=True):
        for p in self.parameters():
            p.requires_grad = flag
    
    def extract_feats(self, x,clip_img=False):
        # breakpoint()
        if clip_img:
            x = un_norm_clip(x)
            x = TFF.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        x = self.face_pool_1(x)  if x.shape[2]!=256 else  x # (1) resize to 256 if needed
        x = x[:, :, 35:223, 32:220]  # (2) Crop interesting region
        x = self.face_pool_2(x) # (3) resize to 112 to fit pre-trained model
        # breakpoint()
        x_feats = self.facenet(x, multi_scale=self.multiscale )

        return x_feats

    def forward(self, x,clip_img=False):
        x_feats_ms = self.extract_feats(x,clip_img=clip_img)
        return x_feats_ms[-1]

################## PulID model ##################
import math

import torch
import torch.nn as nn


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttentionCA(nn.Module):
    def __init__(self, *, dim=3072, dim_head=128, heads=16, kv_dim=2048, bias=False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads # 2048

        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=bias)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1) # (b, D) -> (b, 1, D)
        if torch.isnan(x).any():
            raise RuntimeError('PerceiverAttentionCA: input x is nan')
        x = self.norm1(x)
        # NAN check
        if torch.isnan(x).any():
            raise RuntimeError('PerceiverAttentionCA: x is nan')
        latents = self.norm2(latents)
        # NAN check
        if torch.isnan(latents).any():
            raise RuntimeError('PerceiverAttentionCA: latents is nan')

        b, seq_len, _ = latents.shape

        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        if torch.isnan(q).any():
            raise RuntimeError('PerceiverAttentionCA: q is nan')
        if torch.isnan(k).any():
            raise RuntimeError('PerceiverAttentionCA: k is nan')
        if torch.isnan(v).any():
            raise RuntimeError('PerceiverAttentionCA: v is nan')

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        ### attention
        # scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        # weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        # weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        # if torch.isnan(weight).any():
        #     raise RuntimeError('PerceiverAttentionCA: attention weight is nan')
        # out = weight @ v

        ### F.scaled_dot_product_attention
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )
        if torch.isnan(out).any():
            raise RuntimeError('PerceiverAttentionCA: output out after attention is nan')

        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)

        if torch.isnan(out).any():
            raise RuntimeError('PerceiverAttentionCA: output out is nan')

        return self.to_out(out)


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, kv_dim=None):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)

        return self.to_out(out)


class IDFormer(nn.Module):
    """
    - perceiver resampler like arch (compared with previous MLP-like arch)
    - we concat id embedding (generated by arcface) and query tokens as latents
    - latents will attend each other and interact with vit features through cross-attention
    - vit features are multi-scaled and inserted into IDFormer in order, currently, each scale corresponds to two
      IDFormer layers
    """
    def __init__(
            self,
            dim=1024,
            depth=10,
            dim_head=64,
            heads=16,
            num_id_token=5,
            num_queries=32,
            output_dim=2048,
            ff_mult=4,
            use_target_clip=False,
    ):
        super().__init__()

        self.num_id_token = num_id_token
        self.dim = dim
        self.num_queries = num_queries
        assert depth % 5 == 0
        self.depth = depth // 5
        scale = dim ** -0.5
        self.use_target_clip = use_target_clip

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) * scale) # torch.Size([1, 32, 1024])
        self.proj_out = nn.Parameter(scale * torch.randn(dim, output_dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        for i in range(5):
            setattr(
                self,
                f'mapping_{i}',
                nn.Sequential( # dim : from 1024 -> dim = 1024
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, dim),
                ),
            )
            
            
        if self.use_target_clip:
            for i in range(5):
                setattr(
                    self,
                    f'trg_mapping_{i}',
                    nn.Sequential( # dim : 768 -> dim = 1024
                        nn.Linear(768, 1024),
                        nn.LayerNorm(1024),
                        nn.LeakyReLU(),
                        nn.Linear(1024, 1024),
                        nn.LayerNorm(1024),
                        nn.LeakyReLU(),
                        nn.Linear(1024, dim),
                    ),
                )

        self.id_embedding_mapping = nn.Sequential( # from 1280 to dim * num_id_token
            nn.Linear(1280, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, dim * num_id_token),
        )

    def forward(self, x, y, z=None):
        # z : target clip cls token for target clip condition 

        latents = self.latents.repeat(x.size(0), 1, 1) # shape : (1, 32, 1024) -> (b, 32, 1024)

        num_duotu = x.shape[1] if x.ndim == 3 else 1

        x = self.id_embedding_mapping(x) # shape : (b, 1280) -> (b, dim * num_id_token) = (b, 5 * 1024)
        x = x.reshape(-1, self.num_id_token * num_duotu, self.dim) # shape : (b, dim * num_id_token) -> (b, num_id_token, dim), (b, 5, 1024)

        latents = torch.cat((latents, x), dim=1) # shape : (b, 32, 1024) + (b, 5, 1024) -> (b, 37, 1024), token concat

        for i in range(5):
            vit_feature = getattr(self, f'mapping_{i}')(y[i]) # shape
            ctx_feature = torch.cat((x, vit_feature), dim=1) # token concat : (Pdb) x.shape torch.Size([1, 5, 1024]) (Pdb) vit_feature.shape torch.Size([1, 577, 1024])
            if self.use_target_clip and (z is not None):
                trg_clip_cls = getattr(self, f'trg_mapping_{i}')(z) # shape : (b, 768) -> (b, 1024)
                trg_clip_cls = trg_clip_cls.unsqueeze(1) # (b, 1, 1024)
                ctx_feature = torch.cat((ctx_feature, trg_clip_cls), dim=1) # (b, 5 + 577 + 1, 1024)
            for attn, ff in self.layers[i * self.depth: (i + 1) * self.depth]:
                latents = attn(ctx_feature, latents) + latents
                latents = ff(latents) + latents

        latents = latents[:, :self.num_queries]
        latents = latents @ self.proj_out
        return latents


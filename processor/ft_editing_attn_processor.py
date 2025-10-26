from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.pipelines import DiffusionPipeline


class ScoreParams:
    def __init__(self, gap, match, mismatch):
        self.gap = gap
        self.match = match
        self.mismatch = mismatch

    def mis_match_char(self, x, y):
        if x != y:
            return self.mismatch
        else:
            return self.match


def get_matrix(size_x, size_y, gap):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = (np.arange(size_y) + 1) * gap
    matrix[1:, 0] = (np.arange(size_x) + 1) * gap
    return matrix


def get_traceback_matrix(size_x, size_y):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = 1
    matrix[1:, 0] = 2
    matrix[0, 0] = 4
    return matrix


def global_align(x, y, score):
    matrix = get_matrix(len(x), len(y), score.gap)
    trace_back = get_traceback_matrix(len(x), len(y))
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            left = matrix[i, j - 1] + score.gap
            up = matrix[i - 1, j] + score.gap
            diag = matrix[i - 1, j - 1] + score.mis_match_char(x[i - 1], y[j - 1])
            matrix[i, j] = max(left, up, diag)
            if matrix[i, j] == left:
                trace_back[i, j] = 1
            elif matrix[i, j] == up:
                trace_back[i, j] = 2
            else:
                trace_back[i, j] = 3
    return matrix, trace_back


def get_aligned_sequences(x, y, trace_back):
    x_seq = []
    y_seq = []
    i = len(x)
    j = len(y)
    mapper_y_to_x = []
    while i > 0 or j > 0:
        if trace_back[i, j] == 3:
            x_seq.append(x[i - 1])
            y_seq.append(y[j - 1])
            i = i - 1
            j = j - 1
            mapper_y_to_x.append((j, i))
        elif trace_back[i][j] == 1:
            x_seq.append("-")
            y_seq.append(y[j - 1])
            j = j - 1
            mapper_y_to_x.append((j, -1))
        elif trace_back[i][j] == 2:
            x_seq.append(x[i - 1])
            y_seq.append("-")
            i = i - 1
        elif trace_back[i][j] == 4:
            break
    mapper_y_to_x.reverse()
    return x_seq, y_seq, torch.tensor(mapper_y_to_x, dtype=torch.int64)


def get_mapper(x: str, y: str, tokenizer, max_len=77):
    x_seq = tokenizer.encode(x)
    y_seq = tokenizer.encode(y)
    score = ScoreParams(0, 1, -1)
    matrix, trace_back = global_align(x_seq, y_seq, score)
    mapper_base = get_aligned_sequences(x_seq, y_seq, trace_back)[-1]
    alphas = torch.ones(max_len)
    alphas[: mapper_base.shape[0]] = mapper_base[:, 1].ne(-1).float()
    mapper = torch.zeros(max_len, dtype=torch.int64)
    mapper[: mapper_base.shape[0]] = mapper_base[:, 1]
    mapper[mapper_base.shape[0] :] = len(y_seq) + torch.arange(max_len - len(y_seq))
    return mapper, alphas


def get_refinement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]
    mappers, alphas = [], []
    for i in range(1, len(prompts)):
        mapper, alpha = get_mapper(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
        alphas.append(alpha)
    return torch.stack(mappers), torch.stack(alphas)


def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def get_replacement_mapper_(x: str, y: str, tokenizer, max_len=77):
    words_x = x.split(" ")
    words_y = y.split(" ")
    if len(words_x) != len(words_y):
        raise ValueError(
            f"attention replacement edit can only be applied on prompts with the same length"
            f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words."
        )
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]
    inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]
    mapper = np.zeros((max_len, max_len))
    i = j = 0
    cur_inds = 0
    while i < max_len and j < max_len:
        if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:
            inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]
            if len(inds_source_) == len(inds_target_):
                mapper[inds_source_, inds_target_] = 1
            else:
                ratio = 1 / len(inds_target_)
                for i_t in inds_target_:
                    mapper[inds_source_, i_t] = ratio
            cur_inds += 1
            i += len(inds_source_)
            j += len(inds_target_)
        elif cur_inds < len(inds_source):
            mapper[i, j] = 1
            i += 1
            j += 1
        else:
            mapper[j, j] = 1
            i += 1
            j += 1

    return torch.from_numpy(mapper).float()


def get_replacement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]
    mappers = []
    for i in range(1, len(prompts)):
        mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
    return torch.stack(mappers)


def update_alpha_time_word(
    alpha,
    bounds: Union[float, Tuple[float, float]],
    prompt_ind: int,
    word_inds: Optional[torch.Tensor] = None,
):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start:end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(
    prompts,
    num_steps,
    cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
    tokenizer,
    max_num_words=77,
):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0.0, 1.0)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"], i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, max_num_words, 1)
    return alpha_time_words


def calc_mean_std(feat, eps: float = 1e-5):
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def expand_first(
    feat,
    scale=1.0,
):
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def adaln(src_feat, tar_feat):
    src_mean, src_std = calc_mean_std(src_feat)
    tar_mean, tar_std = calc_mean_std(tar_feat)

    feat = (src_feat - src_mean) / src_std
    feat = feat * tar_std + tar_mean

    return feat


class AdalayernormReplace:
    def __init__(self, prompts, num_steps, ly_ratio, tokenizer, device, num_adanorm):
        self.cur_step = 0
        self.cur_layer = 0
        self.num_adanorm = num_adanorm
        self.batch_size = len(prompts)
        if type(ly_ratio) is float:
            ly_ratio = 0, ly_ratio
        self.num_self_replace = int(num_steps * ly_ratio[0]), int(num_steps * ly_ratio[1])

        # get t5 mappers
        self.mapper, alphas = get_refinement_mapper(prompts, tokenizer, max_len=512)
        self.mapper, alphas = self.mapper.squeeze().to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], alphas.shape[1], 1)

    def replace_adaptive_layernorm(self, base, replace):
        select_unchange_base_t5 = base[:, self.mapper, :]
        replace = select_unchange_base_t5 * self.alphas + replace * (1 - self.alphas)

        return replace

    def forward(self, x):
        if self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]:
            # No CFG, use all batches
            txt_prpt_embed = x  # batch_size = 2
            txt_prpt_embed_base = txt_prpt_embed[0:1, :]
            txt_prpt_embed_replace = txt_prpt_embed[1:2, :]
            txt_prpt_embed[1:2, :] = self.replace_adaptive_layernorm(
                txt_prpt_embed_base, txt_prpt_embed_replace
            )
        return x

    def __call__(self, x):
        x = self.forward(x)
        self.cur_layer += 1
        if self.cur_layer == self.num_adanorm:
            self.cur_step += 1
            self.cur_layer = 0
        return x


class FluxAttentionReplace:
    def __init__(self, prompts, num_steps, attn_ratio, num_att_layers):
        self.cur_step = 0
        self.cur_layer = 0
        self.num_att_layers = num_att_layers
        self.batch_size = len(prompts)
        if type(attn_ratio) is float:
            attn_ratio = 0, attn_ratio
        self.num_self_replace = int(num_steps * attn_ratio[0]), int(num_steps * attn_ratio[1])

    def replace_self_attention(
        self,
        attn_base,
        att_replace,
    ):
        attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        return attn_base

    def forward(self, x):
        if self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]:
            x_base, x_repalce = x[0], x[1:]
            x[1:] = self.replace_self_attention(
                x_base,
                x_repalce,
            )

        return x

    def __call__(self, x):
        x = self.forward(x)
        self.cur_layer += 1
        if self.cur_layer == self.num_att_layers:
            self.cur_step += 1
            self.cur_layer = 0
        return x


class P2PFlux_JointAttnProcessor2_0:
    def __init__(self, controller, block_idx: int = -1):
        super().__init__()
        self.block_idx = block_idx
        self.controller = controller
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "P2PFlux_JointAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        do_replace: bool = False,
    ) -> torch.FloatTensor:
        batch_size, _, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

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

        # since flux does not need CFG, replace all
        if do_replace:
            value = self.controller(value)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
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


def register_norm_control_flux(pipe: DiffusionPipeline, controller: AdalayernormReplace | None = None):
    def forward_norm1_cxt_with_controller(self):
        def forward(
            x: torch.Tensor,
            timestep: Optional[torch.Tensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            hidden_dtype: Optional[torch.dtype] = None,
            emb: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            if self.emb is not None:
                emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
            emb = self.linear(self.silu(emb))
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
            x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
            x = controller(x)
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

        return forward

    def forward_norm1_cxt_without_controller(self):
        def forward(
            x: torch.Tensor,
            timestep: Optional[torch.Tensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            hidden_dtype: Optional[torch.dtype] = None,
            emb: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            if self.emb is not None:
                emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
            emb = self.linear(self.silu(emb))
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
            x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

        return forward

    for _, net in pipe.transformer.transformer_blocks.named_children():
        for sub_net_name, sub_net in net.named_children():
            if sub_net_name == "norm1_context":
                if controller is not None:
                    sub_net.forward = forward_norm1_cxt_with_controller(sub_net)
                else:
                    sub_net.forward = forward_norm1_cxt_without_controller(sub_net)

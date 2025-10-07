from typing import Callable, List, Optional, Tuple
import copy
import os

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import time

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .configuration_mixtral import MixtralConfig, get_mixtral_config

class MixtralRMSNorm(nn.Module):
    def __init__(self, hidden_size, device, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.device = device

    def init_weights(self, path):
        self.weight.data = torch.load(path, map_location=self.device, weights_only=True)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{Tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class MixtralRotaryEmbedding(nn.Module):
    def __init__(self, config: MixtralConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class MixtralAttention(nn.Module):

    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.hidden_size=config.hidden_size
        self.num_heads=config.num_attention_heads
        self.num_key_value_heads=config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
    def init_weights(self, path):
        path = path + f"/original/model.layers.{self.layer_idx}.self_attn."
        self.q_proj.weight.data = torch.load(path+"q_proj.weight", map_location=self.config.device, weights_only=True)
        self.k_proj.weight.data = torch.load(path+"k_proj.weight", map_location=self.config.device, weights_only=True)
        self.v_proj.weight.data = torch.load(path+"v_proj.weight", map_location=self.config.device, weights_only=True)
        self.o_proj.weight.data = torch.load(path+"o_proj.weight", map_location=self.config.device, weights_only=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.w1 = self.w2 = self.w3 = None

        self.act_fn = ACT2FN[config.hidden_act]

    def init_weights(self, path, idx=None):        
        path_0 = path + f"/original/model.layers.{self.layer_idx}.block_sparse_moe.experts.{idx}."
        self.w1_path = path_0 + f"w1.weight"
        self.w2_path = path_0 + f"w2.weight"
        self.w3_path = path_0 + f"w3.weight"

        self.w1 = torch.load(self.w1_path, map_location=self.config.device, weights_only=True)
        self.w2 = torch.load(self.w2_path, map_location=self.config.device, weights_only=True)
        self.w3 = torch.load(self.w3_path, map_location=self.config.device, weights_only=True)
        
    def forward(self, x):
        return F.linear(F.silu(F.linear(x, self.w1)) * F.linear(x, self.w3), self.w2) 


class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.layer_idx = layer_idx
        self.device = config.device

        # gating
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MixtralBlockSparseTop2MLP(config, layer_idx) for _ in range(self.num_experts)]
        )

    def init_weights(self, path):
        gate_path = path + f"/original/model.layers.{self.layer_idx}.block_sparse_moe.gate.weight"
        self.gate.weight.data = torch.load(gate_path, map_location=self.device, weights_only=True)
        
        for idx in range(self.num_experts):
            self.experts[idx].init_weights(path, idx)
         
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class MixtralDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = MixtralAttention(config, layer_idx)
        self.block_sparse_moe = MixtralSparseMoeBlock(config, layer_idx)

        self.input_layernorm = MixtralRMSNorm(config.hidden_size, config.device, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MixtralRMSNorm(config.hidden_size, self.config.device, eps=config.rms_norm_eps)
        
    def init_weights(self, path):
        self.self_attn.init_weights(path)
        self.block_sparse_moe.init_weights(path)

        input_ln_path = path + f"/original/model.layers.{self.layer_idx}.input_layernorm.weight"
        post_ln_path = path + f"/original/model.layers.{self.layer_idx}.post_attention_layernorm.weight"
        self.input_layernorm.init_weights(input_ln_path)
        self.post_attention_layernorm.init_weights(post_ln_path)
    
    def forward(
        self, hidden_states, attention_mask = None, position_ids = None, past_key_value = None,
        cache_position = None, position_embeddings= None, **kwargs,
    ):
        residual = hidden_states

        attn_start = time.perf_counter()
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cache_position=cache_position,
        )
        attn_time = time.perf_counter() - attn_start

        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        mlp_start = time.perf_counter()
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)
        mlp_time = time.perf_counter() - mlp_start

        hidden_states = residual + hidden_states

        outputs = (hidden_states, attn_time, mlp_time)

        return outputs


class MixtralModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = MixtralDecoderLayer(config)
        self.norm = MixtralRMSNorm(config.hidden_size, config.device, eps=config.rms_norm_eps)
        self.rotary_emb = MixtralRotaryEmbedding(config=config)
        
    def init_weights(self, path):
        self.layers.init_weights(path)
        
        ln_path = path + "/original/model.norm.weight"
        embed_path = path + "/original/model.embed_tokens.weight"
        self.norm.init_weights(ln_path)
        self.embed_tokens.weight.data = torch.load(embed_path, map_location=self.config.device, weights_only=True)
    
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> Tuple:
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        dec_start = time.perf_counter()
        layer_outputs = self.layers(
            hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_value=past_key_values,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )

        hidden_states = layer_outputs[0]

        dec_time = time.perf_counter() - dec_start
            
        hidden_states = self.norm(hidden_states)

        return hidden_states, past_key_values, layer_outputs[1], layer_outputs[2], dec_time
    
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Mixtral. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
    
    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: MixtralConfig,
        past_key_values: Cache,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask

class MixtralForCausalLM(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.config = get_mixtral_config(args.model)
        self.config.device = args.device
        self.device = args.device
        self.min_length = args.min_length
        self.max_length = args.max_length
        self.early_stopping = args.early_stopping
        self.path = args.path
        self.config._attn_implementation = "sdpa"
 
        self.model = MixtralModel(self.config)
        self.vocab_size = self.config.vocab_size
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        self.num_experts = self.config.num_local_experts
        self.num_experts_per_tok = self.config.num_experts_per_tok

        self.init_weights(args)

    def init_weights(self, args):
        if "/" in args.model:
            model_name = args.model.split("/")[1].lower()
        expanded_path = os.path.abspath(os.path.expanduser(os.path.join(self.path, model_name)))
        lm_head_path = os.path.join(expanded_path, "original/lm_head.weight")
        self.model.init_weights(expanded_path)
        self.lm_head.weight.data = torch.load(lm_head_path, map_location=self.config.device, weights_only=True)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Tuple:

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        # output = (logits,) + outputs[1:]
        return logits, outputs[1], outputs[2], outputs[3], outputs[4]
    
    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None):
        seq_len = input_ids.shape[1]
        past_key_values = DynamicCache()
        if attention_mask is None:
            attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=self.config.device)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.config.device).unsqueeze(0)
        cache_position = torch.arange(0, seq_len, dtype=torch.long, device=self.config.device)

        attn_total = 0
        mlp_total = 0
        dec_total = 0

        for i in range(32):
        # for i in tqdm(range(32), desc="Infer."):
            logits, past_key_values, attn_time, mlp_time, dec_time = self.forward(input_ids=input_ids, 
                                                attention_mask=attention_mask,
                                                position_ids=position_ids,
                                                past_key_values=past_key_values,
                                                cache_position=cache_position,
                                                )
            logits = F.softmax(logits, dim=-1)

            attn_total += attn_time
            mlp_total += mlp_time
            dec_total += dec_time

            # greedy search:
            input_ids = torch.argmax(logits, dim=-1)
            if i == 0:
                output = copy.deepcopy(input_ids[:, -1:])
            else:
                output = torch.cat((output, input_ids), dim=1)

            input_ids = input_ids[:, -1:]
            # if self.early_stopping and i > self.min_length and input_ids.item() == self.config.eos_token_id:
            #     return output
  
            # prepare for next decoding.
            position_ids = (position_ids[:, -1] + 1).unsqueeze(-1)
            cache_position = (cache_position[-1] + 1).unsqueeze(-1)
            attention_mask = torch.cat([attention_mask, torch.ones(1, 1, dtype=torch.long, device=self.config.device)], dim=-1)

                # profiler.step()
        return (output, dec_total/32, attn_total/32, mlp_total/32)

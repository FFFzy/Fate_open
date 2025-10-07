from transformers.configuration_utils import PretrainedConfig

class DeepseekConfig(PretrainedConfig):
    
    model_type = "deepseek"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        name="deepseek-moe-16b-base",
        vocab_size=102400,
        hidden_size=2048,
        intermediate_size=10944,
        moe_intermediate_size = 1408,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=16,
        n_shared_experts = 2,
        n_routed_experts = 64,
        num_experts_per_tok = 6,
        moe_layer_freq = 1,
        first_k_dense_replace = 1,
        norm_topk_prob = False,
        scoring_func = 'softmax',
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=100001,
        bos_token_id=100000,
        eos_token_id=100001,
        # tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        torch_dtype="bfloat16",
        int2_rate=0.5,
        # **kwargs,
    ):
        self.name = name
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.torch_dtype = torch_dtype
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.int2_rate = int2_rate

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")


def get_Deepseek_config(name):
    if "/" in name:
        name = name.split("/")[1]
    name = name.lower()

    if name == "deepseek-moe-16b-base":
        config = DeepseekConfig()
    else:
        raise ValueError(f"Invalid model name: {name}")
    return config

if __name__ == "__main__":
   config = get_Deepseek_config("deepseek-moe-16b-base")
   config.offload = True
   print(config.intermediate_size)
   print(config.eos_token_id)
   print(config.pad_token_id)
import torch
import GPUtil
from utils.utils import MB

def memory_cost_qwen(config, memory_budget):
    if memory_budget == 0:
        device = config.device
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**2)
        # allocated_memory = torch.cuda.memory_allocated(device) / (1024**2)
        gpus = GPUtil.getGPUs()
        target_gpu = next((gpu for gpu in gpus if f"cuda:{gpu.id}" == device), None)
        allocated_memory = target_gpu.memoryUsed
        
        memory_budget = total_memory - allocated_memory
    else:
        memory_budget = memory_budget * 1024

    # Reserve space for 1024
    seq_len = 1024
    num_hidden_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    moe_intermediate_size = config.moe_intermediate_size
    shared_expert_intermediate_size = config.shared_expert_intermediate_size
    num_experts = config.num_experts

    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_key_value_heads = config.num_key_value_heads
    max_position_embeddings = config.max_position_embeddings

    # layer
    embed = (vocab_size * hidden_size * 2 * 2)/MB
    attention = (2 * (hidden_size * num_heads * head_dim + 
                        hidden_size * num_key_value_heads * head_dim) 
                        * num_hidden_layers * 2)/MB
    attn_bias = ((num_heads * head_dim + 2 * num_key_value_heads * head_dim) * num_hidden_layers * 2)/MB
    rotaryEmbedding = ((head_dim // 2 + head_dim * max_position_embeddings * 2) * num_hidden_layers * 4)/MB
    norm = ((2 * hidden_size * num_hidden_layers + hidden_size) * 2)/MB

    shared_expert = (3 * hidden_size * shared_expert_intermediate_size * 2)/MB
    shared_expert_gate = (hidden_size * num_hidden_layers * num_hidden_layers * 2)/MB

    expert = (3 * hidden_size * moe_intermediate_size * 2)/MB
    expert_gate = (hidden_size * num_experts * num_hidden_layers * 2)/MB
    
    # during infer.
    kv = (2 * seq_len * num_hidden_layers * hidden_size * 2)/MB
    hidden = (2 * seq_len * hidden_size * 2)/MB

    # Remove frequently accessed layers
    available_memory = (memory_budget - embed - attention - attn_bias - rotaryEmbedding - norm -
                        shared_expert_gate - expert_gate - kv - hidden)
    available_memory = available_memory - (shared_expert * 24)

    # remove meta_data, prefetch
    meta_data = 0.3 * (3 * 60 + 3 + 4 + 2 + 2 + 2) * 24
    available_memory = available_memory - meta_data

    if available_memory < 0:
        assert False, f"{available_memory}, memory is not enough for dense."

    zero_scale = (2 * (moe_intermediate_size // 64 * hidden_size) * 1 * 4)/MB
    expert_int4 = expert/4 + 3 * zero_scale

    quan_map = {}
    offload_map = {}
    for i in range(num_hidden_layers):
        quan_map[i] = 0
        offload_map[i] = 0
    
    # all in mem.
    if available_memory > num_hidden_layers * 60 * expert:
        return (offload_map, quan_map)
    # all in mem. with int4
    elif available_memory > (num_hidden_layers * 60 * expert_int4):
        available_memory = available_memory - (num_hidden_layers * 60 * expert_int4)
        fp16_layers = available_memory // (60 * expert - 60 * expert_int4)
        for i in range(num_hidden_layers):
            if i == fp16_layers or i > fp16_layers:
                quan_map[i] = 4
        return (offload_map, quan_map)
    # offload
    else:
        cache_num = available_memory // expert_int4
        all_cache_layers = cache_num // 60
        # weather shallow can be all cached
        if all_cache_layers < 4:
            for i in range(num_hidden_layers):
                quan_map[i] = 4
                if i < all_cache_layers:
                    offload_map[i] = 0
                elif i == all_cache_layers:
                    offload_map[i] = 60 - (cache_num - 60 * (all_cache_layers))
                else:
                    offload_map[i] = 60
        else:
            cache_deep = (cache_num - 4 * 60) // (num_hidden_layers - 4)
            for i in range(num_hidden_layers):
                quan_map[i] = 4
                if i < 4:
                    offload_map[i] = 0
                else:
                    offload_map[i] = 60 - cache_deep
        return (offload_map, quan_map)

def memory_cost_mixtral(config, memory_budget):
    if memory_budget == 0:
        device = config.device
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**2)
        # allocated_memory = torch.cuda.memory_allocated(device) / (1024**2)
        gpus = GPUtil.getGPUs()
        target_gpu = next((gpu for gpu in gpus if f"cuda:{gpu.id}" == device), None)
        allocated_memory = target_gpu.memoryUsed
        
        memory_budget = total_memory - allocated_memory
    else:
        memory_budget = memory_budget * 1024

    # Reserve space for 1024
    seq_len = 1024
    num_hidden_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    moe_intermediate_size = config.intermediate_size
    num_experts = config.num_local_experts

    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_key_value_heads = config.num_key_value_heads
    max_position_embeddings = config.max_position_embeddings

    # layer
    embed = (vocab_size * hidden_size * 2 * 2)/MB
    attention = (2 * (hidden_size * num_heads * head_dim + 
                        hidden_size * num_key_value_heads * head_dim) 
                        * num_hidden_layers * 2)/MB
    # attn_bias = ((num_heads * head_dim + 2 * num_key_value_heads * head_dim) * num_hidden_layers * 2)/MB
    rotaryEmbedding = ((head_dim // 2 + head_dim * max_position_embeddings * 2) * num_hidden_layers * 4)/MB
    norm = ((2 * hidden_size * num_hidden_layers + hidden_size) * 2)/MB

    expert = (3 * hidden_size * moe_intermediate_size * 2)/MB
    expert_gate = (hidden_size * num_experts * num_hidden_layers * 2)/MB
    
    # during infer.
    kv = (2 * seq_len * num_hidden_layers * hidden_size * 2)/MB
    hidden = (2 * seq_len * hidden_size * 2)/MB

    # Remove frequently accessed layers
    available_memory = (memory_budget - embed - attention - rotaryEmbedding - norm -
                         - expert_gate - kv - hidden)

    # remove meta_data, prefetch
    meta_data = 0.3 * (3 * 8 + 3 + 4 + 2 + 2 + 2) * 32
    available_memory = available_memory - meta_data + 500

    if available_memory < 0:
        assert False, f"{available_memory}, memory is not enough for dense."

    zero_scale = (2 * (moe_intermediate_size // 64 * hidden_size) * 1 * 4)/MB
    expert_int4 = expert/4 + 3 * zero_scale

    quan_map = {}
    offload_map = {}
    for i in range(num_hidden_layers):
        quan_map[i] = 0
        offload_map[i] = 0
    
    # all in mem.
    if available_memory > num_hidden_layers * 8 * expert:
        return (offload_map, quan_map)
    # all in mem. with int4
    elif available_memory > (num_hidden_layers * 8 * expert_int4):
        available_memory = available_memory - (num_hidden_layers * 8 * expert_int4)
        fp16_layers = available_memory // (8 * expert - 8 * expert_int4)
        for i in range(num_hidden_layers):
            if i == fp16_layers or i > fp16_layers:
                quan_map[i] = 4
        return (offload_map, quan_map)
    # offload
    else:
        cache_num = available_memory // expert_int4
        all_cache_layers = cache_num // 8
        # weather shallow can be all cached
        if all_cache_layers < 4:
            for i in range(num_hidden_layers):
                quan_map[i] = 4
                if i < all_cache_layers:
                    offload_map[i] = 0
                elif i == all_cache_layers:
                    offload_map[i] = 8 - (cache_num - 8 * (all_cache_layers))
                else:
                    offload_map[i] = 8
        else:
            cache_deep = (cache_num - 4 * 8) // (num_hidden_layers - 4)
            for i in range(num_hidden_layers):
                quan_map[i] = 4
                if i < 4:
                    offload_map[i] = 0
                else:
                    offload_map[i] = 8 - cache_deep
        return (offload_map, quan_map)

def memory_cost_deepseek(config, memory_budget):
    if memory_budget == 0:
        device = config.device
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**2)
        # allocated_memory = torch.cuda.memory_allocated(device) / (1024**2)
        gpus = GPUtil.getGPUs()
        target_gpu = next((gpu for gpu in gpus if f"cuda:{gpu.id}" == device), None)
        allocated_memory = target_gpu.memoryUsed
        
        memory_budget = total_memory - allocated_memory
    else:
        memory_budget = memory_budget * 1024

    # Reserve space for 1024
    seq_len = 1024
    num_hidden_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    intermediate_size = config.intermediate_size
    moe_intermediate_size = config.moe_intermediate_size
    shared_expert_intermediate_size = 2 * config.moe_intermediate_size
    num_experts = config.n_routed_experts

    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_key_value_heads = config.num_key_value_heads
    max_position_embeddings = config.max_position_embeddings

    # layer
    embed = (2 * vocab_size * hidden_size * 2)/MB
    attention = (2 * (hidden_size * num_heads * head_dim + 
                        hidden_size * num_key_value_heads * head_dim) 
                        * num_hidden_layers * 2)/MB
    rotaryEmbedding = ((head_dim // 2 + head_dim * max_position_embeddings * 2) * num_hidden_layers * 4)/MB
    norm = ((2 * hidden_size * num_hidden_layers + hidden_size) * 2)/MB

    shared_expert = (3 * hidden_size * shared_expert_intermediate_size * 2)/MB
    dense_expert = (3 * hidden_size * intermediate_size * 2)/MB
    expert = (3 * hidden_size * moe_intermediate_size * 2)/MB
    expert_gate = (hidden_size * num_experts * num_hidden_layers * 2)/MB
    
    # during infer.
    kv = (2 * seq_len * num_hidden_layers * hidden_size * 2)/MB
    hidden = (2 * seq_len * hidden_size * 2)/MB

    # Remove frequently accessed layers
    available_memory = (memory_budget - embed - attention - rotaryEmbedding - norm - expert_gate - kv - hidden)
    available_memory = available_memory - (shared_expert * num_hidden_layers) - dense_expert

    # remove meta_data, prefetch
    meta_data = 0.3 * (3 * 64 + 3 + 4 + 2 + 2 + 2) * num_hidden_layers
    available_memory = available_memory - meta_data + 1400

    if available_memory < 0:
        assert False, f"{available_memory}, memory is not enough for dense."

    zero_scale = (2 * (moe_intermediate_size // 64 * hidden_size) * 1 * 4)/MB
    expert_int4 = expert/4 + 3 * zero_scale

    #
    quan_map = {}
    offload_map = {}
    for i in range(num_hidden_layers):
        quan_map[i] = 0
        offload_map[i] = 0
    num_hidden_layers = num_hidden_layers - 1
    # all in mem.
    if available_memory > num_hidden_layers * 64 * expert:
        return (offload_map, quan_map)
    # all in mem. with int4
    elif available_memory > (num_hidden_layers * 64 * expert_int4):
        available_memory = available_memory - (num_hidden_layers * 64 * expert_int4)
        fp16_layers = available_memory // (64 * expert - 64 * expert_int4)
        for i in range(1, num_hidden_layers + 1):
            if i == fp16_layers or i > fp16_layers:
                quan_map[i] = 4
        return (offload_map, quan_map)
    # offload
    else:
        cache_num = available_memory // expert_int4
        all_cache_layers = cache_num // 64
        # weather shallow can be all cached
        if all_cache_layers < 4:
            for i in range(1, num_hidden_layers+1):
                quan_map[i] = 4
                if i < all_cache_layers or i == all_cache_layers:
                    offload_map[i] = 0
                elif i == all_cache_layers + 1:
                    offload_map[i] = 64 - (cache_num - 64 * (all_cache_layers))
                else:
                    offload_map[i] = 64
        else:
            cache_deep = (cache_num - 4 * 64) // (num_hidden_layers - 4)
            for i in range(1, num_hidden_layers+1):
                quan_map[i] = 4
                if i < 5:
                    offload_map[i] = 0
                else:
                    offload_map[i] = 64 - cache_deep
        return (offload_map, quan_map)

def memory_cost_deepseekv2(config, memory_budget):
    if memory_budget == 0:
        device = config.device
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**2)
        # allocated_memory = torch.cuda.memory_allocated(device) / (1024**2)
        gpus = GPUtil.getGPUs()
        target_gpu = next((gpu for gpu in gpus if f"cuda:{gpu.id}" == device), None)
        allocated_memory = target_gpu.memoryUsed
        
        memory_budget = total_memory - allocated_memory
    else:
        memory_budget = memory_budget * 1024

    # Reserve space for 1024
    seq_len = 1024
    num_hidden_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    intermediate_size = config.intermediate_size
    moe_intermediate_size = config.moe_intermediate_size
    shared_expert_intermediate_size = 2 * config.moe_intermediate_size
    num_experts = config.n_routed_experts

    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    max_position_embeddings = config.max_position_embeddings
    kv_lora_rank = config.kv_lora_rank
    qk_rope_head_dim = config.qk_rope_head_dim
    q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    v_head_dim = config.v_head_dim

    # layer
    embed = (2 * vocab_size * hidden_size * 2)/MB
    attention = ((hidden_size * num_heads * q_head_dim + 
                    kv_lora_rank * qk_rope_head_dim +
                    kv_lora_rank * num_heads * (q_head_dim - qk_rope_head_dim + v_head_dim) +
                    hidden_size * num_heads * v_head_dim) 
                    * num_hidden_layers * 2)/MB
    attention_q_a_layernorm = ((kv_lora_rank * num_hidden_layers) * 2)/MB
    # 需要重算
    rotaryEmbedding = ((head_dim // 2 + head_dim * max_position_embeddings * 2) * num_hidden_layers * 4)/MB
    norm = ((2 * hidden_size * num_hidden_layers + hidden_size) * 2)/MB

    shared_expert = (3 * hidden_size * shared_expert_intermediate_size * 2)/MB
    dense_expert = (3 * hidden_size * intermediate_size * 2)/MB
    expert = (3 * hidden_size * moe_intermediate_size * 2)/MB
    expert_gate = (hidden_size * num_experts * num_hidden_layers * 2)/MB
    
    # during infer.
    kv = (2 * seq_len * num_hidden_layers * hidden_size * 2)/MB
    hidden = (2 * seq_len * hidden_size * 2)/MB

    # Remove frequently accessed layers
    available_memory = (memory_budget - embed - attention - attention_q_a_layernorm - rotaryEmbedding - norm - expert_gate - kv - hidden)
    available_memory = available_memory - (shared_expert * num_hidden_layers) - dense_expert
    print("available_memory: ", available_memory)

    # remove meta_data, prefetch
    meta_data = 0.3 * (3 * 64 + 3 + 4 + 2 + 2 + 2) * num_hidden_layers
    available_memory = available_memory - meta_data

    if available_memory < 0:
        assert False, f"{available_memory}, memory is not enough for dense."

    zero_scale = (2 * (moe_intermediate_size // 64 * hidden_size) * 1 * 4)/MB
    expert_int4 = expert/4 + 3 * zero_scale

    #
    quan_map = {}
    offload_map = {}
    for i in range(num_hidden_layers):
        quan_map[i] = 0
        offload_map[i] = 0
    num_hidden_layers = num_hidden_layers - 1
    # all in mem.
    if available_memory > num_hidden_layers * 64 * expert:
        return (offload_map, quan_map)
    # all in mem. with int4
    elif available_memory > (num_hidden_layers * 64 * expert_int4):
        available_memory = available_memory - (num_hidden_layers * 64 * expert_int4)
        fp16_layers = available_memory // (64 * expert - 64 * expert_int4)
        for i in range(1, num_hidden_layers + 1):
            if i == fp16_layers or i > fp16_layers:
                quan_map[i] = 4
        return (offload_map, quan_map)
    # offload
    else:
        cache_num = available_memory // expert_int4
        all_cache_layers = cache_num // 64
        # weather shallow can be all cached
        if all_cache_layers < 4:
            for i in range(1, num_hidden_layers + 1):
                quan_map[i] = 4
                if i < all_cache_layers:
                    offload_map[i] = 0
                elif i == all_cache_layers:
                    offload_map[i] = 64 - (cache_num - 64 * (all_cache_layers))
                else:
                    offload_map[i] = 64
        else:
            cache_deep = (cache_num - 4 * 64) // (num_hidden_layers - 4)
            for i in range(1, num_hidden_layers + 1):
                quan_map[i] = 4
                if i < 4:
                    offload_map[i] = 0
                else:
                    offload_map[i] = 64 - cache_deep
        return (offload_map, quan_map)

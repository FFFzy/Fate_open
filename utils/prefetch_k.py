import torch
import time
import os 
import json

from transformers import AutoTokenizer
from models.Qwen.modeling_qwen_moe_time import Qwen2MoeForCausalLM
from models.Mixtral.modeling_mixtral_time import MixtralForCausalLM
from models.DeepSeekMoE.modeling_deepseek_time import DeepseekForCausalLM

def get_prefetch_k(args, time_record=None):

    model_name = args.model
    if model_name == "Qwen/Qwen1.5-MoE-A2.7B":
        model = Qwen2MoeForCausalLM(args)
    elif model_name == "mistralai/Mixtral-8x7B-v0.1":
        model = MixtralForCausalLM(args)
    elif model_name == "deepseek-ai/deepseek-moe-16b-base":
        model = DeepseekForCausalLM(args)

    model_name = model_name.split("/")[1].lower()
    tokenizer = AutoTokenizer.from_pretrained(f"{args.path}/{model_name}/tokenizer")
        
    warmup_prompt = "Hey, are you conscious? Can you talk to me?"
    warmup_ids = tokenizer(warmup_prompt, return_tensors="pt").input_ids.to(args.device)
    model.eval()

    if time_record is None:
        time_record = {}

    time_record[model_name] = {}

    dec_total = 0
    attn_total = 0
    mlp_total = 0

    # print("Warmup for time recoed...")
    for i in range(10):
        model.generate(warmup_ids)

    print("Recording time...")
    # for i in tqdm(range(5), desc="Recording time."):
    for i in range(5):
        input = "Hey, are you conscious? Can you talk to me?"
        input_ids = tokenizer(input, return_tensors="pt").input_ids.to(args.device)

        (_, dec_time, attn_time, mlp_time) = model.generate(input_ids)

        dec_total += dec_time
        attn_total += attn_time
        mlp_total += mlp_time

    time_record[model_name]["decoder_layer_time"] = dec_total * 1000 / 5
    time_record[model_name]["attention_time"] = attn_total * 1000 / 5
    time_record[model_name]["mlp_time"] = mlp_total * 1000 / 5

    transfer_time_record = transfer_time(args)
    time_record[model_name]["transfer_time"] = transfer_time_record

    prefetch_k =  time_record[model_name]["decoder_layer_time"] // transfer_time_record
    prefetch_k = int(prefetch_k)
    time_record[model_name]["prefetch_k"] = prefetch_k

    os.makedirs("time_record", exist_ok=True)    
    hostname = os.popen('hostname').read().strip()
    with open(f"time_record/exec_transfer_time_on_{hostname}.json", "w", encoding="utf-8") as f:
        json.dump(time_record, f, indent=2)

    return prefetch_k


def pin_weights(weight):
    scale_zero = [weight['scale'].flatten(), weight['zero'].flatten()]
    sizes = [t.numel() for t in scale_zero]

    scale_zero = torch.cat(scale_zero, dim=0).pin_memory()

    return {
        'nbits': weight['nbits'],
        'shape': weight['shape'],
        'sizes': sizes,
        'W_q': weight['W_q'].pin_memory(),
        'scale_zero': scale_zero,
        'scale_shape': weight['scale'].shape,
        'zero_shape': weight['zero'].shape,
    }

def merge_scale_zero(weight):
    scale_zero = [weight['scale'].flatten(), weight['zero'].flatten()]
    sizes = [t.numel() for t in scale_zero]

    scale_zero = torch.cat(scale_zero, dim=0)

    return {
        'nbits': weight['nbits'],
        'shape': weight['shape'],
        'sizes': sizes,
        'W_q': weight['W_q'],
        'scale_zero': scale_zero,
        'scale_shape': weight['scale'].shape,
        'zero_shape': weight['zero'].shape,
    }

load_stream = torch.cuda.Stream()
# with torch.cuda.stream(prefetch_stream):
def load_from_cpu(weight, device, non_blocking=True):
    # weight['W_q'] = weight['W_q'].to(device, non_blocking=non_blocking)
    # weight['scale_zero'] = weight['scale_zero'].to(device, non_blocking=non_blocking)
    if non_blocking == True:
        with torch.cuda.stream(load_stream):
            return {
                'nbits': weight['nbits'],
                'shape': weight['shape'],
                'sizes': weight['sizes'],
                'scale_shape': weight['scale_shape'],
                'zero_shape': weight['zero_shape'],
                'W_q': weight['W_q'].to(device, non_blocking=non_blocking),
                'scale_zero': weight['scale_zero'].to(device, non_blocking=non_blocking),
            }
    else:
        return {
            'nbits': weight['nbits'],
            'shape': weight['shape'],
            'sizes': weight['sizes'],
            'scale_shape': weight['scale_shape'],
            'zero_shape': weight['zero_shape'],
            'W_q': weight['W_q'].to(device, non_blocking=non_blocking),
            'scale_zero': weight['scale_zero'].to(device, non_blocking=non_blocking),
        }
    # return weight

def back_to_cpu(weight):
    weight['W_q'] = weight['W_q'].to('cpu')
    weight['scale_zero'] = weight['scale_zero'].to('cpu')
    return weight

def cpu_to_gpu(down, up, gate, device):
    down = load_from_cpu(down, device, non_blocking=False)
    up = load_from_cpu(up, device, non_blocking=False)
    gate = load_from_cpu(gate, device, non_blocking=False)
    return down, up, gate

def gpu_to_cpu(down, up, gate):
    down = back_to_cpu(down)
    up = back_to_cpu(up)
    gate = back_to_cpu(gate)
    return down, up, gate

def transfer_time(args):
    model_name = args.model
    if "/" in model_name:
        name = model_name.split("/")[1]
    model_name = name.lower()

    expanded_path = os.path.abspath(os.path.expanduser(os.path.join(args.path, model_name)))
    if model_name == "mixtral-8x7b-v0.1":
        int4_down_path = os.path.join(expanded_path, "quantized/int4/model.layers.0.block_sparse_moe.experts.0.w1.weight")
        int4_up_path = os.path.join(expanded_path, "quantized/int4/model.layers.0.block_sparse_moe.experts.0.w2.weight")
        int4_gate_path = os.path.join(expanded_path, "quantized/int4/model.layers.0.block_sparse_moe.experts.0.w3.weight")
    else:
        int4_down_path = os.path.join(expanded_path, "quantized/int4/model.layers.1.mlp.experts.0.down_proj.weight")
        int4_up_path = os.path.join(expanded_path, "quantized/int4/model.layers.1.mlp.experts.0.up_proj.weight")
        int4_gate_path = os.path.join(expanded_path, "quantized/int4/model.layers.1.mlp.experts.0.gate_proj.weight")

    down = torch.load(int4_down_path, map_location='cpu', weights_only=True)
    up = torch.load(int4_up_path, map_location='cpu', weights_only=True)
    gate = torch.load(int4_gate_path, map_location='cpu', weights_only=True)    

    down = pin_weights(down)
    up = pin_weights(up)
    gate = pin_weights(gate)

    # warmup
    for i in range(3):
        down, up, gate = cpu_to_gpu(down, up, gate, args.device)
        down, up, gate = gpu_to_cpu(down, up, gate)

    transfer_time_total = 0
    for i in range(5):
        start = time.perf_counter()
        down = load_from_cpu(down, args.device, non_blocking=False)
        up = load_from_cpu(up, args.device, non_blocking=False)
        gate = load_from_cpu(gate, args.device, non_blocking=False)
        transfer_time_total += (time.perf_counter() - start)
        down, up, gate = gpu_to_cpu(down, up, gate)

    transfer_time_recoord = transfer_time_total * 1000 / 5

    return transfer_time_recoord

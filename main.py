import argparse
from utils.utils import str2bool
import os 
from transformers import AutoTokenizer
import time
import torch
from dataset import get_ChatGPT_prompts_inputs, get_gsm8k_inputs, get_openai_humaneval_inputs
from tqdm import tqdm
import json

from models.Qwen.modeling_qwen_moe import Qwen2MoeForCausalLM
from models.Mixtral.modeling_mixtral import MixtralForCausalLM
from models.DeepSeekMoE.modeling_deepseek import DeepseekForCausalLM
# from models.Qwen.modeling_qwen_moe_multiproc import Qwen2MoeForCausalLM
# from models.Qwen.modeling_qwen_moe_marlin import Qwen2MoeForCausalLM

def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-MoE-A2.7B", help="The model name.")
    parser.add_argument("--path", type=str, default="/root/workspace/Fate_before_hw/model_weights", help="The path to the model weights.")
    parser.add_argument("--early_stopping", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--memory_budget", type=int, default=0, help="GB")
    parser.add_argument("--device", type=str, default='cuda:0')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    model_name = args.model
    if model_name == "Qwen/Qwen1.5-MoE-A2.7B":
        model = Qwen2MoeForCausalLM(args)
    elif model_name == "mistralai/Mixtral-8x7B-v0.1":
        model = MixtralForCausalLM(args)
    elif model_name == "deepseek-ai/deepseek-moe-16b-base":
        model = DeepseekForCausalLM(args)

    model_name = model_name.split("/")[1].lower()
    tokenizer = AutoTokenizer.from_pretrained(f"{args.path}/{model_name}/tokenizer")

    model.eval()

    input_prompt = "Hey, are you conscious? Can you talk to me?"
    input_tokenizer = tokenizer(input_prompt, return_tensors="pt")
    input_ids = input_tokenizer.input_ids.to(args.device)
    attention_mask = input_tokenizer.attention_mask.to(args.device)

    # warmup
    for i in range(3):
        output_ids = model.generate(input_ids)

    start = time.perf_counter()
    (output, prefill_time, decoding_token) = model.generate(input_ids)
    end = time.perf_counter()

    print("latency = " + str(end-start))
    print("TTFT = ", prefill_time)
    print("decoding latency = ", (end - start - prefill_time))
    print("output_ids: ", output)

    outputs = tokenizer.batch_decode(output, skip_special_tokens=True)

    print(outputs)
    
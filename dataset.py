import datasets
import random
from typing import List, Optional, Tuple
import os
import torch
from transformers import AutoTokenizer
import json

def get_line_from_dataset(dataset, line):
    # there are some topics and None in dataset
    if len(dataset['train'][line]['text']) < 128:
        line += 1
        return get_line_from_dataset(dataset, line)
    return dataset['train'][line]['text']

def get_inputs():
    if not os.path.exists('datasets/wikitext-103-v1'):
        os.makedirs('datasets/wikitext-103-v1')
        # load from HF
        dataset = datasets.load_dataset('wikitext', 'wikitext-103-v1')
        dataset.save_to_disk('datasets/wikitext-103-v1')
    else:
        # load from local
        dataset = datasets.load_from_disk('datasets/wikitext-103-v1')
    
    # get_line = []
    # for _ in range(num_prompts):
    #     get_line.append(random.randint(0, 1801350-1))

    # inputs = ()
    # for line in get_line:
    #     input_ids = tokenizer(get_line_from_dataset(dataset, line), return_tensors="pt").input_ids
    #     inputs = inputs + (input_ids,) 
    # return inputs

# sum
def get_xsum_inputs():
    if not os.path.exists('datasets/xsum'):
        os.makedirs('datasets/xsum')
        # load from HF
        dataset = datasets.load_dataset('xsum')
        dataset.save_to_disk('datasets/xsum')
    else:
        # load from local
        dataset = datasets.load_from_disk('datasets/xsum')

    return dataset['validation']

# sum
def get_samsum_inputs():
    if not os.path.exists('datasets/samsum'):
        os.makedirs('datasets/samsum')
        # load from HF
        dataset = datasets.load_dataset('Samsung/samsum')
        dataset.save_to_disk('datasets/samsum')
    else:
        # load from local
        dataset = datasets.load_from_disk('datasets/samsum')

    return dataset['test']

# translate
def get_wmt_inputs():
    if not os.path.exists('datasets/wmt16'):
        os.makedirs('datasets/wmt16')
        # load from HF
        dataset = datasets.load_dataset("wmt16", "ro-en")
        dataset.save_to_disk('datasets/wmt16')
    else:
        # load from local
        dataset = datasets.load_from_disk('datasets/wmt16')

    return dataset['test']

# mmlu
def get_mmlu_inputs():
    if not os.path.exists('datasets/mmlu'):
        os.makedirs('datasets/mmlu')
        # load from HF
        dataset = datasets.load_dataset("lighteval/mmlu", "all")
        dataset.save_to_disk('datasets/mmlu')
    else:
        # load from local
        dataset = datasets.load_from_disk('datasets/mmlu')

    return dataset['test']

# gsm8k
def get_gsm8k_inputs():
    if not os.path.exists('datasets/gsm8k'):
        os.makedirs('datasets/gsm8k', exist_ok=True)
        # load from HF
        config = datasets.DownloadConfig(resume_download=True, max_retries=100)
        dataset = datasets.load_dataset("gsm8k", "main", download_config=config)
        dataset.save_to_disk('datasets/gsm8k')
    else:
        # load from local
        dataset = datasets.load_from_disk('datasets/gsm8k')

    return dataset['test']

# ChatGPT-prompts
def get_ChatGPT_prompts_inputs():
    if not os.path.exists('datasets/ChatGPT-prompts'):
        os.makedirs('datasets/ChatGPT-prompts')
        # load from HF
        dataset = datasets.load_dataset("MohamedRashad/ChatGPT-prompts")
        dataset.save_to_disk('datasets/ChatGPT-prompts')
    else:
        # load from local
        dataset = datasets.load_from_disk('datasets/ChatGPT-prompts')

    return dataset['train']

# openai_humaneval
def get_openai_humaneval_inputs():
    if not os.path.exists('datasets/openai_humaneval'):
        os.makedirs('datasets/openai_humaneval')
        # load from HF
        dataset = datasets.load_dataset("openai/openai_humaneval")
        dataset.save_to_disk('datasets/openai_humaneval')
    else:
        # load from local
        dataset = datasets.load_from_disk('datasets/openai_humaneval')

    return dataset['test']

def sample_sharegpt_requests(
        dataset_path,
        requests_num,
        tokenizer,
        modified_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    if modified_output_len is not None and modified_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    # Only keep the first two turns of each conversation.
    dataset = [subset for subset in dataset if len(subset["conversations"]) >= 2]
    dataset = [(subset["conversations"][0]["value"],
                subset["conversations"][1]["value"]) for subset in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # print(len(dataset))
    # print(dataset[0])
    # assert False

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        i = i + 2000
        if len(filtered_dataset) == requests_num:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        output = dataset[i][1]
        output_token_ids = tokenizer(output).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(output_token_ids
                         ) if modified_output_len is None else modified_output_len
        if prompt_len < 4 or output_len < 4:
            # Filter sequences that are too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Filter sequences that are too long.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))
    
    print(len(filtered_dataset))

    return filtered_dataset


if __name__ == "__main__":
    dataset_path = "datasets/ShareGPT_V3_unfiltered_cleaned_split.json"

    model_name = "Qwen/Qwen1.5-MoE-A2.7B"
    model_name = model_name.split("/")[1].lower()
    tokenizer = AutoTokenizer.from_pretrained(f"/mnt/fangzhy/model_weights/{model_name}/tokenizer")

    filtered_dataset = sample_sharegpt_requests(dataset_path, 50, tokenizer)
    print(filtered_dataset[0])

    with open('warmup_data.json', 'w', encoding='utf-8') as f:
        json.dump(filtered_dataset, f, indent=4)  # indent 参数用于格式化输出
    
    with open('warmup_data.json', 'r', encoding='utf-8') as f2:
        loaded_data = json.load(f2)
        print(loaded_data[0])




#    dataset = get_openai_humaneval_inputs()
# #    dataset = dataset['validation']
#    for i in range(len(dataset)):
#        if i < 5:
#         # print(dataset[i])
#         print(dataset[i])

   
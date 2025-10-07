import argparse
import glob
import os
from quantizer import quantize
# from layers.marlin_utils import marlin_quantize
from tqdm import tqdm

def download_Qwen_weights(model_name, path):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    import torch

    if "/" in model_name:
        model_name2 = model_name.split("/")[1].lower()
    print(model_name)

    # # from Hugging Face
    folder = snapshot_download(model_name, cache_dir=f"model_weights/{model_name2}/weights", force_download=True, resume_download=False, allow_patterns="*.safetensor")
    safetensor_files = glob.glob(os.path.join(folder, "*.safetensors"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()

    # load from local
    # weights_path = f"model_weights/qwen1.5-moe-a2.7b/weights/"
    # safetensor_files = glob.glob(os.path.join(weights_path, "*.safetensors"))

    path = os.path.join(path, f"{model_name}")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)
    ori_path = os.path.join(path, 'original')
    quan_path = os.path.join(path, 'quantized')
    # quan_int8_path = os.path.join(quan_path, 'int8')
    quan_int4_path = os.path.join(quan_path, 'int4')
    quan_int2_path = os.path.join(quan_path, 'int2')
    os.makedirs(ori_path, exist_ok=True)
    # os.makedirs(quan_int8_path, exist_ok=True)
    os.makedirs(quan_int4_path, exist_ok=True)
    os.makedirs(quan_int2_path, exist_ok=True)

    # with open('a.txt', 'w+') as txt_file:
    for safetensor_file in tqdm(safetensor_files, desc="Saving and quantizing"):
        state = load_file(safetensor_file)
    
        # for name, _ in tqdm(state.items(), leave=False):
        #     txt_file.write(f"{name}: {111}\n")

        for name, param in tqdm(state.items(), leave=False):

            param_path = os.path.join(ori_path, name)
            torch.save(param, param_path)

            if "expert" in name and "shared" not in name:
                param_int4 = quantize(param, 4)
                param_int2 = quantize(param, 2)

                param_int4_path = os.path.join(quan_int4_path, name)
                param_int2_path = os.path.join(quan_int2_path, name)
                torch.save(param_int4, param_int4_path)
                torch.save(param_int2, param_int2_path)

def download_Mixtral_weights(model_name, path):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    import torch

    if "/" in model_name:
        model_name2 = model_name.split("/")[1].lower()

    # # from Hugging Face
    folder = snapshot_download(model_name, cache_dir=f"model_weights/{model_name2}/weights", force_download=True, resume_download=False, allow_patterns="*.safetensor")
    safetensor_files = glob.glob(os.path.join(folder, "*.safetensors"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()

    # load from local
    # weights_path = os.path.join(path, f"{model_name}/weights")
    # weights_path = f"/mnt/fangzhy/model_weights/mixtral-8x7b-v0.1/weights"
    # safetensor_files = glob.glob(os.path.join(weights_path, "*.safetensors"))

    path = os.path.join(path, f"{model_name}")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)
    ori_path = os.path.join(path, 'original')
    quan_path = os.path.join(path, 'quantized')
    quan_int4_path = os.path.join(quan_path, 'int4')
    quan_int2_path = os.path.join(quan_path, 'int2')
    os.makedirs(ori_path, exist_ok=True)
    os.makedirs(quan_int4_path, exist_ok=True)
    os.makedirs(quan_int2_path, exist_ok=True)

    for safetensor_file in tqdm(safetensor_files, desc="Saving and quantizing"):
        state = load_file(safetensor_file)

        for name, param in tqdm(state.items(), leave=False):
            param_path = os.path.join(ori_path, name)
            torch.save(param, param_path)
            
            if "expert" in name:
                param_int4 = quantize(param, 4)
                param_int2 = quantize(param, 2)

                param_int4_path = os.path.join(quan_int4_path, name)
                param_int2_path = os.path.join(quan_int2_path, name)

                torch.save(param_int4, param_int4_path)
                torch.save(param_int2, param_int2_path)

def download_Deepseek_weights(model_name, path):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    import torch

    if "/" in model_name:
        model_name2 = model_name.split("/")[1].lower()

    # # from Hugging Face
    folder = snapshot_download(model_name, cache_dir=f"model_weights/{model_name2}/weights", force_download=True, resume_download=False,  allow_patterns="*.safetensor")
    safetensor_files = glob.glob(os.path.join(folder, "*.safetensors"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()

    # load from local
    # weights_path = f"model_weights/deepseek-moe-16b-base/weights"
    # safetensor_files = glob.glob(os.path.join(weights_path, "*.safetensors"))

    path = os.path.join(path, f"{model_name}")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)
    ori_path = os.path.join(path, 'original')
    quan_path = os.path.join(path, 'quantized')
    # quan_int8_path = os.path.join(quan_path, 'int8')
    quan_int4_path = os.path.join(quan_path, 'int4')
    quan_int2_path = os.path.join(quan_path, 'int2')
    os.makedirs(ori_path, exist_ok=True)
    # os.makedirs(quan_int8_path, exist_ok=True)
    os.makedirs(quan_int4_path, exist_ok=True)
    os.makedirs(quan_int2_path, exist_ok=True)

    for safetensor_file in tqdm(safetensor_files, desc="Saving and quantizing"):
        state = load_file(safetensor_file)

        for name, param in tqdm(state.items(), leave=False):

            param_path = os.path.join(ori_path, name)
            torch.save(param, param_path)

            if "expert" in name and "shared" not in name:
                param_int4 = quantize(param, 4)
                param_int2 = quantize(param, 2)

                param_int4_path = os.path.join(quan_int4_path, name)
                param_int2_path = os.path.join(quan_int2_path, name)
                torch.save(param_int4, param_int4_path)
                torch.save(param_int2, param_int2_path)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str)
#     parser.add_argument("--path", type=str, default="/mnt/fangzhy/model_weights")
#     args = parser.parse_args()

    # model = "Qwen/Qwen1.5-MoE-A2.7B"
    # download_Qwen_weights(model, args.path)

    # model = "mistralai/Mixtral-8x7B-v0.1"
    # download_Mixtral_weights(model, args.path)

    # model = "deepseek-ai/deepseek-moe-16b-base"
    # download_Deepseek_weights(model, args.path)
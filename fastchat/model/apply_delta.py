"""
Apply the delta weights on top of a base model.
即将delta模型中对应base模型中的层的权重复制给base模型
当内存小于10G时，使用apply_delta_low_cpu_mem；否则使用apply_delta
Usage:
python3 -m fastchat.model.apply_delta --base ~/model_weights/llama-7b --target ~/model_weights/vicuna-7b --delta lmsys/vicuna-7b-delta-v1.1
"""
import argparse
import gc
import glob
import json
import os
import shutil
import tempfile

from huggingface_hub import snapshot_download
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Union

GB = 1 << 30


def split_files(model_path:str, tmp_path:str, split_size:Union[int,float]):
    """将目标模型按照`state_dict`字典进行更细的划分
    划分标准为预定义的`split_size`,若一个或多个`state_dict[name]`
    的大小超过`split_size`，则存到本地的`tmp_path`路径下。
        model_path: 模型的hf repo-id或路径
        tmp_path: 分割后模型的存储路径
        split_size: 每个分割文件的大小
    """
    if not os.path.exists(model_path):
        model_path = snapshot_download(repo_id=model_path)
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    file_pattern = os.path.join(model_path, "pytorch_model-*.bin")
    files = glob.glob(file_pattern)

    part = 0
    try:
        for file_path in tqdm(files):
            state_dict = torch.load(file_path)
            new_state_dict = {}

            current_size = 0
            for name, param in state_dict.items():
                param_size = param.numel() * param.element_size()

                if current_size + param_size > split_size:
                    new_file_name = f"pytorch_model-{part}.bin"
                    new_file_path = os.path.join(tmp_path, new_file_name)
                    torch.save(new_state_dict, new_file_path)
                    current_size = 0
                    new_state_dict = None
                    gc.collect()
                    new_state_dict = {}
                    part += 1

                new_state_dict[name] = param
                current_size += param_size

            new_file_name = f"pytorch_model-{part}.bin"
            new_file_path = os.path.join(tmp_path, new_file_name)
            torch.save(new_state_dict, new_file_path)
            new_state_dict = None
            gc.collect()
            new_state_dict = {}
            part += 1
    except Exception as e:
        print(f"An error occurred during split_files: {e}")
        shutil.rmtree(tmp_path)
        raise


def apply_delta_low_cpu_mem(base_model_path, target_model_path, delta_path):
    """执行delta合并，作用机制是将delta模型对应base模型的层的权重复制给base模型，
    tokenizer和config仍使用delta模型的tokenizer和config.
    low_cpu_mem的意义在于事先将模型文件进行更小粒度的划分。
    base_model_path: base model的路径
    target_model_path: 合并后保存文件的路径
    delta_path: delta权重的路径
    """
    delta_tokenizer = AutoTokenizer.from_pretrained(delta_path, use_fast=False)
    delta_config = AutoConfig.from_pretrained(delta_path)

    if os.path.exists(target_model_path):
        shutil.rmtree(target_model_path)
    os.makedirs(target_model_path)

    split_size = 4 * GB
    # 首先分割base model和delta model，然后找到它们的文件名，
    # 取出划分后delta model的第一个文件记为delta_state_dict
    with tempfile.TemporaryDirectory() as tmp_base_path, tempfile.TemporaryDirectory() as tmp_delta_path:
        print(f"Split files for the base model to {tmp_base_path}")
        split_files(base_model_path, tmp_base_path, split_size)
        print(f"Split files for the delta weights to {tmp_delta_path}")
        split_files(delta_path, tmp_delta_path, split_size)

        base_pattern = os.path.join(tmp_base_path, "pytorch_model-*.bin")
        base_files = glob.glob(base_pattern)
        delta_pattern = os.path.join(tmp_delta_path, "pytorch_model-*.bin")
        delta_files = glob.glob(delta_pattern)
        delta_state_dict = torch.load(delta_files[0])

        print("Applying the delta")
        weight_map = {}
        total_size = 0
        # 针对每个划分后的base model的文件，先加载文件，如果针对该文件的每个层base_f_l，
        # 其层的name在不在delta model第一个文件的层字典中，则遍历delta模型的所有文件，
        # 直至找到包含base_f_l层的delta model文件，将该层的delta model权重，赋值给
        # base model对应该层，并将base model文件的文件名记录到weight_map中
        # 因此delta合并的意义就是将delta模型对应层的权重复制给base模型
        for i, base_file in tqdm(enumerate(base_files)):
            state_dict = torch.load(base_file)
            file_name = f"pytorch_model-{i}.bin"
            for name, param in state_dict.items():
                if name not in delta_state_dict:
                    for delta_file in delta_files:
                        delta_state_dict = torch.load(delta_file)
                        gc.collect()
                        if name in delta_state_dict:
                            break

                state_dict[name] += delta_state_dict[name]
                weight_map[name] = file_name
                total_size += param.numel() * param.element_size()
                gc.collect()
            torch.save(state_dict, os.path.join(target_model_path, file_name))

        with open(
            os.path.join(target_model_path, "pytorch_model.bin.index.json"), "w"
        ) as f:
            json.dump(
                {"weight_map": weight_map, "metadata": {"total_size": total_size}}, f
            )

    print(f"Saving the target model to {target_model_path}")
    delta_tokenizer.save_pretrained(target_model_path)
    delta_config.save_pretrained(target_model_path)


def apply_delta(base_model_path, target_model_path, delta_path):
    """执行delta合并，作用机制是将delta模型对应base模型的层的权重复制给base模型
        tokenizer和config仍使用delta模型的tokenizer和config.
    Args:
        base_model_path (_type_): base模型的路径
        target_model_path (_type_): 合并后模型存储的路径
        delta_path (_type_): delta模型的路径
    """
    print(f"Loading the delta weights from {delta_path}")
    delta_tokenizer = AutoTokenizer.from_pretrained(delta_path, use_fast=False)
    delta = AutoModelForCausalLM.from_pretrained(
        delta_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    print("Applying the delta")
    for name, param in tqdm(base.state_dict().items(), desc="Applying delta"):
        assert name in delta.state_dict()
        param.data += delta.state_dict()[name]

    print(f"Saving the target model to {target_model_path}")
    base.save_pretrained(target_model_path)
    delta_tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)
    parser.add_argument(
        "--low-cpu-mem",
        action="store_true",
        help="Lower the cpu memory usage. This will split large files and use "
        "disk as swap to reduce the memory usage below 10GB.",
    )
    args = parser.parse_args()

    if args.low_cpu_mem:
        apply_delta_low_cpu_mem(
            args.base_model_path, args.target_model_path, args.delta_path
        )
    else:
        apply_delta(args.base_model_path, args.target_model_path, args.delta_path)

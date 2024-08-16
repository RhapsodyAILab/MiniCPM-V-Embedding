# This script is designed for Batched Inference (Chat mode) with QWen-VL with streaming jsonl datasets.

# Input format: Jsonl, each line {"id": "xxx", "base64": "xxxxxxxx"}

import logging
import os
import sys
from pathlib import Path
import glob
import csv
import json
from tqdm import tqdm
import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from transformers import AutoConfig, AutoTokenizer, AutoModel, PreTrainedTokenizer
from transformers import HfArgumentParser, TrainingArguments
from contextlib import nullcontext
from torch.cuda import amp


# 加载jsonl图片数据集
import base64
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)

@dataclass
class InverseQueryMMArguments:
    model_name_or_path: str = field(
        default=None, 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: str = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    prompt_path: str = field(
        default=None,
        metadata={"help": "generation prompt path"},
    )
    data_path: str = field(
        default=None,
        metadata={"help": "Input data path"},
    )
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Output length"},
    )

# 将输入字典转换为目标格式的函数
def reshape_dict(input_dict):
    # 获取字典中第一个键的值的长度，假设所有列表长度一致
    length = len(next(iter(input_dict.values())))
    # 创建空列表，用于存储转换后的字典
    reshaped_list = []
    for i in range(length):
        # 对于每个索引，创建一个新字典，从每个键的列表中取出对应元素
        new_dict = {key: value[i] for key, value in input_dict.items()}
        reshaped_list.append(new_dict)
    return reshaped_list

# 检查文件大小
def is_file_size_over_limit(file_path, limit_mb):
    return os.path.exists(file_path) and os.path.getsize(file_path) > limit_mb * 1024 * 1024


def distributed_parallel_inference(
    data_path: None,
    model: None,
    tokenizer: None,
    data_output_dir: str,
    prompt: str,
    max_new_tokens: int,
    args: TrainingArguments
):
    if data_path.endswith('.jsonl'):
        pass
    elif data_path.endswith('.h5'):
        pass
    else:
        pass
    
    
    os.makedirs(data_output_dir, exist_ok=True)

    file_count = 1
    MAX_OUTPUT_SIZE_MB = 8 # 8MB

    with amp.autocast() if args.fp16 else nullcontext():
        with torch.no_grad():
            for batch in tqdm(dataloader, disable=args.process_index > 0):
                
                images_string = batch["image_base64"]
                
                msgs = [ [{'role': 'user', 'content': prompt}] ] * len(images_string) # [ChatML] * B

                # convert images(base64 strings from jsonl dataset) to PIL.Image
                images = [] # [ [ PIL.Image ] ] * B
                for base64_image_string in images_string: # this may involve sequential processing, waste of time
                    # 解码 base64 字符串
                    image_data = base64.b64decode(base64_image_string)
                    # 使用 BytesIO 从解码的二进制数据创建一个 stream
                    image_stream = BytesIO(image_data)
                    # 使用 Image.open 打开 stream，相当于打开一个文件
                    image = Image.open(image_stream) # jpg or png are all fine.
                    # 转换图像为 RGB
                    image = image.convert("RGB")
                    images.append(image)
                
                # for MiniCPM-V chat batch mode
                response = model.chat(
                    image_list=images,
                    msgs_list=msgs,
                    tokenizer=tokenizer,
                    sampling=True,
                    # temperature=0.8,
                    max_new_tokens=max_new_tokens
                )

                batch["response"] = response # create a new column in JSON blob representing the results
                del batch["image_base64"] # delete image base64 string to save space. (temperal)
                blobs = reshape_dict(batch) # reshape the blob

                current_file = f"{data_output_dir}/{args.local_rank}-{file_count}.jsonl"
                if is_file_size_over_limit(current_file, MAX_OUTPUT_SIZE_MB):
                    file_count += 1
                    current_file = f"{data_output_dir}/{args.local_rank}-{file_count}.jsonl"
                    
                with open(current_file, "a", encoding='utf-8') as f:
                    for blob in blobs:
                        print(blob)
                        f.write(json.dumps(blob, ensure_ascii=False))
                        f.write('\n')
                        print("-"*10)
                
                # del images
                # del blobs
                # del query
                # del response
                print("="*10)
    
    if args.world_size > 1:
        torch.distributed.barrier()

    return


def main():
    parser = HfArgumentParser((InverseQueryMMArguments, TrainingArguments))
    
    self_args, encoding_args = parser.parse_args_into_dataclasses()
    
    if os.path.exists(encoding_args.output_dir) and os.listdir(encoding_args.output_dir):
        if not encoding_args.overwrite_output_dir:
            logger.warning(
                f"Output directory ({encoding_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        else:
            # remove all files in the output directory
            if encoding_args.local_process_index == 0:
                for file in os.listdir(encoding_args.output_dir):
                    os.remove(os.path.join(encoding_args.output_dir, file))

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if encoding_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed inference: %s, 16-bits inference: %s",
        encoding_args.local_rank,
        encoding_args.device,
        encoding_args.n_gpu,
        bool(encoding_args.local_rank != -1),
        encoding_args.fp16,
    )

    with open(self_args.prompt_path, 'r') as f:
        prompt = f.read()
    assert prompt != "", "Prompt must be non-empty"

    logger.info(f"generation prompt: {prompt}")

    with open(os.path.join(self_args.model_name_or_path, "config.json"), "r") as f:
        config_json = json.loads(f.read())
        if "MiniCPM-Llama3-V-2_5" in config_json["_name_or_path"]:
            from openmatch.modeling.modeling_minicpmv_llama.modeling_minicpmv import MiniCPMV as model_cls
            from openmatch.modeling.modeling_minicpmv_llama.modeling_minicpmv import PreTrainedTokenizerFastWrapper as tokenizer_cls
        elif "MiniCPM-V-2.0" in config_json["_name_or_path"]:
            from openmatch.modeling.modeling_minicpmv.modeling_minicpmv import MiniCPMV as model_cls
            from openmatch.modeling.modeling_minicpmv.modeling_minicpmv import LlamaTokenizerWrapper as tokenizer_cls
        else:
            raise NotImplementedError("your model config arch is not supported")
    
    tokenizer = tokenizer_cls.from_pretrained(
        self_args.model_name_or_path
    )

    logging.info("1 - load dataset begin..")
    mm_doc_dataset = InferenceDataset.load(
        tokenizer=tokenizer,
        data_args=None,
        data_files=self_args.data_path,
        # full_tokenization=True,
        mode="raw",
        stream=True,
        batch_size=encoding_args.per_device_eval_batch_size,
        num_processes=encoding_args.world_size,
        process_index=encoding_args.process_index,
    )
    logging.info("1 - load dataset end..")

    logging.info("2 - load model begin..")
    model = model_cls.from_pretrained( # bf16 by default.
        self_args.model_name_or_path, 
        trust_remote_code=True, 
        # torch_dtype=torch.float16
        # config=config,
    )
    logging.info("2 - load model end..")

    # inference mode
    print(f"to encoding_args.device = {encoding_args.device}")
    model.to(encoding_args.device)
    model.eval()

    distributed_parallel_inference(
        dataset=mm_doc_dataset,
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        data_output_dir=encoding_args.output_dir,
        max_new_tokens=self_args.max_new_tokens,
        args=encoding_args,
    )

    if encoding_args.world_size > 1:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()






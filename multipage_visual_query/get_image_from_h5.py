import h5py
import os
import numpy as np
import base64
from PIL import Image
import io
import random
# import random
from collections import defaultdict

# 读取 .h5 文件
dataset_name = 'K12Textbook_visual'
output_directory = '/home/jeeves/xubokai/sampled_images_forocr'
h5_file_path = f'/mnt/data/user/tc_agi/xubokai/{dataset_name}/train_pid_image.h5'

# 不同数据集都有 train_pid_image.h5，可以cd 到/mnt/data/user/tc_agi/xubokai

# 以下数据集需要造query
# CuratedVisualPDF
# K12Textbook_textual
# K12Textbook_visual
# anna_visual
# icml_sampled
# manuallib_sampled
# nips_sampled

# 以下数据集不需要造query
# ChartQA
# arxivqa_new
# docvqa_mp
# infographicsqa
# plot_qa
# slidevqa


def random_chunk_length():
    # 定义长度和概率分布
    lengths = [4, 5, 6] # 4页 5页 6页采样概率
    probabilities = [0.2, 0.3, 0.5]
    return random.choices(lengths, probabilities, k=1)[0]

def sample_collections_random_chunks(data, N):
    # 解析数据并排序
    print(data[:10])
    unique_collection = set([])
    parsed_data = [('_'.join(x.split('_')[:-1]), int(x.split('_')[-1])) for x in data]
    sorted_data = sorted(parsed_data, key=lambda x: (x[0], x[1]))
    
    # 根据collectionname进行分组
    groups = defaultdict(list)
    for name, pid in sorted_data:
        groups[name].append(f"{name}_{pid}")
    
    # 将每个组分成随机长度的块
    chunks = []
    for group in groups.values():
        index = 0
        while index < len(group):
            chunk_length = random_chunk_length()
            if index + chunk_length <= len(group):
                chunks.append(group[index:index + chunk_length])
                index += chunk_length
            else:
                # 处理剩余页数不足以形成一个完整块的情况
                chunks.append(group[index:])
                break

    # 随机选择N个块
    samples = []
    if len(chunks) < N:
        return samples  # 如果没有足够的块可以返回，直接返回空列表
    
    sampled_chunks = random.sample(chunks, N)  # 随机选择N个块
    for chunk in sampled_chunks:
        samples.append(chunk)
    
    return samples

os.makedirs(output_directory, exist_ok=True)

output_subdir = os.path.join(output_directory, dataset_name)
os.makedirs(output_subdir, exist_ok=True)

with h5py.File(h5_file_path, 'r') as file:
    
    # 获取所有 group 名称
    print("reading group_names")
    group_names = list(file.keys())
    print("reading group_names ok")
    
    # 如果 group 数量多于 100，则随机采样 100 个
    # sampled_groups = random.sample(group_names, min(100, len(group_names)))
    
    N = 10
    
    # 调用函数
    samples = sample_collections_random_chunks(group_names, N)
    print(samples)

    for doc_name_list in samples:
        print(doc_name_list)
        for doc_name in doc_name_list:
            print(file[doc_name])
            # 读取 base64 编码的 image 数据
            image_data = file[doc_name]['image'][()]  # 假设 image 存储在每个 group 的 'image' 数据集中
            # base64 解码
            image_bytes = base64.b64decode(image_data)
            # 将 bytes 转换为 image
            image = Image.open(io.BytesIO(image_bytes))
            # 保存图片
            image.save(os.path.join(output_subdir, f'{doc_name}.png'))

print('所有图片已保存至:', output_subdir)

import json
import os, sys
import h5py
import base64
from PIL import Image
import io, re


def extract_number(s):
    match = re.match(r'(\d+)', s)
    return int(match.group(1)) if match else float('inf')

input_dir = '/home/jeeves/rag-v-common/query_cls_result/infographicsVQA_train_v1'
source_path = '/mnt/data/user/tc_agi/xubokai/infographicsqa/infographicsVQA_train_v1.0.json'
h5_path = '/mnt/data/user/tc_agi/xubokai/infographicsqa/train_pid_image.h5'
output_dir = '/home/jeeves/preprocessed_train_data/infographicsqa'
index = 0
batch_index = 0
line_cnt = 0
batch_size = 128

os.makedirs(output_dir, exist_ok=True)

# 先把所有jsonl合并成一个jsonl
all_json_path = os.path.join(input_dir, "all.jsonl")
with open(all_json_path, "w", encoding="utf-8") as all_json_file:
    for jsonl_name in sorted([f for f in os.listdir(input_dir) if f.endswith('.jsonl') and 'all' not in f], key=extract_number):
        if ("all" not in jsonl_name):
                jsonl_path = os.path.join(input_dir, jsonl_name)
                with open(jsonl_path, 'r', encoding="utf-8") as file:
                    for line in file:
                        all_json_file.write(line)

with h5py.File(h5_path) as h5_file, \
    open(source_path, 'r', encoding="utf-8") as source_file, \
    open(all_json_path, "r", encoding="utf-8") as all_json_file:
    data1 = json.load(source_file)
    data1 = data1['data']
    for line2 in all_json_file:
        data2 = json.loads(line2)
        if (data2['response'] == "A"):
            query = data2["query"]
            doc_id = data1[index]["image_local_name"]
            image = h5_file[f'{doc_id}//image']
            image = image[()]
            systhesize_data = {"index": index, "query": query, "pos": {"text": "", "image": image.decode('utf-8'), "instruction": ""}, "neg": []}
            json_line = json.dumps(systhesize_data)
            with open(os.path.join(output_dir, f'batch{batch_index}.jsonl'), 'a', encoding="utf-8") as output_file:
                output_file.write(json_line + '\n')
            line_cnt = line_cnt + 1
            if (line_cnt == batch_size):
                batch_index = batch_index + 1
                line_cnt = 0
        index = index + 1
                    
                
            
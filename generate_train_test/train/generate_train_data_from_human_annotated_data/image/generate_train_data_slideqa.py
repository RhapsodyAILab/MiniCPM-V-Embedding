import json
import os, sys
import h5py
import base64
from PIL import Image
import io
import random
import re


def extract_number(s):
    match = re.match(r'(\d+)', s)
    return int(match.group(1)) if match else float('inf')

def allocate_data(query_pool_dict, B=128):
    random.seed(42)
    piles = []
    current_pile = []
    used_query_current_pile = []
    dead_loop_count = 0

    while len(list(query_pool_dict.keys())) > 0:  # while pages are not exhausted
        if dead_loop_count == 1000:
            break

        query_id = random.choice(list(query_pool_dict.keys()))
        if query_id in used_query_current_pile:
            dead_loop_count += 1
            continue
        datas = query_pool_dict[query_id]
        data = datas.pop(0)
        current_pile.append(data)
        # Once append, check if bucket is full
        if len(current_pile) == B:
            piles.append(current_pile)
            current_pile = []
            used_query_current_pile = []

        used_query_current_pile.append(query_id)
        # Remove this page from dict if it is empty
        if len(datas) == 0:
            del query_pool_dict[query_id]

    return piles



input_dir = '/home/jeeves/rag-v-common/query_cls_result/slidevqa-train_query'
source_path = '/mnt/data/user/tc_agi/xubokai/slidevqa/train_formal.jsonl'
h5_path = '/mnt/data/user/tc_agi/xubokai/slidevqa/train_pid_image.h5'
output_dir = '/home/jeeves/train_data_image/slidevqa'
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

# 建立逆向索引 Dict[query, List[docid]]
doc_id_dict = {}
with open(source_path, "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        query = data["query"]
        if (query not in doc_id_dict):
            doc_id_dict[query] = []
        doc_id_dict[query].append(data["pos"])

# 建立data_pool
data_pool = {}
with h5py.File(h5_path) as h5_file, \
    open(all_json_path, "r", encoding="utf-8") as all_json_file:
    for line2 in all_json_file:
        data2 = json.loads(line2)
        if (data2['response'] == "A"):
            query = data2["query"]
            doc_ids = doc_id_dict[query]
            for doc_id in doc_ids:
                print(doc_id)
                try:
                    image = h5_file[f'{doc_id}//image']
                    image = image[()]
                except:
                    continue
                systhesize_data = {"index": index, "query": query, "pos": {"text": "", "image": image.decode('utf-8'), "instruction": ""}, "neg": []}
                json_line = json.dumps(systhesize_data)
            if (query not in data_pool):
                data_pool[query] = []
            data_pool[query].append(json_line)

piles = allocate_data(data_pool, batch_size)

for batch_index, batch_data in enumerate(piles):
    with open(os.path.join(output_dir, f'batch{batch_index}.jsonl'), 'w', encoding="utf-8") as output_file:
        for line in batch_data:
            output_file.write(line + '\n')
    
                
                
            
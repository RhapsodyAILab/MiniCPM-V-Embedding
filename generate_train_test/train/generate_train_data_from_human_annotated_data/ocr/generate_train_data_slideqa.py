import json
import os, sys
import h5py
import base64
from PIL import Image
import io, re
import argparse, random

def allocate_data(data_pool, batch_size=128):
    random.seed(42)
    piles = []
    current_pile = []
    used_current_pile = []
    dead_loop_count = 0

    while len(list(data_pool.keys())) > 0:  # while pages are not exhausted
        if dead_loop_count == 1000:
            break

        id = random.choice(list(data_pool.keys()))
        if id in used_current_pile:
            dead_loop_count += 1
            continue
        datas = data_pool[id]
        data = datas.pop(0)
        current_pile.append(data)
        # Once append, check if bucket is full
        if len(current_pile) == batch_size:
            piles.append(current_pile)
            current_pile = []
            used_current_pile = []

        used_current_pile.append(id)
        # Remove this page from dict if it is empty
        if len(datas) == 0:
            del data_pool[id]

    return piles

def extract_number(s):
    match = re.match(r'(\d+)', s)
    return int(match.group(1)) if match else float('inf')

batch_size = 128

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, required=True) # sqmd(Single Query Multiple Docs) or sdmq
    parser.add_argument('--doc_type', type=str, required=True) # ocr or image
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--ocr_type', type=str, default=None) # merge or preserve_layout
    parser.add_argument('--threshold', type=int, default = 0)
    args = parser.parse_args()
    dataset_type = args.dataset_type
    doc_type = args.doc_type
    dataset_name = args.dataset_name
    ocr_type = args.ocr_type
    
    threshold = args.threshold

    if (doc_type == 'ocr' and threshold == 0):
        raise Exception
    
    filter_dir = f'/home/jeeves/xubokai/query_cls_result/slidevqa-train_query'
    train_jsonl_path = f'/mnt/data/user/tc_agi/xubokai/slidevqa/train_formal.jsonl'
    h5_path = f'/mnt/data/user/tc_agi/xubokai/slidevqa/train_pid_image.h5'
    if (doc_type == 'ocr'):
        output_dir = f'/home/jeeves/train_data_{doc_type}/{ocr_type}/{dataset_name}'
    else:
        output_dir = f'/home/jeeves/train_data_{doc_type}/{dataset_name}'
    ocr_path = f'/home/jeeves/xubokai/captioner_output/slidevqa_train'
    
    # 首先根据filter_dir下的*.jsonl文件生成dict[query, is_local]
    filter_dict={}
    for filename in os.listdir(filter_dir):
        jsonl_path = os.path.join(filter_dir, filename)
        with open(jsonl_path, "r", encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                filter_dict[data["query"]] = data["response"]
    
    # 然后根据doc_type读取h5或者ocr结果建立dict[docid, result]
    if doc_type == 'ocr':
        ocr_result = {}
        for filename in os.listdir(ocr_path):
            jsonl_path = os.path.join(ocr_path, filename)
            with open(jsonl_path, "r", encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    for key, val in data.items():
                        if (ocr_type in key):
                            ocr_result[data['filename']] = val
                            break
    else:
        h5_file = h5py.File(h5_path)
                    
                
    # 然后根据dataset_type生成相应类型的data_pool
    data_pool = {}
    with open(train_jsonl_path, "r", encoding='utf-8') as train_jsonl_file:
        for line in train_jsonl_file:
            data = json.loads(line)
            query = data["query"]
            if (filter_dict[query] == 'B'):
                continue
            docid = data['pos']
            if dataset_type == 'sdmq':
                if (docid not in data_pool):
                    data_pool[docid] = []
            elif dataset_type == 'sqmd':
                if (query not in data_pool):
                    data_pool[query] = []
            if (doc_type == 'image'):
                image = h5_file[f'{docid}//image']
                image = image[()]
                systhesize_data = {
                    "query": {'text': query, 'image': None, 'instruction': 'Represent this query for retrieving relavant documents: '},
                    "pos": [{
                        "text": "",
                        "image": image.decode('utf-8'),
                        "instruction": ""
                    }],
                    "neg": []
                }
            else:
                try:
                    text = ocr_result[docid]
                except:
                    if (dataset_type == 'sdmq' and data_pool[docid] == []):
                        del data_pool[docid]
                    elif (dataset_type == 'sqmd' and data_pool[query] == []):
                        del data_pool[query]
                    continue
                if (len(text) < threshold):
                    if (dataset_type == 'sdmq' and data_pool[docid] == []):
                        del data_pool[docid]
                    elif (dataset_type == 'sqmd' and data_pool[query] == []):
                        del data_pool[query]
                    continue
                systhesize_data = {
                    "query": {'text': query, 'image': None, 'instruction': 'Represent this query for retrieving relavant documents: '},
                    "pos": [{
                        "text": text,
                        "image": None,
                        "instruction": ""
                    }],
                    "neg": []
                }
            systhesize_data = json.dumps(systhesize_data)
            if dataset_type == 'sdmq':
                data_pool[docid].append(systhesize_data)
            elif dataset_type == 'sqmd':
                data_pool[query].append(systhesize_data)
    piles = allocate_data(data_pool, batch_size)
    
    # 最后将allocate之后的piles写入output_dir
    os.makedirs(output_dir, exist_ok=True)
    for batch_index, batch_data in enumerate(piles):
        with open(os.path.join(output_dir, f'batch{batch_index}.jsonl'), 'w', encoding="utf-8") as output_file:
            print(f"Writing batch{batch_index}.jsonl")
            for line in batch_data:
                output_file.write(line + '\n')
    
    if (doc_type == 'image'):
        h5_file.close()
        
    
# 使用说明
# 给定dataset_type，doc_type, dataset_name, ocr_type, threshold参数（参数含义见下）
# 即可实现由ocr结果（或image.h5）, query文件构建batch_size = 128的训练数据(batch0.jsonl, batch1.jsonl, etc.)
# 最后还需通过openmatch/generate_train&test_data/make_train_batch_0703a.py脚本得到集合所有数据集处理结果的data.jsonl文件


import json
import os, sys
import h5py
import base64
from PIL import Image
import io, re
import argparse, random
import tarfile

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


def read_txt_from_tar(tar, txt_filename):
    # 从tar文件中找到目标txt文件
    member = tar.getmember(txt_filename)
    # 读取txt文件内容
    file = tar.extractfile(member)
    content = file.read().decode('utf-8')
    return content
    
def parse_text(response_string):
    # 提取JSON字符串部分
    start = response_string.find("{")
    end = response_string.rfind("}") + 1
    json_string = response_string[start:end]
    
    return json_string
                

batch_size = 128

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, required=True) # option: sqmd(Single Query Multiple Docs) or sdmq(Single Doc Multiple Queries)
    parser.add_argument('--doc_type', type=str, required=True) # option: ocr or image
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--ocr_type', type=str, default=None) # option: None(if image), merge, preserve_layout, pytesseract
    parser.add_argument('--threshold', type=int, default = 0) # 当doc_type为ocr时，将忽略len(text) < threshold的doc
    args = parser.parse_args()
    dataset_type = args.dataset_type
    doc_type = args.doc_type
    dataset_name = args.dataset_name
    ocr_type = args.ocr_type
    threshold = args.threshold
    
    if (doc_type == 'ocr' and threshold == 0):
        raise Exception
    
    tar_path = f'/home/jeeves/xubokai/{dataset_name}_train.tar.gz'
    train_txt_path = f'/mnt/data/user/tc_agi/xubokai/{dataset_name}/train.noquery.txt'
    h5_path = f'/mnt/data/user/tc_agi/xubokai/{dataset_name}/train_pid_image.h5'
    if (doc_type == 'ocr'):
        output_dir = f'/home/jeeves/train_data_{doc_type}/{ocr_type}/{dataset_name}'
    else:
        output_dir = f'/home/jeeves/xubokai/train_data_{doc_type}/{dataset_name}'
    ocr_path = f'/home/jeeves/xubokai/captioner_output_0717/{dataset_name}_train'
    # 根据ocr类型的不同，此处的ocr_path需要修改
    
    # 首先根据doc_type读取h5或者ocr结果建立dict[docid, result]
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
    
    #打开tar文件
    tar = tarfile.open(tar_path, 'r')
                    
    type1err = 0
    type2err = 0
    type3err = 0
    type4err = 0
    type5err = 0
                
    # 然后根据dataset_type生成相应类型的data_pool, 对于sqmd类型，data_pool的key为query，
    # data_pool[query]储存同一query的多个query_doc对，sdmq类型则相反
    data_pool = {}
    with open(train_txt_path, "r", encoding='utf-8') as train_txt_file:
        for line in train_txt_file:
            docid = line.strip('\n')
            filename = line.strip('\n') + '.txt'
            filename = f"data/{dataset_name}_train_inv_query_output/{filename}"
            try:
                to_be_parsed_content = read_txt_from_tar(tar, filename) # 读取tar中的txt文件
                data = json.loads(to_be_parsed_content)
                extracted_response = parse_text(data["response"])
            except:
                type5err += 1
                continue
            
            if extracted_response is None:
                # presumed that there is an error
                try:  # give it another chance, because it could be a json directly
                    response_parsed_json = json.loads(data["response"])
                except json.JSONDecodeError:
                    type3err += 1
                    print(f"{type3err} type 3 error")
                    continue
            else:
                try:
                    response_parsed_json = json.loads(extracted_response)
                except json.JSONDecodeError:
                    type4err += 1
                    print(f"{type4err} type 4 error")
                    continue

            if "result" not in response_parsed_json:
                type1err += 1
                print(f"{type1err} type 1 error")
                continue

            if not isinstance(response_parsed_json["result"], list):
                type2err += 1
                print(f"{type2err} type 2 error")
                continue

            if len(response_parsed_json["result"]) == 0:
                type2err += 1
                print(f"{type2err} type 2 error")
                continue
            
            # gpt4o是sdmq, 即一个txt文件中有对应于一个doc的多个query，因此需要遍历每个query
            for query_point in response_parsed_json['result']:
                try:
                    query = query_point['query']
                except:
                    try:
                        query = query_point['QUERY']
                    except:
                        continue
                if dataset_type == 'sdmq':
                    if (docid not in data_pool):
                        data_pool[docid] = []   
                elif dataset_type == 'sqmd':
                    if (query not in data_pool):
                        data_pool[query] = []
                if (doc_type == 'image'):
                    try:
                        image = h5_file[f'{docid}//image']
                        image = image[()]
                    except:
                        type5err += 1
                        continue
                    systhesize_data = {
                        'index': 0,
                        "query": {'text': query, 'image': None, 'instruction': 'Represent this query for retrieving relavant documents: '},
                        "pos": [{
                            "text": "",
                            "image": image.decode('utf-8'),
                            "instruction": ""
                        }],
                        "neg": []
                    }
                else:
                    text = ocr_result[docid]
                    
                    # 这里是为了防止最后allocate_data中出现data_pool[docid]或者data_pool[query] == []的情形
                    if (len(text) < threshold):
                        if (dataset_type == 'sdmq' and data_pool[docid] == []):
                            del data_pool[docid]
                        elif (dataset_type == 'sqmd' and data_pool[query] == []):
                            del data_pool[query]    
                        continue
                    systhesize_data = {
                        'index': 0,
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
        
    tar.close()
        
    print(f"type1err: {type1err}")
    print(f"type2err: {type2err}")
    print(f"type3err: {type3err}")
    print(f"type4err: {type4err}")
    print(f"type5err: {type5err}")
    
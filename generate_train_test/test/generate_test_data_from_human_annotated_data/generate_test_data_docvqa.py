import json
import os, sys
import h5py
import base64
from PIL import Image
import io, re, csv
import argparse, random


def extract_number(s):
    match = re.match(r'(\d+)', s)
    return int(match.group(1)) if match else float('inf')

def save_beir_qrels(qrels_list, qrels_path): # qrels: a list
    with open(qrels_path, 'w', newline='') as file:
        fieldnames = ["query-id", "corpus-id", "score"]
        tsvwriter = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')
        
        # Write the header
        tsvwriter.writeheader()
        
        # Write the rows
        for item in qrels_list:
            qid = item["qid"]
            pid = item["pid"]
            # using BEIR qrel format
            tsvwriter.writerow({"query-id": qid, "corpus-id": pid, "score": 1})
    
    return

batch_size = 128

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc_type', type=str, required=True) # ocr or image
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--ocr_type', type=str, default=None) # merge, preserve_layout, pytesseract
    args = parser.parse_args()
    doc_type = args.doc_type
    dataset_name = args.dataset_name
    ocr_type = args.ocr_type
    
    filter_dir = f'/home/jeeves/xubokai/query_cls_result/docvqa_mp-val_query'
    test_jsonl_path = f'/mnt/data/user/tc_agi/xubokai/{dataset_name}/val_formal.jsonl'
    h5_path = f'/mnt/data/user/tc_agi/xubokai/{dataset_name}/val_pid_image.h5'
    error_cnt = 0
    if (doc_type == 'ocr'):
        output_dir = f'/home/jeeves/test_data_{doc_type}/{ocr_type}/{dataset_name}'
    else:
        output_dir = f'/home/jeeves/test_data_{doc_type}/{dataset_name}'
    ocr_path = f'/home/jeeves/xubokai/ocr_output_0717_pytesseract_human_data/docvqa_mp-val_pid_image'
    docid_cnt = {}
    
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
                            ocr_result[data['docid']] = val
                            break
    else:
        h5_file = h5py.File(h5_path)
                    
                
    # 然后根据dataset_type生成相应类型的systhesize_data_qrels_list,
    # systhesize_data_docs_list, systhesize_data_queries_list
    
    systhesize_data_qrels_list = []
    systhesize_data_docs_list = []
    systhesize_data_queries_list = []
    
    
    with open(test_jsonl_path, "r", encoding='utf-8') as test_jsonl_file:
        for line in test_jsonl_file:
            data = json.loads(line)
            query = data["query"]
            try:
                if (filter_dict[query] == 'B'):
                    continue
            except:
                error_cnt += 1
                continue
            docid = data['pos']
            # if dataset_type == 'sdmq':
            #     if (docid not in data_pool):
            #         data_pool[docid] = []
            # elif dataset_type == 'sqmd':
            #     if (query not in data_pool):
            #         data_pool[query] = []
            
            # 产生query json
            if (docid not in docid_cnt):
                    docid_cnt[docid] = 1
            qid = docid + f'-{docid_cnt[docid]}'
            docid_cnt[docid] = docid_cnt[docid] + 1
            systhesize_data_query = {
                "text": query,
                'image': None,
                "id": qid,
                "answer": None
            }
                
            # 产生qrels json
            systhesize_data_qrels = {
                "qid": qid,
                "pid": docid
            }
                
            if (doc_type == 'image'):
                image = h5_file[f'{docid}//image']
                image = image[()]
                # systhesize_data = {
                #     "query": {'text': query, 'image': None, 'instruction': 'Represent this query for retrieving relavant documents: '},
                #     "pos": [{
                #         "text": "",
                #         "image": image.decode('utf-8'),
                #         "instruction": ""
                #     }],
                #     "neg": []
                # }
                
                # 产生doc json
                systhesize_data_doc = {
                    "text": None,
                    'image': image.decode('utf-8'),
                    "id": docid,
                    "answer": None
                }
                
            else:
                text = ocr_result[docid]
                systhesize_data_doc = {
                    "text": text,
                    'image': None,
                    "id": docid,
                    "answer": None
                }
                
            systhesize_data_docs_list.append(systhesize_data_doc)
            systhesize_data_qrels_list.append(systhesize_data_qrels)
            systhesize_data_queries_list.append(systhesize_data_query)
    
    # # 最后将allocate之后的piles写入output_dir
    # os.makedirs(output_dir, exist_ok=True)
    # for batch_index, batch_data in enumerate(piles):
    #     with open(os.path.join(output_dir, f'batch{batch_index}.jsonl'), 'w', encoding="utf-8") as output_file:
    #         print(f"Writing batch{batch_index}.jsonl")
    #         for line in batch_data:
    #             output_file.write(line + '\n')
    qrels_path = os.path.join(output_dir, f"{dataset_name}-eval-qrels.tsv")
    queries_path = os.path.join(output_dir, f"{dataset_name}-eval-queries.jsonl")
    docs_path = os.path.join(output_dir, f"{dataset_name}-eval-docs.jsonl")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 写入qrels文件
    save_beir_qrels(systhesize_data_qrels_list, qrels_path)
    
    # 写入queries文件
    with open(queries_path, 'w') as file:
        for line in systhesize_data_queries_list:
            file.write(json.dumps(line)+'\n')
            
    # 写入docs文件
    with open(docs_path, 'w') as file:
        for line in systhesize_data_docs_list:
            file.write(json.dumps(line)+'\n')
    
    print(f"error_cnt:{error_cnt}")
    
    if (doc_type == 'image'):
        h5_file.close()
        
    
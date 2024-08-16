# 使用说明
# 给定doc_type, dataset_name, ocr_type参数（参数含义见下）
# 即可实现由ocr结果（或image.h5）, query文件构建BEIR评测数据
# ======> xxx-eval-docs.jsonl, xxx-eval-queries.jsonl, xxx-eval-qrels.tsv


import json
import os, sys
import h5py
import base64
from PIL import Image
import io, re, csv
import argparse, random
import tarfile


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
    parser.add_argument('--doc_type', type=str, required=True) # option: ocr or image
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--ocr_type', type=str, default=None) # option: None(if image), merge, preserve_layout, pytesseract
    args = parser.parse_args()
    doc_type = args.doc_type
    dataset_name = args.dataset_name
    ocr_type = args.ocr_type
    
    tar_path = f'/home/jeeves/xubokai/{dataset_name}_val.tar.gz'
    test_txt_path = f'/mnt/data/user/tc_agi/xubokai/{dataset_name}/val.noquery.txt'
    h5_path = f'/mnt/data/user/tc_agi/xubokai/{dataset_name}/val_pid_image.h5'
    if (doc_type == 'ocr'):
        output_dir = f'/home/jeeves/xubokai/test_data_{doc_type}/{ocr_type}/{dataset_name}'
    else:
        output_dir = f'/home/jeeves/xubokai/test_data_{doc_type}/{dataset_name}'
    ocr_path = f'/home/jeeves/xubokai/captioner_output_0717/{dataset_name}_val'
    # 根据ocr类型的不同，此处的ocr_path需要修改
    docid_cnt = {}
    
    # 首先根据doc_type读取h5或者ocr结果建立dict[docid, result]，即docid到base64编码或者ocr结果的映射
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
                    
                
    # 然后生成systhesize_data_qrels_list, systhesize_data_docs_list和systhesize_data_queries_list
    
    systhesize_data_qrels_list = []
    systhesize_data_docs_list = []
    systhesize_data_queries_list = []
    
    type1err = 0
    type2err = 0
    type3err = 0
    type4err = 0
    type5err = 0
    
    
    with open(test_txt_path, "r", encoding='utf-8') as test_txt_file:
        for line in test_txt_file:
            docid = line.strip('\n')
            filename = line.strip('\n') + '.txt'
            filename = f"data/{dataset_name}_val_inv_query_output/{filename}"
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
            
            for query_point in response_parsed_json['result']:
                try:
                    query = query_point['query']
                except:
                    try:
                        query = query_point['QUERY']
                    except:
                        continue
            
                # 产生query json
                if (docid not in docid_cnt):
                        docid_cnt[docid] = 1
                qid = docid + f'-{docid_cnt[docid]}' # 分配qid
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
                systhesize_data_queries_list.append(systhesize_data_query)
                systhesize_data_qrels_list.append(systhesize_data_qrels)
                
                
            if (doc_type == 'image'):
                image = h5_file[f'{docid}//image']
                image = image[()]
                
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
    
    if (doc_type == 'image'):
        h5_file.close()
        
        
    tar.close()
        
    print(f"type1err: {type1err}")
    print(f"type2err: {type2err}")
    print(f"type3err: {type3err}")
    print(f"type4err: {type4err}")
    print(f"type5err: {type5err}")
        
    
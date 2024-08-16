API_BASE = "http://xxxx"
# OpenAI API Key
API_KEY = "sk-xxxx"

WORKERS=32

import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor
import requests
from datetime import datetime
import re
import threading
import json
import base64

lock = threading.Lock()

input_dir = sys.argv[1]
output_dir = sys.argv[2]
prompt_path = sys.argv[3]
os.makedirs(output_dir, exist_ok=True)

from PIL import Image
def save_base64_image(base64_data, save_file_path):
    image_data = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_data))
    image = image.convert('RGB')
    image.save(save_file_path)
    

HEADERS = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {API_KEY}"
}

PROMPT = open(prompt_path, 'r').read()


def parse_text(input_string):
    # 正则表达式匹配以```json开始，然后捕获[TEXT]部分，直到字符串以```结束
    pattern = r"```json\n(.*?)\n```"
    match = re.search(pattern, input_string, re.DOTALL)
    
    if not match:
        return None
    
    # 返回捕获的[TEXT]部分
    return match.group(1)
    

def process(data_path):

    max_retries = 20  # 设置最大重试次数
    retries = 0

    print(os.path.join(input_dir, data_path))
    
    data = json.loads(open(os.path.join(input_dir, data_path)).read())
    
    # print("why don't you read?")
    
    # retries = 0
    while retries < max_retries:
        print(f"{retries}/{max_retries}")
        try:
            img_base64 = data["image_base64"]
            usr_msg = [
                    {
                        "type": "text",
                        "text": PROMPT
                    },
                    {
                        "type": "text",
                        "text": f"This is document I provided:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}",
                            "detail": "auto"
                        }
                    } 
                ]

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": usr_msg
                    }
                ],
                "max_tokens": 2048
            }
            
            # 解码 base64 字符串
            # save_base64_image(img_base64, )
            
            imagebyte_to_save = base64.b64decode(img_base64)

            # # 将解码后的数据写入文件
            with open(f"{output_dir}/{data_path}.png", "wb") as file:
                file.write(imagebyte_to_save)

            response = requests.post(API_BASE, headers=HEADERS, json=payload)
            response_json = response.json()
            print(response_json)
            process_response(response_json, data, data_path)
            
            break  # 成功后退出循环
        
        except Exception as e:
            retries += 1
            print(f"failed retrying... ({retries}/{max_retries})")
            if retries == max_retries:
                print("max try, halting...")
                break

def process_response(response_json, sample_json, data_path):
    response_core = response_json["choices"][0]["message"]["content"]
    core_json_str = parse_text(response_core)
    core_json_json = json.loads(core_json_str)

    if core_json_json is None:
        print("parse failed, no json found.")
        raise Exception
    else:
        json_object = {
            "response": response_core,
            "filename": sample_json["docid"],
            "dataset": sample_json["dataset"]
        }
        
        with lock:
            with open(f"{output_dir}/{data_path}", 'w', encoding='utf-8') as file:
                file.write(json.dumps(json_object, ensure_ascii=False) + '\n')
            print("saved parsed response for reading.")

def list_txt_files(directory):
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    
    # print(txt_files)
    
    return txt_files

def find_additional_elements(a, b):
    a_set = set(a)
    b_set = set(b)
    unique_elements = list(a_set - b_set)
    return unique_elements

def main(): 
    input_data_paths = list_txt_files(input_dir)
    finished_data_paths = list_txt_files(output_dir)
    todos = find_additional_elements(input_data_paths, finished_data_paths)
    
    print(f"todos = {todos}")
    
    # 创建一个包含8个工作线程的线程池
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        # 将每个JSON文件路径提交到线程池中处理
        executor.map(process, todos)
    # process(todos[0])

if __name__ == "__main__":
    main()


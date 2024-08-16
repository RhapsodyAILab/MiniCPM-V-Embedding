
import os
import json
import sys

from concurrent.futures import ThreadPoolExecutor

import requests
from datetime import datetime
import re

OUTPUT_DIR = "./outputs"
OUTPUT_PARSED_DIR = OUTPUT_DIR + "_parsed"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_PARSED_DIR, exist_ok=True)



PROMPT = """Please read the following pages from a document. Your tasks are:
1. Discuss what each page is about, give a description for each page. Note that the page 1 and page 2 may not be relevant to the rest of pages.
2. Then, discuess in what scenario will this page be useful for solveing problems. 
3. Construct `realistic query` for recalling each page (imagine that someone has the same query in real world, and will recall the specific page with that query. Generate one `simple query` focusing on details, and one `hard query` with more considerations and association in it).
4. Compose a `decent problem` -- you know it's like a realistic problem you encountered during the exam, during your work, or just a real-world problem. It needs to be at least several lines of description and some sub questions. Try your best, if no proper `decent problem` is available, just leave blank is fine. Finally, I expect you can compose `decent problem` for at least 1/3 of pages.
5. Finally, view all of the pages (except page 1 and page 2) as a whole chunk, construct `general` description, usage scenario, easy query, hard query for it. 

Tips:
- Generated query should be concise and informative, and should be able to recall the page content. And query for different pages should be different to distinguish them.
- If a page contains visual elements like diagrams or tables, please take good use of them in your response. For description, you should describe the visual elements in detail to preserve more visual information.
- If a page contains no meaningful information or it is not feasible to generate a query for it, you can give empty response ('') for the query part.

Output your response using json format with this schema:
```json
{
    "page_1": {
        "description": "xxxx",
        "usage_scenario": "xxxx",
        "simple_query": "xxxxx",
        "hard_query": "xxxxx",
        "descent_problem": "xxxxx"
    },
    "page_2": {
        "description": "xxxx",
        "usage_scenario": "xxxx",
        "simple_query": "xxxxx",
        "hard_query": "xxxxx",
        "descent_problem": "xxxxx"
    }, ...
    "page_n": {
        "description": "xxxx",
        "usage_scenario": "xxxx",
        "simple_query": "xxxxx",
        "hard_query": "xxxxx",
        "descent_problem": "xxxxx"
    },
    "general": {
        "description": "xxxx",
        "usage_scenario": "xxxx",
        "simple_query": "xxxxx",
        "hard_query": "xxxxx",
        "descent_problem": "xxxxx"
    }
}
```
"""


def parse_text(input_string):
    # 正则表达式匹配以```json开始，然后捕获[TEXT]部分，直到字符串以```结束
    pattern = r"```json\n(.*?)\n```"
    match = re.search(pattern, input_string, re.DOTALL)
    
    if not match:
        return None
    
    # 返回捕获的[TEXT]部分
    return match.group(1)



def process(data_path):
    data = json.loads(open(data_path).read())
    
    
    
    max_retries = 5  # 设置最大重试次数
    retries = 0

    while retries < max_retries:
        print(f"{retries}/{max_retries} processing: {data['pdf_path']} + {data['_id']}")
        try:
            img_base64_list = data["images_base64"]
            global_generation_id = data["global_id"]

            text_prompt_msg = {
                "type": "text",
                "text": PROMPT
            }

            image_base64_msgs = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_str}"
                    }
                } for base64_str in img_base64_list
            ]

            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [text_prompt_msg, *image_base64_msgs]
                    }
                ],
                "max_tokens": 1024+2048
            }

            response = requests.post(API_BASE, headers=HEADERS, json=payload)
            response_json = response.json()
            
            with open(f'{OUTPUT_DIR}/{global_generation_id}.txt', 'w') as f:
                f.write(json.dumps(response_json, indent=4, ensure_ascii=False))

            # 如果到这一步没有异常发生，说明请求成功
            process_response(response_json, global_generation_id)
            break  # 成功后退出循环
        
        except Exception as e:
            retries += 1
            print(f"failed retrying... ({retries}/{max_retries})")
            if retries == max_retries:
                print("max try, halting...")
                break

def process_response(response_json, global_generation_id):
    # try:
    response_core = response_json["choices"][0]["message"]["content"]
    # if failed, raise error here
    
    # the following should not be considered as error
    
    try:
        core_json_str = parse_text(response_core)
        core_json_json = json.loads(core_json_str)

        if core_json_json is None:
            print("parse failed, no json found.")
        else:
            with open(f'{OUTPUT_PARSED_DIR}/{global_generation_id}.txt', 'w') as f:
                f.write(json.dumps(core_json_json, indent=4, ensure_ascii=False))
            print("saved parsed response for reading.")
    except:
        print(f"{global_generation_id} unknown process error")
    # except KeyError as e:
    #     print("parse failed due to KeyError")
    #     print(e)

    # except json.decoder.JSONDecodeError as e:
    #     print("parse failed due to JSONDecodeError")
    #     print(e)

    # except Exception as e:
    #     print("parse failed due to unknown error")
    #     print(e)


def list_json_files(directory):
    """列出指定目录下的所有JSON文件"""
    source = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    source_ids = [i.split('.')[0] for i in source]
    
    completed = [f for f in os.listdir(OUTPUT_PARSED_DIR) if f.endswith('.txt')]
    
    completed_ids = [i.split('.')[0] for i in completed]
    
    todo_ids = list(set(source_ids) - set(completed_ids))
    
    print(f"source = {len(source_ids)}, completed={len(completed_ids)}, todo={len(todo_ids)}")
    
    todo_abs = [os.path.join(directory, f"{i}.json") for i in todo_ids]
    return todo_abs


try:
    WORKERS=int(sys.argv[1])
except:
    WORKERS=4

# WORKERS=1

def main(directory):
    # 获取所有JSON文件路径
    json_files = list_json_files(directory)
    
    print(f"remaining: {len(json_files)} task.")
    
    # json_files.sort() # no sort is ok.

    # 创建一个包含8个工作线程的线程池
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        # 将每个JSON文件路径提交到线程池中处理
        executor.map(process, json_files)


if __name__ == "__main__":
    directory = './0506_all_split'
    main(directory)

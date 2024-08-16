import json
import os, sys
dataset_name = 'infographicsqa'
answer_dir = f'/mnt/data/user/tc_agi/xubokai/infographicsqa'
answer_path = os.path.join(answer_dir, 'infographicsVQA_val_v1.0_withQT.json')
input_file_dir = f'/mnt/data/user/tc_agi/xubokai/visualrag_evaldata/merged_for_evaluation/ocr/ocr_pytesseract/{dataset_name}'
input_file_path = os.path.join(input_file_dir, f'{dataset_name}-eval-queries.jsonl')
save_file_path = os.path.join(input_file_dir, f'{dataset_name}-eval-queries_with_answer.jsonl')

answer_dict = {}
cnt = 0
# with open(answer_path, 'r') as file:
#     for line in file:
#         item = json.loads(line)
#         query = item['query']
#         answer = item['answer']
#         # if (answer != 'A' and answer != 'B' and answer != 'C' and answer != 'D'):
#         #     cnt += 1
#         #     print(cnt)
#         #     print(f"answer:{answer}")
#         #     print('-------------')
#         # answer_dict[query] = (answer, options)
#         answer_dict[query] = answer
with open(answer_path, 'r') as file:
    datas = json.load(file)
    datas = datas['data']
    for data in datas:
        query = data['question']
        answer = data['answers']
        answer_dict[query] = answer

buffer = []
error_cnt = 0
with open(input_file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        query = data['text']
        try:
            data['answer'] = answer_dict[query]
        except:
            error_cnt += 1
            continue
        buffer.append(json.dumps(data))

with open(save_file_path, 'w') as file:
    for line in buffer:
        file.write(line+'\n')
        
print(f"error_cnt:{error_cnt}")
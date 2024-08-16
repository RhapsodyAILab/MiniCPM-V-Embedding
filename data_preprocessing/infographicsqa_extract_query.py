import json

input_jsonl_path = '/mnt/data/user/tc_agi/xubokai/infographicsqa/infographicsVQA_train_v1.0.json'
output_jsonl_path = '/home/jeeves/xubokai/infographicsVQA_train_v1.0-query.jsonl'

input_jsonl_path = '/mnt/data/user/tc_agi/xubokai/infographicsqa/infographicsVQA_test_v1.0.json'
output_jsonl_path = '/home/jeeves/xubokai/infographicsVQA_test_v1.0-query.jsonl'

input_jsonl_path = '/mnt/data/user/tc_agi/xubokai/infographicsqa/infographicsVQA_val_v1.0_withQT.json'
output_jsonl_path = '/home/jeeves/tcy6/infographicsVQA_val_v1.0_withQT.jsonl'


with open(input_jsonl_path, 'r') as f:
    data = json.loads(f.read())
    data = data['data']
    
global_cnt = 0
output_data = []
for data_ in data:
    json_ = {
        "id": global_cnt,
        "query": data_["question"]
    }
    global_cnt += 1
    output_data.append(json_)
    
with open(output_jsonl_path, 'w') as f:
    for data_ in output_data:
        f.write(json.dumps(data_)+'\n')


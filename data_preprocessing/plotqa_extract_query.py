import json

input_jsonl_path = '/mnt/data/user/tc_agi/xubokai/plot_qa/qa_pairs_V1_train_sampled.jsonl'
output_jsonl_path = '/home/jeeves/xubokai/plotqa-qa_pairs_V1_train_sampled-query.jsonl'

input_jsonl_path = '/mnt/data/user/tc_agi/xubokai/plot_qa/qa_pairs_V1_test_sampled.jsonl'
output_jsonl_path = '/home/jeeves/xubokai/plotqa-qa_pairs_V1_test_sampled-query.jsonl'


global_cnt = 0
data = []
with open(input_jsonl_path, 'r') as f:
    for line in f:
        data_ = json.loads(line)
        json_ = {
            "id": global_cnt,
            "query": data_["question_string"]
        }
        data.append(json_)
        
        global_cnt += 1
    
with open(output_jsonl_path, 'w') as f:
    for data_ in data:
        f.write(json.dumps(data_)+'\n')

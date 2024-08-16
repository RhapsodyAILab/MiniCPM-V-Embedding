import json

json_path = '/home/jeeves/xubokai/ChartQA_Dataset/train/train_human.json'
output_jsonl_path = '/home/jeeves/xubokai/ChartQA_Dataset/train/chartqa-train_human-query.jsonl'

json_path = '/home/jeeves/xubokai/ChartQA_Dataset/test/test_human.json'
output_jsonl_path = '/home/jeeves/xubokai/ChartQA_Dataset/test/chartqa-test_human-query.jsonl'


with open(json_path, 'r') as f:
    data = json.loads(f.read())

global_counter = 0

output_data = []
for data_ in data:
    data_ = {
        "id": global_counter,
        "query": data_["query"],
    }
    global_counter += 1

    output_data.append(data_)


with open(output_jsonl_path, 'w') as f:
    for data in output_data:
        f.write(json.dumps(data)+'\n')

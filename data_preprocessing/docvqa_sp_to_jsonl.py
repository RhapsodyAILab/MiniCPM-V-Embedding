import json

in_file = 'train_v1.0_withQT.json'
out_file = 'train_v1.0_withQT.jsonl'

in_file = 'train.json'
out_file = 'train.jsonl'


# 读取 JSON 文件
with open(in_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取 data 键的内容
data_list = data.get('data', [])

# 将每个列表项写入 JSONL 文件
with open(out_file, 'w', encoding='utf-8') as f:
    for item in data_list:
        json_line = json.dumps(item, ensure_ascii=False)
        f.write(json_line + '\n')

# import json
# a=json.load(open('train_v1.0_withQT.json'))
# q = [i['question'] for i in a['data'][:50]]
# 
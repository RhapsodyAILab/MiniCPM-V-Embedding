import json
import random

def sample_qa_pairs(json_path, output_path, sample_size=100000, random_seed=42):
    # 读取 JSON 文件
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # 获取 qa_pairs 列表
    qa_pairs = data['qa_pairs']
    
    # 设置随机种子并采样
    random.seed(random_seed)
    sampled_qa_pairs = random.sample(qa_pairs, sample_size)
    
    # 将采样结果保存为 JSON Lines 文件
    with open(output_path, 'w') as file:
        for qa_pair in sampled_qa_pairs:
            file.write(json.dumps(qa_pair) + '\n')

# 使用示例
json_path = 'qa_pairs_V1_train.json'
output_path = 'qa_pairs_V1_train_sampled.jsonl'
sample_size = 100000

json_path = 'qa_pairs_V1_test.json'
output_path = 'qa_pairs_V1_test_sampled.jsonl'
sample_size = 20000

sample_qa_pairs(json_path, output_path, sample_size, 42)


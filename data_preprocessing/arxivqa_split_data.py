import json
import random

# Function to read JSONL file
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

# Function to write JSONL file
def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Function to split data
def split_data(data, train_ratio=0.8, seed=42):
    random.seed(seed)
    random.shuffle(data)
    train_size = 30000
    eval_size = 10000
    train_data = data[:train_size]
    eval_data = data[train_size:train_size+eval_size]
    return train_data, eval_data

# Load the original data
file_path = './arxivqa.jsonl'  # Replace with actual file path
data = read_jsonl(file_path)

# Split the data
train_data, eval_data = split_data(data)

# Write the split data to files
train_file_path = './train.jsonl'
eval_file_path = './test.jsonl'

write_jsonl(train_file_path, train_data)
write_jsonl(eval_file_path, eval_data)

train_file_path, eval_file_path


import json
import random

def process_jsonl(input_file, output_file, batch_size=128):
    with open(input_file, 'r') as infile:
        # 读取全部数据
        data = infile.readlines()

    # 将数据分成batch
    batches = []
    for i in range(0, len(data) - len(data) % batch_size, batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)

    # 打乱这些batch
    random.shuffle(batches)
    print(f"Number of batches:{len(batches)}")

    # 将打乱的batch写入新的jsonl文件
    with open(output_file, 'w') as outfile:
        for batch in batches:
            for item in batch:
                outfile.write(item)

if __name__ == "__main__":
    input_file = '/mnt/data/user/tc_agi/xubokai/visualrag_traindata/all_data/image/data_without_shuffle.jsonl'  # 替换为你的输入文件路径
    output_file = '/mnt/data/user/tc_agi/xubokai/visualrag_traindata/all_data/image/data.jsonl'  # 替换为你想要输出的文件路径

    process_jsonl(input_file, output_file)

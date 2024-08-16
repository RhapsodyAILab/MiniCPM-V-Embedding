import json
import os
import random

batch_size = 128

def count_lines_in_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for _ in file)

base_dir = '/home/jeeves/train_data_ocr/caption'

output_dir = '/home/jeeves/xubokai/train_data_0718a_ocr_captioner_gpt4o_data'

os.makedirs(output_dir, exist_ok=True)

sub_datasets = os.listdir(base_dir)

output_jsonl_path = os.path.join(output_dir, 'data.jsonl')
output_metadata_path = os.path.join(output_dir, 'metadata.json')

all_batches = []
batch_dataset_map = []

for sub_dataset in sub_datasets:
    
    sub_datasets_abs = os.path.join(base_dir, sub_dataset)
    
    batches = os.listdir(sub_datasets_abs)
    batches = [b for b in batches if b.endswith('.jsonl')]
    
    batches_abs = [os.path.join(sub_datasets_abs, b) for b in batches if b.endswith('.jsonl')]
    
    
    batches_abs = [b for b in batches_abs if count_lines_in_jsonl(b) == batch_size]
    
    all_batches.extend(batches_abs)
    batch_dataset_map.extend([sub_dataset]*len(batches_abs))


random.seed(42)
random.shuffle(all_batches)
random.seed(42)
random.shuffle(batch_dataset_map)

for i in range(len(batch_dataset_map)):
    print(i+1, batch_dataset_map[i])

# exit(0)

with open(output_jsonl_path, 'w') as f_out:
    cnt = 0
    data_cnt = 0
    for b in all_batches:
        cnt += 1
        print(f"batch {cnt} - processing {b}")
        with open(b, 'r') as f_in:
            for line in f_in:
                data_cnt += 1
                f_out.write(line)

with open(output_metadata_path, 'w') as f_out:
    metadata = {
        'length': data_cnt,
        'description': 'human annotated data for visual rag training, no hard negative, bs = 128.',
        'created_time': '2024-07-08'
    }
    f_out.write(
        json.dumps(metadata)
    )

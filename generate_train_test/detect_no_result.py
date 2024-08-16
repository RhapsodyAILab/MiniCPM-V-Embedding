import argparse
import os, sys, json

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset_name', type=str, required=True)
    argparser.add_argument('--threshold', type=int, required=True)
    args = argparser.parse_args()
    dataset_name = args.dataset_name
    threshold = args.threshold
    detect_dir = f'/home/jeeves/train_data_ocr/merge/{dataset_name}'
    
    for jsonl_name in os.listdir(detect_dir):
        jsonl_path = os.path.join(detect_dir, jsonl_name)
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                result = data['pos'][0]['text']
                if (len(result) < threshold):
                    print(result)
                    print('------------')
            
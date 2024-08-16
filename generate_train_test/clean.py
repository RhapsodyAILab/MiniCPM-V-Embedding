import json
from tqdm import tqdm

def check_and_clean_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        valid_lines = []
        invalid_lines = []
        lines = file.readlines()

        for line_num, line in enumerate(tqdm(lines, desc="Processing lines"), start=1):
            try:
                data = json.loads(line)
                
                # Check if 'index' exists
                if 'index' not in data:
                    data['index'] = 0
                    line = json.dumps(data) + '\n'

                # Check if 'query/text' is a string
                query_text = data.get('query', {}).get('text')
                if not isinstance(query_text, str):
                    print(f"Invalid data at line {line_num}: 'query/text' is not a string")
                    invalid_lines.append(line_num)
                    continue

                # Check 'pos' field conditions
                pos_items = data.get('pos', [])
                if not pos_items or not isinstance(pos_items, list):
                    print(f"Invalid data at line {line_num}: 'pos' is not a list or empty")
                    invalid_lines.append(line_num)
                    continue

                pos_item = pos_items[0]  # Assuming we only check the first item
                pos_text = pos_item.get('text')
                pos_image = pos_item.get('image')
                
                if (pos_text and pos_image) or (not pos_text and not pos_image):
                    print(f"Invalid data at line {line_num}: 'pos/text' and 'pos/image' both present or both empty")
                    invalid_lines.append(line_num)
                    continue

                if not (isinstance(pos_text, str) or isinstance(pos_image, str)):
                    print(f"Invalid data at line {line_num}: 'pos/text' or 'pos/image' is not a string")
                    invalid_lines.append(line_num)
                    continue


                # If all checks passed, add line to valid_lines
                valid_lines.append(line)

            except json.JSONDecodeError as e:
                print(f"JSON decoding error at line {line_num}: {e}")
                invalid_lines.append(line_num)
                
    return valid_lines, invalid_lines

def save_cleaned_data(valid_lines, output_file):
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for line in valid_lines:
            out_file.write(line)

if __name__ == "__main__":
    input_file_path = '/mnt/data/user/tc_agi/xubokai/visualrag_traindata/all_data/image/data_without_clean_and_shuffle.jsonl'
    output_file_path = '/mnt/data/user/tc_agi/xubokai/visualrag_traindata/all_data/image/data_without_shuffle.jsonl'
    
    valid_data, invalid_lines = check_and_clean_jsonl(input_file_path)
    
    print(f"Total invalid lines: {len(invalid_lines)}")
    save_cleaned_data(valid_data, output_file_path)
    print(f"Cleaned data saved to {output_file_path}")

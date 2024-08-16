import os
import json
import base64

source_dir = './jsonl'
destination_dir = './all_pdfs'

# 如果目标目录不存在，则创建它
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# 获取源目录中的所有 JSONL 文件
jsonl_files = [file for file in os.listdir(source_dir) if file.endswith('.jsonl')]

for jsonl_file in jsonl_files:
    jsonl_path = os.path.join(source_dir, jsonl_file)
    print(f"processing {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file):
            print(f"-- processing {line_num}")
            # 解析 JSON 行
            data = json.loads(line)
            
            # 获取 PDF 的 base64 编码数据
            pdf_base64 = data.get('pdf_data')
            if pdf_base64:
                # 解码 base64 数据
                pdf_data = base64.b64decode(pdf_base64)
                
                # 生成 PDF 文件名
                pdf_filename = f"{os.path.splitext(jsonl_file)[0]}_{line_num}.pdf"
                pdf_path = os.path.join(destination_dir, pdf_filename)
                
                # 保存 PDF 文件
                with open(pdf_path, 'wb') as pdf_file:
                    pdf_file.write(pdf_data)

print(f"Successfully extracted PDF files to {destination_dir}")


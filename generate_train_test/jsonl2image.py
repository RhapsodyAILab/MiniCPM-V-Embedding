import json
import random
import base64
from PIL import Image
from io import BytesIO

import os

# 读取JSONL文件并随机抽取10行
file_path = 'data.jsonl'
file_path = '/home/jeeves/train_data_image/plot_qa/batch0.jsonl'

output_dir = 'decoded_images_0701a_arxiv'
output_dir = 'decoded_images_0701a_plotqa'

os.makedirs(output_dir, exist_ok=True)

with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# selected_lines = random.sample(lines, 10)
selected_lines = lines

# 提取query键的文本内容并保存到单独的Markdown文件
for index, line in enumerate(selected_lines):
    data = json.loads(line)
    query = data['query']
    
    markdown_content = f"# Query {index + 1}\n\n{query}"
    md_file_path = os.path.join(output_dir, f'query_{index + 1}.md')
    
    with open(md_file_path, 'w', encoding='utf-8') as md_file:
        md_file.write(markdown_content)
    
    # 提取pos键的第一个元素中的image并保存为图片文件
    if 'pos' in data and len(data['pos']) > 0:
        image_base64 = data['pos'][0]['image']
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        image_file_path = os.path.join(output_dir, f'image_{index + 1}.png')
        image.save(image_file_path)
        
        # 在Markdown文件中添加图片引用
        with open(md_file_path, 'a', encoding='utf-8') as md_file:
            md_file.write(f"\n\n![Image {index + 1}](./{image_file_path})")

print("Tasks completed successfully!")

import os
import random
import shutil

# 定义源目录和目标目录
source_dir = '/mnt/data/user/tc_agi/user/xubokai/ocrdata/pdf_manuallib'
destination_dir = '/home/jeeves/xubokai/data/manuallib50'

# 如果目标目录不存在，则创建它
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# 获取源目录中的所有文件
files = os.listdir(source_dir)

# 设置随机种子
random.seed(42)

# 随机选择50个文件
sampled_files = random.sample(files, 50)

# 复制文件到目标目录
for file in sampled_files:
    print(f"copying {file}")
    shutil.copy(os.path.join(source_dir, file), destination_dir)

print(f"Successfully copied {len(sampled_files)} files to {destination_dir}")

import os
import json
import base64
import pandas as pd
import h5py


# from IPython import embed

# 指定Parquet文件所在的目录和文件路径
directory = './test'
output_dir = './'
metadata_file = 'test.jsonl'
h5_path = 'test_pid_image.h5'

# directory = './train'
# output_dir = './'
# metadata_file = 'train.jsonl'
# h5_path = 'train_pid_image.h5'

# os.makedirs(output_dir, exist_ok=True)


def process_file(filename, h5f):
    metadata = []
    # 确定完整的文件路径
    file_path = os.path.join(directory, filename)
    
    # 读取Parquet文件
    df = pd.read_parquet(file_path)
    cnt = 0
    # 处理DataFrame中的每一行
    for idx in range(len(df)):
        cnt += 1
        # print(cnt)
        
        document_df = df['caption_images'][idx]
        
        document_num_figs = document_df.__len__()
        
        print(f"#Figs in this document -> {document_num_figs}")
        
        # embed()
        
        for fig_id in range(document_num_figs):
            fig_dict = document_df[fig_id]
            fig_caption = fig_dict['caption']
            
            subfig_num = fig_dict['cil_pairs'].__len__()
            
            if subfig_num > 1:
                print(f"subfiger {subfig_num} > 1, skipped")
                continue # we can't use graph with >1 subfigs
            else:
                print("only 1 figure, used")
            
            fig_byte = fig_dict['cil_pairs'][0]['image']['bytes']
            
            # 将字节字符串转换为base64编码
            base64_string = base64.b64encode(fig_byte).decode('utf-8')

            doc_id = f"{filename}_{idx}_{fig_id}"
            
            # 保存到metadata.json
            metadata.append({
                "query": fig_caption,
                "doc_id": doc_id,
                "source": f"{filename}_{idx}"
            })

            # 存储图片到HDF5
            group = h5f.create_group(doc_id)
            group.create_dataset('image', data=base64_string)

    return metadata

# 获取所有Parquet文件
files = [f for f in os.listdir(directory) if f.endswith('.parquet')]

all_metadata = []

# 打开HDF5文件并处理每个文件
with h5py.File(h5_path, 'w') as h5f:
    for file in files:
        print(f" =============== processing {file} ===============")
        metadata_ = process_file(file, h5f)
        all_metadata.extend(metadata_)


# 写入metadata.json
print()
with open(metadata_file, 'w') as metaf:
    for item in all_metadata:
        metaf.write(json.dumps(item, ensure_ascii=False) + '\n')

print("All data processed and saved")


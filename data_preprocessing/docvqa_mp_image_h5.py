import tarfile
import random
import os
from PIL import Image
import io
import base64
import json

import h5py


# train Image 数据
h5_path = 'train_pid_image.h5'
metadata_path = 'train.json'

# test Image 数据
h5_path = 'val_pid_image.h5'
metadata_path = 'val.json'

# 原始 Image
image_source_path = 'images.tar.gz'

with open(metadata_path, 'r') as f:
    train_metadata = json.load(f)

image_path_2_docId = {}

for i in range(len(train_metadata['data'])):
    data = train_metadata['data'][i]
    # image_suffix = data['image'].split('/')[-1]
    for j in data['page_ids']:
        docId = str(j)+'.jpg'
        image_path_2_docId[docId] = 1

print(f"len of image_path_2_docId = {image_path_2_docId}")

with h5py.File(h5_path, 'w') as f:
    with tarfile.open(image_source_path, 'r:gz') as tar:
        cnt = 0
        print("begin iterating gzip")
        for member in tar.getmembers():
            print(member.name)
            # print(cnt)
            cnt += 1
            if member.isfile():
                # print(member.name)
                # input('>')
                suffix = member.name.split('/')[-1]
                # suffix = member.name
                if suffix in image_path_2_docId:
                    docId = suffix
                    # docId
                    print(cnt, "docid", docId, member.name)
                    fileobj = tar.extractfile(member)
                    
                    if fileobj is not None:
                        file_byte = fileobj.read()
                        file_base64_string = base64.b64encode(file_byte).decode('utf-8')
                        
                        # 创建 HDF5 文件并写入数据
                        group = f.create_group(docId)
                        group.create_dataset('image', data=file_base64_string)



# pip install h5py==3.8.0

# with h5py.File('/mnt/data/user/tc_agi/docvqa_sp/train_pid_image.h5', 'r') as f:
#     # image = f[doc_id_to_read]['image']
#     image = f['1891']['image']
#     # 现在 image 变量中存储了对应 doc-id 的图像数据
#     print(image[()])
#     # image = f['1892']['image']
#     # # 现在 image 变量中存储了对应 doc-id 的图像数据
#     # print(image[()])
#     # image = f['1893']['image']
#     # # 现在 image 变量中存储了对应 doc-id 的图像数据
#     # print(image[()])


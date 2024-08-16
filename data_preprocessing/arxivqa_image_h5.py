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
metadata_path = 'train.jsonl'

# test Image 数据
h5_path = 'test_pid_image.h5'
metadata_path = 'test.jsonl'

# 原始 Image
image_source_path = 'images.tar.gz'

train_metadata = []
with open(metadata_path, 'r') as f:
    for line in f:
        train_metadata.append(json.loads(line))

print("metadata load ok")

images = {}
for i in train_metadata:
    images[i['image'].split('/')[-1]] = 1


with h5py.File(h5_path, 'w') as f:
    with tarfile.open(image_source_path, 'r:gz') as tar:
        cnt = 0
        for member in tar.getmembers():
            if member.isfile():
                suffix = member.name.split('/')[-1]
                if suffix in images:
                    # docId = image_path_2_docId[suffix]
                    docId = suffix
                    if cnt % 1000 == 0:
                        print(cnt, "docid", docId, member.name)
                    fileobj = tar.extractfile(member)
                    
                    if fileobj is not None:
                        # print("writing fileobj")
                        file_byte = fileobj.read()
                        file_base64_string = base64.b64encode(file_byte).decode('utf-8')
                        
                        # 创建 HDF5 文件并写入数据
                        group = f.create_group(docId)
                        group.create_dataset('image', data=file_base64_string)
            
            cnt += 1



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


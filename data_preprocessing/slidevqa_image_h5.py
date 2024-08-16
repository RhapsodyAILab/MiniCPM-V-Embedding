import tarfile
import random
import os
from PIL import Image
import io
import base64
import json

import h5py
import zipfile

# train Image 数据
h5_path = 'train_pid_image.h5'
metadata_path = 'train.jsonl'
image_source_path = 'slidevqa_train_image.zip'


# test Image 数据
# h5_path = 'test_pid_image.h5'
# metadata_path = 'test.jsonl'
# image_source_path = 'slidevqa_test_image.zip'

train_metadata = []
with open(metadata_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        train_metadata.append(data)

image_path_2_docId = {}

for i in range(len(train_metadata)):
    data = train_metadata[i]
    # image_suffix = data['image'].split('/')[-1]
    for j in data['image_urls']:
        # print(j)
        # print('j', j)
        elements = j.split('/')[-3:]
        
        assert len(elements) == 3
        docId =  elements[0] + '_' + elements[1] + '__' + elements[2]
        # print('docId', docId)
        # if docId is None:
        #     print(j)
        #     input('found none>')
        # print(docId)
        # assert len(docId) == 2
        # docId = '/'.join(docId)
        # print(docId)
        # input('>')
        # assert docId not in image_path_2_docId
        # input('>')
        image_path_2_docId[docId] = 1

print(f"len of image_path_2_docId = {image_path_2_docId}")

input('>')

# with h5py.File(h5_path, 'w') as f:
#     with tarfile.open(image_source_path, 'r:gz') as tar:
    
#         cnt = 0
#         print("begin iterating gzip")
#         for member in tar.getmembers():
#             print(member.name)
#             # print(cnt)
#             cnt += 1
#             if member.isfile():
#                 # print(member.name)
#                 # input('>')
#                 suffix = member.name.split('/')[-1]
#                 # suffix = member.name
#                 if suffix in image_path_2_docId:
#                     docId = suffix
#                     # docId
#                     print(cnt, "docid", docId, member.name)
#                     fileobj = tar.extractfile(member)
                    
#                     if fileobj is not None:
#                         file_byte = fileobj.read()
#                         file_base64_string = base64.b64encode(file_byte).decode('utf-8')
                        
#                         # 创建 HDF5 文件并写入数据
#                         group = f.create_group(docId)
#                         group.create_dataset('image', data=file_base64_string)



with h5py.File(h5_path, 'w') as f:
    with zipfile.ZipFile(image_source_path, 'r') as zip_file:
        cnt = 0
        print("begin iterating zip file")
        
        for member in zip_file.infolist():
            print(member.filename)
            cnt += 1
            if not member.is_dir():
                # suffix = member.filename.split('/')[-1]
                # suffix = member.filename
                
                suffix = member.filename.split('/')[-2:]
                # print("suffix1", suffix)
                assert len(suffix) == 2
                suffix = '__'.join(suffix)
                
                # print("suffix2", suffix)
                # input('>')
                
                if suffix in image_path_2_docId:
                    # docId = image_path_2_docId[suffix]
                    docId = suffix
                    print(cnt, "docid", docId, member.filename)
                    
                    with zip_file.open(member) as file:
                        file_byte = file.read()
                        file_base64_string = base64.b64encode(file_byte).decode('utf-8')
                        
                        # Create HDF5 file and write data
                        
                        # if docId in f:
                            
                        #     print(f"!!!! member.filename={member.filename} suffix={suffix}")
                        #     input(f"found duplicate {docId} >")
                        #     image_content = f[docId]['image'][()]
                        #     assert image_content == file_base64_string
                            
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


import tarfile
import random
import os
from PIL import Image
import io
import base64
import json
import zipfile
import h5py


def convert_to_png_and_base64(fileobj):
    try:
        # img = Image.open(fileobj)
        # img = img.convert("RGB")  # 确保图像是 RGB 格式
        # buffer = io.BytesIO()
        # img.save(buffer, format="PNG", dpi=(100, 100))
        # img_str = base64.b64encode(fileobj).decode('utf-8')
        
        file_content = fileobj.read()
        img_str = base64.b64encode(file_content).decode('utf-8')

        return img_str
    except Exception as e:
        print(f"Error converting: {e}")
        return None


# train Image 数据
h5_path = 'visual_train_pid_image.h5'
metadata_path = './raw/500K_train.json.visual.images.json'

h5_path = 'textual_train_pid_image.h5'
metadata_path = './raw/500K_train.json.textual.images.json'

h5_path = 'visual_test_pid_image.h5'
metadata_path = './raw/500K_test.json.visual.images.json'

h5_path = 'textual_test_pid_image.h5'
metadata_path = './raw/500K_test.json.textual.images.json'

# test Image 数据
# h5_path = 'test_pid_image.h5'
# metadata_path = 'test.json'

# 原始 Image
image_source_path = './raw/DocBank_500K_ori_img.zip'

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# image_path_2_docId = {}

# for i in range(len(train_metadata['data'])):
#     data = train_metadata['data'][i]
#     # image_suffix = data['image'].split('/')[-1]
#     for j in data['page_ids']:
#         docId = str(j)+'.jpg'
#         image_path_2_docId[docId] = 1

print(f"metadata = {metadata}")


image_path_abs = ['DocBank_500K_ori_img/' + i for i in metadata]


with h5py.File(h5_path, 'w') as f:
    # with tarfile.open(image_source_path, 'r:gz') as tar:
    with zipfile.ZipFile(image_source_path, 'r') as zipf:
        cnt = 0
        # print("begin iterating gzip")
        for member_, member in zip(metadata, image_path_abs):
        # for member in tar.getmembers():
            # print(member.name)
            # print(cnt)
            
            if cnt % 1000 == 0:
                print(cnt, member_, member)
            
            cnt += 1
            
            with zipf.open(member) as fileobj:
                img_base64 = convert_to_png_and_base64(fileobj)
                
                # 创建 HDF5 文件并写入数据
                group = f.create_group(member_)
                group.create_dataset('image', data=img_base64)

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


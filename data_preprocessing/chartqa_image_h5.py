import os
import h5py
import base64

def save_pngs_to_hdf5_as_base64(directory_path, hdf5_path):
    # 创建 HDF5 文件
    with h5py.File(hdf5_path, 'w') as h5f:
        # 遍历目录中的所有文件
        for filename in os.listdir(directory_path):
            if filename.endswith('.png'):
                # 获取文件路径和 doc_id
                file_path = os.path.join(directory_path, filename)
                # doc_id = os.path.splitext(filename)[0]
                doc_id = filename
                
                print(doc_id)
                
                # 读取 PNG 文件并转换为 base64 字符串
                with open(file_path, 'rb') as img_file:
                    img_data = img_file.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                # 创建一个以 doc_id 命名的组
                group = h5f.create_group(doc_id)
                
                # 将图像数据存储在组中
                group.create_dataset('image', data=img_base64)

# 使用示例
directory_path = './png'
# hdf5_path = 'train_pid_image.h5'
hdf5_path = 'test_pid_image.h5'

save_pngs_to_hdf5_as_base64(directory_path, hdf5_path)


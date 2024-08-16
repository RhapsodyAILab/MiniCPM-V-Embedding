import os
import h5py
import base64
import pandas as pd

def save_parquet_to_hdf5(directory_path, split, hdf5_path):
    file_list = sorted([f for f in os.listdir(directory_path) if f.endswith('.parquet') and f.startswith(split)])
    row_id = 0
    
    with h5py.File(hdf5_path, 'w') as h5f:
        for filename in file_list:
            print(f"processing {filename}")
            file_path = os.path.join(directory_path, filename)
            df = pd.read_parquet(file_path)

            for _, row in df.iterrows():
                doc_id = str(row_id)
                img_data = row['image']['bytes']
                img_base64 = base64.b64encode(img_data).decode('utf-8')

                group = h5f.create_group(doc_id)
                group.create_dataset('image', data=img_base64)

                row_id += 1

# split = 'train'
# directory_path = './data'
# hdf5_path = 'train_pid_image.h5'

split = 'train'
directory_path = './data'
hdf5_path = 'test_pid_image.h5'

save_parquet_to_hdf5(directory_path, split, hdf5_path)

import h5py
import json
import base64
import os

def hdf5_to_jsonl(txt_path, hdf5_path, output_dir):
    # 读取txt文件中的groupname
    with open(txt_path, 'r') as file:
        groupnames = [line.rstrip('\n') for line in file]

    data = []
    cnt = 0
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        for groupname in groupnames:
            cnt += 1
            print(f"{cnt} fetching {groupname}")
            
            if groupname in hdf5_file:
                image_data = hdf5_file[groupname]['image'][()]
                # pdf_data = hdf5_file[groupname]['pdf'][()]
                image_base64 = image_data.decode('utf-8')
                
                # image_base64 = image_data
                
                record = {
                    'dataset': hdf5_path,
                    'docid': groupname,
                    'image_base64': image_base64
                }
                
                # 将数据写入jsonl文件
                with open(os.path.join(output_dir, f"{groupname}.txt"), 'w') as f:
                    f.write(json.dumps(record) + '\n')
                
            else:
                print(f"Group {groupname} not found in HDF5 file.")
    
    

base_path = "/mnt/data/user/tc_agi/xubokai"
output_base_dir = "/home/jeeves/makedata_inputs"


# dataset_name = "K12Textbook_visual"
# split_name = "train"

# dataset_name = "icml_sampled"
# split_name = "train"

# dataset_name = "icml_sampled"
# split_name = "val"

dataset_name = "manuallib_sampled"
split_name = "train"

dataset_name = "manuallib_sampled"
split_name = "val"



dataset_name = "nips_sampled"
split_name = "val"


dataset_name = "nips_sampled"
split_name = "train"

dataset_name = "K12Textbook_textual"
split_name = "train"

dataset_name = "K12Textbook_textual"
split_name = "val"

dataset_name = "CuratedVisualPDF"
split_name = "val"

# dataset_name = "anna_visual"
# split_name = "val"


index_path = f"{base_path}/{dataset_name}/{split_name}.noquery.txt"
h5_path = f"{base_path}/{dataset_name}/{split_name}_pid_image.h5"
output_dir = f'{output_base_dir}/{dataset_name}_{split_name}_input_jsons'

os.makedirs(output_base_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

hdf5_to_jsonl(index_path, h5_path, output_dir)

import os
import json
import h5py

def process_jsonl_files(split, input_dir, output_file):
    # 创建一个 HDF5 文件
    group_names = []
    with h5py.File(output_file, 'w') as h5f:
        # 遍历目录中的所有 JSONL 文件
        for file_name in os.listdir(input_dir):
            print(f"processing {file_name}")
            if file_name.endswith('.jsonl'):
                file_path = os.path.join(input_dir, file_name)
                
                with open(file_path, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        doc_id = data.pop('doc_id')  # 获取 doc_id 作为索引
                        
                        # 创建一个以 doc_id 命名的组
                        group = h5f.create_group(doc_id)
                        
                        group_names.append(doc_id)
                        
                        # 将每个 key-value 对存储在组中
                        for key, value in data.items():
                            group.create_dataset(key, data=value)
    with open(f"{split}_pid.txt", "w") as f:
        for pid in group_names:
            f.write(pid+'\n')
    

if __name__ == "__main__":
    
    if os.path.exists('./train_pdf_with_metadata_rendered'):
        print("=======processing train images=======")
        input_directory = './train_pdf_with_metadata_rendered'
        output_hdf5_file = 'train_pid_image.h5'
        split = 'train'
        process_jsonl_files(split, input_directory, output_hdf5_file)
    
    if os.path.exists('./test_pdf_with_metadata_rendered'):
        print("=======processing test images=======")
        input_directory = './test_pdf_with_metadata_rendered'
        output_hdf5_file = 'test_pid_image.h5'
        split = 'test'
        process_jsonl_files(split, input_directory, output_hdf5_file)

    if os.path.exists('./val_pdf_with_metadata_rendered'):
        print("=======processing val images=======")
        input_directory = './val_pdf_with_metadata_rendered'
        output_hdf5_file = 'val_pid_image.h5'
        split = 'val'
        process_jsonl_files(split, input_directory, output_hdf5_file)
    
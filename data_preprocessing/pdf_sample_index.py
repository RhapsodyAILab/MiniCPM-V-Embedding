import h5py
import json
import random
import os

def main(hdf5_file, output_json, sample_num_docs, seed):
    
    random.seed(seed)
    
    with h5py.File(hdf5_file, 'r') as f:
        # 获取所有group的名称
        group_names = list(f.keys())

    print(f"#of group_names = {len(group_names)}")

    real_sample_size = min(sample_num_docs, len(group_names))

    print(f"# of real real_sample_size = {real_sample_size}")

    # 随机采样5000个group
    sampled_groups = random.sample(group_names, real_sample_size)

    # 将采样的group_name列表保存为JSON文件
    with open(output_json, 'w') as f:
        for group_name in sampled_groups:
            f.write(group_name+'\n')

    print(f'Sampled group names saved to {output_json}')


# sample_num_docs = 5000
# hdf5_file = '/mnt/data/user/tc_agi/xubokai/K12Textbook_visual/train_pid_image.hdf5'
# output_json = 'train.noquery.txt'

# sample_num_docs = 5000
# hdf5_file = '/mnt/data/user/tc_agi/xubokai/K12Textbook_visual/test_pid_image.hdf5'
# output_json = 'test.noquery.txt'

# sample_num_docs = 20000
# hdf5_file = '/mnt/data/user/tc_agi/xubokai/manuallib_sampled/train_pid_image.hdf5'
# output_json = 'train.noquery.txt'

# sample_num_docs = 5000
# hdf5_file = '/mnt/data/user/tc_agi/xubokai/manuallib_sampled/test_pid_image.hdf5'
# output_json = 'test.noquery.txt'


seed = 42


train_num_sampled_docs = 30000
test_num_sampled_docs = 5000
val_num_sampled_docs = 10000


if __name__ == "__main__":
    if os.path.exists('./train_pid_image.h5'):
        hdf5_file = './train_pid_image.h5'
        output_json = './train.noquery.txt'
        main(hdf5_file, output_json, train_num_sampled_docs, seed)
    
    if os.path.exists('./test_pid_image.h5'):
        hdf5_file = './test_pid_image.h5'
        output_json = './test.noquery.txt'
        main(hdf5_file, output_json, test_num_sampled_docs, seed)
    
    if os.path.exists('./val_pid_image.h5'):
        hdf5_file = './val_pid_image.h5'
        output_json = './val.noquery.txt'
        main(hdf5_file, output_json, val_num_sampled_docs, seed)


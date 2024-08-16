import h5py

# 读取指定 doc-id 对应的 image
doc_id_to_read = '12001'


# with h5py.File('/mnt/data/user/tc_agi/xubokai/train_pid_image.h5', 'r') as f:
with h5py.File('/mnt/data/user/tc_agi/xubokai/docvqa_mp/train_pid_image.h5', 'r') as f:
    
    # image = f[doc_id_to_read]['image']
    image = f['rmwn0226_p84.jpg']['image']
    # 现在 image 变量中存储了对应 doc-id 的图像数据
    print(image[()])
    # image = f['1892']['image']
    # # 现在 image 变量中存储了对应 doc-id 的图像数据
    # print(image[()])
    # image = f['1893']['image']
    # # 现在 image 变量中存储了对应 doc-id 的图像数据
    # print(image[()])


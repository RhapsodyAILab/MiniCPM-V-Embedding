import json
import os
import shutil
import random
# import tarfile
import random
import os
# from PIL import Image
import io
import base64
import zipfile
# from IPython import embed

visual_keywords = {'figure', 'table', 'equation'}

def judge_tier(fileobj):
    lines = fileobj.readlines()
    # base64.b64encode(lines).decode('utf-8')
    
    # embed()
    # lines.decode('utf-8')
    
    if lines:  # Check if the file is not empty
        for line in lines:
            line_ = line.decode('utf-8')
            last_column = line_.split()[-1]
            if any(keyword in last_column for keyword in visual_keywords):
                return "visual"
            # else:
    
    return "textual"




def stream_zip_metadata(zip_file, split_data):
    visual_count = {}
    textual_count = {}
    collection_to_images = {}
    
    split_data_abs = [
        'DocBank_500K_txt/' + i + '.txt' for i in split_data
    ]

    print("reading zip file...")
    with zipfile.ZipFile(zip_file, 'r') as zipf:
        cnt = 0
        for member_, member in zip(split_data, split_data_abs):
            cnt += 1
            
            if cnt == 100000: # for train and test, both 50k is enough
                break
            
            # if True:
            #     print(cnt, member_)
            
            with zipf.open(member) as fileobj:
                judge = judge_tier(fileobj)
                collection_name = '_'.join(member_.split('_')[:-1]) # 134.tar_1607.04853.gz_coling2016_2 -> 134.tar_1607.04853.gz_coling2016
                
                if cnt % 1000 == 0:
                # if True:
                    print(cnt, collection_name, member_)
                
                previous_collection_to_images = collection_to_images.get(collection_name, [])
                previous_collection_to_images.append(member_+"_ori.jpg")
                collection_to_images[collection_name] = previous_collection_to_images
                
                if judge == "visual":
                    previous_visual_count = visual_count.get(collection_name, 0)
                    visual_count[collection_name] = previous_visual_count + 1
                else:
                    previous_textual_count = textual_count.get(collection_name, 0) 
                    textual_count[collection_name] = previous_textual_count + 1
    
    print("determining visual or textual...")
    visual_list = []
    textual_list = []
    for collection in collection_to_images.keys():
        visual_count_ = visual_count.get(collection, 0)
        textual_count_ = textual_count.get(collection, 0)
        
        if visual_count_ > 0.8 * (textual_count_ + visual_count_):
            visual_list.append(collection)
        else:
            textual_list.append(collection)
    
    print("gathering...")
    visual_list_images = []
    textual_list_images = []
    
    visual_collection_to_images = {}
    textual_collection_to_images = {}
    
    for c in visual_list:
        images_ = collection_to_images.get(c)
        visual_list_images.extend(images_)
        visual_collection_to_images[c] = images_
        
    for c in textual_list:
        images_ = collection_to_images.get(c)
        textual_list_images.extend(images_)
        textual_collection_to_images[c] = images_
    
    return visual_list_images, textual_list_images, visual_collection_to_images, textual_collection_to_images


if __name__ == "__main__":
    
    # random.seed(42)
    
    split_file = './raw/500K_train.json'
    # split_file = './raw/500K_test.json'
    
    metadata_zipfile = './raw/DocBank_500K_txt.zip'
    
    split_data = json.load(open(split_file))
    
    images = [b['file_name'] for b in split_data['images']]
    names = [
        i.rstrip('_ori.jpg') for i in images
    ] # 134.tar_1607.04817.gz_global_r2_34_ori.jpg -> 134.tar_1607.04817.gz_global_r2_34
    
    visual_list_images, textual_list_images, visual_collection_to_images, textual_collection_to_images = stream_zip_metadata(metadata_zipfile, names)
    
    print("saving results...")
    
    with open(split_file+'.visual.collections.json', 'w') as f:
        f.write(json.dumps(visual_collection_to_images))
    
    with open(split_file+'.textual.collections.json', 'w') as f:
        f.write(json.dumps(textual_collection_to_images))
    
    with open(split_file+'.visual.images.json', 'w') as f:
        f.write(json.dumps(visual_list_images))
            
    with open(split_file+'.textual.images.json', 'w') as f:
        f.write(json.dumps(textual_list_images))
    
    
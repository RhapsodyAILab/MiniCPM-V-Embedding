import json
import random
random.seed(42)

input_jsonl = './train.jsonl'
output_jsonl = './train_formal.jsonl'

input_jsonl = './test.jsonl'
output_jsonl = './test_formal.jsonl'


def url2docid(url):
    elements = url.split('/')[-3:]
    assert len(elements) == 3
    docId =  elements[0] + '_' + elements[1] + '__' + elements[2]
    return docId


all_data = []
with open(input_jsonl, 'r') as f:
    multiple_evidence = 0
    for line in f:
        data = json.loads(line)
        if len(data['evidence_pages']) > 1:
            print("multiple evidence", multiple_evidence)
            print(line)
            # input(">")
            multiple_evidence += 1
        if data['evidence_pages'][0] == 1:
            print("start from 0")
        all_data.append(data)


output_data = []
metadata_cnt = 0
global_counter = 0

for data in all_data:
    
    neg_pool = []
    cnt_ = 0
    for url in data["image_urls"]:
        cnt_ += 1
        if cnt_ in data["evidence_pages"]: # if this url in evidence pages, this is not negative.
            continue
        else:
            neg_pool.append(url2docid(url))
    
    for doc_id in data["evidence_pages"]: # this starts from 1, so -1 when using
        data_ = {
            "id": global_counter,
            "metadata_source": metadata_cnt,
            "query": data["question"],
            "answer": data["answer"],
            "pos": url2docid(data["image_urls"][doc_id-1]),
            "neg": neg_pool, # can be []
        }
        output_data.append(data_)
        global_counter += 1
        
    metadata_cnt += 1

with open(output_jsonl, 'w') as f:
    for data in output_data:
        f.write(json.dumps(data)+'\n')


with open(output_jsonl+'.onlyq.jsonl', 'w') as f:
    for data in output_data:
        data_ = {
            "id": data["id"],
            "query": data["query"],
        }
        f.write(json.dumps(data_)+'\n')

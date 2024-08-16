import json
import random

random.seed(42)

input_json = './train.json'
output_jsonl = './train_formal.jsonl'

input_json = './val.json'
output_jsonl = './val_formal.jsonl'


def url2docid(url):
    docId = url + ".jpg"
    return docId

# all_data = []
# with open(input_jsonl, 'r') as f:
#     multiple_evidence = 0
#     for line in f:
#         data = json.loads(line)
#         if len(data['evidence_pages']) > 1:
#             print("multiple evidence", multiple_evidence)
#             print(line)
#             # input(">")
#             multiple_evidence += 1
#         if data['evidence_pages'][0] == 1:
#             print("start from 0")
#         all_data.append(data)

all_data = json.load(open(input_json))['data']

output_data = []
metadata_cnt = 0
global_counter = 0

for data in all_data:
    
    neg_pool = []
    cnt_ = 0
    for url in data["page_ids"]:
        
        if cnt_ in data["page_ids"]: # if this url in evidence pages, this is not negative.
            pass
        else:
            neg_pool.append(url2docid(url))
        
        cnt_ += 1 # start from 0
    
    # for doc_id in data["evidence_pages"]: # this starts from 1, so -1 when using
    data_ = {
        "id": global_counter,
        "metadata_source": metadata_cnt,
        "query": data["question"],
        "answer": random.choice(data["answers"]),
        "pos": url2docid(data["page_ids"][data["answer_page_idx"]]), # index start from 0
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

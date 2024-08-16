
import argparse
import os
import json
from vllm import LLM, SamplingParams

# https://docs.vllm.ai/en/latest/serving/distributed_serving.html#distributed-inference-and-serving
# pip install ray


# /home/xvbokai/miniconda3/envs/vllm3/bin/python /home/xvbokai/query_cls/main.py --model_path /home/xvbokai/Meta-Llama-3-8B-Instruct --prompt_path cls_v3.txt --n_gpu 1 --max_new_tokens 1 --batch_size 32 --output_path /home/xvbokai/query_cls_result --data_path /home/xvbokai/query_raw/docvqa_mp-val_query.jsonl

# /home/xvbokai/miniconda3/envs/vllm3/bin/python /home/xvbokai/query_cls/main.py --model_path /home/xvbokai/Meta-Llama-3-8B-Instruct --prompt_path cls_v3.txt --n_gpu 1 --max_new_tokens 1 --batch_size 256 --output_path /home/xvbokai/query_cls_result --data_path /home/xvbokai/query_raw/docvqa_mp-train_query.jsonl


"""


/home/xvbokai/miniconda3/envs/vllm3/bin/python /home/xvbokai/query_cls/main.py --model_path /home/xvbokai/Meta-Llama-3-8B-Instruct --prompt_path cls_v3.txt --n_gpu 1 --max_new_tokens 1 --batch_size 256 --output_path /home/xvbokai/query_cls_result --data_path /home/xvbokai/query_raw_0623/infographicsVQA_test_v1.0-query.jsonl

/home/xvbokai/miniconda3/envs/vllm3/bin/python /home/xvbokai/query_cls/main.py --model_path /home/xvbokai/Meta-Llama-3-8B-Instruct --prompt_path cls_v3.txt --n_gpu 1 --max_new_tokens 1 --batch_size 256 --output_path /home/xvbokai/query_cls_result --data_path /home/xvbokai/query_raw_0623/infographicsVQA_train_v1.0-query.jsonl

/home/xvbokai/miniconda3/envs/vllm3/bin/python /home/xvbokai/query_cls/main.py --model_path /home/xvbokai/Meta-Llama-3-8B-Instruct --prompt_path cls_v3.txt --n_gpu 1 --max_new_tokens 1 --batch_size 256 --output_path /home/xvbokai/query_cls_result --data_path /home/xvbokai/query_raw_0623/plotqa-qa_pairs_V1_test_sampled-query.jsonl

/home/xvbokai/miniconda3/envs/vllm3/bin/python /home/xvbokai/query_cls/main.py --model_path /home/xvbokai/Meta-Llama-3-8B-Instruct --prompt_path cls_v3.txt --n_gpu 1 --max_new_tokens 1 --batch_size 256 --output_path /home/xvbokai/query_cls_result --data_path /home/xvbokai/query_raw_0623/plotqa-qa_pairs_V1_train_sampled-query.jsonl


"""

# Create the parser
parser = argparse.ArgumentParser(description="Reverse query generation")

# Add arguments
parser.add_argument("--data_path", help="Raw data directory.", type=str)
parser.add_argument("--output_path", help="Output data directory.", type=str)
parser.add_argument("--model_path", help="SFT model path.", type=str)
parser.add_argument("--n_gpu", help="Number of GPUs.", type=int, default=1)
parser.add_argument("--max_new_tokens", help="Max num of new tokens.", type=int, default=16)
parser.add_argument("--batch_size", help="Batch size.", type=int, default=32)
parser.add_argument("--prompt_path", help="Generation prompt path.", type=str)
# "/home/jeeves/Yi-34B-Chat"

# TEMPERATURE = 0.8
# TOP_P = 0.95

if __name__ == "__main__":
    # Parse the arguments
    args = parser.parse_args()
    
    if args.n_gpu > 1:
        llm = LLM(model=args.model_path, tensor_parallel_size=args.n_gpu)
    else:
        llm = LLM(model=args.model_path)

    sampling_params = SamplingParams(
        # temperature=TEMPERATURE, 
        # top_p=TOP_P, 
        # use_beam_search=True,
        # best_of=2,
        temperature=0,
        max_tokens=args.max_new_tokens, 
        stop=['```']
    )
    
    output_base_path = args.output_path
    
    os.makedirs(output_base_path, exist_ok=True)
    subdir_name = args.data_path.split("/")[-1].split(".")[0]
    print(f"=============== output subdir name {subdir_name} ===============")
    output_path_subdir = os.path.join(output_base_path, subdir_name)
    os.makedirs(output_path_subdir, exist_ok=True)
    # Load dataset, args.data_path is a directory of jsons
    # json_data = os.listdir(args.data_path)
    # json_data_effective = [d for d in json_data if d.endswith('.json')]
    
    # print(json_data_effective)
    
    total_data = []
    # total_data_splitter = [0]
    
    # for json_path in json_data_effective:
    #     json_path_abs = os.path.join(args.data_path, json_path)
    #     print(f"loading {json_path_abs}")
    #     data = json.load(open(json_path_abs, 'r'))
    #     total_data.extend(data)
    #     total_data_splitter.append(len(total_data))
    
    with open(args.data_path, 'r') as f:
        for line in f:
            total_data.append(json.loads(line))
    
    print(f"total # of data = {len(total_data)}")
    
    # 获取当前脚本的完整路径
    script_path = __file__

    # 获取脚本所在的目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(script_path))

    # apply prompt
    prompt_abs_path = os.path.join(script_dir, args.prompt_path)
    print(f"prompt path = {prompt_abs_path}")
    prompt_text = open(prompt_abs_path, 'r').read()
    
    # Perform inference with LLM
    batch_size = args.batch_size
    n_batches = len(total_data) // batch_size + 1
    for batch_idx in range(n_batches):
        print(f"batch_idx = {batch_idx} / {n_batches}")
        batch_d = total_data[batch_idx*batch_size: (batch_idx+1)*batch_size]
        
        filled_prompt = [prompt_text.format(passage=d["query"]) for d in batch_d]
        
        print(filled_prompt[0])
        
        outputs = llm.generate(filled_prompt, sampling_params)

        responses = [o.outputs[0].text for o in outputs]
        
        for idx in range(len(batch_d)):
            batch_d[idx]["response"] = responses[idx]
        
        with open(os.path.join(output_path_subdir, f'{batch_idx}.jsonl'), 'w') as f:
            for o_data in batch_d:
                print(o_data)
                f.write(json.dumps(o_data, ensure_ascii=False)+'\n')
        
        # input(">")
        # print(queries)
    
    
    
    

import logging
import os
import sys
import gc
import glob
import pickle
from contextlib import nullcontext
from typing import Dict, List, Union

# import faiss
import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer
import pytrec_eval

from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
from transformers import BatchEncoding

from openmatch.arguments import DataArguments
from openmatch.arguments import InferenceArguments as EncodingArguments
from openmatch.arguments import ModelArguments
from openmatch.dataset import InferenceDataset, DRInferenceCollator
from openmatch.modeling import DRModelForInference
# from openmatch.retriever import Retriever
from openmatch.utils import save_as_trec

import time

logger = logging.getLogger(__name__)



def to_device(data, device):
    """
    Recursively move tensors in a nested list, tuple, or dictionary to the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_device(item, device) for item in data)
    elif isinstance(data, BatchEncoding):
        return data.to(device)
    else:
        return data  # 如果不是上述类型，则原样返回，比如Image, str等无需to device


def naive_collator(batch_input):
    assert isinstance(batch_input, list)
    assert len(batch_input) > 0
    
    keys = list(batch_input[0].keys())
    collated = {key: [] for key in keys}
    for item in batch_input:
        for key in keys:
            collated[key].append(item[key])
    
    return collated
        

def distributed_parallel_embedding_inference(
    dataset: InferenceDataset,
    model: DRModelForInference,
    args: EncodingArguments,
    dataset_type: str = "corpus", # corpus or query
    split_save: bool = True, # whether to save the embeddings in separate files
    model_additional_args = None, # additionak keyword arguments passing to model
):
    # Note: during evaluation, there's no point in wrapping the model
    # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
    if dataset is None:
        raise ValueError("No dataset provided")
    
    dataloader = DataLoader(
        dataset, # this dataset can be sharded (data parallel)
        batch_size=args.per_device_eval_batch_size,
        
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
        drop_last=False, # we don't want to drop the last batch, this is evaluation
        
        collate_fn=naive_collator, # because dataset has handled the length!
    )

    os.makedirs(args.output_dir, exist_ok=True)
    
    encoded = []
    lookup_indices = []
    idx = 0
    prev_idx = 0
    
    batch_cnt = 0
    for batch in tqdm(dataloader, disable=args.process_index > 0):
        # print(batch)
        
        batch_ids = batch['id']
        
        lookup_indices.extend(batch_ids)
        
        idx += len(batch_ids)
        
        if batch_cnt == 0:
            if args.fp16:
                logger.info("using bf16, will use autocast during inference")
            elif args.bf16:
                raise ValueError("bf16 is not yet validated, please self-check.")
            else:
                raise ValueError("datatype not specified")
        
        with amp.autocast() if args.fp16 else nullcontext():
            with torch.no_grad():
                # t1 = time.time()
                for k, v in batch.items():
                    batch[k] = to_device(v, args.device) # a BatchEncoding object, can contain strings.
                
                # print(batch['input_ids'].shape)
                # t2 = time.time()
                # print("dataloader - elspased", t2-t1)
                
                if dataset_type == "corpus":
                    model_output: DROutput = model(passage=batch, **model_additional_args)
                    encoded_ = model_output.p_reps.cpu().detach().numpy()
                elif dataset_type == "query":
                    model_output: DROutput = model(query=batch, **model_additional_args)
                    encoded_ = model_output.q_reps.cpu().detach().numpy()
                else:
                    raise ValueError(f"dataset_type: {dataset_type} is not valid.")
                
                if batch_cnt == 0:
                    logger.info(f"encoded_ dtype = {encoded_.dtype}")
                    contains_nan = np.isnan(encoded_).any()
                    assert contains_nan != True, "vital error, model output has nan, please check."
                # logging.info(f"you got {encoded_.dtype} for embedding")
                
                encoded.append(encoded_)
        
        if len(lookup_indices) >= args.max_inmem_docs // args.world_size:
            if split_save:
                encoded = np.concatenate(encoded)
                with open(
                    os.path.join(
                        args.output_dir,
                        "embeddings.{}.rank.{}.{}-{}".format(
                            dataset_type,
                            args.process_index, 
                            prev_idx, idx
                        ),
                    ),
                    "wb",
                ) as f:
                    pickle.dump((encoded, lookup_indices), f, protocol=4)
                encoded = []
                lookup_indices = []
                prev_idx = idx
                gc.collect()
        
        # t1 = time.time()
        batch_cnt += 1

    # this is to handle the last batch
    if len(lookup_indices) > 0:
        if split_save:
            encoded = np.concatenate(encoded)
            with open(
                os.path.join(
                    args.output_dir,
                    "embeddings.{}.rank.{}.{}-{}".format(
                        dataset_type,
                        args.process_index, 
                        prev_idx, idx
                    ),
                ),
                "wb",
            ) as f:
                pickle.dump((encoded, lookup_indices), f, protocol=4)

    # if save to a whole file (each rank only save one file)
    if not split_save:
        encoded = np.concatenate(encoded)
        with open(
            os.path.join(
                args.output_dir,
                "embeddings.{}.rank.{}".format(
                    dataset_type,
                    args.process_index, 
                    # prev_idx, idx
                ),
            ),
            "wb",
        ) as f:
            pickle.dump((encoded, lookup_indices), f, protocol=4)
    
    del encoded
    del lookup_indices

    if args.world_size > 1:
        torch.distributed.barrier()

    return





# def qwen_vl_distributed_parallel_embedding_inference(
#     dataset: InferenceDataset,
#     model: torch.nn.Module,
#     args: None,
#     split_save: bool = True, # whether to save the embeddings in separate files
# ):
#     # Note: during evaluation, there's no point in wrapping the model
#     # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
#     if dataset is None:
#         raise ValueError("No dataset provided")
#     dataloader = DataLoader(
#         dataset,
#         batch_size=args.per_device_eval_batch_size,
#         collate_fn=DRInferenceCollator(),
#         num_workers=args.dataloader_num_workers,
#         pin_memory=args.dataloader_pin_memory,
#         drop_last=False, # we don't want to drop the last batch, this is evaluation
#     )

#     os.makedirs(args.output_dir, exist_ok=True)
    
#     encoded = []
#     lookup_indices = []
#     idx = 0
#     prev_idx = 0
#     for batch_ids, batch in tqdm(dataloader, disable=args.process_index > 0):
#         lookup_indices.extend(batch_ids)
#         idx += len(batch_ids)
#         with amp.autocast() if args.fp16 else nullcontext():
#             # print(f"{args.fp16}!!!")
#             with torch.no_grad():
#                 for k, v in batch.items():
#                     batch[k] = v.to(args.device)
                
#                 # print(batch['input_ids'].shape)
                
#                 if dataset_type == "corpus":
#                     model_output: DROutput = model(passage=batch)
#                     encoded_ = model_output.p_reps.cpu().detach().numpy()
#                 elif dataset_type == "query":
#                     model_output: DROutput = model(query=batch)
#                     encoded_ = model_output.q_reps.cpu().detach().numpy()
#                 else:
#                     raise ValueError(f"dataset_type: {dataset_type} is not valid.")
                
#                 # logging.info(f"you got {encoded_.dtype} for embedding")
                
#                 encoded.append(encoded_)
        
#         if len(lookup_indices) >= args.max_inmem_docs // args.world_size:
#             if split_save:
#                 encoded = np.concatenate(encoded)
#                 with open(
#                     os.path.join(
#                         args.output_dir,
#                         "embeddings.{}.rank.{}.{}-{}".format(
#                             dataset_type,
#                             args.process_index, 
#                             prev_idx, idx
#                         ),
#                     ),
#                     "wb",
#                 ) as f:
#                     pickle.dump((encoded, lookup_indices), f, protocol=4)
#                 encoded = []
#                 lookup_indices = []
#                 prev_idx = idx
#                 gc.collect()

#     # this is to handle the last batch
#     if len(lookup_indices) > 0:
#         if split_save:
#             encoded = np.concatenate(encoded)
#             with open(
#                 os.path.join(
#                     args.output_dir,
#                     "embeddings.{}.rank.{}.{}-{}".format(
#                         dataset_type,
#                         args.process_index, 
#                         prev_idx, idx
#                     ),
#                 ),
#                 "wb",
#             ) as f:
#                 pickle.dump((encoded, lookup_indices), f, protocol=4)

#     # if save to a whole file (each rank only save one file)
#     if not split_save:
#         encoded = np.concatenate(encoded)
#         with open(
#             os.path.join(
#                 args.output_dir,
#                 "embeddings.{}.rank.{}".format(
#                     dataset_type,
#                     args.process_index, 
#                     # prev_idx, idx
#                 ),
#             ),
#             "wb",
#         ) as f:
#             pickle.dump((encoded, lookup_indices), f, protocol=4)
    
#     del encoded
#     del lookup_indices

#     if args.world_size > 1:
#         torch.distributed.barrier()

#     return

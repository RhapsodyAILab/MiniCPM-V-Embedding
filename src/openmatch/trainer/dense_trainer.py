# Adapted from Tevatron (https://github.com/texttron/tevatron)

import logging
import os
from itertools import repeat
from typing import Any, Dict, List, Optional, Tuple, Union
import shutil
from contextlib import nullcontext

import datasets
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler # for distributed training 

from transformers.file_utils import is_datasets_available
from transformers.trainer import TRAINING_ARGS_NAME, Trainer
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers import BatchEncoding

# from ..loss import DistributedContrastiveLoss, SimpleContrastiveLoss

logger = logging.getLogger(__name__)

try:
    from grad_cache import GradCache
    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False


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


class DRTrainer(Trainer):
    def __init__(
        self, 
        *args, 
        model_name_or_path, 
        # processor,
        **kwargs,
        
    ):
        super(DRTrainer, self).__init__(*args, **kwargs)
        
        self._dist_loss_scale_factor = dist.get_world_size() if self.args.negatives_x_device else 1
        
        self.model_name_or_path = model_name_or_path
        # process_rank and world_size work well for multi nodes and multi gpu setting
        # for example, 4 nodes, each node has 8 gpus, then each node has 8 processes, each process control one gpu.
        # for node 2, gpu 3, the get_rank will return rank=2*8+3
        
        # for multimodal model training
        # self.processor = processor
        
        self.process_rank = dist.get_rank()
        # world_size = 4*8 = 32
        self.world_size = dist.get_world_size()
        
        self.metric_hook = {
            "accuracy": [],
        }

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        
        # this state_dict is contructed by deepspeed or others, not the model we know.
        # because the parameter of each process is not the full parameter
        
        if state_dict is not None: # with deepspeed training
            # step1: remove prefix "lm_q."
            
            # if self.model.base_model_arch == "SmartCPMModel": # Text embedding
            #     state_dict = {k.replace("lm_q", "model"): v for k, v in state_dict.items()}
            if self.model.base_model_arch == "MiniCPMVEmbedding": # Multimodal embedding
                state_dict = {k.replace("lm_q.", ""): v for k, v in state_dict.items()}
            # elif self.model.base_model_arch == "MiniCPMEmbedding": # text embedding
            #     state_dict = {k.replace("lm_q.", ""): v for k, v in state_dict.items()}
            # elif self.model.base_model_arch == "SiglipModel": # Multimodal embedding
            #     state_dict = {k.replace("lm_q.", ""): v for k, v in state_dict.items()}
            # elif "Mistral" in self.model.base_model_arch:
            #     state_dict = {k.replace("lm_q", "model"): v for k, v in state_dict.items()}
            # elif "Bert" in self.model.base_model_arch:
            #     state_dict = {k.replace("lm_q", "bert"): v for k, v in state_dict.items()}
            # elif "QWen" in self.model.base_model_arch:
            #     state_dict = {k.replace("lm_q", "transformer"): v for k, v in state_dict.items()}
            # elif "Llama" in self.model.base_model_arch:
            #     state_dict = {k.replace("lm_q", "model"): v for k, v in state_dict.items()}
            else:
                # raise NotImplementedError
                state_dict = {k.replace("lm_q.", ""): v for k, v in state_dict.items()}
                logger.warning("model arch not handled, so save the state_dict after removing lm_q. directly.")
            
            # step2: save this state_dict as model paramters
            super(DRTrainer, self)._save(output_dir=output_dir, state_dict=state_dict)
            
            # step3: copy config file (important for identifying model arch)
            config_file_path = os.path.join(self.model_name_or_path, 'config.json')
            shutil.copy(config_file_path, output_dir)
        
        else: # normal training 
            # step1: save model params
            self.model.lm_q.save_pretrained(output_dir)
            # step2: save tokenizer
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            # step3: save training args
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    
    # def _prepare_inputs(
    #     self, inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    # ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
    #     prepared = []
    #     for x in inputs:
    #         if isinstance(x, torch.Tensor):
    #             prepared.append(x.to(self.args.device))
    #         else:
    #             prepared.append(super()._prepare_inputs(x))
    #     return prepared
    
    def _prepare_inputs(
        self, inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> Tuple[Dict[str, Union[torch.Tensor, Any]]]:
        # move to device
        query, passages = inputs
        # print(f"input query keys {query.keys()}")
        # here query and passage should only include:
        # - input_ids & attention_mask for text model
        # - inputs_embeds & attention_mask for multimodal model
        query = to_device(query, self.args.device)
        passages = to_device(passages, self.args.device)
        
        # print(f"output query keys {query.keys()}")
        
        query = BatchEncoding(query)
        passages = BatchEncoding(passages)
        
        # assert isinstance(query, BatchEncoding), "query should be a BatchEncoding object"
        # assert isinstance(passages, BatchEncoding), "passages should be a BatchEncoding object"
        
        return query, passages

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            
            logger.info(f"self.args.train_batch_size = {self.args.train_batch_size}")
            logger.info(f"self.world_size = {self.world_size}")
            logger.info(f"self.process_rank = {self.process_rank}")
            
            if self.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.world_size, # this world_size is global (multi gpu and multi nodes).
                    process_index=self.process_rank, # this process_rank is global (multi gpu and multi nodes).
                )
                
                # No worry! IterableDatasetShard will not duplicate for each process, though the dataset.process_fn will be called for each data point on each process!

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory
            )
        else:
        
            # train_sampler = self._get_train_sampler() # what is this sampler? distributed? [exception: this does not work for distributed training]
            
            train_sampler = DistributedSampler(
                train_dataset, 
                shuffle=False, # each epoch will shuffle once by default
                num_replicas=self.world_size, 
                rank=self.process_rank
            )
            
            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

    def log(self, log_dict): # overwrite Trainer.log [hacked]
        if len(self.metric_hook["accuracy"]) != 0:
            log_dict["accuracy"] = sum(self.metric_hook["accuracy"]) / len(self.metric_hook["accuracy"])
            self.metric_hook["accuracy"] = []
            
        # we need to restore the true global CrossEntropyLoss (global batch)
        # when logging
        if "loss" in log_dict:
            log_dict["loss"] = log_dict["loss"] / self.world_size
                
        super().log(log_dict)
    
    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)] # gpus across all nodes , counted
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
    
    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`torch.nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        
        # logger.info(f"training model dtype: {model.module.lm_q.dtype}")
        
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        if self.model.base_model_arch == "MiniCPMVEmbedding":
            if self.state.global_step == 0:
                logger.info("You are training MiniCPMVEmbedding, we are injecting additional arguments `tokenizer` and `max_inp_length` to the forward function.")
            
            assert self.train_dataset.data_args.q_max_len == self.train_dataset.data_args.p_max_len, "If you are training MiniCPMVEmbedding, `max_q_len` should be equal to `p_max_len`."
            
            additional_args = {
                "tokenizer": self.tokenizer,
                "max_inp_length": self.train_dataset.data_args.p_max_len
            }
        elif self.model.base_model_arch == "MiniCPMEmbedding":
            if self.state.global_step == 0:
                logger.info("You are training MiniCPMEmbedding, we are injecting additional arguments `tokenizer` and `max_inp_length` to the forward function.")
            
            assert self.train_dataset.data_args.q_max_len == self.train_dataset.data_args.p_max_len, "If you are training MiniCPMEmbedding, `max_q_len` should be equal to `p_max_len`."
            
            additional_args = {
                "tokenizer": self.tokenizer,
                "max_inp_length": self.train_dataset.data_args.p_max_len
            }
        else:
            additional_args = {}
        

        # gradient cache mode
        if self.args.grad_cache_enable:
            # Reference:
            # Original implementation: https://github.com/luyug/GradCache
            # Paper: https://arxiv.org/abs/2101.06983
            # This is a re-implementation of GradCache, which is a technique to reduce the memory consumption of large models.
            if self.state.global_step == 0:
                logger.info("Gradient Cache is enabled now.")
            assert self.args.negatives_x_device == True, "--negatives_x_device should be enabled if you use --grad_cache_enable， please check"
            
            # step1: compute all the representations of query and passage. in fact, only passage is needed, the query is not necessary, but for convenience and simplicity, we compute both.
            query, passage = inputs
            # divide `minibatch` into `microbatches`, recall what is mini-batch and what is micro-batch?
            mini_bsz = self.args.per_device_train_batch_size
            micro_bsz = self.args.grad_cache_micro_batch_size
            global_micro_bsz = micro_bsz * self.world_size
            
            # assert query["input_ids"].shape[0] == mini_bsz, "input `query` batch size should be equal to --per_device_train_batch_size, please check your dataloader and data collator."
            assert mini_bsz % micro_bsz == 0, "batch size should be divisible by --grad_cache_mini_batch_size"
            num_microbatches = mini_bsz // micro_bsz
            # assert passage["input_ids"].shape[0] % query["input_ids"].shape[0] == 0, "expect #of passage to be multiple of #of query"
            # n_passages = passage["input_ids"].shape[0] // query["input_ids"].shape[0]
            
            n_passages = self.train_dataset.data_args.train_n_passages
            
            # get all the passage representations, but with no gradient
            q_reps_cache = []
            p_reps_cache = []
            rng_state_cache = []
            
            with torch.no_grad(): 
                # Here we don't preserve any gradient and intermediate states.
                # the memory consumption for step1 is model.hidden_dim * 1(one vector for each text) * sizeof(dtype, bf16=2) * mini_bsz * world_size
                # For example:
                # hidden_dim=2304, sizeof(bf16)=2, mini_bsz=512, world_size=8(#gpu=8), then the memory consumption is 18MB for query
                # hidden_dim=2304, sizeof(bf16)=2, mini_bsz=512, world_size=8(#gpu=8), n_passages=6 then the memory consumption is 108MB for query
                # totally 18+108=126MB for query and passage cache.
                for i in range(0, num_microbatches):
                    if self.state.global_step == 0:
                        logger.info(f"* without gradient forward # {i}")
                    # step 1a: prepare micro-batch of query & passage
                    micro_query = BatchEncoding({k: v[i * micro_bsz: (i + 1) * micro_bsz] for k, v in query.items()})
                    micro_passage = BatchEncoding({k: v[i * micro_bsz * n_passages: (i + 1) * micro_bsz * n_passages] for k, v in passage.items()})
                    
                    # step 1b: compute representations of mini-batch of query & passage
                    rng_state_ = torch.get_rng_state() # but have to save the rng_state for w/ grad forward
                    rng_state_cache.append(rng_state_)
                    outputs_ = model(query=micro_query, passage=micro_passage, **additional_args)
                    q_reps_ = outputs_.q_reps
                    p_reps_ = outputs_.p_reps
                    
                    q_reps_ = self.dist_gather_tensor(q_reps_) # all of the gathered large tensor do not have gradient
                    p_reps_ = self.dist_gather_tensor(p_reps_) # all of the gathered large tensor do not have gradient
                    
                    # ========= Composition of q_reps_ =========
                    # [   q_reps_ from GPU 0, microbatch 0   ]
                    # [   q_reps_ from GPU 1, microbatch 0   ]
                    # [   q_reps_ from GPU 2, microbatch 0   ]
                    # [   q_reps_ from GPU 3, microbatch 0   ]
                    # ----------------- cat ------------------
                    # [   q_reps_ from GPU 0, microbatch 1   ]
                    # [   q_reps_ from GPU 1, microbatch 1   ]
                    # [   q_reps_ from GPU 2, microbatch 1   ]
                    # [   q_reps_ from GPU 3, microbatch 1   ]
                    # ----------------- cat ------------------
                    # [   q_reps_ from GPU 0, microbatch 2   ]
                    # [   q_reps_ from GPU 1, microbatch 2   ]
                    # [   q_reps_ from GPU 2, microbatch 2   ]
                    # [   q_reps_ from GPU 3, microbatch 2   ]
                    # ==========================================

                    # expect_p_reps_shape = micro_passage["input_ids"].shape[0] * self.world_size
                    # assert p_reps_.shape[0] == expect_p_reps_shape, f"expect passage representations output to be gathered across all gpus, you have {self.args.n_gpu} gpu, expect {expect_p_reps_shape}, but got {p_reps_.shape[0]} p_reps, check the parallelism of training."
                    
                    # step 1c: cache the representations of this micro-batch of passage
                    q_reps_cache.append(q_reps_)
                    p_reps_cache.append(p_reps_)
            
            # cat cache to be tensors List[Tensor[B, ...]] -> Tensor[B*n_gpus, ...]
            q_reps_cache = torch.cat(q_reps_cache, dim=0)
            p_reps_cache = torch.cat(p_reps_cache, dim=0)
            
            # assert passage["input_ids"].shape[0] % query["input_ids"].shape[0] == 0, "expect #of passage to be multiple of #of query"
            # n_passages = passage["input_ids"].shape[0] // query["input_ids"].shape[0]

            # step2: compute query, corresponding passage (both with gradient) and calculate loss, and perform backward to get gradient for parameters
            for i in range(0, num_microbatches):
                if self.state.global_step == 0:
                    logger.info(f"* with gradient forward # {i}")
                
                # we only want to synchronize gradient in the last iteration, because for non-last iteration, to all-reduce the gradient is trivial, a waste of time.
                if i == (num_microbatches - 1):
                    context = nullcontext()
                    if self.state.global_step == 0:
                        logger.info("    this is the last micro-batch iteration, no_sync is disabled.")
                else:
                    context = self.accelerator.no_sync(model) # <- is a feature of accelerate, and for pure DDP, it should be model.no_sync()
                    if self.state.global_step == 0:
                        logger.info("    this is not the last micro-batch, no_sync is enabled.")
                
                with context:
                    # step 2a: prepare micro-batch again
                    micro_query = BatchEncoding({k: v[i * micro_bsz: (i + 1) * micro_bsz] for k, v in query.items()})
                    micro_passage = BatchEncoding({k: v[i * micro_bsz * n_passages: (i + 1) * micro_bsz * n_passages] for k, v in passage.items()})
                    
                    # if self.state.global_step == 0: # <- this is for checking if the weight changes in each micro-batch, it should not change across micro-batch. the gradient seems to be invisible with deepspeed.
                    #     sampled_weight = model.lm_q.layers[0].self_attn.q_proj.weight[0, 0].item() # <- this is different from model to model
                    #     if model.lm_q.layers[0].self_attn.q_proj.weight.grad is not None:
                    #         sampled_weight_grad = model.lm_q.layers[0].self_attn.q_proj.weight.grad[0, 0].item()
                    #     else:
                    #         sampled_weight_grad = None
                    #     print(f"gpu={self.process_rank} iteration={i}: weight={sampled_weight}, grad={sampled_weight_grad}")

                    with self.compute_loss_context_manager():
                        rng_state_ = rng_state_cache.pop(0)
                        torch.set_rng_state(rng_state_) # for dropout..
                        outputs_ = model(query=micro_query, passage=micro_passage, **additional_args) # <- this model call is with gradient and intermediate states
                        q_reps_ = outputs_.q_reps
                        p_reps_ = outputs_.p_reps
                        
                        if self.state.global_step == 0:
                            logger.info(f"    process #{self.process_rank}: {i * global_micro_bsz + self.process_rank * micro_bsz} -> {i * global_micro_bsz + (self.process_rank + 1) * micro_bsz}")
                        
                        # for this, please refer to the above "Composition of q_reps_"
                        q_reps_tmp = q_reps_cache.clone()
                        q_reps_tmp[i * global_micro_bsz + self.process_rank * micro_bsz: i * global_micro_bsz + (self.process_rank + 1) * micro_bsz] = q_reps_
                        p_reps_tmp = p_reps_cache.clone()
                        p_reps_tmp[(i * global_micro_bsz + self.process_rank * micro_bsz) * n_passages: (i * global_micro_bsz + (self.process_rank + 1) * micro_bsz) *  n_passages] = p_reps_
                        
                        # compute loss
                        if self.args.biaxial_loss:
                            raise NotImplementedError
                        else:
                            scores = torch.matmul(q_reps_tmp, p_reps_tmp.transpose(0, 1))
                            if self.state.global_step == 0:
                                logger.info(f"    scores.shape = {scores.shape}")
                                logger.info(f"    softmax_temperature = {self.args.softmax_temperature}")
                                
                            scores = scores / self.args.softmax_temperature
                            
                            # one query, multiple passage (only one positive passage, others are all negatives)
                            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long) # shape [B, 1] where 1 is an index from 0 to B*n_train_docs
                            target = target * n_passages
                            
                            # reflections:
                            # if we have a dataset where one doc corresponds to multiple query, we could distribute the pairs to different batches, rather than put into the same batch
                            
                            
                            if self.state.global_step == 0:
                                logger.info(f"    target.shape = {target.shape}")
                            
                            loss = F.cross_entropy(scores, target, reduction='mean')
                            
                            # In distributed training, global CE backward will give one gpu its own gradient, 
                            # But DDP will average the gradient of all gpus, here, an extra mean is introduced. 
                            # then, we need to multiply loss by self.world_size to make the loss the same AS IF it is trained on a single LARGE GPU
                            loss = loss * self.world_size # the value in each gradcache step is the same, but the computational graph is different
                    
                    # For deepspeed, the behavior is different <- a feature of `accelerate` library
                    # Reference:
                    # https://github.com/huggingface/accelerate/blob/cd7df4117d92f660965d5f737364395b5693a535/src/accelerate/accelerator.py#L2015
                    # https://github.com/huggingface/accelerate/blob/cd7df4117d92f660965d5f737364395b5693a535/src/accelerate/accelerator.py#L1705
                    # https://github.com/huggingface/accelerate/blob/cd7df4117d92f660965d5f737364395b5693a535/src/accelerate/utils/deepspeed.py#L153
                    # https://github.com/huggingface/accelerate/blob/cd7df4117d92f660965d5f737364395b5693a535/src/accelerate/utils/deepspeed.py#L175
                    # Here is the hack for this:
                    if self.args.deepspeed is not None:
                        if self.state.global_step == 0:
                            logger.info("    You are using deepspeed+accelerate+transformers to train the model, calling self.accelerator.deepspeed_engine_wrapped.engine.backward(loss)")
                        # using `deepspeed+accelerate+transformers`, hack 1 of 2
                        self.accelerator.deepspeed_engine_wrapped.engine.backward(loss)
                    else:
                        if self.state.global_step == 0:
                            logger.info("    You are using DDP+accelerate+transformers to train the model.")
                        # input(">>> ")
                        self.accelerator.backward(loss)
            
            # For deepspeed, the behavior is different <- a feature of `accelerate` library
            # this step is for optimizer.step and schedular.step, which is maintained by deepspeed, not huggingface `transformers` Trainer.
            if self.args.deepspeed is not None:
                # using `deepspeed+accelerate+transformers`, hack 2 of 2
                self.accelerator.deepspeed_engine_wrapped.engine.step()
                if self.state.global_step == 0:
                    logger.info("    You are using deepspeed+accelerate+transformers to train the model, calling self.accelerator.deepspeed_engine_wrapped.engine.step()")
            
            # Compute accuracy of global mini-batch, only once is fine.
            with torch.no_grad():
                predicted_indices = torch.argmax(scores, axis=1)
                accuracy = torch.mean((predicted_indices == target).float())
            self.metric_hook["accuracy"].append(accuracy.item())
            
        else: # this is normal training, without gradient cache.
            with self.compute_loss_context_manager():
                query, passage = inputs
                outputs = model(query=query, passage=passage, **additional_args)
                q_reps = outputs.q_reps
                p_reps = outputs.p_reps
                
                if self.args.negatives_x_device:
                    # gather q_reps, p_reps across all gpus
                    
                    q_reps = self.dist_gather_tensor(q_reps) # <- all of the gathered large tensor do not have gradient, except the partition of this gpu
                    p_reps = self.dist_gather_tensor(p_reps) # <- all of the gathered large tensor do not have gradient, except the partition of this gpu
                    
                    scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
                    logger.info(f"scores.shape = {scores.shape}")
                    scores = scores / self.args.softmax_temperature
                    target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                    
                    # assert passage["input_ids"].shape[0] % query["input_ids"].shape[0] == 0, "expect #of passage to be multiple of #of query"
                    
                    # n_passages = passage["input_ids"].shape[0] // query["input_ids"].shape[0]
                    n_passages = self.train_dataset.data_args.train_n_passages
                    
                    if self.args.biaxial_loss:
                        raise NotImplementedError("Biaxial loss is not suggested to use.")

                        # query对doc的 CE loss
                        doc_query_loss = F.cross_entropy(scores.t()[::n_passages], target, reduction='mean')
                        target = target * n_passages
                        query_doc_loss = F.cross_entropy(scores, target, reduction='mean')
                        # doc对query的 CE loss(hard negative 的doc没有对应的query，算CE需要去掉，也就是只保留是postive的doc算第二轴的loss)
                        loss = (query_doc_loss + doc_query_loss) / 2
                    else:
                        target = target * n_passages
                        loss = F.cross_entropy(scores, target, reduction='mean')
                        
                    # In distributed training, global CE backward will give one gpu its own gradient, 
                    # But DDP will average the gradient of all gpus, here, an extra mean is introduced. 
                    # then, we need to multiply loss by self.world_size to make the loss the same AS IF it is trained on a single LARGE GPU
                    loss = loss * self.world_size
                    
                    with torch.no_grad():
                        predicted_indices = torch.argmax(scores, axis=1)
                        accuracy = torch.mean((predicted_indices == target).float())
                else:
                    raise NotImplementedError("simple contrastive learning is not implemented yet")
                
            # hacked metric logging:
            self.metric_hook["accuracy"].append(accuracy.item())
                
            # if self.args.n_gpu > 1: # this seems useless in our case?
            #     loss = loss.mean()  # mean() to average on multi-gpu parallel training

            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps



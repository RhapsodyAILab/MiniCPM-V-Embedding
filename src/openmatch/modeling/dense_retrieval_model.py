# Adapted from Tevatron (https://github.com/texttron/tevatron)

import copy
import importlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    AutoConfig,
    AutoModel,
    BatchEncoding,
    PreTrainedModel,
    T5EncoderModel,
)

from transformers.modeling_outputs import ModelOutput

from ..arguments import DataArguments
from ..arguments import DRTrainingArguments as TrainingArguments
from ..arguments import ModelArguments
from ..utils import mean_pooling
from .linear import LinearHead

# import SiglipProcessor
from openmatch.modeling.modeling_siglip.processing_siglip import SiglipProcessor

logger = logging.getLogger(__name__)


# torch.set_printoptions(threshold=10_000)
# torch.set_printoptions(threshold=10_000)  # 设置阈值为10000，对于大多数情况下应该足够

# For decoder-only models
def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


@dataclass
class DROutput(ModelOutput):
    q_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None
    accuracy: Tensor = None # loss绝对值不可参考，因为loss与temperature有关


class DRModel(nn.Module):
    def __init__(
        self,
        lm_q: PreTrainedModel,
        # lm_p: PreTrainedModel,
        # tied: bool = True,
        feature: str = "last_hidden_state",
        pooling: str = "lasttoken",
        attention: str = "causal",
        head_q: nn.Module = None,
        head_p: nn.Module = None,
        normalize: bool = False,
        model_args: ModelArguments = None,
        data_args: DataArguments = None,
        train_args: TrainingArguments = None,
        base_model_arch: str = "Llama",
        # model_name_or_path: str = "",
    ):
        super().__init__()

        # self.tied = tied
        self.lm_q = lm_q
        
        # clip: text(query); 
        # vision causallm: text(query)+image(passage); 
        # pure-text embedding text(query)+text(passage);
        
        # self.lm_p = lm_p
        # clip: image(passage)
        
        self.head_q = head_q
        self.head_p = head_p

        self.feature = feature
        self.pooling = pooling
        self.normalize = normalize
        self.attention = attention

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        
        self.base_model_arch = base_model_arch
        # self.model_name_or_path = model_name_or_path

        if train_args is not None:
            if train_args.distillation:
                self.loss_fn = (
                    nn.MSELoss() if train_args.distil_mode == "pairwise" else nn.KLDivLoss()
                )
            else:
                self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

            if train_args.negatives_x_device:
                if not dist.is_initialized():
                    raise ValueError(
                        "Distributed training has not been initialized for representation all gather."
                    )
                
                # process_rank and world_size work well for multi nodes and multi gpu setting
                # for example, 4 nodes, each node has 8 gpus, then each node has 8 processes, each process control one gpu.
                # for node 2, gpu 3, the get_rank will return rank=2*8+3
                self.process_rank = dist.get_rank()
                
                print(f"process_rank = {self.process_rank}")
                
                # world_size = 4*8 = 32
                self.world_size = dist.get_world_size()
                
                print(f"world_size = {self.world_size}")
        else:
            # raise ValueError("Please specify train_args.")
            logger.info("train_args not specified.")
        
        return

    def _get_config_dict(self):
        config = {
            # "tied": self.tied,
            "plm_backbone": {
                "type": type(self.lm_q).__name__,
                "feature": self.feature,
            },
            "pooling": self.pooling,
            "linear_head": bool(self.head_q),
            "normalize": self.normalize,
        }
        return config

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
        **kwargs
    ):  
        
        if query is not None:
            _, q_reps = self.encode_query(query, **kwargs)
        else:
            q_reps = None
        
        if passage is not None:
            if self.train_args.passage_stop_grad: # 可以选择对passage的梯度停止，inspired by GritLM & Kaiming He's Siamese network
                with torch.no_grad():
                    logger.info("you are stopping gradient of passages")
                    _, p_reps = self.encode_passage(passage, **kwargs)
            else:
                _, p_reps = self.encode_passage(passage, **kwargs)
        else:
            p_reps = None
            
        # print("p_reps shape:", p_reps.shape)
        # print("p_reps tensor:", p_reps)
        
        # print("q_reps shape:", q_reps.shape)
        # print("q_reps tensor:", q_reps)

        return DROutput(q_reps=q_reps, p_reps=p_reps)


    def encode(self, items, model, head, is_query=False, **kwargs):
        if items is None:
            return None, None # for Inferences
        items_out = model(is_query = is_query, **items, **kwargs)
        
        if (self.pooling == 'bge_pooling'):
            return None, items_out.pooler_output
        
        hidden = getattr(items_out, self.feature) # usually "last_hidden_state", if no exist, error will happen
        
        if self.pooling in [
            "lasttoken", 
            "wmean", "drop_wmean", 
            "drop_mean", "mean",
            "lasttoken_simcse",
            "siglip_pooling",
            "bge_pooling"
        ]:
            # we need attention mask in this case
            if "attention_mask" in items:
                attention_mask = getattr(items, "attention_mask")
            elif "attention_mask" in items_out:
                attention_mask = getattr(items_out, "attention_mask")
            else:
                raise ValueError("failed to get attention mask.")
        
        if (self.pooling == 'siglip_pooling'):
            return None, items_out.pooled_output
        
        if self.pooling == "lasttoken":
            # print(f"attention_mask = {attention_mask}")
            reps = last_token_pool(
                last_hidden_states=hidden,
                attention_mask=attention_mask
            )
        
        elif self.pooling == "simple_lasttoken":
            reps = hidden[:, -1, :]
        
        elif self.pooling == "wmean":
            attention_mask_ = attention_mask * attention_mask.cumsum(dim=1) # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
            # print(attention_mask.shape)
            # print(attention_mask_.shape)
            # print(attention_mask_)
            s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
            d = attention_mask_.sum(dim=1, keepdim=True).float()
            reps = s / d
        
        elif self.pooling == "drop_wmean":
            vector_dropout = nn.Dropout1d(0.3)
            # attention_mask = items["attention_mask"]
            attention_mask_ = attention_mask * attention_mask.cumsum(dim=1) # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
            # print(attention_mask.shape)
            # print(attention_mask_.shape)
            # print(attention_mask_)
            hidden_masked = hidden * attention_mask_.unsqueeze(-1).float()
            hidden_masked  = vector_dropout(hidden_masked)
            s = torch.sum(hidden_masked, dim=1)
            d = attention_mask_.sum(dim=1, keepdim=True).float()
            reps = s / d
        
        elif self.pooling == "drop_mean":
            vector_dropout = nn.Dropout1d(0.3)
            # attention_mask = getattr(items, "attention_mask")
            # print(attention_mask.shape)
            # print(attention_mask_.shape)
            # print(attention_mask_)
            hidden_masked = hidden * attention_mask.unsqueeze(-1).float()
            hidden_masked  = vector_dropout(hidden_masked)
            s = torch.sum(hidden_masked, dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            reps = s / d
        
        elif self.pooling == "mean":
            # attention_mask = getattr(items, "attention_mask")
            s = torch.sum(hidden * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            reps = s / d
        
        elif self.pooling == "lasttoken_simcse":
            # print(f"attention_mask = {attention_mask}")
            reps = last_token_pool(
                last_hidden_states=hidden,
                attention_mask=attention_mask
            )
            # SimCSE only add dropout to the passage.
            if not is_query: # passage add dropout to augment data
                reps = F.dropout(reps, p=0.1, training=True)
                
        elif self.pooling == "cls":
            # Bert-style model, use cls
            reps = hidden[:, 0, :]
        else:
            raise ValueError("Unknown pooling type: {}".format(self.pooling))
        
        assert self.normalize == True, "Normalize must be true"
        reps = F.normalize(reps, dim=1)

        return None, reps # None is hidden
        # return hidden, reps

    def encode_passage(self, psg, **kwargs):
        return self.encode(psg, self.lm_q, self.head_p, is_query=False, **kwargs)

    def encode_query(self, qry, **kwargs):
        return self.encode(qry, self.lm_q, self.head_q, is_query=True, **kwargs)

    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        model_name_or_path: str = None,
        data_args: DataArguments = None,
        train_args: TrainingArguments = None,
        **hf_kwargs,
    ):
        
        model_name_or_path = model_name_or_path or model_args.model_name_or_path
        
        # load local
        config = None
        head_q = head_p = None
        # if os.path.exists(os.path.join(model_name_or_path, "openmatch_config.json")):
        #     with open(os.path.join(model_name_or_path, "openmatch_config.json")) as f:
        #         config = json.load(f)

        # tied = not model_args.untie_encoder
        # config_json = json.load(open(os.path.join(model_name_or_path, 'config.json')))
        if 'bge' in model_name_or_path:
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            config_json = config.to_dict()
        else:
            config_json = json.load(open(os.path.join(model_name_or_path, 'config.json')))
        
        # raise NotImplementedError
        # hacked
        assert "_name_or_path" in config_json, "building model will need to determine the modeling file, please make sure _name_or_path exists in config.json"
        
        # ------------- config and model --------------
        # if "siglip" in config_json['_name_or_path'] or "SigLIP" in config_json['_name_or_path']:
        #     logging.info("using SIGLIP model, load modeling from openmatch.modeling.modeling_siglip")
        #     from openmatch.modeling.modeling_siglip.configuration_siglip import SiglipConfig as config_cls
        #     from openmatch.modeling.modeling_siglip.configuration_siglip import SiglipTextConfig, SiglipVisionConfig
        #     from openmatch.modeling.modeling_siglip.modeling_siglip import SiglipModel as model_class
        # elif "bge" in config_json['_name_or_path']:
        #     from openmatch.modeling.modeling_bge.modeling_bge import Bge_modified as model_class
        # elif "CPM-2B" in config_json["_name_or_path"]: # in base model config.json
        #     # print(config_json["architectures"])
        #     # logging.info("using CPM model, load modeling from openmatch.modeling.modeling_smartcpm")
        #     # from openmatch.modeling.modeling_smartcpm.configuration_smartcpm import SmartCPMConfig as config_cls
        #     # from openmatch.modeling.modeling_smartcpm.modeling_smartcpm import SmartCPMModel # what we use in DRTrainer
        #     # model_class = SmartCPMModel
        #     logging.info("using CPM model, load modeling from openmatch.modeling.modeling_minicpm")
        #     from openmatch.modeling.modeling_minicpm.modeling_minicpm import MiniCPMEmbedding as model_class
        #     from openmatch.modeling.modeling_minicpm.configuration_minicpm import MiniCPMConfig as config_cls
        # elif "MiniCPM-Llama3-V-2_5" in config_json["_name_or_path"]:
        #     from openmatch.modeling.modeling_minicpmv_llama.modeling_minicpmv import MiniCPMVEmbedding as model_class
        #     config_cls = AutoConfig
        if "MiniCPM-V-2.0" in config_json["_name_or_path"]:
            from openmatch.modeling.modeling_minicpmv.modeling_minicpmv import MiniCPMVEmbedding as model_class
            from openmatch.modeling.modeling_minicpmv.configuration_minicpm import MiniCPMVConfig as config_cls
        else: # other model
            logging.info("using AutoModel model")
            config_cls = AutoConfig
            model_class = AutoModel
            hf_kwargs["trust_remote_code"]=True
        
        logger.info(f"model class = {model_class}")
        
        # if "siglip" in config_json['_name_or_path'] or "SigLIP" in config_json['_name_or_path']:
        #     text_config = SiglipTextConfig.from_pretrained(model_name_or_path)
        #     vision_config = SiglipVisionConfig.from_pretrained(model_name_or_path)
        #     # kwargs = None
        #     # kwargs = dict(use_bfloat16 = False, torch_dtype = "bfloat16")
        #     config = config_cls.from_text_vision_configs(text_config, vision_config)
        #     # print("打印config", config)
        # elif "bge" in config_json['_name_or_path']:
        #     pass
        # else:
        config = config_cls.from_pretrained(model_name_or_path)
        
        # add attention pattern 
        if model_args.attention == "bidirectional":
            config.is_causal = False
        elif model_args.attention == "causal":
            # config.is_causal = True
            pass
        else:
            raise ValueError(f"attention type {model_args.attention} is not valid")
        
        # Create raw hf model
        # if ("siglip" in config_json['_name_or_path']):
        #     lm_q = model_class.from_pretrained(
        #         model_name_or_path, 
        #         trust_remote_code=True,
        #         config=config,
        #         **hf_kwargs
        #     )
        # else:
        if ('trust_remote_code' in hf_kwargs):
            del hf_kwargs['trust_remote_code']
        
        if model_args.dtype == 'bfloat16':
            lm_q = model_class.from_pretrained(
                model_name_or_path, 
                trust_remote_code=True,
                attn_implementation=model_args.attn_implementation, 
                config=config,
                dtype=torch.bfloat16
                **hf_kwargs
            )
        else:
            lm_q = model_class.from_pretrained(
                model_name_or_path, 
                trust_remote_code=True,
                attn_implementation=model_args.attn_implementation, 
                config=config,
                **hf_kwargs
            )
        
        if "siglip" in config_json['_name_or_path'] or "SigLIP" in config_json['_name_or_path']:
            lm_q.processor = SiglipProcessor.from_pretrained(model_name_or_path)
        
        
        base_model_arch = type(lm_q).__name__ # in case LoRA will replace class name
        logger.info(f"base model type = {base_model_arch}")
        
        # Inject LoRA if use LoRA
        if model_args.lora:
            logger.info("Using LoRA")
            from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
            
            # FEATURE_EXTRACTION: Feature extraction. 
            # Provides the hidden states which can be used as embeddings or features for downstream tasks.
            # https://huggingface.co/docs/peft/package_reference/peft_types
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, 
                r=model_args.lora_r, lora_alpha=32, lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
            )
            
            # Transform the model to LoRA model
            lm_q = get_peft_model(lm_q, peft_config)
            logger.info("trainable parameters")
            lm_q.print_trainable_parameters()
        
        # lm_p = copy.deepcopy(lm_q) if not tied else lm_q
        
        # Finally add linear head 
        if model_args.add_linear_head:
            # head_q = LinearHead(model_args.projection_in_dim, model_args.projection_out_dim)
            raise NotImplementedError
            # head_p = copy.deepcopy(head_q) if not tied else head_q

        model = cls(
            lm_q=lm_q,
            feature=model_args.feature,
            pooling=model_args.pooling,
            attention=model_args.attention,
            head_q=head_q,
            head_p=head_p,
            normalize=model_args.normalize,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
            base_model_arch=base_model_arch,
            # model_name_or_path=model_name_or_path,
        )
        
        # model.bfloat16()
        
        return model

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        # logger.info(f"*** gradient_checkpointing_kwargs = {gradient_checkpointing_kwargs}")
        gradient_checkpointing_kwargs["use_reentrant"] = False # handle a bug with DDP
        # tied = not self.model_args.untie_encoder
        # logger.info(f"*** tied = {tied}")
        self.lm_q.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs) # this model should be transformers model
        # if not tied:
        #     self.lm_p.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs) # this model should be transformers model
        return
    
    # def save(self, output_dir: str): # not compatible with deepspeed!
    #     print("save model")
    #     if not self.tied:
    #         # os.makedirs(os.path.join(output_dir, "query_model"))
    #         # os.makedirs(os.path.join(output_dir, "passage_model"))
    #         # self.lm_q.save_pretrained(os.path.join(output_dir, "query_model"))
    #         # self.lm_p.save_pretrained(os.path.join(output_dir, "passage_model"))
    #         # if self.head_q is not None:
    #         #     self.head_q.save(os.path.join(output_dir, "query_head"))
    #         #     self.head_p.save(os.path.join(output_dir, "passage_head"))
    #         raise NotImplementedError
    #     else:
    #         self.lm_q.save_pretrained(output_dir) # work for peft model
    #         if self.head_q is not None:
    #             self.head_q.save(output_dir)

        with open(os.path.join(output_dir, "openmatch_config.json"), "w") as f:
            json.dump(self._get_config_dict(), f, indent=4)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)] # gpus across all nodes , counted
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class DRModelForInference(DRModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval()

    @torch.no_grad()
    def encode_passage(self, psg, **kwargs):
        return super(DRModelForInference, self).encode_passage(psg, **kwargs)

    @torch.no_grad()
    def encode_query(self, qry, **kwargs):
        return super(DRModelForInference, self).encode_query(qry, **kwargs)

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
        **kwargs # can pass tokenizer, etc.
    ):
        q_hidden, q_reps = self.encode_query(query, **kwargs) # can pass tokenizer, etc.
        p_hidden, p_reps = self.encode_passage(passage, **kwargs) # can pass tokenizer, etc.
        return DROutput(q_reps=q_reps, p_reps=p_reps)

class DRModelForGDCache(DRModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):
        q_hidden, q_reps = self.encode_query(query)
        p_hidden, p_reps = self.encode_passage(passage)
        return DROutput(q_reps=q_reps, p_reps=p_reps)


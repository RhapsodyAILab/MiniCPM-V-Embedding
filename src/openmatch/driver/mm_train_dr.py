# Adapted from Tevatron (https://github.com/texttron/tevatron)

import logging
import os
import sys
import json

from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed

from openmatch.arguments import DataArguments
from openmatch.arguments import DRTrainingArguments as TrainingArguments
from openmatch.arguments import ModelArguments
# from openmatch.dataset import MappingDRTrainDataset, QPCollator, StreamDRTrainDataset
from openmatch.dataset import MappingMMDRTrainDataset, StreamMMDRTrainDataset, MMQPCollator
from openmatch.modeling import DRModel
from openmatch.trainer import DRTrainer as Trainer

# from openmatch.utils import get_delta_model_class
import torch

logger = logging.getLogger(__name__)



def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    print(f"training_args.local_rank = {training_args.local_rank}")
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    # config = AutoConfig.from_pretrained(
    #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    #     num_labels=num_labels,
    #     # cache_dir=model_args.cache_dir,
    #     trust_remote_code=True,
    # )
    if 'bge' in model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        config_json = config.to_dict()
    else:
        config_json = json.load(open(os.path.join(model_args.model_name_or_path, 'config.json')))
    
    # hacked config
    # config_json = json.load(open(os.path.join(model_args.model_name_or_path, 'config.json')))
    # if "siglip" in config_json['_name_or_path'] or "SigLIP" in config_json['_name_or_path']:
    #     from openmatch.modeling.modeling_siglip.tokenization_siglip import SiglipTokenizer as tokenizer_cls
    # elif "MiniCPM-Llama3-V-2_5" in config_json["_name_or_path"]:
    #     # from openmatch.modeling.modeling_minicpmv_llama.modeling_minicpmv import MiniCPMV as model_cls
    #     from openmatch.modeling.modeling_minicpmv_llama.modeling_minicpmv import PreTrainedTokenizerFastWrapper as tokenizer_cls
    if "MiniCPM-V-2.0" in config_json["_name_or_path"]:
        # from openmatch.modeling.modeling_minicpmv.modeling_minicpmv import MiniCPMV as model_cls
        from openmatch.modeling.modeling_minicpmv.modeling_minicpmv import LlamaTokenizerWrapper as tokenizer_cls
    else:
        # raise NotImplementedError("your model config arch is not supported")
        tokenizer_cls = AutoTokenizer

    # ----- tokenizer -----
    tokenizer = tokenizer_cls.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False, 
    )
    
    # ----- model ----
    
    model = DRModel.build(
        model_args=model_args,
        data_args=data_args,
        train_args=training_args,
        cache_dir=model_args.cache_dir
    )
    
    # ---- processor ----
    
    # config_json = json.load(open(os.path.join(model_args.model_name_or_path, 'config.json')))
    # if "siglip" in config_json['_name_or_path']:
        # processor = if model.lm_q.processor
    #     logging.info("using SIGLIP model, load processor from openmatch.modeling.processing_siglip")
    #     from openmatch.modeling.modeling_siglip.processing_siglip import SiglipProcessor as processor_class
        
    #     processor = processor_class.from_pretrained(model_name_or_path, tokenizer=tokenizer)
    
    # else:
    #     logging.info("no need to load processor, use model() directly.")
    #     processor_class = None
    #     processor = None
    
    
    
    # logger.info(f"processor class = {processor_class}")
    
    # streaming or not
    # if data_args.dataset_class == "text":
    #     train_dataset_cls = (
    #         MappingDRTrainDataset if training_args.use_mapping_dataset else StreamDRTrainDataset
    #     )
    # elif data_args.dataset_class == "multimodal":
    
    train_dataset_cls = ( # for multimodal dense retrieval
        MappingMMDRTrainDataset if training_args.use_mapping_dataset else StreamMMDRTrainDataset
    )
    # else:
    #     raise NotImplementedError("dataset_class not supported.")
    
    train_dataset = train_dataset_cls(
        tokenizer,
        data_args,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir,
    )
    
    logger.info(f"DataArgs: {data_args}")
    
    eval_dataset = (
        train_dataset_cls(
            tokenizer,
            data_args,
            is_eval=True,
            cache_dir=data_args.data_cache_dir or model_args.cache_dir,
        )
        if data_args.eval_path is not None
        else None
    )
    
    # Handle data collator (for handling batch truncation and padding)
    # data_collator = None

    data_collator = MMQPCollator(
        # tokenizer,
    )

    # data_collator = MMQPCollator(tokenizer, max_q_len=data_args.q_max_len, max_p_len=data_args.p_max_len)
    
    # else:
    #     if (data_args.q_max_len != 0) or (data_args.p_max_len != 0):
    #         raise ValueError("--q_max_len and --p_max_len are not supported with collator none, please remove them.")
    # def identity(*args):
    #     return args
    # data_collator = identity
    
    # trainer_cls = GCDenseTrainer if training_args.grad_cache else Trainer
    
    trainer_cls = Trainer
    
    trainer = trainer_cls(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        model_name_or_path=model_args.model_name_or_path,
        # processor = model.lm_q.processor
    )
    train_dataset.trainer = trainer

    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero(): # should be LOCAL_RANK=0
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()

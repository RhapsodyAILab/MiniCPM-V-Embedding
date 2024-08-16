# Adapted from Tevatron (https://github.com/texttron/tevatron)

import glob
import logging
import os
import random
from typing import Callable, Dict, List, Union, Optional
import json

from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers.trainer_pt_utils import IterableDatasetShard

from ..arguments import DataArguments, DRPretrainingDataArguments
from ..data_augmentation_strategy import Cropping, NullStrategy, SequentialStrategies
from ..trainer import DRTrainer

import torch.distributed as dist

import torch

import base64
from PIL import Image
from io import BytesIO
# from torchvision import transforms
# from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
# import math

logger = logging.getLogger(__name__)


class TrainDatasetBase:
    """
    Abstract base class for all train datasets in Openmatch.\n
    This implants arguments and data preparation, but should be mostly used for identifying an OpenMatch Train Dataset.\n
    All future dataset ABCs would subclass this and `(Iterable)Dataset`.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        trainer: DRTrainer = None,
        is_eval: bool = False,
        shuffle_seed: int = None,
        cache_dir: str = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.trainer = trainer
        self.is_eval = is_eval
        self._prepare_data(data_args, shuffle_seed, cache_dir)

    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        if not self.is_eval:
            self.data_files = (
                [data_args.train_path]
                if data_args.train_dir is None
                else glob.glob(os.path.join(data_args.train_dir, "*.jsonl"))
            )
        else:
            self.data_files = [data_args.eval_path]

    def get_process_fn(self, epoch, hashed_seed):
        raise NotImplementedError


class StreamTrainDatasetMixin(IterableDataset):
    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        super()._prepare_data(data_args, shuffle_seed, cache_dir)
        self.dataset = load_dataset(
            "json", data_files=self.data_files, streaming=True, cache_dir=cache_dir
        )["train"]
        sample = list(self.dataset.take(1))[0]
        self.all_columns = sample.keys()

    def __len__(self):
        if self.data_args.train_dir is not None:
            logger.info("--train_dir is used, so first attempting to load dataset length from metadata...")
            metadata_path = os.path.join(self.data_args.train_dir, "metadata.json")
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.loads(f.read())
                    length = int(metadata["length"])
                    logger.info(f"loaded metadata, using length = {length}")
            except Exception as e:
                logger.warning(e)
                logger.info(f"--train_dir is used, but could not load 'length' key from {metadata_path}, now use wc -l to get the length, which could be very slow...")

            return length
        
        logger.info("attempting to load dataset length using wc -l, which could be very slow...")
        
        concat_filenames = " ".join(self.data_files)
        count = 0
        with os.popen("wc -l {}".format(concat_filenames)) as f:
            for line in f:
                lc, filename = line.strip().split()
                lc = int(lc)
                if filename != "total":
                    count += lc
        return count

    def __iter__(self):
        # rank = dist.get_rank()
        # print(f"rank = {rank}, Fetching once")
        
        # if not self.is_eval:
        #     epoch = int(self.trainer.state.epoch)
        #     _hashed_seed = hash(self.trainer.args.seed)
        #     self.dataset.set_epoch(epoch)
        #     return iter(
        #         self.dataset.map(
        #             self.get_process_fn(epoch, _hashed_seed), remove_columns=self.all_columns
        #         )
        #     )
        return iter(self.dataset.map(self.get_process_fn(0, None), remove_columns=self.all_columns))


class MappingTrainDatasetMixin(Dataset):
    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        super()._prepare_data(data_args, shuffle_seed, cache_dir)
        self.dataset = load_dataset(
            "json", data_files=self.data_files, streaming=False, cache_dir=cache_dir
        )["train"]
        
        # manually shuffle here
        # logger.info(f"shuffle_seed = {shuffle_seed}")
        # self.dataset = (
        #     self.dataset.shuffle(seed=shuffle_seed)
        #     if shuffle_seed is not None
        #     else self.dataset
        # )
        
        sample = self.dataset[0]
        self.all_columns = sample.keys()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        group = self.dataset[index]
        # if not self.is_eval:
        #     epoch = int(self.trainer.state.epoch)
        #     _hashed_seed = hash(index + self.trainer.args.seed)
        #     return self.get_process_fn(epoch, _hashed_seed)(group) # 负例采样
        return self.get_process_fn(0, None)(group) # 指定负例


class DRTrainDataset(TrainDatasetBase):
    def create_one_example(self, text: str, is_query=False):
        return self.tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            """
            example format:
            {
                "query": ["query instruction", "query text"],
                "pos": ["passage instruction", "positve document 1"], # usually 1
                "neg": ["passage instruction", 
                    "negative document 1", 
                    "negative document 2", 
                    "negative document 3", 
                ...] # can be set by --train_n_passages
            }
            """
            
            query: str = " ".join(example["query"]) if self.data_args.query_instruction else example["query"][1] # with query instruction
            pos: str = " ".join(example["pos"]) if self.data_args.corpus_instruction else example["pos"][1]  # without passage instruction
            
            # TODO: this assertion could be eliminated
            assert self.data_args.train_n_passages >= 1 # we have hard negative
            
            negs: List[str] = [" ".join([example["neg"][0], example["neg"][i]]) if self.data_args.corpus_instruction else example["neg"][i] for i in range(1, self.data_args.train_n_passages)]  # without passage instruction
            
            # assert self.data_args.train_n_passages == 2 # MEDI and more data
            
            # print("query", query, "pos", pos)
            
            encoded_query = [self.create_one_example(query, is_query=True)]
            
            # print(encoded_query)
            
            # print("encoded_query", encoded_query)
            encoded_passages = self.create_one_example(pos)
            encoded_passages.extend([self.create_one_example(neg) for neg in negs])
            # print("encoded_passages", encoded_passages)
            # raise Exception
            # Avoid name conflict with query in the original dataset
            return {"query_": encoded_query, "passages": encoded_passages}

        return process_fn


# For multimodal Dense Retrieval Model
class MMDRTrainDataset(TrainDatasetBase):
    def convert_base64string_to_image(self, base64_string):
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        image = image.convert("RGB")
        return image
    
    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            # here we don't do tokenization.
            
            # MM example format:
            # {
            #     "query": {
            #             "instruction": string,
            #             "image": None or base64_string,
            #             "text": string
            #         }
            #     ]
            #     "pos": [
            #         {
            #             "instruction": string,
            #             "image": None or base64_string,
            #             "text": string
            #         }, ...
            #     ]
            #     "neg": [
            #         {
            #             "instruction": string,
            #             "image": None or base64_string,
            #             "text": string
            #         }, ...
            #     ]
            # }
            
            # TODO: this assertion could be eliminated
            # assert self.data_args.train_n_passages > 1 # we have hard negative
            # assert self.data_args.train_n_passages <= (len(example["neg"]) + 1), "--train_n_passages should <= number of provided negative samples + number of positive samples "
            
            query = example["query"]
            pos = example["pos"][0]
            neg = example["neg"][0: self.data_args.train_n_passages - 1] # we can exactly have self.data_args.train_n_passages-1 negative passages
            
            # Step1: merge instructions into texts if requested
            if self.data_args.query_instruction:
                query["text"] = query["instruction"] + query["text"]
                # query["image"] = pos["image"] # test

            if self.data_args.corpus_instruction: # test
                pos["text"] = pos["instruction"] + pos["text"]
                for idx in range(len(neg)):
                    neg[idx]["text"] = neg[idx]["instruction"] + neg[idx]["text"]
            
            # Step2: convert base64_string to images
            if query["image"] is not None:
                query["image"] = self.convert_base64string_to_image(query["image"])
            
            if pos["image"] is not None:
                pos["image"] = self.convert_base64string_to_image(pos["image"])
            else:
                # image is None
                if pos["text"] == "":
                    logger.warning("a passage data has neither image nor text, please make sure it is expected, now will replace it with a random string.")
                    pos["text"] = "empty passage"
            
            for idx in range(len(neg)):
                if neg[idx]["image"] is not None:
                    neg[idx]["image"] = self.convert_base64string_to_image(neg[idx]["image"])
                else:
                    if neg[idx]["text"] == "":
                        logger.warning("a passage data has neither image nor text, please make sure it is expected, now will replace it with a random string.")
                        neg[idx]["text"] = "empty passage"
            
            
            
            query_ = [query]
            passages = [pos, *neg]
            
            return {"query_": query_, "passages": passages}

        return process_fn


class StreamDRTrainDataset(StreamTrainDatasetMixin, DRTrainDataset):
    pass


class MappingDRTrainDataset(MappingTrainDatasetMixin, DRTrainDataset):
    pass


class StreamMMDRTrainDataset(StreamTrainDatasetMixin, MMDRTrainDataset):
    pass


class MappingMMDRTrainDataset(MappingTrainDatasetMixin, MMDRTrainDataset):
    pass


class DRPretrainDataset(TrainDatasetBase):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DRPretrainingDataArguments,
        trainer: DRTrainer = None,
        is_eval: bool = False,
        shuffle_seed: int = None,
        cache_dir: str = None,
    ) -> None:
        super(DRPretrainDataset, self).__init__(
            tokenizer, data_args, trainer, is_eval, shuffle_seed, cache_dir
        )
        pretrain_strategies_str = (
            data_args.pretrain_strategies.split(",")
            if data_args.pretrain_strategies is not None
            else []
        )
        strategies = []
        for strategy_str in pretrain_strategies_str:
            if strategy_str == "null":
                strategies.append(NullStrategy())
                logger.info("Adding NullStrategy")
            elif strategy_str == "crop":
                strategies.append(
                    Cropping(
                        ratio_min=data_args.cropping_ratio_min,
                        ratio_max=data_args.cropping_ratio_max,
                    )
                )
                logger.info(
                    "Adding Cropping, ratio_min={}, ratio_max={}".format(
                        data_args.cropping_ratio_min, data_args.cropping_ratio_max
                    )
                )
            else:
                raise ValueError("Unknown pretraining strategy: {}".format(strategy_str))
        self.apply_strategy = SequentialStrategies(*strategies)

    def create_one_example(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
        text_encoding = self.apply_strategy(text_encoding)
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation="only_first",
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            content = example[self.data_args.pretrain_target_field]
            encoded_query = self.create_one_example(content, is_query=True)
            encoded_passages = [self.create_one_example(content)]

            return {"query_": encoded_query, "passages": encoded_passages}

        return process_fn


class StreamDRPretrainDataset(StreamTrainDatasetMixin, DRPretrainDataset):
    pass


class MappingDRPretrainDataset(MappingTrainDatasetMixin, DRPretrainDataset):
    pass


class RRTrainDataset(TrainDatasetBase):
    def create_one_example(self, qry_encoding, psg_encoding) -> BatchEncoding:
        if self.data_args.encode_as_text_pair:
            item = self.tokenizer.encode_plus(
                qry_encoding,
                psg_encoding,
                truncation="longest_first",
                max_length=self.data_args.q_max_len + self.data_args.p_max_len + 2,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=True,
            )
        else:
            item = self.tokenizer.encode_plus(
                qry_encoding + psg_encoding,
                truncation="longest_first",
                max_length=self.data_args.q_max_len + self.data_args.p_max_len + 2,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
        return item

    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            qry = example["query"]
            group_positives = example["positives"]
            group_negatives = example["negatives"]

            if self.data_args.positive_passage_no_shuffle or hashed_seed is None:
                pos_psg = group_positives[0]
            else:
                pos_psg = group_positives[(hashed_seed + epoch) % len(group_positives)]
            encoded_pos_pair = self.create_one_example(qry, pos_psg)

            if hashed_seed is None:
                neg_psg = group_negatives[0]
            else:
                neg_psg = group_negatives[(hashed_seed + epoch) % len(group_negatives)]
            encoded_neg_pair = self.create_one_example(qry, neg_psg)
            return {"pos_pair": encoded_pos_pair, "neg_pair": encoded_neg_pair}

        return process_fn


class StreamRRTrainDataset(StreamTrainDatasetMixin, RRTrainDataset):
    pass


class MappingRRTrainDataset(MappingTrainDatasetMixin, RRTrainDataset):
    pass


class QGTrainDataset(TrainDatasetBase):
    def create_one_example(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation="only_first",
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            qry = example["query"]
            group_positives = example["positives"]
            if self.data_args.positive_passage_no_shuffle or hashed_seed is None:
                pos_psg = group_positives[0]
            else:
                pos_psg = group_positives[(hashed_seed + epoch) % len(group_positives)]

            encoded_query = self.create_one_example(qry, is_query=True).input_ids
            encoded_query[encoded_query == self.tokenizer.pad_token_id] == -100
            encoded_psg = self.create_one_example(pos_psg)
            psg_input_ids, psg_attention_mask = encoded_psg.input_ids, encoded_psg.attention_mask
            return {
                "input_ids": psg_input_ids[0],
                "attention_mask": psg_attention_mask[0],
                "labels": encoded_query[0],
            }

        return process_fn


class StreamQGTrainDataset(StreamTrainDatasetMixin, QGTrainDataset):
    pass


class MappingQGTrainDataset(MappingTrainDatasetMixin, QGTrainDataset):
    pass


class CQGTrainDataset(TrainDatasetBase):
    def create_one_example(
        self,
        qry_encoding: List[int] = None,
        psg_encoding_pos: List[int] = None,
        psg_encoding_neg: List[int] = None,
    ) -> BatchEncoding:
        if qry_encoding is not None:
            return self.tokenizer.encode_plus(
                qry_encoding,
                truncation="only_first",
                max_length=self.data_args.q_max_len,
                padding="max_length",
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors="pt",
            )
        return self.tokenizer.encode_plus(
            psg_encoding_pos + psg_encoding_neg,
            truncation="only_first",
            max_length=self.data_args.p_max_len * 2,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )

    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            qry = example["query"]
            group_positives = example["positives"]
            group_negatives = example["negatives"]
            if self.data_args.positive_passage_no_shuffle or hashed_seed is None:
                pos_psg = group_positives[0]
            else:
                pos_psg = group_positives[(hashed_seed + epoch) % len(group_positives)]

            if hashed_seed is None:
                neg_psg = group_negatives[0]
            else:
                neg_psg = group_negatives[(hashed_seed + epoch) % len(group_negatives)]

            encoded_query = self.create_one_example(qry_encoding=qry).input_ids
            encoded_query[encoded_query == self.tokenizer.pad_token_id] == -100
            encoded_psg_pair = self.create_one_example(
                psg_encoding_pos=pos_psg, psg_encoding_neg=neg_psg
            )
            psg_input_ids, psg_attention_mask = (
                encoded_psg_pair.input_ids,
                encoded_psg_pair.attention_mask,
            )
            return {
                "input_ids": psg_input_ids[0],
                "attention_mask": psg_attention_mask[0],
                "labels": encoded_query[0],
            }

        return process_fn


class StreamCQGTrainDataset(StreamTrainDatasetMixin, CQGTrainDataset):
    pass


class MappingCQGTrainDataset(MappingTrainDatasetMixin, CQGTrainDataset):
    pass


class PairwiseDistillationTrainDataset(TrainDatasetBase):
    def create_one_example(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation="only_first",
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            qry = self.create_one_example(example["query"], is_query=True)
            pos = self.create_one_example(example["positive"])
            neg = self.create_one_example(example["negative"])
            score = example["score"]
            return {"query_": qry, "positive_": pos, "negative_": neg, "score_": score}

        return process_fn


class StreamPairwiseDistillationTrainDataset(
    StreamTrainDatasetMixin, PairwiseDistillationTrainDataset
):
    pass


class MappingPairwiseDistillationTrainDataset(
    MappingTrainDatasetMixin, PairwiseDistillationTrainDataset
):
    pass


class ListwiseDistillationTrainDataset(TrainDatasetBase):
    def create_one_example(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation="only_first",
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            qry = example["query"]
            encoded_query = self.create_one_example(qry, is_query=True)
            encoded_passages = []
            passages = example["docs"]
            scores = example["scores"]
            passages_and_scores = list(zip(passages, scores))

            if len(passages) < self.data_args.train_n_passages:
                if hashed_seed is not None:
                    psgs = random.choices(passages_and_scores, k=self.data_args.train_n_passages)
                else:
                    psgs = [x for x in passages_and_scores]
                    psgs = psgs * 2
                    psgs = psgs[: self.data_args.train_n_passages]
            else:
                _offset = epoch * self.data_args.train_n_passages % len(passages)
                psgs = [x for x in passages_and_scores]
                if hashed_seed is not None:
                    random.Random(hashed_seed).shuffle(psgs)
                psgs = psgs * 2
                psgs = psgs[_offset : _offset + self.data_args.train_n_passages]

            for psg in psgs:
                encoded_passages.append(self.create_one_example(psg[0]))

            assert len(encoded_passages) == self.data_args.train_n_passages

            return {
                "query_": encoded_query,
                "passages": encoded_passages,
                "scores_": [x[1] for x in psgs],
            }  # Avoid name conflict with query in the original dataset

        return process_fn


class StreamListwiseDistillationTrainDataset(
    StreamTrainDatasetMixin, ListwiseDistillationTrainDataset
):
    pass


class MappingListwiseDistillationTrainDataset(
    MappingTrainDatasetMixin, ListwiseDistillationTrainDataset
):
    pass



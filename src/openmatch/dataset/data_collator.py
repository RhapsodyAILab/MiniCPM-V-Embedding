# Adapted from Tevatron (https://github.com/texttron/tevatron)

from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding, DefaultDataCollator
from transformers import BatchEncoding

# last = ""
def print_structure(struct, indent=0):
    # global last
    prefix = ' ' * indent
    if isinstance(struct, dict):
        print(f"{prefix}"+"{")
        for key, value in struct.items():
            print(f'{prefix}{key}:')
            print_structure(value, indent + 4)
        # if last == "Tensor":
            # print("\n")
        print(f"{prefix}"+"},")
    elif isinstance(struct, list):
        print(f"{prefix}[")
        for item in struct:
            print_structure(item, indent + 4)
        # if last == "Tensor":
            # print("\n")
        print(f"{prefix}],")
    elif "Tensor" in str(type(struct)):  # 检查是否为张量
        # if last == "Tensor":
            # print(f"{struct.shape}, ", end='')
        # else:
        print(f'{prefix}{struct.shape},')  # 打印张量的形状和内容
        # last = "Tensor"
    else:
        # print(f'{prefix}{struct}')
        pass


@dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq = [f["query_"] for f in features]
        dd = [f["passages"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])
            if isinstance(dd[0], list):
                dd = sum(dd, [])
        
        q_collated = self.tokenizer.pad(
            qq,
            padding="max_length", # here we have no padding, padding with `data collator`
            max_length=self.max_q_len,
            return_tensors="pt",
            # truncation=True,
        )
        
        d_collated = self.tokenizer.pad(
            dd,
            padding="max_length",
            max_length=self.max_p_len,
            return_tensors="pt",
            # truncation=True,
        )
        
        return q_collated, d_collated

def reshape(input_list):
    # print(input_list)
    
    keys = input_list[0].keys()
    output_dict = {key: [] for key in keys}
    # print(f"output_dict, {keys}, {output_dict}")
    for data in input_list:
        # print(data)
        # print("---")
        for key in keys:
            output_dict[key].append(data[key]) # this is not append, it is extend
    # print('===')
    # print(output_dict)
    return output_dict

def stack(list_of_lists):
    merged_list = []
    for sublist in list_of_lists:
        merged_list.extend(sublist)
    return merged_list


@dataclass
class MMQPCollator(DefaultDataCollator):

    def __call__(self, features):
        query = [f["query_"] for f in features]
        passages = [f["passages"] for f in features]

        # print('>>>>')
        # print("query raw", query)
        # print('>>>>')
        
        # reshape
        query = reshape(stack(query)) # List[Dict[str, Any]] -> Dict[str, List[Any]]
        passages = reshape(stack(passages)) # List[Dict[str, Any]] -> Dict[str, List[Any]]
        
        # padding
        # query_padded_inputs = self.pad(query["input_ids"], padding_side="right")
        # query["input_ids"] = query_padded_inputs["input_ids"]
        # query["attention_mask"] = query_padded_inputs["attention_mask"]
        
        
        # print(query["attention_mask"])
        # print(query["input_ids"])
        # print_structure(query)
        
        # from IPython import embed
        # embed()
        # print(query)
        
        # query = BatchEncoding(query)
        
        # passage_padded_inputs = self.pad(passages["input_ids"], padding_side="right")
        # passages["input_ids"] = passage_padded_inputs["input_ids"]
        # passages["attention_mask"] = passage_padded_inputs["attention_mask"]
        
        # passage = BatchEncoding(passages)
        
        return query, passages


# @dataclass
# class MMQPCollatorForInference(DefaultDataCollator):

#     def __call__(self, features):
#         image = [f["image"] for f in features]
#         text = [f["text"] for f in features]

#         return {
#             "image": image,
#             "text": text
#         }

@dataclass
class PairCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        pos_pairs = [f["pos_pair"] for f in features]
        neg_pairs = [f["neg_pair"] for f in features]

        if isinstance(pos_pairs[0], list):
            pos_pairs = sum(pos_pairs, [])
        if isinstance(neg_pairs[0], list):
            neg_pairs = sum(neg_pairs, [])

        pos_pair_collated = self.tokenizer.pad(
            pos_pairs,
            padding="max_length",
            max_length=self.max_q_len + self.max_p_len + 2,
            return_tensors="pt",
        )
        neg_pair_collated = self.tokenizer.pad(
            neg_pairs,
            padding="max_length",
            max_length=self.max_q_len + self.max_p_len + 2,
            return_tensors="pt",
        )

        return pos_pair_collated, neg_pair_collated


@dataclass
class PairwiseDistillationCollator(DataCollatorWithPadding):
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq = [f["query_"] for f in features]
        positives = [f["positive_"] for f in features]
        negatives = [f["negative_"] for f in features]
        scores = [f["score_"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(positives[0], list):
            positives = sum(positives, [])
        if isinstance(negatives[0], list):
            negatives = sum(negatives, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding="max_length",
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        positives_collated = self.tokenizer.pad(
            positives,
            padding="max_length",
            max_length=self.max_p_len,
            return_tensors="pt",
        )
        negatives_collated = self.tokenizer.pad(
            negatives,
            padding="max_length",
            max_length=self.max_p_len,
            return_tensors="pt",
        )
        scores_collated = torch.tensor(scores)

        return q_collated, positives_collated, negatives_collated, scores_collated


@dataclass
class ListwiseDistillationCollator(DataCollatorWithPadding):
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq = [f["query_"] for f in features]
        dd = [f["passages"] for f in features]
        scores = [f["scores_"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding="max_length",
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding="max_length",
            max_length=self.max_p_len,
            return_tensors="pt",
        )
        scores_collated = torch.tensor(scores)

        return q_collated, d_collated, scores_collated


@dataclass
class DRInferenceCollator(DefaultDataCollator):
    def __call__(self, features):
        text_ids = [x["text_id"] for x in features]
        collated_features = super().__call__(features)
        return text_ids, collated_features


@dataclass
class RRInferenceCollator(DefaultDataCollator):
    def __call__(self, features):
        query_ids = [x["query_id"] for x in features]
        doc_ids = [x["doc_id"] for x in features]
        collated_features = super().__call__(features)
        return query_ids, doc_ids, collated_features


@dataclass
class CQGInferenceCollator(DefaultDataCollator):
    def __call__(self, features):
        query_ids = [x["query_id"] for x in features]
        pos_doc_ids = [x["pos_doc_id"] for x in features]
        neg_doc_ids = [x["neg_doc_id"] for x in features]
        collated_features = super().__call__(features)
        return query_ids, pos_doc_ids, neg_doc_ids, collated_features

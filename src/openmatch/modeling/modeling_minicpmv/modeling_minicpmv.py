import math
from typing import List, Optional
import json
import timm
import torch

import time

import torchvision
from PIL import Image
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms
from transformers import LlamaTokenizer

from transformers import BatchEncoding # note that, MiniCPMV do padding during forward, not before forward

from transformers.utils import ModelOutput
from typing import Optional, Tuple

from dataclasses import dataclass

from .configuration_minicpm import MiniCPMVConfig
from .modeling_minicpm import MiniCPMForCausalLM, MiniCPMPreTrainedModel
from .resampler import Resampler

# for faster batch inference
from concurrent.futures import ThreadPoolExecutor


class MiniCPMVPreTrainedModel(MiniCPMPreTrainedModel):
    config_class = MiniCPMVConfig


class MiniCPMV(MiniCPMVPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.llm = MiniCPMForCausalLM(config)
        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.llm.config.hidden_size
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)
        self.transform = self.init_transform()
    
    # self.lm_q.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        print(gradient_checkpointing_kwargs)
        print(f"MiniCPMV.gradient_checkpointing enbale called: {gradient_checkpointing_kwargs}")
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        print("self.llm.gradient_checkpointing_enable ... OK")
        self.vpm.set_grad_checkpointing(enable=True)
        print("self.vpm.gradient_checkpointing_enable ... OK")
        return

    def init_vision_module(self):
        model = timm.create_model(
            self.config.vision_encoder,
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=True
        )

        if isinstance(model, timm.models.VisionTransformer):
            if model.attn_pool is not None:
                model.attn_pool = torch.nn.Identity()

        if self.config.drop_vision_last_layer:
            model.blocks = model.blocks[:-1]

        return model

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(
            grid_size=int(math.sqrt(self.config.query_num)),
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True
        )

    def init_transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
                ),
            ]
        )

    # @Vision encoder 把raw pixels变成visual tokens
    def get_vision_embedding(self, pixel_values): 
        res = []
        dtype = self.vpm.pos_embed.data.dtype
        
        # first slice
        H, W = pixel_values[0].shape[-2:]
        tgt_size = (
            math.ceil(H / self.vpm.patch_embed.patch_size[0]), math.ceil(W / self.vpm.patch_embed.patch_size[0])
        )
        
        # t1 = time.time()
        vision_embedding = self.vpm.forward_features(pixel_values[0].unsqueeze(0).type(dtype))
        # print("vision_embedding.shape - 1", vision_embedding.shape)
        res.append(self.resampler(vision_embedding, tgt_size))
        # t2 = time.time()
        
        # print("first slice", t2-t1)
        
        # t1 = time.time()
        # remaining slices
        if len(pixel_values) > 1:
        
            H, W = pixel_values[1].shape[-2:]
            tgt_size = (
                math.ceil(H / self.vpm.patch_embed.patch_size[0]), math.ceil(W / self.vpm.patch_embed.patch_size[0])
            )
            vision_embedding = self.vpm.forward_features(torch.stack(pixel_values[1:], dim=0).type(dtype))
            # print("vision_embedding.shape - 2", vision_embedding.shape)
            res.append(self.resampler(vision_embedding, tgt_size))
        # t2 = time.time()
        
        # print("remaining slices", t2-t1)
        
        # for pixel_value in pixel_values:
        #     H, W = pixel_value.shape[-2:]
        #     tgt_size = (
        #         math.ceil(H / self.vpm.patch_embed.patch_size[0]), math.ceil(W / self.vpm.patch_embed.patch_size[0])
        #     )
            
        #     vision_embedding = self.vpm.forward_features(pixel_value.type(dtype))
        #     # if hasattr(self.vpm, 'num_prefix_tokens') and self.vpm.num_prefix_tokens > 0:
        #     #     vision_embedding = vision_embedding[:, self.vpm.num_prefix_tokens:]
        #     # print(vision_embedding.shape, tgt_size)
        #     # raise Exception
        #     res.append(self.resampler(vision_embedding, tgt_size))
        
        
        
        # vision_embedding = self.vpm.forward_features(pixel_value.unsqueeze(0).type(dtype))
        return torch.vstack(res)

    # 传入input_ids(includes image placeholder), pixel_values, image_bound，传出unified inputs_embeds
    def get_vllm_embedding(self, data):
        if "vision_hidden_states" not in data:
            pixel_values_list = data["pixel_values"]
            vision_hidden_states = []
            # print("self.training", self.training)
            # raise Exception
            
            # 这一部分的并行性很差，是因为UHD切分出的小图的数目不定，需要一个一个处理，暂时不能并行处理
            # cnt = 0
            for pixel_values in pixel_values_list:
                # print(cnt)
                # cnt += 1
                # print(f"UHD Image#: {len(pixel_values)}")
                # reference:
                # arxivcap原始图会产生10张小图 -> A100单卡micro_bsz=2
                # dpi100的pdf教科书会产生7张小图 -> A100单卡micro_bsz=4
                # 多卡deepspeed并行也许会提高micro_bsz
                
                # print(len(pixel_values))
                # for item in pixel_values:
                    # print(item.shape)
                
                if len(pixel_values) > 0:
                    vision_hidden_states.append(self.get_vision_embedding(pixel_values))
                    # raise Exception
                
                # elif self.training:
                #     dtype = self.vpm.pos_embed.data.dtype
                #     device = self.vpm.pos_embed.data.device
                #     dummy_image = torch.zeros(
                #         (1, 3, 224, 224), device=device, dtype=dtype
                #     )
                #     vision_hidden_states.append(self.get_vision_embedding(dummy_image))
                else:
                    vision_hidden_states.append([])
            
            # self.get_vision_embedding(pixel_values)

        else:
            vision_hidden_states = data["vision_hidden_states"]

        vllm_embedding = (
            self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
        )
        
        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i
            for i in vision_hidden_states
        ]

        bs = len(data["input_ids"])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data["image_bound"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [
                            torch.arange(r[0], r[1], dtype=torch.long)
                            for r in cur_image_bound
                        ]
                    ).to(vllm_embedding.device)

                    cur_vllm_emb.scatter_(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                    )
                elif self.training:
                    cur_vllm_emb += cur_vs_hs[0].mean() * 0
                    

        return vllm_embedding, vision_hidden_states

    # def get_vision_embedding_(self, pixel_values):
    #     res = []
    #     dtype = self.vpm.pos_embed.data.dtype
    #     for pixel_value in pixel_values: # 这里为何会sequential？
    #         H, W = pixel_value.shape[-2:]
    #         tgt_size = (
    #             math.ceil(H / self.vpm.patch_embed.patch_size[0]), math.ceil(W / self.vpm.patch_embed.patch_size[0])
    #         )
    #         vision_embedding = self.vpm.forward_features(pixel_value.unsqueeze(0).type(dtype))
    #         # if hasattr(self.vpm, 'num_prefix_tokens') and self.vpm.num_prefix_tokens > 0:
    #         #     print("num_prefix_tokens")
    #         #     raise Exception
    #         #     vision_embedding = vision_embedding[:, self.vpm.num_prefix_tokens:]
    #         res.append(self.resampler(vision_embedding, tgt_size))
    #     return torch.vstack(res)

    # def get_vllm_embedding_(self, data):
    #     pixel_values_list = data["pixel_values"]
    #     vision_hidden_states_splitter = []
    #     pixel_values_list_flattened = []

    #     begin = 0
    #     # end = 0
    #     for pixel_values in pixel_values_list:
            
    #         if len(pixel_values) > 0:
    #             pixel_values_list_flattened.extend(pixel_values)
    #             vision_hidden_states_splitter.append([begin, begin+len(pixel_values)])
    #             begin += len(pixel_values)
    #             # vision_hidden_states.append(self.get_vision_embedding(pixel_values))
    #         elif self.training:
    #             dtype = self.vpm.pos_embed.data.dtype
    #             device = self.vpm.pos_embed.data.device
    #             dummy_image = torch.zeros(
    #                 (1, 3, 224, 224), device=device, dtype=dtype
    #             )
    #             pixel_values_list_flattened.extend(dummy_image)
    #             vision_hidden_states_splitter.append([begin, begin+1])
    #             begin += 1
    #             # vision_hidden_states.append(self.get_vision_embedding(dummy_image))
    #         else:
    #             raise NotImplementedError
        
    #     # batch forward of vpm
    #     dtype = self.vpm.pos_embed.data.dtype
        
    #     pixel_values_list_flattened = torch.stack(pixel_values_list_flattened) # List[Tensor] -> Tensor[B, ...]
  
    #     vision_hidden_states_ = self.vpm.forward_features(pixel_values_list_flattened.type(dtype))

    #     tgt_sizes = [
    #         (
    #             math.ceil(pixel_value.shape[-2] / self.vpm.patch_embed.patch_size[0]), 
    #             math.ceil(pixel_value.shape[-1] / self.vpm.patch_embed.patch_size[0]) 
    #         ) for pixel_value in pixel_values_list_flattened
    #     ]

    #     print("vision_hidden_states_.shape", vision_hidden_states_.shape)
    #     vision_hidden_states = []
    #     assert len(vision_hidden_states_.shape) == 3
    #     for vision_embedding, tgt_size in zip(vision_hidden_states_, tgt_sizes):
    #         print("vision_embedding.shape", vision_embedding.shape)
    #         vision_hidden_states.append(self.resampler(vision_embedding.unsqueeze(0), tgt_size))

    #     # self.resampler(vision_embedding, tgt_size)

    #     vllm_embedding = (
    #         self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
    #     )

    #     vision_hidden_states = [
    #         i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i
    #         for i in vision_hidden_states
    #     ]

    #     bs = len(data["input_ids"])
    #     for i in range(bs):
    #         cur_vs_hs = vision_hidden_states[i]
    #         if len(cur_vs_hs) > 0:
    #             cur_vllm_emb = vllm_embedding[i]
    #             cur_image_bound = data["image_bound"][i]
    #             if len(cur_image_bound) > 0:
    #                 image_indices = torch.stack(
    #                     [
    #                         torch.arange(r[0], r[1], dtype=torch.long)
    #                         for r in cur_image_bound
    #                     ]
    #                 ).to(vllm_embedding.device)

    #                 cur_vllm_emb.scatter_(
    #                     0,
    #                     image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
    #                     cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
    #                 )
    #             elif self.training:
    #                 cur_vllm_emb += cur_vs_hs[0].mean() * 0

    #     raise Exception("...")
    #     return vllm_embedding, vision_hidden_states

    def _convert_to_tensors(
        self, tokenizer, input_str, max_inp_length: Optional[int] = None):
        if tokenizer.add_bos_token:
            input_ids = tokenizer.encode(input_str)
        else:
            input_ids = [tokenizer.bos_id] + tokenizer.encode(input_str)
        if max_inp_length is not None:
            input_ids = input_ids[:max_inp_length]
        input_ids = torch.tensor(input_ids, dtype=torch.int32)

        image_start_tokens = torch.where(input_ids == tokenizer.im_start_id)[0]
        # 跳过 im_start
        image_start_tokens += 1
        image_end_tokens = torch.where(input_ids == tokenizer.im_end_id)[0]
        valid_image_nums = max(len(image_start_tokens), len(image_end_tokens))
        image_bound = torch.hstack(
            [
                image_start_tokens[:valid_image_nums].unsqueeze(-1),
                image_end_tokens[:valid_image_nums].unsqueeze(-1),
            ]
        )

        model_input = {}
        model_input["input_ids"] = input_ids.unsqueeze(0).to(self.device)
        model_input["image_bound"] = image_bound


        return model_input
    
    def _process_list( # pad input tensors
        self, tokenizer, data_list: List[str], max_inp_length: Optional[int] = None, padding_side: str = "right"
    ):
        # pad_keys = ["input_ids"]
        input_tensors = []
        for data in data_list:
            input_tensors.append(
                self._convert_to_tensors(tokenizer, data, max_inp_length)
            )

            
        padded = pad([i["input_ids"] for i in input_tensors], padding_side=padding_side)
        
        padded = padded.to(self.device)
        padded["image_bound"] = [i["image_bound"] for i in input_tensors]
        return padded

    def _decode(self, model_inputs, tokenizer, **kwargs): # fixed version of _decode
        output = self.llm.generate(
            inputs_embeds=model_inputs["inputs_embeds"],
            attention_mask=model_inputs["attention_mask"],
            pad_token_id=0,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs
        )
        return self._decode_text(output, tokenizer)

    def _decode_text(self, result_ids, tokenizer):
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_id:
                result = result[1:]
            if result[-1] == tokenizer.eos_id:
                result = result[:-1]
            result_text.append(tokenizer.decode(result).strip())
        return result_text

    def slice_image(self, image):
        return slice_image(
            image,
            self.config.max_slice_nums,
            self.config.scale_resolution,
            self.config.patch_size,
        )

    def get_slice_image_placeholder(self, image, tokenizer):
        image_placeholder = (
            tokenizer.im_start
            + tokenizer.unk_token * self.config.query_num
            + tokenizer.im_end
        )

        slice_images = []

        source_image, patches, best_grid = slice_image(
            image,
            self.config.max_slice_nums,
            self.config.scale_resolution,
            self.config.patch_size,
        )

        slice_images.append(source_image)
        final_placeholder = image_placeholder

        if len(patches) > 0:
            for i in range(len(patches)):
                for j in range(len(patches[0])):
                    slice_images.append(patches[i][j])

            final_placeholder += get_grid_placeholder(
                tokenizer, best_grid, self.config.query_num
            )
        return slice_images, final_placeholder

    def generate(
        self,
        data_list=None, # List[str]
        img_list=None, # List[List[PIL.Image]]
        tokenizer=None,
        max_inp_length: Optional[int] = None,
        vision_hidden_states=None, # default None
        return_vision_hidden_states=False,
        **kwargs):

        assert data_list is not None
        bs = len(data_list)
        if img_list == None:
            img_list = [[] for i in range(bs)]
        assert bs == len(img_list)
        
        # t1 = time.time()
        model_inputs = self._process_list(tokenizer, data_list, max_inp_length, padding_side="right") # will add attention mask
        # t2 = time.time()
        # print("_process_list - elapsed time", t2-t1)
        
        # t1 = time.time()
        if vision_hidden_states is None:
            pixel_values = []
            for i in range(bs):
                img_inps = []
                for img in img_list[i]:
                    img_inps.append(self.transform(img).to(self.device))
                if img_inps:
                    pixel_values.append(img_inps)
                else:
                    pixel_values.append([])
            model_inputs["pixel_values"] = pixel_values
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        # t2 = time.time()
        # print("pre - elapsed time", t2-t1)
        
        with torch.inference_mode():
            (
                model_inputs["inputs_embeds"],
                vision_hidden_states,
            ) = self.get_vllm_embedding(model_inputs)
            # here model_inputs["inputs_embeds"] have merged text and image embeddings.
            # if bs > 1:
            #     torch.save(model_inputs["inputs_embeds"], '/home/jeeves/tmp/batch.pt')
            # else:
            #     torch.save(model_inputs["inputs_embeds"], '/home/jeeves/tmp/single.pt')
            # print(model_inputs["inputs_embeds"].shape)
            # for embedding, should be forward()
            result = self._decode(model_inputs, tokenizer, **kwargs)
            # print(result)
            # exit(0)

        if return_vision_hidden_states:
            return result, vision_hidden_states

        return result
    
    def chat(
        self,
        image_list, # List[ PIL.Image ] B*PIL.Image, one image for each data
        msgs_list, # List[Dict[str, str]] B*ChatML, one ChatML Dict for each data
        tokenizer,
        vision_hidden_states=None,
        max_new_tokens=1024, 
        sampling=True,
        max_inp_length=2048,
        **kwargs):
        
        processed_image_list = []
        processed_msgs_list = []
        
        for msgs, image in zip(msgs_list, image_list):
            if not isinstance(msgs, list):
                raise NotImplementedError(f"chatml format expected, expect outmost type to be list but got {type(msgs)}")
            
            # msgs to prompt
            prompt = ""
            for i, msg in enumerate(msgs):
                role = msg["role"]
                content = msg["content"]
                assert role in ["user", "assistant"]
                if i == 0:
                    assert role == "user", "The role of first msg should be user"
                    if self.config.slice_mode:
                        images, final_placeholder = self.get_slice_image_placeholder(
                            image, tokenizer
                        ) # crop one image into multiple sub images -> List[Image]
                        content = final_placeholder + "\n" + content
                        # print(f"len pages = {len(images)}")
                    else:
                        images = [image] # only keep one image without cropping -> List[Image]
                        content = (
                            tokenizer.im_start
                            + tokenizer.unk_token * self.config.query_num
                            + tokenizer.im_end
                            + "\n"
                            + content
                        )
                prompt += "<用户>" if role == "user" else "<AI>"
                prompt += content
            # prompt += "<AI>"
            final_input = prompt
            
            
            processed_msgs_list.append(final_input)
            processed_image_list.append(images)
        
        if sampling:
            generation_config = {
                # "top_p": 0.8,
                # "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.02
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        generation_config.update(
            (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
        )

        with torch.inference_mode():
            res, vision_hidden_states = self.generate(
                data_list=processed_msgs_list,
                max_inp_length=max_inp_length,
                img_list=processed_image_list,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states, # this is None by default.
                return_vision_hidden_states=True,
                **generation_config
            )
        # answer = res[0]
        answers = res
        
        return answers





class LlamaTokenizerWrapper(LlamaTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.im_start = "<image>"
        self.im_end = "</image>"
        self.ref_start = "<ref>"
        self.ref_end = "</ref>"
        self.box_start = "<box>"
        self.box_end = "</box>"
        self.quad_start = "<quad>"
        self.quad_end = "</quad>"
        self.point_start = "<point>"
        self.point_end = "</point>"
        self.slice_start = "<slice>"
        self.slice_end = "</slice>"

    @property
    def eos_id(self):
        return self.sp_model.eos_id()

    @property
    def bos_id(self):
        return self.sp_model.bos_id()

    @property
    def unk_id(self):
        return self.sp_model.unk_id()

    @property
    def im_start_id(self):
        return self._convert_token_to_id(self.im_start)

    @property
    def im_end_id(self):
        return self._convert_token_to_id(self.im_end)

def pad(orig_items, max_length=None, padding_value=0, padding_side="right"):
    """
    Args:
        orig_items: a list of input_ids, each input_ids should be [1, length_i]
    """
    # assert padding_value == 1 # bug fix
    
    padding_value = 0
    # padding_side = "left"
    
    assert isinstance(orig_items, list)
    assert isinstance(orig_items[0], torch.Tensor)
    
    items = [t.squeeze() for t in orig_items]

    batch_size = len(items)
    shape = items[0].shape
    
    # print(f"items[0].shape = {items[0].shape}")
    
    dim = len(shape)
    
    # print("shape", shape)
    assert dim == 1, "This pad function only expect B*Tensor([seq_len]) input."  # Assuming 1D tensors for simplicity

    if max_length is None:
        max_length = max(item.shape[0] for item in items)

    tensor = torch.full((batch_size, max_length), padding_value, dtype=items[0].dtype)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.int8)

    for i, item in enumerate(items):
        length = item.shape[0]
        if padding_side == "left":
            raise NotImplementedError("Using left padding will cause model training error.")
            tensor[i, -length:] = item
            attention_mask[i, -length:] = 1
        else:
            tensor[i, :length] = item
            attention_mask[i, :length] = 1

    return_dict = {
        "input_ids": tensor,
        "attention_mask": attention_mask,
    }
    
    return BatchEncoding(return_dict)


def slice_image(
    image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []
    
    # t1=time.time()

    if multiple <= 1 or never_split:
        # dont need to slice, upsample
        best_size = find_best_resize(
            original_size, scale_resolution, patch_size, allow_upscale=True
        )
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(original_size, scale_resolution, patch_size)
        
        # t2 = time.time()
        
        # t1 = time.time()
        
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []
        
        # t2 = time.time()
        
        # print("resize1 - elapsed", t2-t1)
        
        # t3 =time.time()

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(
            original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
        )
        
        # t4 =time.time()

        # t1 = time.time()
        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        # t2 = time.time()
        
        # print("resize2 - elapsed", t2-t1)
        
        patches = split_to_patches(refine_image, best_grid)
        
        # print(patches)
        
        # t5 =time.time()
        
        # print(f"{t1}, {t2}, {t3}, {t4}, {t5}")

    return source_image, patches, best_grid


def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)


def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)


def get_refine_size(
    original_size, grid, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(  
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size


def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches


def get_grid_placeholder(tokenizer, grid, query_num):
    image_placeholder = (
        tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end
    )

    cols = grid[0]
    rows = grid[1]
    slices = []
    for i in range(rows):
        lines = []
        for j in range(cols):
            lines.append(image_placeholder)
        slices.append("".join(lines))
    slice_placeholder = tokenizer.slice_start + "\n".join(slices) + tokenizer.slice_end
    return slice_placeholder


def transform_image_mp(img_list, transform, device, max_workers=None):
    pixel_values = []
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     for img_batch in img_list:
    #         img_inps = list(executor.map(transform, img_batch))
    #         pixel_values.append(img_inps if img_inps else [])
    
    # 使用ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for img_batch in img_list:
            img_inps = list(executor.map(transform, img_batch))
            for i in range(len(img_inps)):
                img_inps[i] = img_inps[i].to(device)
            pixel_values.append(img_inps if img_inps else [])

    return pixel_values


@dataclass
class BaseModelOutputWithAttentionMask(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    # hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.Tensor] = None

class MiniCPMVEmbedding(MiniCPMV): # -> MiniCPMV ->  Ultimately a CausalLM
    def fused_tokenize(
        self,
        data_list=None, # List[str]
        img_list=None, # List[List[PIL.Image]]
        tokenizer=None,
        max_inp_length: Optional[int] = None,
        vision_hidden_states=None, # default None
        return_vision_hidden_states=False,
        **kwargs):

        t1 = time.time()
        
        assert data_list is not None
        bs = len(data_list)
        if img_list == None:
            img_list = [[] for i in range(bs)]
        assert bs == len(img_list)

        model_inputs = self._process_list(tokenizer, data_list, max_inp_length, padding_side="right")

        t2 = time.time()
        
        # print("_process_list - elapsed time", t2-t1)
        
        t1 = time.time()
        
        if vision_hidden_states is None:
            
            # pixel_values = []
            # for i in range(bs):
            #     print('pre ', i)
            #     img_inps = []
            #     for img in img_list[i]:
            #         print(img)
            #         img_inps.append(self.transform(img).to(self.device))
            #     if img_inps:
            #         pixel_values.append(img_inps)
            #     else:
            #         pixel_values.append([])
            
            # 先进行图像转换
            pixel_values = transform_image_mp(img_list, self.transform, self.device, max_workers=8)
            
            # pixel_values = [[]]*bs

            model_inputs["pixel_values"] = pixel_values
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states
        
        t2 = time.time()
        
        # print("pre - elapsed time", t2-t1)
        
        return model_inputs
    
    def prepare_context(self, inputs, tokenizer):
        text_, image_ = inputs
        if not isinstance(text_, str):
            raise NotImplementedError(f"chatml format expected, expect outmost type to be str but got {type(text_)}")
        
        # 1.add text
        content = text_ 
        
        # 2. add image
        if image_:
            if self.config.slice_mode:
                images, final_placeholder = self.get_slice_image_placeholder(
                    image_, tokenizer
                ) # crop one image into multiple sub images -> List[Image]
                content = final_placeholder + "\n" + content
                # print(f"len pages = {len(images)}")
            else:
                images = [image_] # only keep one image without cropping -> List[Image]
                content = (
                    tokenizer.im_start
                    + tokenizer.unk_token * self.config.query_num
                    + tokenizer.im_end
                    + "\n"
                    + content
                )
        else:
            images = []
        
        return content, images
    
    def forward(
        self,
        text, # List[str] B*str
        image, # List[ PIL.Image ] B*PIL.Image, one image for each data
        tokenizer,
        vision_hidden_states=None,
        max_inp_length=2048,
        **kwargs):
        
        processed_image = []
        processed_text = []
        
        # cnt = 0
        
        t1 = time.time()
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            contexts = list(executor.map(lambda inputs: self.prepare_context(inputs, tokenizer), zip(text, image)))
        
        for context in contexts:
            content_, image_ = context
            processed_text.append(content_)
            processed_image.append(image_)
        
        # for text_, image_ in zip(text, image):
        #     # cnt += 1
        #     # print(cnt, "text", text_)
        #     # print(cnt, "image", image_)
        #     if not isinstance(text_, str):
        #         raise NotImplementedError(f"chatml format expected, expect outmost type to be str but got {type(text)}")
        
        #     # 1.add text
        #     content = text_ 
        
        #     # 2. add image
        
        #     if image_:
        #         if self.config.slice_mode:
        #             images, final_placeholder = self.get_slice_image_placeholder(
        #                 image_, tokenizer
        #             ) # crop one image into multiple sub images -> List[Image]
        #             content = final_placeholder + "\n" + content
        #             # print(f"len pages = {len(images)}")
        #         else:
        #             images = [image_] # only keep one image without cropping -> List[Image]
        #             content = (
        #                 tokenizer.im_start
        #                 + tokenizer.unk_token * self.config.query_num
        #                 + tokenizer.im_end
        #                 + "\n"
        #                 + content
        #             )
        #     else:
        #         images = []
            
        #     processed_text.append(content)
        #     processed_image.append(images)
        
        t2 = time.time()
        
        # print("slicing - elapsed", t2-t1)
        model_inputs = self.fused_tokenize(
            data_list=processed_text, # List[str]
            img_list=processed_image, # List[List[PIL.Image]]
            tokenizer=tokenizer,
            max_inp_length=max_inp_length
        )
        
        # this is vision encoder forward.
        # print("vision begin")
        t1 = time.time()
        model_inputs["inputs_embeds"], vision_hidden_states = self.get_vllm_embedding(model_inputs)
        t2 = time.time()
        # print("vision end")
        
        # print(model_inputs["inputs_embeds"])
        # with torch.no_grad():
        #     contains_nan = torch.isnan(model_inputs["inputs_embeds"]).any()
        #     # print(f"-------------- contains nan {contains_nan} --------------")
        #     assert contains_nan != True, "vital error, model input 'input_embeds' has nan, please check."
            
        # print("vision elapsed", t2-t1)
        # model_inputs = BatchEncoding(model_inputs)
        
        # if not return_dict:
        #     raise NotImplementedError("must return_dict in MiniCPMVEmbedding.")
        
        # t1 = time.time()
        vlm_outputs = self.llm.model(
            input_ids=None, # because image and text have been merged into model_inputs["inputs_embeds"] here, we don't give input_ids
            position_ids=None,
            inputs_embeds=model_inputs["inputs_embeds"],
            attention_mask=model_inputs["attention_mask"],
            return_dict=True
        )
        # t2 = time.time()
        # print("llm elapsed", t2-t1)

        # print(vlm_outputs.last_hidden_state)
        # with torch.no_grad():
        #     contains_nan = torch.isnan(vlm_outputs.last_hidden_state).any()
        #     assert contains_nan != True, "vital error, model output 'last_hidden_state' has nan, please check."
            
        return BaseModelOutputWithAttentionMask(
            last_hidden_state=vlm_outputs.last_hidden_state,
            attention_mask=model_inputs.attention_mask
        )
        

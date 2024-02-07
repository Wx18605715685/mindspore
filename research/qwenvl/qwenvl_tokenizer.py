# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file was refer to project:
# https://huggingface.co/Qwen/Qwen-VL
# ============================================================================
"""QwenVL tokenizer"""

import re
from typing import List, Any, Union, Tuple, Callable, Set, Collection, Dict

from tokenizers import AddedToken
import unicodedata
from qwen.qwen_tokenizer import QwenTokenizer, _load_tiktoken_bpe

from mindformers import MindFormerRegister, MindFormerModuleType

try:
    import tiktoken
except ImportError:
    raise ImportError("Package 'tiktoken' required to run Qwen. please install it with pip.")

PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
ENDOFTEXT = "<|endoftext|>"
IMSTART = "<|im_start|>"  # used in Qwen-7B-chat
IMEND = "<|im_end|>"  # used in Qwen-7B-chat
EXTRAS = tuple((f"<|extra_{i}|>" for i in range(205)))
SPECIAL_TOKENS = (ENDOFTEXT, IMSTART, IMEND,) + EXTRAS
IMG_TOKEN_SPAN = 256


def _list_find(
        input_list: List[Any],
        candidates: Tuple[Any],
        start: int = 0,
):
    for i in range(start, len(input_list)):
        if input_list[i] in candidates:
            return i
    return -1


def _replace_closed_tag(
        input_tokens: List[Any],
        start_tags: Union[Any, Tuple[Any]],
        end_tags: Union[Any, Tuple[Any]],
        inclusive_replace_func: Callable,
        exclusive_replace_func: Callable = lambda x: x,
):
    if isinstance(start_tags, (str, int)):
        start_tags = (start_tags,)
    if isinstance(end_tags, (str, int)):
        end_tags = (end_tags,)
    assert len(start_tags) == len(end_tags)

    output_tokens = []
    end = 0
    while True:
        start = _list_find(input_tokens, start_tags, end)
        if start == -1:
            break
        output_tokens.extend(exclusive_replace_func(input_tokens[end: start]))
        tag_idx = start_tags.index(input_tokens[start])
        end = _list_find(input_tokens, (end_tags[tag_idx],), start)
        if end == -1:
            raise ValueError("Unclosed image token")
        output_tokens.extend(inclusive_replace_func(input_tokens[start: end + 1]))
        end += 1
    output_tokens.extend(exclusive_replace_func(input_tokens[end:]))
    return output_tokens


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class QwenVLTokenizer(QwenTokenizer):
    # _support_list = MindFormerBook.get_tokenizer_support_list()['qwen-vl']

    def __init__(self, vocab_file="qwen.tiktoken",
                 errors="replace",
                 pad_token="<|endoftext|>",
                 image_start_tag='<img>',
                 image_end_tag='</img>',
                 image_pad_tag='<imgpad>',
                 ref_start_tag='<ref>',
                 ref_end_tag='</ref>',
                 box_start_tag='<box>',
                 box_end_tag='</box>',
                 quad_start_tag='<quad>',
                 quad_end_tag='</quad>',
                 **kwargs):
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        super().__init__(vocab_file=vocab_file, pad_token=pad_token, **kwargs)

        self.image_start_tag = image_start_tag
        self.image_end_tag = image_end_tag
        self.image_pad_tag = image_pad_tag
        self.ref_start_tag = ref_start_tag
        self.ref_end_tag = ref_end_tag
        self.box_start_tag = box_start_tag
        self.box_end_tag = box_end_tag
        self.quad_start_tag = quad_start_tag
        self.quad_end_tag = quad_end_tag
        self.IMAGE_ST = (
            ref_start_tag, ref_end_tag,
            box_start_tag, box_end_tag,
            quad_start_tag, quad_end_tag,
            image_start_tag, image_end_tag,
            image_pad_tag
        )

        self.errors = errors  # how to handle errors in decoding
        self.vocab_file = vocab_file
        self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)  # type: dict[bytes, int]

        self.special_tokens = {
            token: index
            for index, token in enumerate(
                SPECIAL_TOKENS + self.IMAGE_ST, start=len(self.mergeable_ranks),
            )
        }

        self.img_start_id = self.special_tokens[self.image_start_tag]
        self.img_end_id = self.special_tokens[self.image_end_tag]
        self.img_pad_id = self.special_tokens[self.image_pad_tag]
        self.ref_start_id = self.special_tokens[self.ref_start_tag]
        self.ref_end_id = self.special_tokens[self.ref_end_tag]
        self.box_start_id = self.special_tokens[self.box_start_tag]
        self.box_end_id = self.special_tokens[self.box_end_tag]
        self.quad_start_id = self.special_tokens[self.quad_start_tag]
        self.quad_end_id = self.special_tokens[self.quad_end_tag]
        self.image_special_tokens = set([
            self.ref_start_id, self.ref_end_id, self.box_start_id, self.box_end_id,
            self.quad_start_id, self.quad_end_id,
        ])

        enc = tiktoken.Encoding(
            "Qwen",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        assert (len(self.mergeable_ranks) + len(
            self.special_tokens) == enc.n_vocab), f"{len(self.mergeable_ranks) + len(self.special_tokens)} != " \
                                                  f"{enc.n_vocab} in encoding"

        self.decoder = {
            v: k for k, v in self.mergeable_ranks.items()
        }  # type: dict[int, bytes|str]
        self.decoder.update({v: k for k, v in self.special_tokens.items()})

        self.tokenizer = enc  # type: tiktoken.Encoding

        self.eod_id = self.tokenizer.eot_token
        self.im_start_id = self.special_tokens[IMSTART]
        self.im_end_id = self.special_tokens[IMEND]

    def tokenize(
            self,
            text: str,
            allowed_special: Union[Set, str] = "all",
            disallowed_special: Union[Collection, str] = (),
            **kwargs,
    ) -> List[Union[bytes, str]]:
        tokens = []
        text = unicodedata.normalize("NFC", text)

        # this implementation takes a detour: text -> token id -> token surface forms
        for t in self.tokenizer.encode(
            text, allowed_special=allowed_special, disallowed_special=disallowed_special
        ):
            tokens.append(self.decoder[t])

        def _encode_imgurl(img_tokens):
            assert img_tokens[0] == self.image_start_tag and img_tokens[-1] == self.image_end_tag
            img_tokens = img_tokens[1:-1]
            img_url = b''.join(img_tokens)
            out_img_tokens = list(map(self.decoder.get, img_url))
            if len(out_img_tokens) > IMG_TOKEN_SPAN:
                raise ValueError("The content in {}..{} is too long".format(
                    self.image_start_tag, self.image_end_tag))
            out_img_tokens.extend([self.image_pad_tag] * (IMG_TOKEN_SPAN - len(out_img_tokens)))
            out_img_tokens = [self.image_start_tag] + out_img_tokens + [self.image_end_tag]
            return out_img_tokens

        return _replace_closed_tag(tokens, self.image_start_tag, self.image_end_tag, _encode_imgurl)

    def from_list_format(self, list_format: List[Dict]):
        text = ''
        num_images = 0
        for ele in list_format:
            if 'image' in ele:
                num_images += 1
                text += f'Picture {num_images}: '
                text += self.image_start_tag + ele['image'] + self.image_end_tag
                text += '\n'
            elif 'text' in ele:
                text += ele['text']
            elif 'box' in ele:
                if 'ref' in ele:
                    text += self.ref_start_tag + ele['ref'] + self.ref_end_tag
                for box in ele['box']:
                    text += self.box_start_tag + '(%d,%d),(%d,%d)' % (box[0], box[1], box[2], box[3]) + self.box_end_tag
            else:
                raise ValueError("Unsupported element: " + str(ele))
        return text

    def post_process(self, result: str, query: List[Dict]):
        begin_tag_idx = []
        end_tag_idx = []
        for match in re.finditer(r'<img>', result):
            begin_tag_idx.append((match.start(), match.end()))

        for match in re.finditer(r'</img>', result):
            end_tag_idx.append((match.start(), match.end()))

        if len(begin_tag_idx) != len(end_tag_idx):
            raise ValueError("the text has unclosed image tag")

        img_paths = [item for item in query if 'image' in item]
        if len(begin_tag_idx) != len(img_paths):
            raise ValueError("The number of image tags != the number of image paths")

        sub_strs = []
        last_end_idx = 0
        for begin, end, path in zip(begin_tag_idx, end_tag_idx, img_paths):
            sub_strs.append(result[last_end_idx:begin[1]])
            sub_strs.append(path['image'])
            sub_strs.append(result[end[0]:end[1]])
            last_end_idx = end[1]
        sub_strs.append(result[last_end_idx:])

        replaced_str = ''.join(sub_strs)
        return replaced_str

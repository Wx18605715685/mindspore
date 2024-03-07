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
# ============================================================================
"""QwenVL DataLoader."""

import os
import random
import numpy as np
from typing import Optional, Union, List, Tuple
from collections import defaultdict
import json
from PIL import Image
from typing import Union, Callable

from mindspore.dataset import GeneratorDataset
from mindspore.dataset import vision

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.dataset.dataloader.sft_dataloader import SFTDataSet
from mindformers.tools import logger
from mindformers.dataset.transforms.vision_transforms import (
    BatchResize,
)


def _dir_concat(dataset_dir, path, check_func):
    for i, _ in enumerate(path):
        path[i] = os.path.join(dataset_dir, path[i])
        if not check_func(path[i]):
            raise ValueError(f"{path[i]} is not a valid path.")
    return path


class BatchResizeV2(BatchResize):
    def __init__(self, img_resolution, interpolation='cubic'):
        super().__init__(img_resolution, interpolation)
        self.resize = vision.Resize(img_resolution, self.interpolation)


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class QwenVLDataLoader:
    _default_column_names = ["image", "text"]

    def __new__(cls,
                dataset_dir: str,
                image_dir: list,
                task_config: dict,
                stage: int,
                column_names: Optional[Union[List[str], Tuple[str]]] = None,
                repeat_images: Optional[bool] = False,
                shuffle: Optional[bool] = True,
                extra_kwargs: Optional[dict] = None,
                **kwargs):

        if stage not in [1, 2, 3]:
            raise ValueError("stage should be 1, 2 or 3.")
        if column_names is None:
            column_names = cls._default_column_names

        # Concat directory with dataset_dir
        image_dir = _dir_concat(dataset_dir, image_dir, os.path.isdir)
        for task, file_path in task_config.items():
            for path_name, path in file_path.items():
                if len(path) != len(image_dir):
                    raise ValueError(f"The number of {path_name} should be equal to image_dir!")
                for i in range(len(path)):
                    path = _dir_concat(dataset_dir, path, lambda x: True)
        if stage == 3:
            qwenvl_dataset = QwenVLSFTDataset(image_dir, task_config, **extra_kwargs)
        else:
            qwenvl_dataset = QwenVLPretrainDataSet(image_dir, task_config)
        return GeneratorDataset(qwenvl_dataset,
                                column_names,
                                shuffle=shuffle,
                                **kwargs)


class QwenVLPretrainDataSet:
    """Stage2 dataset for QwenVL"""

    def __init__(self,
                 image_dir: list,
                 task_config: dict,
                 ):
        self.data = []
        for task, file_path in task_config.items():
            if task == 'caption':
                annotation_files = file_path.get('annotation_files', None)
                if annotation_files is None:
                    raise ValueError(f"annotation_files in {task} should be provided!")
                for i, annotation_file in enumerate(annotation_files):
                    with open(annotation_file, 'r', encoding='utf-8') as file:
                        new_annotation = json.load(file)
                    local_new_annotation = []
                    for new_ann in new_annotation:
                        ann = dict()
                        ann["image"] = os.path.join(image_dir[i], new_ann["image"])
                        caption_dict = {'caption': new_ann["caption"], 'task': task}
                        ann["text"] = caption_dict
                        local_new_annotation.append(ann)
                    self.data.extend(local_new_annotation)
            elif task == 'vqa':
                annotation_files = file_path.get('annotation_files', None)
                question_files = file_path.get('question_files', None)
                if annotation_files is None or question_files is None:
                    raise ValueError(f"annotation_files and question_files in {task} should be provided!")
                for i, annotation_file in enumerate(annotation_files):
                    with open(annotation_file, 'r', encoding='utf-8') as file:
                        new_annotation = json.load(file)
                    with open(question_files[i], 'r', encoding='utf-8') as file:
                        new_question = json.load(file)
                    data_subtype = new_annotation["data_subtype"]
                    new_annotation = new_annotation["annotations"]
                    new_image_dir = os.path.join(image_dir[i], data_subtype)
                    for entry in os.scandir(new_image_dir):
                        if entry.is_file() and entry.name.endswith('jpg'):
                            template = entry.name
                            break
                    template = data_subtype + '/' + template.rsplit('_', 1)[0] + '_{}.jpg'
                    new_question = new_question["questions"]
                    qa_data_list = []
                    for j in range(len(new_annotation)):
                        qa_data = dict()
                        ans_weights = defaultdict(int)
                        ann = new_annotation[j]
                        ques = new_question[j]
                        img_name = template.format(str(ann["image_id"]).zfill(12))
                        qa_data["image"] = os.path.join(image_dir[i], img_name)
                        anno = dict()
                        anno["question"] = ques["question"]
                        for ans in map(lambda x: x.get('answer'), ann["answers"]):
                            ans_weights[ans] += (1 / len(ann["answers"]))
                        ans_weights = dict(sorted(ans_weights.items(), key=lambda x: x[1], reverse=True))
                        # Get the max weight answer
                        anno["answers"] = list(ans_weights.keys())[0]
                        anno["weights"] = list(ans_weights.values())[0]
                        anno["task"] = task
                        qa_data['text'] = anno
                        qa_data_list.append(qa_data)
                    self.data.extend(qa_data_list)

    def __getitem__(self, index):
        ann = self.data[index]
        img = Image.open(ann["image"]).convert("RGB")
        return img, ann['text']

    def __len__(self):
        return len(self.data)

    def display_item(self, index):
        """display item

        Args:
            index (int): index

        Returns:
            out (OrderedDict): item info
        """
        sample, ann = self[index], self.data[index]
        out = {"image": sample[0]}
        out.update(ann)
        return out


class QwenVLSFTDataset(SFTDataSet):
    # TODO: Stage3 datasets only support string dir or len 1 list dir
    def __init__(self,
                 image_dir: str,
                 task_config: dict,
                 file_format: str = None,
                 max_length: int = 1025,
                 read_function: Callable = None,
                 map_function_kwargs: dict = None,
                 out_img_shape: int = 448,
                 max_img_len: int = 5):
        if len(task_config.keys()) != 1 or task_config.get('sft') is None:
            raise ValueError("The task_config should be {'sft': {}} for stage 3 dataset")
        annotation_files = task_config['sft'].get('annotation_files')
        if annotation_files is None:
            raise ValueError("annotation_files should be provided for stage 3 dataset")
        if isinstance(annotation_files, list):
            if len(annotation_files) > 1:
                raise ValueError("Only one annotation file is supported for stage 3 dataset")
            annotation_files = annotation_files[0]
            image_dir = image_dir[0]
        self.out_img_shape = (out_img_shape, out_img_shape)
        self.resize = BatchResizeV2(self.out_img_shape, interpolation='cubic')
        super(QwenVLSFTDataset, self).__init__(annotation_files, None, None, file_format=file_format,
                                               max_length=max_length,
                                               read_function=read_function, map_function=self._qwenvl_map,
                                               map_function_kwargs=map_function_kwargs)
        self.image_dir = image_dir
        self.max_img_len = max_img_len

    def __getitem__(self, i):
        example = self.table.take([i]).to_pylist()[0]
        result = self.map_function(example, **self.map_function_kwargs)
        img_list, img_idx = self._img_padding(result["text"]["img_dir"], result["text"]["img_idx"])
        new_dict = {}
        for key, value in result["text"].items():
            if key == "img_dir" or key == "img_idx":
                continue
            new_dict[key] = value
        new_dict["img_idx"] = img_idx
        return img_list, new_dict

    def _img_padding(self, img_list, img_idx):
        if not len(img_list):
            return [np.ones(self.out_img_shape + (3,), dtype=np.uint8)] * self.max_img_len, [-1] * self.max_img_len
        new_img_list = []
        new_img_idx = []
        for i, img_dir in enumerate(img_list):
            new_img_list.append(self.resize(np.array(Image.open(img_dir).convert("RGB"))))
            new_img_idx.append(img_idx[i])
        if len(new_img_list) <= self.max_img_len:
            for _ in range(self.max_img_len - len(new_img_list)):
                new_img_list.append(new_img_list[-1])
                new_img_idx.append(-1)
        else:
            raise ValueError("The number of images is greater than the maximum number of images.")
        return new_img_list, new_img_idx

    def _find_img_tags(self, s):
        start_tag = '<img>'
        end_tag = '</img>'
        start_positions = []
        end_positions = []
        start = 0
        while start < len(s):
            start = s.find(start_tag, start)
            if start == -1:
                break
            start_positions.append(start + 5)
            start += len(start_tag)
        start = 0
        while start < len(s):
            start = s.find(end_tag, start)
            if start == -1:
                break
            end_positions.append(start)
            start += len(end_tag)
        return start_positions, end_positions

    def _qwenvl_map(self, example, **kwargs):
        data_field = kwargs.get("data_field", "conversations")
        from_keyword, value_keyword = kwargs.get("from_keyword", "from"), kwargs.get("value_keyword", "value")
        user_role_name = kwargs.get("user_role_name", "human")
        assistant_role_name = kwargs.get("assistant_role_name", "gpt")
        user_prompt, assistant_prompt = kwargs.get("user_prompt", ""), kwargs.get("assistant_prompt", "")
        system_message = kwargs.get("system_message", "You are a helpful assistant.")

        raw_data = []
        raw_data_role = []
        img_idx = []
        img_dir = []
        img_pos = 0

        # 增加系统信息
        system = "<|im_start|>system\n" + system_message + "<|im_end|>\n"
        raw_data.append(system)
        raw_data_role.append('system')

        for message in example[data_field]:
            from_ = message[from_keyword]
            value = message[value_keyword]
            if '<img>' in value:
                img_start_pos, img_end_pos = self._find_img_tags(value)
                img_dir, img_string = self._get_img_dir(value, img_start_pos, img_end_pos)
                for i, _ in enumerate(img_string):
                    img_idx.append(img_pos)
                    img_pos += 1
            raw_data_role.append(from_)
            if from_ == user_role_name:
                raw_data.append("<|im_start|>" + from_ + '\n' + user_prompt + value + '<|im_end|>\n')
            elif from_ == assistant_role_name:
                raw_data.append("<|im_start|>" + from_ + '\n' + assistant_prompt + value + '<|im_end|>\n')
            else:
                raise ValueError(f"Incorrect role name: {from_}. Check the values of `user_role_name` "
                                 f"and `assistant_role_name` in `map_function_kwargs`.")
        return dict(
            text=dict(img_dir=img_dir, raw_data=raw_data, raw_data_role=raw_data_role, img_idx=img_idx,
                      user_role_name=user_role_name, assistant_role_name=assistant_role_name, task='sft'))

    def _get_img_dir(self, value, img_start_pos, img_end_pos):
        img_dir = []
        img_string = []
        for i in range(len(img_start_pos)):
            img_dir_in_value = value[img_start_pos[i]:img_end_pos[i]]
            img_dir.append(os.path.join(self.image_dir, img_dir_in_value))
            img_string.append(img_dir_in_value)
        return img_dir, img_string

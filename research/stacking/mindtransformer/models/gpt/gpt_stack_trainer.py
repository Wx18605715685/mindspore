# Copyright 2022 Huawei Technologies Co., Ltd
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

"""GPT Trainer"""
from mindtransformer.models.gpt import GPTConfig, GPTWithLoss
from mindtransformer.trainer import Trainer, TrainingConfig, parse_config
from mindtransformer.data import create_gpt_dataset
from mindspore import load_checkpoint


class GPTTrainingConfig(TrainingConfig):
    """
    GPTTrainingConfig
    """

    def __init__(self, *args, **kwargs):
        super(GPTTrainingConfig, self).__init__(*args, **kwargs)
        self.epoch_size = 1
        self.train_data_path = ""
        self.optimizer = "adam"
        self.parallel_mode = "stand_alone"
        self.full_batch = False
        self.global_batch_size = 4
        self.checkpoint_prefix = "gpt"


class GPTTrainer(Trainer):
    """
    GPTTrainer
    """

    def build_model_config(self):
        model_config = GPTConfig()
        return model_config

    def build_model(self, model_config):
        net_with_loss = GPTWithLoss(model_config)
        return net_with_loss

    def build_dataset(self):
        return create_gpt_dataset(self.config)

    def double_model_weights(self, pre_load_model_path, pre_num_layers):
        """
        stack GPT model parameters
        """
        param_dict = load_checkpoint(pre_load_model_path)
        lst = []
        for k, v in param_dict.items():
            k_split = k.split('.')
            if k_split[0] == 'backbone' and k_split[1] == "transformer" \
                    and k_split[2] == 'encoder' and k_split[3] == 'blocks':
                l_id = int(k_split[4])
                k_split[4] = str(l_id + pre_num_layers)
                new_k = ".".join(k_split)
                lst.append([new_k, v.clone()])
        # Add new stacked weights to param_dict
        for k, v in lst:
            param_dict[k] = v
        return param_dict


if __name__ == "__main__":
    config = GPTTrainingConfig()
    parse_config(config)
    trainer = GPTTrainer(config)
    trainer.stack_train()
    
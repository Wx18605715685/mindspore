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
"""LMTask"""
import math
from mindspore import load_checkpoint, load_param_into_net
from mindtransformer.tasks import Task, TaskConfig
from mindtransformer.trainer import parse_config
from mindtransformer.data import create_language_model_dataset
from mindtransformer.utils import get_newest_ckpt

class LMTask(Task):
    """
    LMTask
    """

    def preprocess(self):
        """
        process input dataset
        """
        self.config.dataset_batch_size = self.config.eval_batch_size
        self.config.dataset_path = self.config.eval_data_path
        self.config.dataset_do_shuffle = self.config.eval_data_shuffle
        self.config.is_training = False
        self.config.dataset_device_num = 1
        self.config.dataset_rank = 0
        self.config.repeat_count = 1
        return create_language_model_dataset(self.config)

    def process(self, preprocess_output, model):
        """
        process inference result
        """

        if self.config.metric.lower() == "ppl":
            print("Prepare to calculate the ppl score ...")

            columns_list = ["input_ids", "input_mask", "label_ids"]
            print("==================== [PPL] Testing ====================")
            num_data = 1
            total_loss = 0.0
            avg_loss = 0.0
            for data in preprocess_output.create_dict_iterator():
                input_data = []
                for i in columns_list:
                    input_data.append(data[i])
                input_ids, input_mask, label_ids = input_data
                loss = model.predict(input_ids, input_mask, label_ids)
                loss = float(loss.asnumpy())
                total_loss += loss
                avg_loss = float(total_loss / num_data)
                num_data += 1

            print("\n\n")
            print("**************************************************************")
            print("Average Loss: {:.6f}".format(avg_loss))
            print("Average PPL: {:.6f}".format(math.exp(avg_loss)))
            print("********************** Testing Finished **********************")
        else:
            raise ValueError("metric method not supported, support: [ppl]")

    def load_checkpoint(self, net_with_loss):
        """load checkpoint"""
        if self.config.load_checkpoint_path == "" and self.config.save_checkpoint_path != "" \
                and self.config.checkpoint_prefix != "":
            self.config.load_checkpoint_path = get_newest_ckpt(self.config.save_checkpoint_path,
                                                               self.config.checkpoint_prefix)

        if self.config.load_checkpoint_path != "":
            if self.config.load_checkpoint_path.endswith('.ckpt'):
                self.logger.info("Start to load the ckpt from %s", self.config.load_checkpoint_path)
            else:
                self.config.load_checkpoint_path = get_newest_ckpt(self.config.load_checkpoint_path,
                                                                   self.config.checkpoint_prefix)

        if self.config.eval_type == "zero-shot":
            final_param_dict = {}
            param_dict = load_checkpoint(self.config.load_checkpoint_path)
            for name, _ in param_dict.items():
                final_param_dict['gpt2.' + name] = param_dict[name]
            final_param_dict['gpt2.dense1.weight'] = param_dict['backbone.word_embedding.embedding_table']
            load_param_into_net(net_with_loss, final_param_dict)
        else:

            ckpt = load_checkpoint(self.config.load_checkpoint_path)
            load_param_into_net(net_with_loss, ckpt)


class LMTaskConfig(TaskConfig):
    """
    LMTaskConfig
    """
    def __init__(self, *args, **kwargs):
        super(LMTaskConfig, self).__init__(*args, **kwargs)
        self.auto_model = "gpt_language_model"
        self.device_target = "GPU"
        self.device_id = 0
        self.epoch_num = 3
        self.eval_data_shuffle = False
        self.eval_batch_size = 1
        self.checkpoint_prefix = 'gpt2_language_model'
        self.eval_data_path = './test-mindrecord'

        self.vocab_size = 50257
        self.hidden_size = 768
        self.embedding_size = 768
        self.num_layers = 12
        self.num_heads = 12
        self.seq_length = 1024
        self.max_position_embeddings = 1024
        self.metric = 'ppl'


if __name__ == "__main__":
    config = LMTaskConfig()
    parse_config(config)
    task = LMTask(config)
    task.run()

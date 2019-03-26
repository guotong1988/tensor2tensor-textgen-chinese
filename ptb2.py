# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import tarfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow as tf

PAD = text_encoder.PAD
EOS = text_encoder.EOS


def _read_words(filename):
    """Reads words from a file."""
    all_words = []
    with tf.gfile.GFile(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            for current_char in line:
                if current_char == "，" or \
                        current_char == "。" or \
                        current_char == "？" or \
                        current_char == "！" or \
                        current_char == "、":
                    all_words.append(current_char)
                elif current_char == "“" or current_char == "”" \
                        or current_char == "「" or current_char == "」":
                    all_words.append("'")
                elif current_char  != "\n" and \
                        current_char != "\u3000" and \
                        current_char != " " and \
                        current_char != "\\'":
                    all_words.append(current_char)
    return all_words


def _build_vocab(filename, vocab_path, vocab_size):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    words = list(words[:vocab_size])
    words = ["<UNK>"] + words # 2
    words = [EOS] + words # 1
    words = [PAD] + words # 0
    with open(vocab_path, mode = "w", encoding="utf-8") as f:
        f.write("\n".join(words))


def _get_token_encoder(vocab_dir, vocab_name, filename):
    """Reads from file and returns a `TokenTextEncoder` for the vocabulary."""
    vocab_path = os.path.join(vocab_dir, vocab_name)
    if not tf.gfile.Exists(vocab_path):
        _build_vocab(filename, vocab_path, 10000)
    return text_encoder.TokenTextEncoder(vocab_path,replace_oov="<UNK>")


def get_data_files(tmp_dir):
    result_files = []
    file_list = tf.gfile.ListDirectory(tmp_dir)
    for file_name in file_list:
        if file_name=="train.txt" or file_name=="valid.txt":
            result_files.append(file_name)
    return result_files


@registry.register_problem
class LanguagemodelChinese(text_problems.Text2SelfProblem):
    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 10,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def is_generate_per_split(self):
        return True

    @property
    def vocab_filename(self):
        return "vocab.lm_cn.10000"

    @property
    def vocab_type(self):
        return text_problems.VocabType.TOKEN

    @property
    def has_inputs(self):
        return True

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        generator,encoder = self.generate_samples(data_dir, tmp_dir, dataset_split)
        return text_problems.text2text_generate_encoded(generator, encoder,
                                                        has_inputs=True,
                                                        inputs_prefix=self.inputs_prefix,
                                                        targets_prefix=self.targets_prefix)

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        files = get_data_files(tmp_dir)

        train_file, valid_file = None, None
        for filename in files:
            if "train" in filename:
                train_file = os.path.join(tmp_dir, filename)
            elif "valid" in filename:
                valid_file = os.path.join(tmp_dir, filename)

        assert train_file, "Training file not found"
        assert valid_file, "Validation file not found"

        encoder = _get_token_encoder(data_dir, self.vocab_filename, train_file)

        train = dataset_split == problem.DatasetSplit.TRAIN
        filepath = train_file if train else valid_file

        def _generate_samples():
            with tf.gfile.GFile(filepath, "r") as f:
                last_result = None
                result = []
                for line in f:
                    for char in line:
                        # if char=="。":
                        #     result.append(EOS)
                        # else:
                        result.append(char)
                        if len(result)==30: # 1 sentence 30 char
                            result_ = " ".join(result)
                            if last_result!=None:
                                result = []
                                yield {"targets": result_,
                                       "inputs": last_result}
                            else:
                                result = []
                            last_result = result_

        return _generate_samples(), encoder



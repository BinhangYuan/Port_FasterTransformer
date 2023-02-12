# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import gpt


class BloomWeight(gpt.GPTWeights):

    def __init__(self, head_num, size_per_head, layer_num, vocab_size,
                 tensor_para_size, pipeline_para_size, weights_data_type, inference_data_type,
                 int8_mode=0):
        super().__init__(
            head_num, size_per_head, layer_num, vocab_size, 0,
            tensor_para_size, pipeline_para_size, weights_data_type,
            inference_data_type,
            has_adapters=False,
            adapter_inter_size=0,
            has_positional_encoding=False,
            has_pre_decoder_layernorm=True,
            has_post_decoder_layernorm=True,
            int8_mode=int8_mode)


@dataclasses.dataclass
class BloomParam:
    num_heads: int
    size_per_head: int
    inter_size: int
    num_layers: int
    vocab_size: int
    start_id: Optional[int] = None
    end_id: Optional[int] = None
    tensor_para_size: int = 1
    pipeline_para_size: int = 1
    remove_padding: bool = True
    shared_contexts_ratio: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.shared_contexts_ratio <= 1.0:
            raise ValueError(
                f'Got an invalid value of shared_context_ratio '
                f'{self.shared_contexts_ratio} - range: [0.0, 1.0]')

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(num_heads=args.num_heads,
                   size_per_head=args.size_per_head,
                   inter_size=args.inter_size,
                   num_layers=args.num_layers,
                   vocab_size=args.vocab_size,
                   start_id=args.start_id,
                   end_id=args.end_id,
                   tensor_para_size=args.tensor_para_size,
                   pipeline_para_size=args.pipeline_para_size,
                   shared_contexts_ratio=args.shared_contexts_ratio)

    @staticmethod
    def add_args_group(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Bloom Model Configuration')
        group.add_argument(
            '--num-heads', type=int, metavar='N', default=None,
            help='The number of attention heads.')
        group.add_argument(
            '--size-per-head', type=int, metavar='N', default=None,
            help='The dimension of an attention head.')
        group.add_argument(
            '--inter-size', type=int, metavar='N', default=None,
            help='The intermediate dimension of the MLP block. If None, '
                 'it will be 4 * num_heads * size_per_head as default.')
        group.add_argument(
            '--num-layers', type=int, metavar='N', default=None,
            help='The number of bloom layers.')
        group.add_argument(
            '--vocab-size', type=int, metavar='N', default=None,
            help='The vocabulary size.')
        group.add_argument(
            '-tp', '--tensor-para-size', type=int, metavar='N', default=1,
            help='The size of tensor parallelism.')
        group.add_argument(
            '-pp', '--pipeline-para-size', type=int, metavar='N', default=1,
            help='The size of pipeline parallelism.')
        group.add_argument(
            '--no-remove-padding', action='store_false', dest='remove_padding',
            help='Disable the optimization feature that skips padded tokens'
                 ' during context computation.')
        group.add_argument(
            '--shared-contexts-ratio', type=float, metavar='M', default=1.0,
            help='The threshold of the duplication ratio to apply the context'
                 ' sharing. If less than shared_context_ratio * batch_size '
                 'sentences are duplicated among inputs of size batch_size, '
                 'the model shares those inputs during context computation.')

    def asdict(self):
        return dataclasses.asdict(self)


class Bloom(gpt.ParallelGPT):

    def __init__(self,
                 head_num, size_per_head,
                 vocab_size, start_id, end_id, layer_num,
                 tensor_para_size: int,
                 pipeline_para_size: int,
                 lib_path: str | Path,
                 inference_data_type: str,
                 weights_data_type: str | np.dtype = np.float16,
                 layernorm_eps: float = 1e-5,
                 shared_contexts_ratio: float = 1.0,
                 int8_mode: int = 0):
        super().__init__(
            head_num, size_per_head, vocab_size, start_id, end_id, layer_num,
            0, tensor_para_size,  pipeline_para_size,
            lib_path=lib_path,
            inference_data_type=inference_data_type,
            layernorm_eps=layernorm_eps,
            # gpt_variant_params
            layernorm_type="pre_layernorm",
            activation_type="Gelu",
            has_positional_encoding=False,
            has_pre_decoder_layernorm=True,
            has_post_decoder_layernorm=True,
            has_adapters=False,
            adapter_inter_size=0,
            use_attention_linear_bias=True,
            int8_mode=int8_mode,
            weights_data_type=weights_data_type,
            shared_contexts_ratio=shared_contexts_ratio)

    def set_input_tensor(self, input_tensor: Optional[torch.Tensor]):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func
        """
        self.input_tensor = input_tensor

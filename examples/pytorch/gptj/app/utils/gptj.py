import os
import typing
import gc
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import time

def _profiling_torch_tensor_memory():
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:
                    # print(type(obj), obj.size(), obj.element_size())
                    total_size += obj.nelement() * obj.element_size()
        except: 
            pass
    print(f"<profiling_torch_tensor_memory>: total HBM of torch tensors: {total_size/1073741824} GB.")


class GPTJWeights(object):
    def __init__(self, head_num, size_per_head, layer_num, vocab_size, max_seq_len, tensor_para_size,
                 pipeline_para_size, weights_data_type=typing.Union[str, np.float16]):
        assert (head_num % tensor_para_size == 0)
        self.head_num = head_num
        self.size_per_head = size_per_head
        self.layer_num = layer_num
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.layers_per_device = layer_num // pipeline_para_size

        local_head_num = head_num // tensor_para_size
        global_head_num = head_num
        local_hidden_units = local_head_num * size_per_head
        global_hidden_units = global_head_num * size_per_head
        local_inter_size = local_hidden_units * 4

        self.local_head_num = local_head_num
        self.global_head_num = global_head_num
        self.local_hidden_units = local_hidden_units
        self.global_hidden_units = global_hidden_units
        self.local_inter_size = local_inter_size

        if isinstance(weights_data_type, str):
            try:
                weights_data_type = {
                    "fp16": np.float16,
                    "fp32": np.float32,
                    "float16": np.float16,
                    "float32": np.float32,
                }[weights_data_type]
            except KeyError:
                raise ValueError(f"Don't know how to interpret weights_data_type: {weights_data_type}")

        assert weights_data_type in [np.float32, np.float16]
        self.weights_data_type = weights_data_type

        self.w = []
        # Transformer blocks
        # self_layernorm_gamma
        self.w.extend([torch.zeros(global_hidden_units, dtype=torch.float16)] * layer_num)
        # self_layernorm_beta
        self.w.extend([torch.zeros(global_hidden_units, dtype=torch.float16)] * layer_num)
        # self_kernel (k, q, v)
        self.w.extend([torch.zeros(global_hidden_units, local_hidden_units * 3, dtype=torch.float16)] * layer_num)
        # self_bias (k, q, v) GPT-J does not have bias for k,q,v. This is just a placeholder.
        self.w.extend([torch.zeros(local_hidden_units * 3, dtype=torch.float16)] * layer_num)
        # self_output_kernel
        self.w.extend([torch.zeros(local_hidden_units, global_hidden_units, dtype=torch.float16)] * layer_num)
        # self_output_bias no bias for output
        # self.w.extend([torch.zeros(global_hidden_units, dtype=torch.float16)] * layer_num)
        # ffn_kernel1
        self.w.extend([torch.zeros(global_hidden_units, local_inter_size, dtype=torch.float16)] * layer_num)
        # ffn_bias1 
        self.w.extend([torch.zeros(local_inter_size, dtype=torch.float16)] * layer_num)
        # ffn_kernel2
        self.w.extend([torch.zeros(local_inter_size, global_hidden_units, dtype=torch.float16)] * layer_num)
        # ffn_bias2
        self.w.extend([torch.zeros(global_hidden_units, dtype=torch.float16)] * layer_num)
        
        # After Transformer blocks wte
        self.w.append(torch.zeros(vocab_size, global_hidden_units, dtype=torch.float16))
        # After Transformer blocks 1 layernorm_gamma
        self.w.append(torch.zeros(global_hidden_units, dtype=torch.float16))
        # After Transformer blocks 1 layernorm_beta
        self.w.append(torch.zeros(global_hidden_units, dtype=torch.float16))
        # After Transformer blocks lm kernel
        self.w.append(torch.zeros(vocab_size, global_hidden_units, dtype=torch.float16))
        # After Transformer blocks lm bias
        self.w.append(torch.zeros(vocab_size, dtype=torch.float16))

        # Initialization
        # self._map(lambda w: torch.nn.init.normal_(w, mean=0., std=1.))

    def __getitem__(self, idx):
        return self.w[idx]

    def __setitem__(self, idx, val):
        self.w[idx] = val

    def __len__(self):
        return len(self.w)

    def _map(self, func):
        for i in range(len(self.w)):
            if isinstance(self.w[i], list):
                for j in range(len(self.w[i])):
                    self.w[i][j] = func(self.w[i][j])
            else:
                self.w[i] = func(self.w[i])

    def load(self, ckpt_path, tensor_para_rank, pipeline_para_rank):
        if not os.path.exists(ckpt_path):
            return False
        w = []
        type_map = {np.float32: torch.float32, np.float16: torch.float16}

        # Load
        def is_load(i):
            return self.layers_per_device * pipeline_para_rank <= i < self.layers_per_device * (pipeline_para_rank + 1)

        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.input_layernorm.weight.bin".format(i),
                                               dtype=self.weights_data_type))
                  if is_load(i) else torch.empty(0).to(type_map[self.weights_data_type])
                  for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.input_layernorm.bias.bin".format(i),
                                               dtype=self.weights_data_type))
                  if is_load(i) else torch.empty(0).to(type_map[self.weights_data_type])
                  for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.attention.query_key_value.weight.{}.bin"
                                               .format(i, tensor_para_rank), dtype=self.weights_data_type))
                  if is_load(i) else torch.empty(0).to(type_map[self.weights_data_type])
                  for i in range(self.layer_num)])
        # GPT-J has no bias for query key and value. 
        w.extend([torch.zeros(self.local_hidden_units * 3).to(type_map[self.weights_data_type])
                  for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.attention.dense.weight.{}.bin"
                                               .format(i, tensor_para_rank), dtype=self.weights_data_type))
                  if is_load(i) else torch.empty(0).to(type_map[self.weights_data_type])
                  for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_h_to_4h.weight.{}.bin"
                                               .format(i, tensor_para_rank), dtype=self.weights_data_type))
                  if is_load(i) else torch.empty(0).to(type_map[self.weights_data_type])
                  for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_h_to_4h.bias.{}.bin"
                                               .format(i, tensor_para_rank), dtype=self.weights_data_type))
                  if is_load(i) else torch.empty(0).to(type_map[self.weights_data_type])
                  for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_4h_to_h.weight.{}.bin"
                                               .format(i, tensor_para_rank), dtype=self.weights_data_type))
                  if is_load(i) else torch.empty(0).to(type_map[self.weights_data_type])
                  for i in range(self.layer_num)])
        w.extend([torch.from_numpy(np.fromfile(ckpt_path + "/model.layers.{}.mlp.dense_4h_to_h.bias.bin"
                                               .format(i), dtype=self.weights_data_type))
                  if is_load(i) else torch.empty(0).to(type_map[self.weights_data_type])
                  for i in range(self.layer_num)])

        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.wte.bin",
                                              dtype=self.weights_data_type)))
        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.final_layernorm.weight.bin",
                                              dtype=self.weights_data_type)))
        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.final_layernorm.bias.bin",
                                              dtype=self.weights_data_type)))
        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.lm_head.weight.bin", dtype=self.weights_data_type)))
        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.lm_head.bias.bin", dtype=self.weights_data_type)))

        # Reshape
        try:
            total_size = 0
            for i in range(len(w)):
                if w[i].nelement() > 0:
                    # print(f"<{i}> Expected shape: {self.w[i].shape} loaded shape: {w[i].shape})")
                    self.w[i] = w[i].reshape(self.w[i].shape)
                    total_size += (w[i].nelement() * w[i].element_size())
                else:
                    self.w[i] = w[i]
            print(f"Weight type: {self.weights_data_type}, Total_para_size: {total_size/1024/1024/1024} GB.")

        except RuntimeError:
            raise RuntimeError(
                f"head_num, size_per_head, vocab_size, and max_seq_len must be the same as the ones during training "
                f"(idx: {i} expected shape: {self.w[i].shape} got shape: {w[i].shape})."
            )

        # transpose calibrate quantize the kernel
        layer_num = self.layer_num
        return True


class GPTJ(nn.Module):
    def __init__(self,
                 head_num,
                 size_per_head,
                 layer_num,
                 vocab_size,
                 rotary_embedding_dim,
                 start_id,
                 end_id,
                 max_seq_len,
                 tensor_para_size, pipeline_para_size,
                 lib_path,
                 weights_data_type: np.dtype = np.float16,
                 device_index = None):
        super().__init__()
        self.head_num = head_num
        self.size_per_head = size_per_head
        self.vocab_size = vocab_size
        self.rotary_embedding_dim = rotary_embedding_dim
        self.start_id = start_id
        self.end_id = end_id
        self.layer_num = layer_num
        # multi-gpu params
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.use_sparse_gemm = False
        self.build_model = False
        self.weights_data_type = weights_data_type

        assert torch.cuda.is_available(), "CUDA is required for this model."

        assert head_num % tensor_para_size == 0, "head_num must be a multiple of tensor_para_size."
        assert layer_num % pipeline_para_size == 0, "layer_num must be a multiple of pipeline_para_size."

        # Load the C++ model into Pytorch model.
        print(f"<GPTJ>:__init__: load lib starts.")
        torch.classes.load_library(os.path.abspath(lib_path))
        print(f"<GPTJ>:__init__: load lib ends.")

        # Prepare weights

        print(f"<GPTJ>:__init__: init weight starts.")
        start_time = time.time()
        self.weights = GPTJWeights(head_num, size_per_head, layer_num, vocab_size, max_seq_len, tensor_para_size,
                                   pipeline_para_size, weights_data_type)
        end_time = time.time()
        _profiling_torch_tensor_memory()
        print(f"<GPTJ>:__init__: init weight ends. init takes {end_time - start_time} seconds.")

        # Prepare for tensor/pipeline parallel
        try:
            if not dist.is_initialized():
                dist.init_process_group(backend='mpi')
        except:
            print("[INFO] WARNING: Have initialized the process group")
        self.rank = dist.get_rank()
        self.device_count = torch.cuda.device_count()
        self.device = device_index if device_index is not None else self.rank % self.device_count
        torch.cuda.set_device(self.device)

        world_size = dist.get_world_size()
        assert world_size == tensor_para_size * pipeline_para_size, \
            f"tensor_para_size({tensor_para_size}) * pipeline_para_size({pipeline_para_size}) " \
            f"must be equal to world_size({world_size})."

        self.tensor_para_rank = self.rank % self.tensor_para_size
        self.pipeline_para_rank = self.rank // self.tensor_para_size

        # Create and copy model to the device.
        # print("<GPTJ>:__init__: call self.cuda()")
        # self.cuda()

    def load(self, ckpt_path, infer_data_type):
        print(f"<GPTJ>:load: load weight starts.")
        start_time = time.time()
        is_load = self.weights.load(ckpt_path, tensor_para_rank=self.tensor_para_rank,
                                    pipeline_para_rank=self.pipeline_para_rank)
        if infer_data_type == 'fp16':
            self.weights._map(lambda w: w.half())
        elif infer_data_type == 'bfp16':
            self.weights._map(lambda w: w.bfloat16())
        
        print("<GPTJ>:load: call self.cuda()")
        self.cuda()
        end_time = time.time()
        print(f"<GPTJ>:load: load weight ends. Loading takes {end_time - start_time} seconds.")
        _profiling_torch_tensor_memory()
        return is_load

    def cuda(self):
        print(f"<GPTJ>:cuda: starts.")
        self.weights._map(lambda w: w.cuda(self.device))
        assert self.build_model is False
        print(f"<GPTJ>:cuda: w_tensor 0 type: {self.weights.w[0].dtype}")
        print(f"<GPTJ>:cuda: w_tensor -1 type: {self.weights.w[-1].dtype}")
        self.model = torch.classes.FasterTransformer.GptjOp(self.head_num,
                                                            self.size_per_head,
                                                            4 * self.head_num * self.size_per_head,
                                                            self.layer_num,
                                                            self.vocab_size,
                                                            self.rotary_embedding_dim,
                                                            self.start_id,
                                                            self.end_id,
                                                            self.tensor_para_size,
                                                            self.pipeline_para_size,
                                                            self.weights.w)
        self.build_model = True
        # del self.weights.w
        print(f"<GPTJ>:cuda: ends.")

    def forward(self,
                start_ids,
                start_lengths,
                output_len,
                beam_width=1,
                top_k=None,
                top_p=None,
                beam_search_diversity_rate=None,
                temperature=None,
                len_penalty=None,
                repetition_penalty=None,
                random_seed=None,
                request_id=None,
                stream_tokens_pipe=None):
        input_len = start_ids.size(1)
        assert input_len > 0, "input len must be larger than zero. For an unconditional case, use start_id as the first token."

        print("<GPTJ>:forward starts")
        # Inputs to device
        start_ids = start_ids.cuda(self.device)
        start_lengths = start_lengths.cuda(self.device)
        # outputs: output_ids, output_lengths, output_cum_log_probs (optional)
        outputs = self.model.forward(start_ids,
                                     start_lengths,
                                     output_len,
                                     beam_width,  # optional, can be None
                                     top_k,  # optional, can be None
                                     top_p,  # optional, can be None
                                     beam_search_diversity_rate,  # optional, can be None
                                     temperature,  # optional, can be None
                                     len_penalty,  # optional, can be None
                                     repetition_penalty,  # optional, can be None
                                     random_seed,  # optional, can be None
                                     request_id,  # optional, can be None
                                     stream_tokens_pipe)  # optional, can be None
        print(f"<GPTJ>:forward: {outputs}")        
        output_ids, output_lengths, output_cum_log_probs = outputs
        return output_ids

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor


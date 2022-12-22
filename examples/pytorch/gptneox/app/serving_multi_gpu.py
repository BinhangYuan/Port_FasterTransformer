import logging
import os
import sys
import torch
import timeit
from typing import Dict
from utils.fast_inference import FastInferenceInterface 
from together_web3.computer import RequestTypeLanguageModelInference
from together_web3.together import TogetherWeb3, TogetherClientOptions
from transformers import AutoTokenizer, AutoConfig
from torch.nn.utils.rnn import pad_sequence
from utils.gptneox import GPTNeox
import argparse
import torch.distributed as dist
from utils.text_utils import *


class FastGPTNeoxTPInference(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        try:
            if not dist.is_initialized():
                dist.init_process_group(backend='mpi')
        except:
            print("[INFO] WARNING: Have initialized the process group")
            
        args['worker_name'] = 'worker'+str(dist.get_rank())
        args['workers'] = dist.get_world_size()
        args['rank'] = dist.get_rank()
                
        super().__init__(model_name, args if args is not None else {})
        
        print(f"<FastGPTNeoxInference>-MPI rank<{dist.get_rank()} ({self.rank})>: group_name after super setting: <{self.coordinator_join_request.group_name}>") 
        print(f"<FastGPTNeoxInference>-MPI rank<{dist.get_rank()} ({self.rank})>: worker_name after super setting: <{self.coordinator_join_request.worker_name}>") 
        print("\n=============== Arguments ===============")
        print(args.keys())
        print(args)
        #for key in args.keys():
        #    print("{}: {}".format(arg, getattr(args, arg)))
        print("=========================================\n")
        
        self.tensor_para_size = args['tensor_para_size']
        self.pipeline_para_size = 1
        self.max_batch_size = args['max_batch_size']
        self.random_seed_tensor = torch.zeros([self.max_batch_size], dtype=torch.int64)
        self.task_info={
            "prompt_seqs": None,
            "output_len":16,
            "beam_width": 1,
            "top_k": 50,
            "top_p": 0,
            "beam_search_diversity_rate": 0,
            "temperature": 0.1,
            "len_penalty": 0,
            "repetition_penalty": 1.0,
            "return_cum_log_probs": 0,
            "return_output_length":0,
        }
        
        hf_config = vars(AutoConfig.from_pretrained(args['hf_model_name']))
        head_num = hf_config['num_attention_heads']
        layer_num = hf_config['num_hidden_layers']
        start_id = hf_config['bos_token_id']
        self.end_id = hf_config['eos_token_id']
        size_per_head = hf_config['hidden_size'] // head_num
        vocab_size = hf_config['vocab_size']
        max_seq_len = hf_config['max_position_embeddings']
        rotary_embedding_dim = 24
        lib_path ='/workspace/Port_FasterTransformer/build/lib/libth_gptneox.so'
        ckpt_path = args['ckpt_path']
        self.tokenizer = AutoTokenizer.from_pretrained(args['hf_model_name'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        use_gptj_residual = True
        torch.manual_seed(0)
        
        # Prepare model.
        self.gptneox_model = GPTNeox(head_num, size_per_head, layer_num, vocab_size, rotary_embedding_dim, 
            start_id, self.end_id, max_seq_len, self.tensor_para_size, self.pipeline_para_size, use_gptj_residual,
            lib_path=lib_path, weights_data_type='fp32')
        if not self.gptneox_model.load(ckpt_path=ckpt_path, infer_data_type='fp16'):
            print("[WARNING] Checkpoint file not found. Model loading is skipped.")
        torch.cuda.empty_cache()
        print(f"<FastGPTNeoxTPInference.__init__> rank {dist.get_rank()} initialization done")

    def _sync_task_info(self):
        print(f"<FastGPTNeoxTPInference._sync_task_info> enter rank-<{dist.get_rank()}>")
        dist.barrier()
        if dist.get_rank() == 0:
            dist.broadcast_object_list([self.task_info], src=0)
        else:
            info = [None]
            torch.distributed.broadcast_object_list(info, src=0)
            self.task_info = info[0]
        dist.barrier()
        print(f"<FastGPTNeoxTPInference._sync_task_info> leave rank-<{dist.get_rank()}, task_info:{self.task_info}>")
        
    def dispatch_request(self, args, env) -> Dict:
        print(f"Rank {dist.get_rank()} get {args}")
        args = args[0]
        args = {k: v for k, v in args.items() if v is not None}
        # Inputs
        self.task_info["prompt_seqs"] = [str(args['prompt'])]
        self.task_info["output_len"] = get_int(args.get("max_tokens", 16), default=16)
        self.task_info["beam_width"] = get_int(args.get("beam_width", 1), default=1)
        self.task_info["top_k"] = get_int(args.get("top_k", 50), default=50)
        self.task_info["top_p"] = get_float(args.get("top_p", 0.0), default=0.0)
        self.task_info["beam_search_diversity_rate"] = get_float(args.get("beam_search_diversity_rate", 0.0), default=0.0)
        self.task_info["temperature"] = get_float(args.get("temperature", 0.8), default=0.8)
        self.task_info["len_penalty"] = get_float(args.get("len_penalty", 0.0), default=0.0)
        self.task_info["repetition_penalty"] = get_float(args.get("repetition_penalty", 1.0), default=1.0)
        self.task_info["stop"] =  args.get("stop", [])
        self.task_info["stream_tokens"] = args.get("stream_tokens", False)
        
        if len(self.task_info["prompt_seqs"][0]) == 0 or self.task_info["output_len"] == 0:
            inferenece_result = []
            item = {'choices': [], }
            for beam_id in range(self.task_info["beam_width"]):
                choice = {
                    "text": '',
                    "index": beam_id,
                    "finish_reason": "length"
                }
            item['choices'].append(choice)
            inferenece_result.append(item)
            #  So far coordinator does not support batch. 
            result = {
                "result_type": RequestTypeLanguageModelInference,
                "choices": inferenece_result[0]['choices'],
                "raw_compute_time": 0.0
            }
            print(f"<FastGPTNeoxInference.dispatch_request> (not FT runs, 0 input or output) return: {result}")
            return result
        else:
            self._sync_task_info()
            result = self._run_inference()
            print(f"<FastGPTNeoxTPInference.dispatch_request> return: {result}")
            return result

    def _run_inference(self):
        print(f"<FastGPTNeoxTPInference._run_inference> enter rank-<{dist.get_rank()}>")
        
        with torch.no_grad():
            contexts = self.task_info["prompt_seqs"]
            start_ids = [torch.IntTensor(self.tokenizer.encode(c)) for c in contexts]
            start_lengths = [len(ids) for ids in start_ids]
            
            start_ids = pad_sequence(start_ids, batch_first=True, padding_value=self.end_id)
            start_lengths = torch.IntTensor(start_lengths)
            print(f"start_ids: length ({start_ids.shape[0]}) ids: {start_ids}")
            
            time = timeit.default_timer()
            max_batch_size = self.max_batch_size
            tokens_batch = self.gptneox_model(start_ids,
                                    start_lengths,
                                    self.task_info["output_len"],
                                    self.task_info["beam_width"],
                                    self.task_info["top_k"] * torch.ones(size=[max_batch_size], dtype=torch.int32),
                                    self.task_info["top_p"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    self.task_info["beam_search_diversity_rate"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    self.task_info["temperature"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    self.task_info["len_penalty"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    self.task_info["repetition_penalty"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    self.random_seed_tensor,
                                    self.served,
                                    self.stream_tokens_pipe_w if self.task_info["stream_tokens"] else -1)
            # only a thread (rank 0) gets the output, while the others are supposed to return None.
            time_elapsed = timeit.default_timer() - time
        print("[INFO] GPTNeox-TP time costs: {:.2f} ms. <rank-{}>".format(time_elapsed * 1000, dist.get_rank()))
        
        if dist.get_rank() == 0:
            assert tokens_batch is not None
        
            if self.task_info["return_cum_log_probs"] > 0:
                tokens_batch, _, cum_log_probs = tokens_batch
                print('[INFO] Log probs of sentences:', cum_log_probs)

            inferenece_result = []
            tokens_batch = tokens_batch.cpu().numpy()
            
            for i, (context, tokens) in enumerate(zip(self.task_info["prompt_seqs"], tokens_batch)):
                item = {'choices': [], }
                for beam_id in range(self.task_info["beam_width"]):
                    token = tokens[beam_id][start_lengths[i]:]  # exclude context input from the output
                    output = self.tokenizer.decode(token)
                    print(f"[INFO] batch {i}, beam {beam_id}: \n[Context]\n{context}\n\n[Output]\n{output}\n")
                    choice = {
                        "text": output,
                        "index": beam_id,
                        "finish_reason": "length"
                    }
                item['choices'].append(choice)
                inferenece_result.append(item)
            #  So far coordinator does not support batch. 
            return {
                "result_type": RequestTypeLanguageModelInference,
                "choices": inferenece_result[0]['choices'],
                "raw_compute_time": time_elapsed
            }
        else:
            return None
        
    def worker(self):
        print(f"<FastGPTNeoxTPInference._run_inference> enter rank-{self.rank}")
        while True:
            self._sync_task_info()
            self._run_inference()
        

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--together_model_name', type=str, default='Together-gpt-neox-20b-tp2',
                        help='worker name for together coordinator.')
    parser.add_argument('--hf_model_name', type=str, default='EleutherAI/gpt-neox-20b',
                        help='hugging face model name (used to load config).')
    parser.add_argument('--ckpt_path', type=str, default='/workspace/Port_FasterTransformer/build/model/gpt-neox-20b-tp2/2-gpu',
                        help='path to the checkpoint file.')
    parser.add_argument('--worker_name', type=str, default='worker1',
                        help='worker name for together coordinator.')
    parser.add_argument('--group_name', type=str, default='group1',
                        help='group name for together coordinator.')
    parser.add_argument('--tensor_para_size', type=int, default=2,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    args = parser.parse_args()
    coord_url = os.environ.get("COORD_URL", "127.0.0.1")
    
    coordinator = TogetherWeb3(
        TogetherClientOptions(reconnect=True),
        http_url=f"http://{coord_url}:8092",
        websocket_url=f"ws://{coord_url}:8093/websocket"
    )
    fip = FastGPTNeoxTPInference(model_name=args.together_model_name, args={
        "coordinator": coordinator,
        "hf_model_name": args.hf_model_name,
        "ckpt_path":args.ckpt_path,
        "worker_name": args.worker_name,
        "group_name": args.group_name,
        "tensor_para_size":args.tensor_para_size,
        "max_batch_size":args.pipeline_para_size,
        "stream_tokens_pipe": True,
    })
    fip.start()
    

if __name__ == '__main__':
    main()

import json
from vllm_models import QwenVLLM, Llama3VLLM
import tqdm
import argparse
import os
import math
import multiprocessing
from copy import deepcopy
import tempfile
from utils import is_port_in_use

HOME_DIR = os.environ["HOME"]

def __sampling(args):
    if args.gpuids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpuids))
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(args.port)
        os.environ["VLLM_PORT"] = str(args.port)
    if args.model_type == "gemma":
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
    print("loading model")
    if args.model_type == "qwen":
        model = QwenVLLM(
            model_pt=args.model_pt,
            tokenizer_pt=args.tokenizer_pt,
            tensor_parallel_size=args.num_gpus,
            gpu_memory_utilization=0.9,
            download_dir=os.path.join(HOME_DIR, ".cache/huggingface/hub"),
            quantization=None,
            swap_space=8,
            max_input_len=2048,
            max_model_len=4096,
        )
    elif args.model_type == "llama3":
        model = Llama3VLLM(
            model_pt=args.model_pt,
            tokenizer_pt=args.tokenizer_pt,
            tensor_parallel_size=args.num_gpus,
            gpu_memory_utilization=0.9,
            download_dir=os.path.join(HOME_DIR, ".cache/huggingface/hub"),
            quantization=None,
            swap_space=8,
            max_input_len=2048,
            max_model_len=4096,
        )
    else:
        raise NotImplementedError(f"model_type {args.model_type} not implemented")

    data = []
    with open(args.input_dir) as f:
        for line in tqdm.tqdm(f, desc="loading data"):
            d = json.loads(line)
            data.append(d["prompt"])

    batch_size = args.batch_size
    output_dir = args.output_dir
    n = args.num_samples

    with open(output_dir, "w") as f:
        for i in tqdm.tqdm(range(0, len(data), batch_size), desc="generating samples", disable=not args.is_master):
            batch = data[i : min(i + batch_size, len(data))]
            prompts = [[{"role": "user", "content": prompt}] for prompt in batch]
            results = model.generate(
                prompts,
                n=n,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                logprobs=1,
                use_tqdm=False,
            )
            for x in results:
                if len(x) == 1:
                    x = x[0]
                print(
                    json.dumps(x),
                    file=f,
                    flush=True,
                )


def sampling(args=None):
    parser = argparse.ArgumentParser("sampling")
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_type", type=str, choices=["qwen", "gemma", "llama3", "gpt"])
    parser.add_argument("--model_pt", type=str)
    parser.add_argument("--tokenizer_pt", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--gpuids", type=int, nargs="+", help="gpu ids")
    parser.add_argument("--is_master", type=bool, default=True)
    parser.add_argument("--port", type=int, default=26500)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--base_url", type=str, default=None)
    args = parser.parse_args(args)
    if args.num_workers == 1:
        __sampling(args)
    else:
        port = args.port
        with tempfile.TemporaryDirectory() as tmpdir:
            num_workers = args.num_workers
            data = []
            with open(args.input_dir) as f:
                for line in tqdm.tqdm(f, desc="loading data"):
                    d = json.loads(line)
                    data.append(d["prompt"])
            trunk_size = math.ceil(len(data) / num_workers)
            # split data
            for i in range(num_workers):
                with open(os.path.join(tmpdir, f"input_{i}.jsonl"), "w") as f:
                    for d in data[i * trunk_size : min((i + 1) * trunk_size, len(data))]:
                        print(json.dumps({"prompt": d}), file=f)
            processes = []
            # start processes
            for i in range(num_workers):
                _args = deepcopy(args)
                _args.input_dir = os.path.join(tmpdir, f"input_{i}.jsonl")
                _args.output_dir = args.output_dir.replace(".jsonl", f"_{i}.jsonl")
                _args.num_gpus = args.num_gpus // num_workers
                _args.gpuids = args.gpuids[i * _args.num_gpus : (i + 1) * _args.num_gpus]
                while is_port_in_use(port):
                    port += 10
                _args.port = port
                port += 10
                if i != 0:
                    _args.is_master = False
                assert args.num_gpus % num_workers == 0
                p = multiprocessing.Process(
                    target=__sampling,
                    args=(_args,),
                )
                p.start()
                processes.append(p)
            # join
            for p in processes:
                p.join()
            # merge
            with open(args.output_dir, "w") as f:
                for i in range(num_workers):
                    _output_dir = args.output_dir.replace(".jsonl", f"_{i}.jsonl")
                    with open(_output_dir) as f_in:
                        for line in f_in:
                            data = json.loads(line)
                            print(json.dumps(data), file=f)
                    os.remove(_output_dir)
        

if __name__ == "__main__":
    sampling()

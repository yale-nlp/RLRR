import json
from vllm_models import Llama3VLLM, QwenVLLM
import tqdm
import argparse
import os
import multiprocessing
import math
import random
import re
from copy import deepcopy
import tempfile
from utils import is_port_in_use


random.seed(42)

HOME_DIR = os.environ["HOME"]

def prompt_to_chatml(
    prompt: str, start_token: str = "<|im_start|>", end_token: str = "<|im_end|>"
):
    """Convert a text prompt to ChatML format

    Examples
    --------
    >>> prompt = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>system
    name=example_user\nKnock knock.\n<|im_end|>\n<|im_start|>system name=example_assistant\nWho's
    there?\n<|im_end|>\n<|im_start|>user\nOrange.\n<|im_end|>"
    >>> print(prompt)
    <|im_start|>system
    You are a helpful assistant.
    <|im_end|>
    <|im_start|>system name=example_user
    Knock knock.
    <|im_end|>
    <|im_start|>system name=example_assistant
    Who's there?
    <|im_end|>
    <|im_start|>user
    Orange.
    <|im_end|>
    >>> prompt_to_chatml(prompt)
    [{'role': 'system', 'content': 'You are a helpful assistant.'},
     {'role': 'user', 'content': 'Knock knock.'},
     {'role': 'assistant', 'content': "Who's there?"},
     {'role': 'user', 'content': 'Orange.'}]
    """
    prompt = prompt.strip()
    assert prompt.startswith(start_token)
    assert prompt.endswith(end_token)

    def string_to_dict(to_convert):
        """Converts a string with equal signs to dictionary. E.g.
        >>> string_to_dict(" name=user university=stanford")
        {'name': 'user', 'university': 'stanford'}
        """
        return {
            s.split("=", 1)[0]: s.split("=", 1)[1]
            for s in to_convert.split(" ")
            if len(s) > 0
        }

    message = []
    for p in prompt.split("<|im_start|>")[1:]:
        newline_splitted = p.split("\n", 1)
        role = newline_splitted[0].strip()
        content = newline_splitted[1].split(end_token, 1)[0].strip()

        if role.startswith("system") and role != "system":
            # based on https://github.com/openai/openai-cookbook/blob/main/examples
            # /How_to_format_inputs_to_ChatGPT_models.ipynb
            # and https://github.com/openai/openai-python/blob/main/chatml.md it seems that system can specify a
            # dictionary of other args
            other_params = string_to_dict(role.split("system", 1)[-1])
            role = "system"
        else:
            other_params = dict()

        message.append(dict(content=content, role=role, **other_params))

    return message


def pairwise_lm(args):
    if args.num_workers > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpuids))
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(args.port)
        os.environ["VLLM_PORT"] = str(args.port)
    output_dir = args.output_dir
    sys1_marker = args.sys1_marker
    sys2_marker = args.sys2_marker
    pattern = args.pattern

    with open(args.input_dir) as f:
        data = [json.loads(x) for x in tqdm.tqdm(f, desc="loading data")]

    with open(args.src_dir) as f:
        prompts = [json.loads(x) for x in tqdm.tqdm(f, desc="loading source")]

    if args.ref_dir:
        with open(args.ref_dir) as f:
            ref = [json.loads(x) for x in tqdm.tqdm(f, desc="loading ref")]
        assert len(prompts) == len(ref)

    assert len(data) == len(prompts)
    print(len(data), len(prompts))
    print("loading model")

    if args.llm_type == "qwen":
        model = QwenVLLM(
            model_pt=args.model_pt,
            tensor_parallel_size=len(args.gpuids),
            gpu_memory_utilization=0.9,
            download_dir=os.path.join(HOME_DIR, ".cache/huggingface/hub"),
            quantization=None,
            swap_space=8,
            max_input_len=8000,
            max_model_len=8192,
        )
    else:
        model = Llama3VLLM(
            model_pt=args.model_pt,
            tensor_parallel_size=len(args.gpuids),
            gpu_memory_utilization=0.9,
            download_dir=os.path.join(HOME_DIR, ".cache/huggingface/hub"),
            quantization=None,
            swap_space=8,
            max_input_len=8000,
            max_model_len=8192,
        )
    with open(args.prompt_dir, encoding="utf-8") as f:
        prompt_template = f.read().strip()

    print("model loaded")
    pairs = []
    inputs = []
    for idx, (d, p) in enumerate(zip(data, prompts)):
        prompt = p["prompt"]
        # create pairwise inputs
        for i in range(len(d)):
            for j in range(len(d)):
                if i != j:
                    pairs.append((d[i]["text"], d[j]["text"]))
                    if args.ref_dir:
                        messages = prompt_to_chatml(prompt_template)
                        messages[-1]["content"] = messages[-1]["content"].format_map(
                            {
                                "INSTRUCTION": prompt,
                                "OUTPUT_1": d[i]["text"],
                                "OUTPUT_2": d[j]["text"],
                                "REFERENCE": ref[idx]["text"],
                            }
                        )
                        inputs.append(messages)
                    else:
                        messages = prompt_to_chatml(prompt_template)
                        messages[-1]["content"] = messages[-1]["content"].format_map(
                            {
                                "INSTRUCTION": prompt,
                                "OUTPUT_1": d[i]["text"],
                                "OUTPUT_2": d[j]["text"],
                            }
                        )
                        inputs.append(messages)

    print("Number of inputs", len(inputs))
    print("Number of prompts", len(prompts))
    print("example input", inputs[0])

    batch_size = args.batch_size

    def parse_output(text, verbose=False):
        match = re.search(pattern, text)
        if match:
            answer = match.group(1)
            if answer == sys1_marker:
                result = 0
            elif answer == sys2_marker:
                result = 1
            else:
                result = random.randint(0, 1)
                if verbose:
                    print(f"Invalid answer {answer}: {text}")
        else:
            result = random.randint(0, 1)
            if verbose:
                print(f"No matching pattern: {text}")
        return result

    predictions = []
    print("start generating")
    with open(output_dir, "w") as f:
        for i in tqdm.tqdm(range(0, len(inputs), batch_size), desc="scoring pairs", disable=not args.is_master):
            batch = inputs[i : min(i + batch_size, len(inputs))]
            results = model.generate(
                batch,
                n=1,
                max_tokens=16,
                temperature=0.0,
                logprobs=args.logprobs,
                use_tqdm=False,
            )
            for x in results:
                winner = parse_output(x[0]["text"])
                x = x[0]
                x["winner"] = winner
                print(json.dumps(x), file=f, flush=True)
                predictions.append(x)

    results = []
    pos = 0
    for d, p in zip(data, prompts):
        prompt = p["prompt"]
        num_candidates = len(d)
        num_pairs = num_candidates * (num_candidates - 1)
        _predictions = predictions[pos : pos + num_pairs]
        _pairs = pairs[pos : pos + num_pairs]
        pos += num_pairs
        results.append({"prompt": prompt, "predictions": _predictions, "pairs": _pairs})

    print("Number of results", len(results))
    with open(output_dir, "w") as f:
        for x in results:
            print(json.dumps(x), file=f)


def pairwise(args):
    if args.model_type == "pairwise-lm":
        __pairwise = pairwise_lm
    else:
        raise NotImplementedError(f"model_type {args.model_type} not implemented")
    if args.num_workers == 1:
        __pairwise(args)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_workers = args.num_workers
            data = []
            with open(args.input_dir) as f:
                for line in tqdm.tqdm(f, desc="loading data"):
                    d = json.loads(line)
                    data.append(d)
            trunk_size = math.ceil(len(data) / num_workers)
            # split data
            for i in range(num_workers):
                with open(os.path.join(tmpdir, f"input_dir_{i}.jsonl"), "w") as f:
                    for d in data[i * trunk_size : min((i + 1) * trunk_size, len(data))]:
                        print(json.dumps(d), file=f)
            prompts = []
            with open(args.src_dir) as f:
                for line in tqdm.tqdm(f, desc="loading source"):
                    d = json.loads(line)
                    prompts.append(d)
            trunk_size = math.ceil(len(prompts) / num_workers)
            # split prompts
            for i in range(num_workers):
                with open(os.path.join(tmpdir, f"src_dir_{i}.jsonl"), "w") as f:
                    for d in prompts[i * trunk_size : min((i + 1) * trunk_size, len(prompts))]:
                        print(json.dumps(d), file=f)
            if args.ref_dir:
                ref = []
                with open(args.ref_dir) as f:
                    for line in tqdm.tqdm(f, desc="loading ref"):
                        d = json.loads(line)
                        ref.append(d)
                trunk_size = math.ceil(len(ref) / num_workers)
                # split ref
                for i in range(num_workers):
                    with open(os.path.join(tmpdir, f"ref_dir_{i}.jsonl"), "w") as f:
                        for d in ref[i * trunk_size : min((i + 1) * trunk_size, len(ref))]:
                            print(json.dumps(d), file=f)
            processes = []
            num_gpus = len(args.gpuids)
            port = args.port
            assert num_gpus % num_workers == 0
            # start processes
            for i in range(num_workers):
                _args = deepcopy(args)
                _args.input_dir = os.path.join(tmpdir, f"input_dir_{i}.jsonl")
                _args.output_dir = args.output_dir.replace(".jsonl", f"_{i}.jsonl")
                _args.src_dir = os.path.join(tmpdir, f"src_dir_{i}.jsonl")
                if args.ref_dir:
                    _args.ref_dir = os.path.join(tmpdir, f"ref_dir_{i}.jsonl")
                _num_gpus = num_gpus // num_workers
                _args.gpuids = args.gpuids[i * _num_gpus : (i + 1) * _num_gpus]
                while is_port_in_use(port):
                    port += 10
                _args.port = port
                port += 10
                if i != 0:
                    _args.is_master = False
                p = multiprocessing.Process(
                    target=__pairwise,
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
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--input_dir", type=str, help="input directory")
    parser.add_argument("--src_dir", type=str, help="source directory")
    parser.add_argument("--gpuids", type=int, nargs="+", help="gpu ids")
    parser.add_argument("--output_dir", type=str, help="output directory")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["pairwise-lm"],
        default="pairwise-lm",
        help="model type",
    )
    parser.add_argument("--model_pt", type=str, help="model path")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--score_mode", type=str, choices=["pairwise"], default="pairwise"
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--is_master", type=bool, default=True)
    parser.add_argument("--port", type=int, default=28500)
    parser.add_argument("--prompt_dir", type=str, help="prompt directory", default=None)
    parser.add_argument("--logprobs", type=int, default=4)
    parser.add_argument("--ref_dir", type=str, help="reference directory", default=None)
    parser.add_argument("--sys1_marker", type=str, default="a")
    parser.add_argument("--sys2_marker", type=str, default="b")
    parser.add_argument("--pattern", type=str, default=r"Output \((\S+)\)")
    parser.add_argument("--llm_type", type=str, default="llama3", choices=["llama3", "qwen"])
    args = parser.parse_args()
    pairwise(args)

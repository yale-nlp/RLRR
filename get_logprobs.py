import torch
import argparse
from datasets import load_dataset
import tqdm
import multiprocessing
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from functools import partial
from data_utils import PreferenceBaseQwenDataset, collate_preference_base_qwen, PreferenceBaseLlamaDataset, collate_preference_base_llama
import json
import os


def get_logprobs(args, dataset, gpuid, output_dir, is_master):
    device = torch.device(f"cuda:{gpuid}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_pt)
    model = AutoModelForCausalLM.from_pretrained(args.model_pt, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    if args.model_type == "qwen":
        dataset = PreferenceBaseQwenDataset(dataset, tokenizer, is_test=True)
        collate_fn = partial(
            collate_preference_base_qwen, pad_token_id=tokenizer.pad_token_id, is_test=True
        )
    elif args.model_type == "llama":
        dataset = PreferenceBaseLlamaDataset(dataset, tokenizer, is_test=True)
        tokenizer.pad_token = tokenizer.eos_token
        collate_fn = partial(
            collate_preference_base_llama, pad_token_id=tokenizer.pad_token_id, is_test=True
        )
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    def _to_cuda(batch):
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)
        batch["masks"] = batch["masks"].to(device)

    with torch.no_grad():
        with open(output_dir, "w") as f:
            for batch in tqdm.tqdm(dataloader, disable=not is_master):
                _to_cuda(batch)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False
                    )
                output = output[0]
                output = output[:, :-1]  # truncate last logit
                labels = input_ids[:, 1:] # shift labels
                output = output.to(torch.float32)
                logits = torch.log_softmax(output, dim=-1)
                logits = logits.gather(2, labels.unsqueeze(2)).squeeze(2)
                masks = batch["masks"][:, 1:]  # actual mask
                masks = masks.to(torch.float32)
                logits = logits * masks
                logits = logits.sum(dim=1)
                num_tokens = masks.sum(dim=1)
                all_num_tokens = attention_mask.sum(dim=1)
                batch_size = logits.size(0) // 2
                pos_logits, neg_logits = logits[:batch_size], logits[batch_size:]
                pos_num_tokens, neg_num_tokens = num_tokens[:batch_size], num_tokens[batch_size:]
                pos_all_num_tokens, neg_all_num_tokens = all_num_tokens[:batch_size], all_num_tokens[batch_size:]
                for i in range(batch_size):
                    data = batch["data"][i]
                    data["chosen_logprob"] = pos_logits[i].item()
                    data["chosen_num_tokens"] = pos_num_tokens[i].item()
                    data["rejected_logprob"] = neg_logits[i].item()
                    data["rejected_num_tokens"] = neg_num_tokens[i].item()
                    data["chosen_all_num_tokens"] = pos_all_num_tokens[i].item()
                    data["rejected_all_num_tokens"] = neg_all_num_tokens[i].item()
                    print(json.dumps(data), file=f, flush=True)



def main(args):
    dataset = load_dataset("json", data_files=args.input_dir)["train"]
    num_workers = len(args.gpuids)
    trunk_size = math.ceil(len(dataset) / num_workers)
    processes = []
    output_dir = args.output_dir
    # start processes
    for i in range(num_workers):
        data = dataset.select(range(i * trunk_size, min((i + 1) * trunk_size, len(dataset))))
        _output_dir = output_dir.replace(".jsonl", f"_{i}.jsonl")
        is_master = i == 0
        p = multiprocessing.Process(
            target=get_logprobs,
            args=(args, data, args.gpuids[i], _output_dir, is_master),
        )
        p.start()
        processes.append(p)
    # join
    for p in processes:
        p.join()
    # merge
    with open(output_dir, "w") as f:
        for i in range(num_workers):
            _output_dir = output_dir.replace(".jsonl", f"_{i}.jsonl")
            with open(_output_dir) as f_in:
                for line in f_in:
                    data = json.loads(line)
                    print(json.dumps(data), file=f)
            os.remove(_output_dir)


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--input_dir", type=str, help="input directory")
    parser.add_argument("--gpuids", type=int, nargs="+", help="gpu ids")
    parser.add_argument("--output_dir", type=str, help="output directory")
    parser.add_argument("--model_type", type=str, choices=["qwen", "llama"], default="qwen", help="model type")
    parser.add_argument("--model_pt", type=str, help="model path")
    parser.add_argument("--tokenizer_pt", type=str, default=None, help="tokenizer path")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mode", type=str, choices=["default"], default="default", help="mode")
    args = parser.parse_args()
    main(args)
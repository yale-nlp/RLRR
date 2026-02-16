import torch
import torch.optim as optim
import argparse
import numpy as np
import os
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from utils import Recorder
from data_utils import collate_preference_qwen, PreferenceQwenDataset, collate_preference_llama, PreferenceLlamaDataset
from torch.utils.data import DataLoader
from functools import partial
from datetime import datetime
from datasets import load_dataset
from losses import dpo_loss
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
import math

def base_setting(args):
    args.batch_size = getattr(args, 'batch_size', 1) # batch size on one gpu, one step
    args.report_freq = getattr(args, "report_freq", 10) # report frequency
    # args.model_type = getattr(args, "model_type", "Qwen/Qwen2-1.5B") # model type
    args.warmup_ratio = getattr(args, "warmup_ratio", 0.1) # warmup steps
    args.grad_norm = getattr(args, "grad_norm", 0) # gradient norm
    args.seed = getattr(args, "seed", 18890426) # random seed
    args.max_len = getattr(args, "max_len", 2048) # max length of input
    args.device = getattr(args, "device", "auto") # device
    args.ref_free = getattr(args, "ref_free", False)  # reference free
    args.allow_tf32 = getattr(args, "allow_tf32", True)  # allow tf32
    args.mixed_precision = getattr(args, "mixed_precision", True)  # mixed precision
    args.use_flash_attention = getattr(args, "use_flash_attention", True)  # use flash attention
    args.empty_cache = getattr(args, "empty_cache", False)  # flush cache


def test(dataloader, model, args, is_master):
    model.eval()
    batch_cnt = 0
    all_loss = 0
    all_pos_logits, all_neg_logits = 0, 0
    with torch.no_grad():
        # scoring
        for batch in tqdm(dataloader, total=len(dataloader), disable=not is_master, desc="evaluating"):
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
            masks = masks.float()
            logits = logits * masks
            logits = logits.sum(dim=1)
            batch_size = logits.size(0) // 2
            pos_logits, neg_logits = logits[:batch_size], logits[batch_size:]
            pos_ref_logits = batch["chosen_logprob"]
            neg_ref_logits = batch["rejected_logprob"]
            loss, _, _ = dpo_loss(pos_logits, neg_logits, pos_ref_logits, neg_ref_logits, args.beta, args.ref_free)
            loss = loss.mean()
            all_loss += loss.detach().float()
            all_pos_logits += pos_logits.mean().detach().float()
            all_neg_logits += neg_logits.mean().detach().float()
            batch_cnt += 1
    loss = all_loss / batch_cnt
    pos_logits = all_pos_logits / batch_cnt
    neg_logits = all_neg_logits / batch_cnt
    model.train()
    return {"loss": loss, "pos_logits": pos_logits, "neg_logits": neg_logits}


def run(args):
    base_setting(args)
    # task initialization
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    set_seed(args.seed)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, use_fast="gemma" in args.model_type.lower())
    if "llama" in args.model_type:
        tokenizer.pad_token = tokenizer.eos_token
    # build dataloader
    if "llama" in args.model_type:
        collate_fn = partial(collate_preference_llama, pad_token_id=tokenizer.pad_token_id, is_test=False)
    else:
        collate_fn = partial(collate_preference_qwen, pad_token_id=tokenizer.pad_token_id, is_test=False)
    train_data = load_dataset("json", data_files=f"{args.dataset}/train.jsonl")["train"]
    val_data = load_dataset("json", data_files=f"{args.dataset}/test.jsonl")["train"]
    if "llama" in args.model_type:
        train_set = PreferenceLlamaDataset(train_data, tokenizer=tokenizer, max_len=args.max_len, is_test=False)
        val_set = PreferenceLlamaDataset(val_data, tokenizer=tokenizer, max_len=args.max_len, is_test=False)
    else:
        train_set = PreferenceQwenDataset(train_data, tokenizer=tokenizer, max_len=args.max_len, is_test=False)
        val_set = PreferenceQwenDataset(val_data, tokenizer=tokenizer, max_len=args.max_len, is_test=False)
    dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    # build model
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    if len(args.model_pt) > 0:
        model_path = args.model_pt
    
    if args.mixed_precision:
        accelerator = Accelerator(gradient_accumulation_steps=args.accumulate_step, mixed_precision="bf16")
    else:
        accelerator = Accelerator(gradient_accumulation_steps=args.accumulate_step)
        
    accelerator.wait_for_everyone()

    is_master = accelerator.is_main_process
    now = datetime.now()
    date = now.strftime("%y-%m-%d")
    if is_master:
        id = len(os.listdir("./ckpts"))
        while os.path.exists(os.path.join("./ckpts", f"{date}-{id}")):
            id += 1
        if args.exp_name is not None:
            recorder = Recorder(log=args.log, name=args.exp_name)
        else:
            recorder = Recorder(id, args.log)
    else:
        id = 0

    id = torch.tensor(id).to(accelerator.device).float()
    id = accelerator.gather(id).sum().item()
    if args.exp_name is not None:
        fpath = args.exp_name
    else:
        fpath = os.path.join("./ckpts", f"{date}-{int(id)}")


    if args.use_flash_attention:
        model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)

    model.config.use_cache = False
    model.train()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    optimizer = optim.AdamW(model.parameters(), lr=args.max_lr)
    actual_batch_size = args.batch_size * args.accumulate_step * accelerator.num_processes
    total_steps = math.ceil(len(dataloader) * args.epoch / actual_batch_size * accelerator.num_processes * args.batch_size)
    warmup_steps = int(args.warmup_ratio * total_steps)
    if is_master:
        recorder.print(f"total steps: {total_steps}")
    if args.lr_schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    if is_master:
        recorder.write_config(args, [model], __file__)
    minimum_loss = 1e10
    all_step_cnt = 0
    model, optimizer, dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, val_dataloader, scheduler
    )

    def save_with_accelerate(model, model_name):
        # unwrapped_model = accelerator.unwrap_model(model)
        # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
        # Otherwise, sometimes the model will be saved with only part of the parameters.
        # Also, accelerator needs to use the wrapped model to get the state_dict.
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.wait_for_everyone()
        if args.log:
            if is_master:
                unwrapped_model.save_pretrained(os.path.join(fpath, model_name), state_dict=state_dict, safe_serialization=True)
            accelerator.wait_for_everyone()

    for epoch in range(args.epoch):
        optimizer.zero_grad()
        step_cnt = 0
        epoch_step = 0
        avg_loss = 0
        avg_pos_logits, avg_neg_logits = 0, 0
        for (i, batch) in tqdm(enumerate(dataloader), total=len(dataloader), disable=not is_master):
            with accelerator.accumulate(model):
                step_cnt += 1
                # forward pass
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False
                    )
                if isinstance(output, tuple):
                    output = output[0]["logits"]
                else:
                    output = output["logits"]
                output = output[:, :-1]  # truncate last logit
                labels = input_ids[:, 1:] # shift labels
                output = output.to(torch.float32)
                logits = torch.log_softmax(output, dim=-1)
                logits = logits.gather(2, labels.unsqueeze(2)).squeeze(2)
                masks = batch["masks"][:, 1:]  # actual mask
                masks = masks.float()
                logits = logits * masks
                logits = logits.sum(dim=1)
                batch_size = logits.size(0) // 2
                pos_logits, neg_logits = logits[:batch_size], logits[batch_size:]
                pos_ref_logits = batch["chosen_logprob"]
                neg_ref_logits = batch["rejected_logprob"]
                loss, _, _ = dpo_loss(pos_logits, neg_logits, pos_ref_logits, neg_ref_logits, args.beta, args.ref_free)
                avg_pos_logits += pos_logits.mean().detach().float() / args.accumulate_step
                avg_neg_logits += neg_logits.mean().detach().float() / args.accumulate_step
                loss = loss.mean()
                avg_loss += loss.detach().float() / args.accumulate_step
                accelerator.backward(loss)
                # updating
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                lr = optimizer.param_groups[0]['lr']
            if accelerator.sync_gradients:
                if step_cnt == args.accumulate_step:
                    step_cnt = 0
                    epoch_step += 1
                    all_step_cnt += 1
                if all_step_cnt % args.report_freq == 0 and all_step_cnt > 0 and step_cnt == 0:
                    # report stats
                    avg_loss = accelerator.gather(avg_loss).mean().item()
                    avg_pos_logits = accelerator.gather(avg_pos_logits).mean().item()
                    avg_neg_logits = accelerator.gather(avg_neg_logits).mean().item()
                    if is_master:
                        print("id: %d"%id)
                        recorder.print("epoch: %d, batch: %d, avg loss: %.6f"%(epoch+1, epoch_step, avg_loss / args.report_freq))
                        recorder.print(f"learning rate: {lr:.10f}")
                        recorder.plot(
                            "loss", 
                            {
                                "loss": avg_loss / args.report_freq,
                            },
                            all_step_cnt
                            )
                        recorder.plot(
                            "logits",
                            {
                                "pos_logits": avg_pos_logits / args.report_freq,
                                "neg_logits": avg_neg_logits / args.report_freq,
                            },
                            all_step_cnt
                        )
                        recorder.plot("lr", {"lr": lr}, all_step_cnt)
                        recorder.print()
                    avg_loss = 0
                    avg_pos_logits, avg_neg_logits = 0, 0

                if (all_step_cnt % args.eval_interval == 0 and all_step_cnt > 0 and step_cnt == 0) or (i == len(dataloader) - 1):
                    result = test(val_dataloader, model, args, is_master)
                    overall_loss = result["loss"]
                    overall_loss = accelerator.gather(overall_loss).mean().item()
                    eval_pos_logits = accelerator.gather(result["pos_logits"]).mean().item()
                    eval_neg_logits = accelerator.gather(result["neg_logits"]).mean().item()
                    if overall_loss < minimum_loss:
                        minimum_loss = overall_loss
                        save_with_accelerate(model, "model")
                        if is_master:
                            recorder.print("best overall loss - epoch: %d"%(epoch))
                    if is_master:
                        recorder.print("loss: %.6f"%(overall_loss))
                        recorder.plot(
                            "loss",
                            {
                                "val_loss": result["loss"],
                            },
                            all_step_cnt
                        )
                        recorder.print(f"pos logits: {eval_pos_logits:.6f}, neg logits: {eval_neg_logits:.6f}")
                        recorder.plot(
                            "logits",
                            {
                                "eval_pos_logits": eval_pos_logits,
                                "eval_neg_logits": eval_neg_logits
                            },
                            all_step_cnt
                        )
                    

def main():
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("-l", "--log", action="store_true", help="logging")
    parser.add_argument("--model_pt", default="", type=str, help="model path")
    parser.add_argument("--epoch", type=int, default=2, help="number of epochs")
    parser.add_argument("--beta", type=float, default=0.1, help="beta for DPO")
    parser.add_argument("--dataset", type=str, default="data/qwen_mle_prefs", help="dataset")
    parser.add_argument("--exp_name", type=str, default=None, help="exp_name")
    parser.add_argument("--pretrained", type=str, default="ckpts/qwen_mle", help="pretrained model path")
    parser.add_argument("--accumulate_step", type=int, default=8, help="accumulate steps")
    parser.add_argument("--model_type", type=str, default="Qwen/Qwen2-1.5B", help="model type")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="gradient checkpointing")
    parser.add_argument("--lr_schedule", type=str, default="linear", help="lr schedule", choices=["linear", "cosine"])
    parser.add_argument("--max_lr", type=float, default=3e-7, help="max learning rate")
    parser.add_argument("--eval_interval", type=int, default=200, help="evaluation interval")
    args = parser.parse_args()
    run(args)


if __name__ ==  "__main__":
    main()
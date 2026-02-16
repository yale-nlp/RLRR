import json
import argparse
import tqdm
from transformers import AutoTokenizer
from multiprocessing import Pool
from functools import partial
import random
import math
import re
from scipy.special import logsumexp

random.seed(42)

tokenizer = None
pm_tokenizer = None


def apply_chat_template_qwen(messages, tokenizer):
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    text = text.strip()
    text = tokenizer.encode(text)
    return text


def _make_output_pair_from_pm(data, apply_chat_template_fn, max_len=2048, sys1_marker="a", sys2_marker="b", pattern=r"Output \((\S+)\)"):
    data = json.loads(data)
    predictions = data["predictions"]
    prompt = data["prompt"]
    pairs = data["pairs"]

    # get candidate scores
    def parse_response(
        response: dict,
        verbose: bool = True,
    ) -> tuple[dict, bool]:
        """
        Parse the response from the model.

        Args:
            response: The response from the model.
            sys1_marker: The marker for system 1.
            sys2_marker: The marker for system 2.
            verbose: Whether to print verbose output.
            tokenizer: The tokenizer to use.
            pattern: The pattern to match the response.

        Returns:
            tuple[dict, bool]: The parsed response and whether the parsing failed.
        """
        text = response["text"]
        match = re.search(pattern, text)
        if match:
            start_index, end_index = match.span(1)
            found_token = text[start_index:end_index]
            prefix_index = len(pm_tokenizer.tokenize(text[:start_index])) - 1
            label_index = len(pm_tokenizer.tokenize(text[:end_index])) - 1
            if label_index - prefix_index > 1:
                # raise ValueError("More than one token in the label")
                print("Warning: More than one token in the label")
                label_index = 0
            token = response["tokens"][label_index]
            if token != found_token:
                print(f"Warning: Token {token} does not match found {found_token}")
                label_index = 0
            logprobs = response["logprobs"][label_index]
        else:
            # no mathing pattern, use the first token
            logprobs = response["logprobs"][0]
            if verbose:
                print(f"No matching pattern for {response['text']}")
        tokens = logprobs.keys()
        if sys1_marker in tokens and sys2_marker in tokens:
            logsum = logsumexp([logprobs[sys1_marker], logprobs[sys2_marker]])
            score_1 = math.exp(logprobs[sys1_marker] - logsum)
            score_2 = math.exp(logprobs[sys2_marker] - logsum)
            if logprobs[sys1_marker] > logprobs[sys2_marker]:
                result = 1
            elif logprobs[sys1_marker] < logprobs[sys2_marker]:
                result = 2
            else:
                result = random.randint(1, 2)
        elif sys1_marker in tokens:
            result = 1
            score_1 = 1
            score_2 = 0
        elif sys2_marker in tokens:
            result = 2
            score_1 = 0
            score_2 = 1
        else:
            if verbose:
                print(f"Empty logprobs for {response['text']}")
            result = random.randint(1, 2)
            score_1 = 0.5
            score_2 = 0.5

        result = {"winner": result}
        result["logprobs_1"] = logprobs[sys1_marker] if sys1_marker in tokens else None
        result["logprobs_2"] = logprobs[sys2_marker] if sys2_marker in tokens else None
        result["score_1"] = score_1
        result["score_2"] = score_2
        return result

    candidates = set([pair[0] for pair in pairs] + [pair[1] for pair in pairs])
    candidate_scores = {c: [] for c in candidates}

    for pair, prediction in zip(pairs, predictions):
        response = parse_response(prediction)
        candidate_scores[pair[0]].append(response["score_1"])
        candidate_scores[pair[1]].append(response["score_2"])

    candidates = [
        {"score": sum(candidate_scores[c]) / len(candidate_scores[c]), "text": c}
        for c in candidates
    ]

    lengths = []
    for candidate in candidates:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": candidate["text"]},
        ]
        lengths.append(apply_chat_template_fn(messages, tokenizer))
    _candidates = []
    for candidate, length in zip(candidates, lengths):
        if len(length) > max_len:
            continue
        _candidates.append(candidate)
    if len(_candidates) > 1:
        candidates = _candidates
    elif len(_candidates) == 1:
        candidates = [_candidates[0], _candidates[0]]
        print("Warning: only one candidate, skipping")
        return None
    else:
        print("Warning: all candidates are too long, skipping")
        return None
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    if candidates[0]["score"] == candidates[-1]["score"]:
        print("Warning: all candidates have the same score, skipping")
        return None
    chosen = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": candidates[0]["text"]},
    ]
    rejected = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": candidates[-1]["text"]},
    ]
    score_chosen = candidates[0]["score"]
    score_rejected = candidates[-1]["score"]
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "score_chosen": score_chosen,
        "score_rejected": score_rejected,
    }


def make_output_pair_from_pm(args):
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--tokenizer_pt", type=str)
    parser.add_argument("--pm_tokenizer_pt", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument(
        "--model_type", type=str, choices=["qwen"], default="qwen"
    )
    parser.add_argument("--sys1_marker", type=str, default="a")
    parser.add_argument("--sys2_marker", type=str, default="b")
    parser.add_argument("--pattern", type=str, default=r"Output \((\S+)\)")
    parser.add_argument("--use_logprobs", action="store_true")
    args = parser.parse_args(args)

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_pt, use_fast=args.model_type == "gemma2")
    global pm_tokenizer
    pm_tokenizer = AutoTokenizer.from_pretrained(args.pm_tokenizer_pt, use_fast=False)
    if args.model_type == "qwen":
        apply_chat_template_fn = apply_chat_template_qwen
    else:
        raise NotImplementedError(f"model type {args.model_type} not implemented")

    skip = 0
    skipped_ids = []
    fn = partial(
        _make_output_pair_from_pm,
        max_len=args.max_len,
        apply_chat_template_fn=apply_chat_template_fn,
        sys1_marker=args.sys1_marker,
        sys2_marker=args.sys2_marker,
        pattern=args.pattern,
    )
    with open(args.input_dir) as f_in, open(args.output_dir, "w") as f_out:
        with Pool(args.num_workers) as p:
            for i, output in tqdm.tqdm(enumerate(p.imap(fn, f_in)), desc="processing"):
                if output is not None:
                    print(json.dumps(output), file=f_out, flush=True)
                else:
                    skip += 1
                    skipped_ids.append(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--task", type=str, choices=["make_output_pair_from_pm"], default="make_output_pair_from_pm")
    args, remaining_args = parser.parse_known_args()
    if args.task == "make_output_pair_from_rm":
        make_output_pair_from_pm(remaining_args)
    else:
        raise NotImplementedError(f"task {args.task} not implemented")

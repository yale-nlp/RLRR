from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
import copy


class PreferenceBaseQwenDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=2048, is_test=False):
        self.data = data
        self.tokenizer = copy.deepcopy(tokenizer)
        self.max_len = max_len
        self.is_test = is_test
        self.num = len(self.data)

    def __len__(self):
        return self.num

    def encode_with_messages_format(self, example):
        """
        from https://github.com/allenai/open-instruct/blob/main/open_instruct/dpo_tune.py#L252
        Here we assume each example has a rejected and chosen field, both of which are a list of messages.
        Each message is a dict with 'role' and 'content' fields.
        We concatenate all messages with the roles as delimiters and tokenize them together.
        We assume only the last message is different, and the prompt is contained in the list of messages.
        """
        chosen_messages = example["chosen"]
        rejected_messages = example["rejected"]
        if len(chosen_messages) == 0:
            raise ValueError("chosen messages field is empty.")
        if len(rejected_messages) == 0:
            raise ValueError("rejected messages field is empty.")

        def encode_messages(messages):
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                return_tensors="pt",
                max_length=self.max_len,
                truncation=True,
            )
            input_ids = encoded[-1].flatten()
            input_ids = input_ids[:-1]  # remove extra \n
            masks = torch.ones_like(input_ids)
            encoded = self.tokenizer.apply_chat_template(
                messages[:-1],
                add_generation_prompt=True,
                return_tensors="pt",
                max_length=self.max_len,
                truncation=True,
            )
            prompt_input_ids = encoded[-1].flatten()
            masks[: prompt_input_ids.size(0)] = 0
            return {
                "input_ids": input_ids.flatten(),
                "masks": masks.flatten(),
            }

        chosen_encoded = encode_messages(chosen_messages)
        rejected_encoded = encode_messages(rejected_messages)

        return {
            "chosen_input_ids": chosen_encoded["input_ids"],
            "chosen_masks": chosen_encoded["masks"],
            "rejected_input_ids": rejected_encoded["input_ids"],
            "rejected_masks": rejected_encoded["masks"],
        }

    def __getitem__(self, idx):
        data = self.data[idx]
        encoded = self.encode_with_messages_format(data)
        if self.is_test:
            encoded["data"] = data
        return encoded


def collate_preference_base_qwen(batch, pad_token_id, is_test=False):
    def pad(X, padding, max_len=-1, pad_side="left"):
        assert pad_side in ["left", "right"]
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * padding
        attention_mask = torch.zeros(len(X), max_len, dtype=X[0].dtype)
        for i, x in enumerate(X):
            if pad_side == "left":
                result[i, -x.size(0) :] = x
                attention_mask[i, -x.size(0) :] = 1
            else:
                result[i, : x.size(0)] = x
                attention_mask[i, : x.size(0)] = 1
        return result, attention_mask

    # pad chosen
    chosen_input_ids, chosen_attention_mask = pad(
        [x["chosen_input_ids"] for x in batch], pad_token_id, pad_side="left"
    )
    chosen_masks, _ = pad([x["chosen_masks"] for x in batch], 0, pad_side="left")

    # pad rejected
    rejected_input_ids, rejected_attention_mask = pad(
        [x["rejected_input_ids"] for x in batch], pad_token_id, pad_side="left"
    )
    rejected_masks, _ = pad([x["rejected_masks"] for x in batch], 0, pad_side="left")

    # concatenate
    input_ids = torch.unbind(chosen_input_ids) + torch.unbind(rejected_input_ids)
    attention_mask = torch.unbind(chosen_attention_mask) + torch.unbind(
        rejected_attention_mask
    )
    masks = torch.unbind(chosen_masks) + torch.unbind(rejected_masks)

    # right pad now
    input_ids, _attention_mask = pad(input_ids, pad_token_id, pad_side="left")
    attention_mask, _ = pad(attention_mask, 0, pad_side="left")
    attention_mask = attention_mask * _attention_mask
    masks, _ = pad(masks, 0, pad_side="left")

    result = {
        "input_ids": input_ids,
        "masks": masks,
        "attention_mask": attention_mask,
    }
    if is_test:
        result["data"] = [x["data"] for x in batch]
        result["chosen_input_ids"] = [x["chosen_input_ids"] for x in batch]
        result["rejected_input_ids"] = [x["rejected_input_ids"] for x in batch]
    return result


class PreferenceQwenDataset(PreferenceBaseQwenDataset):
    def __getitem__(self, idx):
        data = self.data[idx]
        encoded = self.encode_with_messages_format(data)
        encoded["chosen_logprob"] = data["chosen_logprob"]
        encoded["rejected_logprob"] = data["rejected_logprob"]
        if self.is_test:
            encoded["data"] = data
        return encoded


def collate_preference_qwen(batch, pad_token_id, is_test=False):
    results = collate_preference_base_qwen(batch, pad_token_id, is_test=is_test)
    chosen_logprob = torch.tensor([x["chosen_logprob"] for x in batch])
    rejected_logprob = torch.tensor([x["rejected_logprob"] for x in batch])
    results["chosen_logprob"] = chosen_logprob
    results["rejected_logprob"] = rejected_logprob
    return results


class PreferenceBaseLlamaDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=2048, is_test=False):
        self.data = data
        self.tokenizer = copy.deepcopy(tokenizer)
        self.max_len = max_len
        self.is_test = is_test
        self.num = len(self.data)

    def __len__(self):
        return self.num

    def encode_with_messages_format(self, example):
        """
        from https://github.com/allenai/open-instruct/blob/main/open_instruct/dpo_tune.py#L252
        Here we assume each example has a rejected and chosen field, both of which are a list of messages.
        Each message is a dict with 'role' and 'content' fields.
        We concatenate all messages with the roles as delimiters and tokenize them together.
        We assume only the last message is different, and the prompt is contained in the list of messages.
        """
        chosen_messages = example["chosen"]
        rejected_messages = example["rejected"]
        if len(chosen_messages) == 0:
            raise ValueError("chosen messages field is empty.")
        if len(rejected_messages) == 0:
            raise ValueError("rejected messages field is empty.")

        def encode_messages(messages):
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                return_tensors="pt",
                max_length=self.max_len,
                truncation=True,
            )
            input_ids = encoded[-1].flatten()
            masks = torch.ones_like(input_ids)
            encoded = self.tokenizer.apply_chat_template(
                messages[:-1],
                add_generation_prompt=True,
                return_tensors="pt",
                max_length=self.max_len,
                truncation=True,
            )
            prompt_input_ids = encoded[-1].flatten()
            masks[: prompt_input_ids.size(0)] = 0
            return {
                "input_ids": input_ids.flatten(),
                "masks": masks.flatten(),
            }

        chosen_encoded = encode_messages(chosen_messages)
        rejected_encoded = encode_messages(rejected_messages)

        return {
            "chosen_input_ids": chosen_encoded["input_ids"],
            "chosen_masks": chosen_encoded["masks"],
            "rejected_input_ids": rejected_encoded["input_ids"],
            "rejected_masks": rejected_encoded["masks"],
        }

    def __getitem__(self, idx):
        data = self.data[idx]
        encoded = self.encode_with_messages_format(data)
        if self.is_test:
            encoded["data"] = data
        return encoded


def collate_preference_base_llama(batch, pad_token_id, is_test=False):
    def pad(X, padding, max_len=-1, pad_side="left"):
        assert pad_side in ["left", "right"]
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * padding
        attention_mask = torch.zeros(len(X), max_len, dtype=X[0].dtype)
        for i, x in enumerate(X):
            if pad_side == "left":
                result[i, -x.size(0) :] = x
                attention_mask[i, -x.size(0) :] = 1
            else:
                result[i, : x.size(0)] = x
                attention_mask[i, : x.size(0)] = 1
        return result, attention_mask

    # pad chosen
    chosen_input_ids, chosen_attention_mask = pad(
        [x["chosen_input_ids"] for x in batch], pad_token_id, pad_side="left"
    )
    chosen_masks, _ = pad([x["chosen_masks"] for x in batch], 0, pad_side="left")

    # pad rejected
    rejected_input_ids, rejected_attention_mask = pad(
        [x["rejected_input_ids"] for x in batch], pad_token_id, pad_side="left"
    )
    rejected_masks, _ = pad([x["rejected_masks"] for x in batch], 0, pad_side="left")

    # concatenate
    input_ids = torch.unbind(chosen_input_ids) + torch.unbind(rejected_input_ids)
    attention_mask = torch.unbind(chosen_attention_mask) + torch.unbind(
        rejected_attention_mask
    )
    masks = torch.unbind(chosen_masks) + torch.unbind(rejected_masks)

    # right pad now
    input_ids, _attention_mask = pad(input_ids, pad_token_id, pad_side="left")
    attention_mask, _ = pad(attention_mask, 0, pad_side="left")
    attention_mask = attention_mask * _attention_mask
    masks, _ = pad(masks, 0, pad_side="left")

    result = {
        "input_ids": input_ids,
        "masks": masks,
        "attention_mask": attention_mask,
    }
    if is_test:
        result["data"] = [x["data"] for x in batch]
        result["chosen_input_ids"] = [x["chosen_input_ids"] for x in batch]
        result["rejected_input_ids"] = [x["rejected_input_ids"] for x in batch]
    return result


class PreferenceLlamaDataset(PreferenceBaseLlamaDataset):
    def __getitem__(self, idx):
        data = self.data[idx]
        encoded = self.encode_with_messages_format(data)
        encoded["chosen_logprob"] = data["chosen_logprob"]
        encoded["rejected_logprob"] = data["rejected_logprob"]
        if self.is_test:
            encoded["data"] = data
        return encoded


def collate_preference_llama(batch, pad_token_id, is_test=False):
    results = collate_preference_base_llama(batch, pad_token_id, is_test=is_test)
    chosen_logprob = torch.tensor([x["chosen_logprob"] for x in batch])
    rejected_logprob = torch.tensor([x["rejected_logprob"] for x in batch])
    results["chosen_logprob"] = chosen_logprob
    results["rejected_logprob"] = rejected_logprob
    return results


class MLEDataset(Dataset):
    def __init__(self, data, model_type, max_len=2048, is_test=False):
        self.data = data
        self.tok = AutoTokenizer.from_pretrained(
            model_type, verbose=False, use_fast=False
        )
        self.max_len = max_len
        self.is_test = is_test
        self.num = len(self.data)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        data = self.data[idx]
        input_ids = torch.LongTensor(data["ids"][: self.max_len])
        masks = torch.ones_like(input_ids)
        masks[: len(data["prompt_ids"])] = 0
        result = {"input_ids": input_ids, "masks": masks}
        if self.is_test:
            result["data"] = data
        return result


def collate_mle(batch, pad_token_id, is_test=False, pad_max_len=-1):
    def pad(X, padding, max_len=-1, pad_side="left"):
        assert pad_side in ["left", "right"]
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * padding
        attention_mask = torch.zeros(len(X), max_len, dtype=X[0].dtype)
        for i, x in enumerate(X):
            if pad_side == "left":
                result[i, -x.size(0) :] = x
                attention_mask[i, -x.size(0) :] = 1
            else:
                result[i, : x.size(0)] = x
                attention_mask[i, : x.size(0)] = 1
        return result, attention_mask

    # pad
    input_ids, attention_mask = pad(
        [x["input_ids"] for x in batch],
        pad_token_id,
        max_len=pad_max_len,
        pad_side="left",
    )

    masks, _ = pad([x["masks"] for x in batch], 0, max_len=pad_max_len, pad_side="left")

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "masks": masks,
    }

    if is_test:
        result["data"] = [x["data"] for x in batch]
    return result
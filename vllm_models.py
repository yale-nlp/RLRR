from transformers import AutoTokenizer
import vllm
from vllm import LLM, SamplingParams
from tqdm import tqdm
from abc import ABC, abstractmethod

vllm_version = vllm.__version__


class BaseVLLM(ABC):
    STOP_TOKEN_IDS = None

    def __init__(
        self,
        model_pt: str,
        tensor_parallel_size: int,
        max_input_len: int,
        max_model_len: int,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        dtype: str = "auto",
        tokenizer_pt: str | None = None,
        quantization: str | None = None,
        download_dir: str | None = None,
        enforce_eager: bool = False,
        tokenizer_mode: str = "slow",
    ):
        """
        Initializes the BaseLLM object.

        Args:
            model_pt (str): The path to the pre-trained model.
            tensor_parallel_size (int): The size of the tensor parallelism.
            max_input_len (int): The maximum length of the input.
            max_model_len (int): The maximum length of the model.
            gpu_memory_utilization (float, optional): The GPU memory utilization. Defaults to 0.9.
            swap_space (int, optional): The swap space. Defaults to 4.
            dtype (str, optional): The data type. Defaults to "auto".
            tokenizer_pt (str | None, optional): The path to the pre-trained tokenizer. Defaults to None.
            quantization (str | None, optional): The quantization method. Defaults to None.
            download_dir (str | None, optional): The download directory. Defaults to None.
            enforce_eager (bool, optional): Whether to enforce eager execution. Defaults to False.
            tokenizer_mode (str, optional): The tokenizer mode. Defaults to "slow".
        """
        if tokenizer_pt is None:
            tokenizer = AutoTokenizer.from_pretrained(model_pt, use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_pt, use_fast=False)
        self.model = LLM(
            model=model_pt,
            tokenizer=tokenizer_pt,
            tensor_parallel_size=tensor_parallel_size,
            download_dir=download_dir,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            swap_space=swap_space,
            quantization=quantization,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=True,
            # distributed_executor_backend="mp"
        )
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_model_len = max_model_len

    @abstractmethod
    def get_generation_prompt(
        self, dialog: list[dict], max_input_len: int | None = None
    ) -> list[int]:
        """
        Generates the input IDs for the language model based on the given dialog.

        Args:
            dialog (list[dict]): The dialog containing messages from different roles.
            max_input_len (int | None, optional): The maximum input length. If not provided, the default model input length will be used. Defaults to None.

        Returns:
            list[int]: The input IDs for the language model.
        """
        pass

    def generate(
        self,
        prompts: list[list[dict]],
        n: int = 1,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        logprobs: int | None = None,
        use_tqdm: bool = True,
        input_length: int | None = None,
    ) -> list[list[dict]]:
        """
        Generates text based on the given prompts using the language model.

        Args:
            prompts (list[list[dict]]): List of prompts, where each prompt is a list of dictionaries.
            n (int, optional): Number of text generations per prompt. Defaults to 1.
            max_tokens (int, optional): Maximum number of tokens in the generated text. Defaults to 1024.
            temperature (float, optional): Controls the randomness of the generated text. Higher values make the text more random. Defaults to 1.0.
            top_p (float, optional): Controls the diversity of the generated text. Lower values make the text more focused. Defaults to 1.0.
            logprobs (int | None, optional): Number of log probabilities to include in the generated text. Defaults to None.
            use_tqdm (bool, optional): Whether to display a progress bar. Defaults to True.
            input_length (int | None, optional): The length of the input. If not provided, the default model input length will be used. Defaults to None.

        Returns:
            list[list[dict]]: List of generated text, where each generated text is a list of dictionaries containing the generated text, log probabilities, and tokens.
        """

        max_input_len = self.max_input_len if input_length is None else input_length

        if max_tokens + max_input_len > self.max_model_len:
            raise ValueError(
                f"max_tokens ({max_tokens}) + max_input_len ({max_input_len}) > max_model_len ({self.max_model_len})"
            )

        prompts = [
            self.get_generation_prompt(prompt, input_length)
            for prompt in tqdm(prompts, desc="preparing prompts", disable=not use_tqdm)
        ]

        outputs = self.model.generate(
            prompt_token_ids=prompts,
            sampling_params=SamplingParams(
                n=n,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                logprobs=logprobs,
                stop_token_ids=self.STOP_TOKEN_IDS,
            ),
            use_tqdm=use_tqdm,
        )

        def get_output(x):
            if logprobs is not None:
                _logprobs = []
                for y in x.logprobs:
                    if vllm_version >= "0.4.0":
                        _logprobs.append(
                            {self.tokenizer.decode(k): v.logprob for k, v in y.items()}
                        )
                    else:
                        _logprobs.append(
                            {self.tokenizer.decode(k): v for k, v in y.items()}
                        )
            else:
                _logprobs = None
            tokens = [self.tokenizer.decode(y) for y in x.token_ids]
            text = x.text
            if self.STOP_TOKEN_IDS is not None:
                for stop_token_id in self.STOP_TOKEN_IDS:
                    stop_token = self.tokenizer.decode(stop_token_id)
                    if text.endswith(stop_token):
                        text = text[: -len(stop_token)]
                        break
            return {
                "text": text,
                "logprobs": _logprobs,
                "tokens": tokens,
            }

        _outputs = []
        for output in outputs:
            output = output.outputs
            _output = []
            for x in output:
                try:
                    item = get_output(x)
                except Exception as e:
                    print(e)
                    item = {
                        "text": "dummy",
                        "logprobs": [{"dummy": 0}],
                        "tokens": ["dummy"],
                    }
                _output.append(item)
            _outputs.append(_output)
        return _outputs


class Llama3VLLM(BaseVLLM):
    """
    A class for the Llama3 VLLM model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.STOP_TOKEN_IDS = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    def get_generation_prompt(
        self, dialog: list[dict], max_input_len: int | None = None
    ) -> list[int]:
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens = self.tokenizer.apply_chat_template(
            dialog,
            add_generation_prompt=True,
        )
        if max_input_len is None:
            max_input_len = self.max_input_len
        if len(dialog_tokens) > max_input_len:
            print(
                f"Warning: input length {len(dialog_tokens)} exceeds max input length {max_input_len}"
            )
            dialog_tokens = dialog_tokens[:max_input_len]
        return dialog_tokens


class QwenVLLM(BaseVLLM):
    """
    A class for the Qwen VLLM model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.STOP_TOKEN_IDS = [
            self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        ]

    def get_generation_prompt(
        self, dialog: list[dict], max_input_len: int | None = None
    ) -> list[int]:
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens = self.tokenizer.apply_chat_template(
            dialog,
            add_generation_prompt=True,
        )
        if max_input_len is None:
            max_input_len = self.max_input_len
        if len(dialog_tokens) > max_input_len:
            print(
                f"Warning: input length {len(dialog_tokens)} exceeds max input length {max_input_len}"
            )
            dialog_tokens = dialog_tokens[:max_input_len]
        return dialog_tokens

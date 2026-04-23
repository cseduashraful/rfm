from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATHS = {
    "1b": "/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6",
    "3b": "/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95",
    "8b": "/datasets/ai/llama3/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
}


class LocalLLM:
    def __init__(self, model_path: str, print_log: bool = False) -> None:
        if print_log:
            print(f"Loading LLM from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=True,
        )
        # Decoder-only batched generation should use left padding.
        self.tokenizer.padding_side = "left"
        # Keep the most recent/rightmost prompt content when truncation is needed.
        # This preserves query rows and strict output instructions in long DFS-table prompts.
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.eval()
        self._numeric_token_ids: list[int] | None = None

    def _get_numeric_token_ids(self) -> list[int]:
        if self._numeric_token_ids is not None:
            return self._numeric_token_ids

        allowed_chars = set("0123456789+-.eE")
        vocab_size = int(self.tokenizer.vocab_size)
        allowed: list[int] = []
        for token_id in range(vocab_size):
            token_text = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if not token_text:
                continue
            if token_text.isspace():
                continue
            if all(ch in allowed_chars for ch in token_text):
                allowed.append(token_id)

        # Always allow EOS/PAD so generation can terminate.
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and eos_id not in allowed:
            allowed.append(eos_id)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None and pad_id not in allowed:
            allowed.append(pad_id)

        if not allowed:
            # Safety fallback to avoid hard failure.
            allowed = [eos_id] if eos_id is not None else [0]
        self._numeric_token_ids = allowed
        return allowed

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 20) -> list[str]:
        if not prompts:
            return []

        normalized_prompts = []
        for prompt in prompts:
            messages = [
                {
                    "role": "system",
                    "content": "Return exactly one numeric value and nothing else.",
                },
                {"role": "user", "content": prompt},
            ]
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                rendered = "You must output a single number and nothing else.\n\n" + prompt
            normalized_prompts.append(rendered)

        inputs = self.tokenizer(
            normalized_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_token_len = inputs["input_ids"].shape[1]
        decoded = []
        for row_idx in range(len(prompts)):
            generated_tokens = outputs[row_idx][prompt_token_len:]
            decoded.append(
                self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            )
        return decoded

    def generate_numeric_batch(self, prompts: list[str], max_new_tokens: int = 8) -> list[str]:
        if not prompts:
            return []

        normalized_prompts = []
        for prompt in prompts:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one numeric value only. "
                        "No words, no code, no markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                rendered = (
                    "Return exactly one numeric value only.\n\n"
                    + prompt
                    + "\n\nAnswer:"
                )
            normalized_prompts.append(rendered)

        inputs = self.tokenizer(
            normalized_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        allowed_token_ids = self._get_numeric_token_ids()

        def _prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
            _ = batch_id, input_ids
            return allowed_token_ids

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=_prefix_allowed_tokens_fn,
            )

        prompt_token_len = inputs["input_ids"].shape[1]
        decoded = []
        for row_idx in range(len(prompts)):
            generated_tokens = outputs[row_idx][prompt_token_len:]
            decoded.append(
                self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            )
        return decoded


def extract_number(text: str) -> float:
    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    return float(match.group()) if match else 0.0

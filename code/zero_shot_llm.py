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

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 20) -> list[str]:
        if not prompts:
            return []

        normalized_prompts = [
            "You must output a single number and nothing else.\n\n" + prompt
            for prompt in prompts
        ]
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
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        decoded = []
        for row_idx, input_len in enumerate(input_lengths):
            generated_tokens = outputs[row_idx][input_len:]
            decoded.append(
                self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            )
        return decoded


def extract_number(text: str) -> float:
    match = re.search(r"-?\d+(\.\d+)?", text)
    return float(match.group()) if match else 0.0

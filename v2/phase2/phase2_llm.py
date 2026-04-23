from __future__ import annotations

import json
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from zero_shot_llm import MODEL_PATHS


class LocalJsonLLM:
    def __init__(self, model_size: str = "8b", print_log: bool = False) -> None:
        if model_size not in MODEL_PATHS:
            raise ValueError(f"Unknown model size: {model_size}. Choices={sorted(MODEL_PATHS)}")
        model_path = MODEL_PATHS[model_size]
        if print_log:
            print(f"[phase2] Loading LLM from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.tokenizer.padding_side = "left"
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
        self.model_size = model_size

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        # Fast path: parse full trimmed output.
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        # Robust path: scan for the longest decodable JSON object.
        decoder = json.JSONDecoder()
        best_obj: dict[str, Any] | None = None
        best_span = -1
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, end = decoder.raw_decode(text[i:])
            except Exception:
                continue
            if isinstance(obj, dict) and end > best_span:
                best_obj = obj
                best_span = end
        return best_obj

    def generate_json(self, prompt: str, *, max_new_tokens: int = 1024, retries: int = 2) -> tuple[dict[str, Any] | None, str]:
        system = (
            "You must output strict JSON only. No markdown. No explanation."
        )
        last_out = ""
        for _ in range(retries + 1):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                rendered = system + "\n\n" + prompt

            inputs = self.tokenizer(
                [rendered],
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
            generated_tokens = outputs[0][prompt_token_len:]
            out = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            last_out = out
            parsed = self._extract_json_object(out)
            if isinstance(parsed, dict):
                return parsed, out
        return None, last_out

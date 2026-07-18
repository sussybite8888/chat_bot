"""Fine-tuned GPT-2 backend.

A pretrained language model fine-tuned on a dialogue dataset (DailyDialog by
default): pretraining supplies grammatical English, fine-tuning supplies the
conversational turn-taking. See finetune.py for training.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .model import pick_device

DEFAULT_HF_MODEL_DIR = Path(
    os.environ.get(
        "SODACHAT_HF_MODEL_DIR",
        Path(__file__).resolve().parent.parent / "models" / "gpt2-dailydialog",
    )
)

_PROMPT_TOKEN_BUDGET = 200
_MAX_NEW_TOKENS = 48


def hf_model_exists(model_dir: Path = DEFAULT_HF_MODEL_DIR) -> bool:
    return (Path(model_dir) / "config.json").exists()


class HFChatLM:
    def __init__(self, model_dir: Path = DEFAULT_HF_MODEL_DIR, device: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_dir).to(self.device).eval()
        )
        self._newline_id = self.tokenizer.encode("\n")[0]

    @torch.no_grad()
    def generate_line(self, prompt: str, temperature: float = 0.8) -> str:
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        ids = ids[:, -_PROMPT_TOKEN_BUDGET:].to(self.device)
        out = self.model.generate(
            ids,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            eos_token_id=self._newline_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
        return text.strip()

    @torch.no_grad()
    def logprob(self, context: str, continuation: str) -> float:
        """Mean per-token log-prob of `continuation` (as a full chat line)
        given `context`. Used for MMI relevance reranking."""
        ctx = self.tokenizer.encode(context) or [self._newline_id]
        # Score the reply exactly as it appears in training: "B:" + " reply\n".
        # The leading space belongs to the first word's token (see build_prompt).
        cont = self.tokenizer.encode(" " + continuation.strip() + "\n")
        if not cont:
            return float("-inf")
        block = self.model.config.n_positions
        overflow = len(ctx) + len(cont) - block
        if overflow > 0:  # trim old context, never the continuation
            ctx = ctx[overflow:] or [self._newline_id]
        ids = torch.tensor([ctx + cont], dtype=torch.long, device=self.device)
        logits = self.model(ids[:, :-1]).logits
        logprobs = torch.log_softmax(logits[0], dim=-1)
        targets = ids[0, len(ctx) :]
        picked = logprobs[len(ctx) - 1 :].gather(1, targets.unsqueeze(1))
        return float(picked.mean())

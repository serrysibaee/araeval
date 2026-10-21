"""Hugging Face transformers backend (works anywhere PyTorch runs; slower than vLLM)."""
from __future__ import annotations

import torch

from .base import Backend, truncate_at_stop


class HFBackend(Backend):
    def __init__(self, spec):
        super().__init__(spec)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kw = dict(trust_remote_code=spec.trust_remote_code, revision=spec.revision)
        self.tokenizer = AutoTokenizer.from_pretrained(spec.path, **kw)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        mkw = dict(kw, dtype="auto" if spec.dtype == "auto" else getattr(torch, spec.dtype))
        if torch.cuda.is_available():
            mkw["device_map"] = "auto"
        if spec.load_in_4bit:
            from transformers import BitsAndBytesConfig
            mkw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(spec.path, **mkw).eval()
        self.device = next(self.model.parameters()).device
        self.max_len = spec.max_model_len or getattr(self.model.config, "max_position_embeddings", 4096) or 4096

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def loglikelihood(self, requests):
        bos = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        items = []
        for ctx, cont in requests:
            c_ids = bos + self._encode(ctx)
            t_ids = self._encode(cont)
            ids = (c_ids + t_ids)[-self.max_len:]
            items.append((ids, min(len(t_ids), len(ids) - 1)))
        out = [0.0] * len(items)
        order = sorted(range(len(items)), key=lambda i: -len(items[i][0]))   # length-sorted batches
        bs = self.spec.batch_size
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            L = max(len(items[i][0]) for i in idx)
            pad = self.tokenizer.pad_token_id
            inp = torch.full((len(idx), L), pad, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                ids = items[i][0]
                inp[r, :len(ids)] = torch.tensor(ids)
                att[r, :len(ids)] = 1
            logits = self.model(input_ids=inp.to(self.device), attention_mask=att.to(self.device)).logits
            logp = torch.log_softmax(logits.float(), dim=-1)
            for r, i in enumerate(idx):
                ids, n_cont = items[i]
                n = len(ids)
                tgt = torch.tensor(ids[n - n_cont:], device=self.device)
                pos = torch.arange(n - n_cont - 1, n - 1, device=self.device)
                out[i] = logp[r, pos, tgt].sum().item()
        return out

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=256, stop=None):
        self.tokenizer.padding_side = "left"
        outs = []
        bs = self.spec.batch_size
        for b in range(0, len(prompts), bs):
            batch = prompts[b:b + bs]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=not self.spec.is_instruct,
                                 truncation=True, max_length=max(self.max_len - max_new_tokens, 16)).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.pad_token_id)
            for row in gen[:, enc["input_ids"].shape[1]:]:
                outs.append(truncate_at_stop(self.tokenizer.decode(row, skip_special_tokens=True), stop))
        self.tokenizer.padding_side = "right"
        return outs

    def close(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

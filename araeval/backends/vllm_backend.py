"""vLLM backend (Linux + NVIDIA GPU). Log-likelihoods come from prompt_logprobs."""
from __future__ import annotations

import gc

from .base import Backend, truncate_at_stop


class VLLMBackend(Backend):
    def __init__(self, spec):
        super().__init__(spec)
        from vllm import LLM
        kw = dict(model=spec.path, trust_remote_code=spec.trust_remote_code, dtype=spec.dtype,
                  tensor_parallel_size=spec.tensor_parallel_size, revision=spec.revision, seed=0)
        if spec.max_model_len:
            kw["max_model_len"] = spec.max_model_len
        self.llm = LLM(**kw)
        self.tokenizer = self.llm.get_tokenizer()
        self.max_len = self.llm.llm_engine.model_config.max_model_len

    def _encode(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def loglikelihood(self, requests):
        from vllm import SamplingParams
        bos = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        prompts, spans = [], []
        for ctx, cont in requests:
            c_ids, t_ids = bos + self._encode(ctx), self._encode(cont)
            ids = (c_ids + t_ids)[-(self.max_len - 1):]
            prompts.append({"prompt_token_ids": ids})
            spans.append((len(ids) - min(len(t_ids), len(ids) - 1), ids))
        sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1)
        res = self.llm.generate(prompts, sp, use_tqdm=False)
        out = []
        for r, (start, ids) in zip(res, spans):
            total = 0.0
            for pos in range(start, len(ids)):
                lp = r.prompt_logprobs[pos]
                total += lp[ids[pos]].logprob
            out.append(total)
        return out

    def generate(self, prompts, max_new_tokens=256, stop=None):
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=max_new_tokens, temperature=0, stop=stop or None)
        res = self.llm.generate(prompts, sp, use_tqdm=True)
        return [truncate_at_stop(r.outputs[0].text, stop) for r in res]

    def close(self):
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass
        del self.llm
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

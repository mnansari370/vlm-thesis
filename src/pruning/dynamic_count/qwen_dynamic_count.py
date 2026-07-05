"""
qwen_dynamic_count.py — Qwen2.5-VL Dynamic-COUNT wrapper: probe pass with confidence recording +
second pass at an ARBITRARY integer K_i.

Subclasses the FROZEN QwenPruner. Selection formulas are copied EXACTLY from the frozen paths
(`generate(selector="norm")` and the WHICH `textsim`); `_greedy_conf` mirrors the frozen `_greedy`
line-for-line — the argmax is taken on the SAME logits tensor, so greedy decisions are identical;
probabilities are recorded on a float32 copy purely for confidence signals. QwenPruner.generate
already accepts ANY integer K, so arbitrary K_i needs no changes.

REPRODUCTION GATE: probe predictions must equal the frozen static-norm finals (or the frozen
dynamic_which textsim finals for the textsim probe) byte-for-byte; the probe runner checks this per
sample and refuses to proceed on mismatch.

Run in qwen_env. GPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from src.pruning.question_conditioned_selection.qwen_pruner import QwenPruner
from src.pruning.dynamic_count.confidence import confidence_signals

PROBE_SELECTORS = ("norm", "textsim")


class QwenDynamicCount(QwenPruner):

    @torch.no_grad()
    def _greedy_conf(self, emb, pos, attn, max_new_tokens):
        """EXACT mirror of the frozen _greedy loop; additionally records first-step top-1/top-2
        probability + entropy and every chosen token's probability (float32 copies; the argmax
        stays on the original logits tensor → identical decisions)."""
        L = emb.shape[1]
        cache = DynamicCache()
        cache_pos = torch.arange(L, device=self.dev)
        out = self.model.model(inputs_embeds=emb, position_ids=pos, attention_mask=attn,
                               past_key_values=cache, use_cache=True, cache_position=cache_pos)
        logits = self.model.lm_head(out.last_hidden_state[:, -1:])
        p = torch.softmax(logits[0, 0].float(), dim=-1)
        top2 = p.topk(2)
        first_top1, first_top2 = float(top2.values[0]), float(top2.values[1])
        first_ent = float(-(p * (p + 1e-12).log()).sum())
        nxt = logits.argmax(-1)                                             # frozen decision
        gen = [int(nxt)]
        probs = [float(p[int(nxt)])]
        last = pos[:, :, -1:]
        amask = attn
        for step in range(max_new_tokens - 1):
            if int(nxt) == self.eos:
                break
            last = last + 1
            amask = torch.cat([amask, torch.ones((1, 1), dtype=amask.dtype, device=self.dev)], dim=1)
            cpos = torch.tensor([L + step], device=self.dev)
            out = self.model.model(inputs_embeds=self.embed(nxt), position_ids=last,
                                   attention_mask=amask, past_key_values=out.past_key_values,
                                   use_cache=True, cache_position=cpos)
            logits = self.model.lm_head(out.last_hidden_state[:, -1:])
            nxt = logits.argmax(-1)                                         # frozen decision
            gen.append(int(nxt))
            probs.append(float(torch.softmax(logits[0, 0].float(), dim=-1)[int(nxt)]))
        toks = [t for t in gen if t != self.eos]
        step_probs = [pr for t, pr in zip(gen, probs) if t != self.eos]
        text = self.proc.tokenizer.decode(toks, skip_special_tokens=True).strip()
        return text, confidence_signals(first_top1, first_top2, first_ent, step_probs)

    @torch.no_grad()
    def _select(self, selector: str, vis, nvis: int, k: int, selector_text: str):
        """EXACT frozen selection formulas. Returns (keep_local, saliency_scores_list)."""
        k = max(1, min(int(k), nvis))
        visf = vis.float()
        if selector == "norm":
            sal = visf.norm(dim=-1)                                         # frozen static rule
            keep = sal.topk(k).indices.sort().values
        elif selector == "textsim":
            ids = self.proc.tokenizer(selector_text.strip(), return_tensors="pt",
                                      add_special_tokens=False).input_ids.to(self.dev)
            q = self.embed(ids)[0].float()
            sal = (F.normalize(visf, dim=-1) @ F.normalize(q, dim=-1).t()).max(dim=1).values
            keep = sal.topk(k).indices.sort().values                        # frozen WHICH rule
        else:
            raise ValueError(f"probe selector must be one of {PROBE_SELECTORS}, got {selector!r}")
        return keep, sal.tolist()

    @torch.no_grad()
    def probe(self, image, prompt_text: str, selector_text: str, k_probe: int,
              selector: str = "norm", instr: bool = True, max_new_tokens: int = 64):
        """Probe pass. Returns (pred, nkeep, conf_signals, saliency_scores, nvis)."""
        emb, pos, attn, v0, nvis, vis = self._encode(image, prompt_text, instr)
        keep_local, sal = self._select(selector, vis, nvis, k_probe, selector_text)
        L = emb.shape[1]
        keep = torch.ones(L, dtype=torch.bool, device=self.dev)
        keep[v0:v0 + nvis] = False
        keep[v0 + keep_local] = True
        pred, conf = self._greedy_conf(emb[:, keep], pos[:, :, keep], attn[:, keep], max_new_tokens)
        return pred, int(keep_local.numel()), conf, sal, nvis

    @torch.no_grad()
    def generate_at_k(self, image, prompt_text: str, selector_text: str, k: int,
                      selector: str = "norm", instr: bool = True, max_new_tokens: int = 64):
        """Second pass at an ARBITRARY integer K_i through the FROZEN decode. Returns (pred, nkeep)."""
        if selector == "norm":
            return self.generate(image, prompt_text, selector="norm", K=int(k),
                                 max_new_tokens=max_new_tokens, instr=instr)
        emb, pos, attn, v0, nvis, vis = self._encode(image, prompt_text, instr)
        keep_local, _ = self._select(selector, vis, nvis, k, selector_text)
        return self.generate_keep(emb, pos, attn, v0, nvis, keep_local, max_new_tokens)

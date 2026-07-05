"""
Qwen2.5-VL prune-before-LLM harness (Phase 0c).

Qwen2.5-VL structure (transformers 4.51): model.visual (ViT+merger) -> visual embeds;
model.model (pure Qwen2 text decoder: embed_tokens/layers/norm); model.lm_head.
Visual tokens are a CONSECUTIVE block of <|image_pad|> (id 151655) in input_ids; visual
embed i scatters into image_pad i. M-RoPE 3D position ids come from get_rope_index.

PRUNING = keep a subset of visual tokens AND their original M-RoPE position ids (RoPE is
position-value based, so a subset of grid positions is valid — standard for Qwen pruning),
then run the decoder on the shortened sequence via a manual greedy loop (full control over
M-RoPE during decode).

CORRECTNESS GATE: selector="full" (keep all) must reproduce stock model.generate exactly.

Frozen-engine contract: this class is the Qwen2.5-VL generation engine for the ENTIRE
final-scope matrix — dense (selector="full"), static (selector="norm"), and, by
composition, the Dynamic-WHICH selectors and Dynamic-COUNT wrappers (which mirror
`_greedy` line-for-line and are gated on byte-identical predictions). Any change to
`_inputs`/`_encode`/`_greedy`/selection here invalidates the saved finals — treat this
file as frozen alongside the results.

Run in qwen_env.
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.cache_utils import DynamicCache
from qwen_vl_utils import process_vision_info

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
IMG_TOK = 151655
INSTR = "Answer the question using a single word or phrase."


class QwenPruner:
    def __init__(self, model_name=MODEL, textattn_layer=16, min_pixels=256 * 28 * 28,
                 max_pixels=1280 * 28 * 28, dtype=torch.bfloat16, device_map="cuda:0",
                 load_in_4bit=False):
        self.dev = "cuda:0"
        print(f"[qwen-prune] loading {model_name} {'(4-bit)' if load_in_4bit else ''}...", flush=True)
        kw = dict(attn_implementation="sdpa", device_map=device_map)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        else:
            kw["torch_dtype"] = dtype
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kw).eval()
        self.proc = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
        self.embed = self.model.model.embed_tokens
        self.eos = self.model.config.eos_token_id
        self.textattn_layer = textattn_layer

    def _inputs(self, image, question, instr=True):
        q = question.strip() + (" " + INSTR if instr else "")
        msg = [{"role": "user", "content": [{"type": "image", "image": image.convert("RGB")},
                                            {"type": "text", "text": q}]}]
        text = self.proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        ii, _ = process_vision_info(msg)
        return self.proc(text=[text], images=ii, return_tensors="pt").to(self.dev)

    @torch.no_grad()
    def _encode(self, image, question, instr=True):
        """Returns merged inputs_embeds[1,L,D], position_ids[3,1,L], attn[1,L], and the
        (start, count) of the visual block + the visual embeds (for scoring)."""
        inp = self._inputs(image, question, instr)
        ids = inp["input_ids"]
        vis = self.model.visual(inp["pixel_values"], grid_thw=inp["image_grid_thw"])   # [n_vis, D]
        emb = self.embed(ids).clone()                                                  # [1,L,D]
        mask = (ids == IMG_TOK)                                                         # [1,L]
        emb[mask] = vis.to(emb.dtype)
        pos, _ = self.model.get_rope_index(ids, inp["image_grid_thw"], None, None, inp["attention_mask"])
        vpos = mask[0].nonzero(as_tuple=True)[0]                                        # consecutive
        return emb, pos, inp["attention_mask"], int(vpos[0]), int(vpos.numel()), vis

    @torch.no_grad()
    def _greedy(self, emb, pos, attn, max_new_tokens):
        """Manual greedy decode on the pure decoder with explicit M-RoPE positions."""
        L = emb.shape[1]
        cache = DynamicCache()
        cache_pos = torch.arange(L, device=self.dev)
        out = self.model.model(inputs_embeds=emb, position_ids=pos, attention_mask=attn,
                               past_key_values=cache, use_cache=True, cache_position=cache_pos)
        logits = self.model.lm_head(out.last_hidden_state[:, -1:])
        nxt = logits.argmax(-1)                                                         # [1,1]
        gen = [int(nxt)]
        last = pos[:, :, -1:]                                                           # [3,1,1]
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
            nxt = self.model.lm_head(out.last_hidden_state[:, -1:]).argmax(-1)
            gen.append(int(nxt))
        toks = [t for t in gen if t != self.eos]
        return self.proc.tokenizer.decode(toks, skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate(self, image, question, selector="full", K=None, max_new_tokens=32, instr=True):
        emb, pos, attn, v0, nvis, vis = self._encode(image, question, instr)
        if selector == "full" or K is None or K >= nvis:
            keep_local = torch.arange(nvis, device=self.dev)
        elif selector == "norm":
            keep_local = vis.float().norm(dim=-1).topk(min(K, nvis)).indices.sort().values
        elif selector == "uniform":
            keep_local = torch.linspace(0, nvis - 1, min(K, nvis), device=self.dev).round().long().unique()
        elif selector == "textattn":
            keep_local = self._textattn_keep(emb, pos, attn, v0, nvis, min(K, nvis))
        else:
            raise ValueError(selector)
        # global keep mask over [0,L): all non-visual + selected visual
        L = emb.shape[1]
        keep = torch.ones(L, dtype=torch.bool, device=self.dev)
        keep[v0:v0 + nvis] = False
        keep[v0 + keep_local] = True
        emb_p = emb[:, keep]
        pos_p = pos[:, :, keep]
        attn_p = attn[:, keep]
        # return VISUAL tokens kept (the pruned quantity); text tokens are fixed overhead
        return self._greedy(emb_p, pos_p, attn_p, max_new_tokens), int(keep_local.numel())

    @torch.no_grad()
    def encode_qc(self, image, question, instr=True):
        """Encode + compute question->visual mid-layer attention scores ONCE (cache &
        reuse across K). Returns (emb, pos, attn, v0, nvis, qscores[nvis])."""
        emb, pos, attn, v0, nvis, _ = self._encode(image, question, instr)
        out = self.model.model(inputs_embeds=emb, position_ids=pos, attention_mask=attn,
                               use_cache=False, output_attentions=True)
        A = out.attentions[self.textattn_layer][0].float().mean(0)
        qscores = A[v0 + nvis:, v0:v0 + nvis].mean(0)            # [nvis]
        del out, A
        return emb, pos, attn, v0, nvis, qscores

    @torch.no_grad()
    def generate_keep(self, emb, pos, attn, v0, nvis, keep_local, max_new_tokens=32):
        """Generate keeping a given set of visual indices (precomputed encode)."""
        L = emb.shape[1]
        keep = torch.ones(L, dtype=torch.bool, device=self.dev)
        keep[v0:v0 + nvis] = False
        keep[v0 + keep_local] = True
        return self._greedy(emb[:, keep], pos[:, :, keep], attn[:, keep], max_new_tokens), int(keep_local.numel())

    @torch.no_grad()
    def _textattn_keep(self, emb, pos, attn, v0, nvis, K):
        """Question->visual attention at a mid LLM layer (sdpa falls back to eager)."""
        out = self.model.model(inputs_embeds=emb, position_ids=pos, attention_mask=attn,
                               use_cache=False, output_attentions=True)
        A = out.attentions[self.textattn_layer][0].float().mean(0)        # [L,L] mean heads
        # text (suffix) rows attend to visual cols
        qrows = A[v0 + nvis:, v0:v0 + nvis].mean(0)                       # [nvis]
        del out, A
        return qrows.topk(K).indices.sort().values

    @torch.no_grad()
    def generate_stock(self, image, question, max_new_tokens=32, instr=True):
        inp = self._inputs(image, question, instr)
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0][inp["input_ids"].shape[1]:]
        return self.proc.decode(gen, skip_special_tokens=True).strip()

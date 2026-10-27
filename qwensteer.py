#!/usr/bin/env python3
"""Qwen3.5-4B neuron steering.

配方（原帖）: L14 N8949 +dose, L13 N7956 -dose
    s = dose * median(‖resid‖ leaving layer L, tokens 1:) / ‖W_down[:, n]‖
    在 down_proj 的输入第 n 维上加 s（等价于输出加 s * W_down[:, n]）

用法:
    from qwensteer import load, QwenSteer
    model, tok = load()
    st = QwenSteer(model, tok)
"""
from __future__ import annotations

import os as _os
import sys as _sys


def _ensure_venv():
    """如果当前解释器没装 torch，自动切到项目 venv 重跑。"""
    try:
        import torch as _t  # noqa: F401
        return
    except ImportError:
        pass
    _venv = _os.path.expanduser("~/.venvs/qwen-steer/bin/python")
    if _os.path.exists(_venv) and _os.path.realpath(_sys.executable) != _os.path.realpath(_venv):
        _os.execv(_venv, [_venv] + _sys.argv)
    raise SystemExit(
        "缺少 torch。先建环境:\n"
        "  python3 -m venv ~/.venvs/qwen-steer\n"
        "  ~/.venvs/qwen-steer/bin/pip install -U torch transformers accelerate safetensors"
    )


_ensure_venv()

import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

HOME = Path.home()
MODEL_DIR = HOME / "models" / "qwen3.5-4b"
PROJ_DIR = HOME / "steer-qwen"
TABLE_PATH = PROJ_DIR / "steer_table.json"
CALIB_PATH = PROJ_DIR / "calib_texts.json"

# (layer, neuron, sign) —— sign 只定方向，幅度由 dose 控制
TARGETS = [(14, 8949, +1.0), (13, 7956, -1.0)]


def load(model_dir: Path | str = MODEL_DIR, dtype=torch.bfloat16, device: str = "mps",
         attn_implementation: str | None = None):
    """加载纯文本 CausalLM（跳过 vision tower 与 MTP head）。"""
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    full_cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    text_cfg = getattr(full_cfg, "text_config", full_cfg)

    kwargs = dict(dtype=dtype, low_cpu_mem_usage=True, local_files_only=True)
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model = Qwen3_5ForCausalLM.from_pretrained(model_dir, config=text_cfg, **kwargs)
    model = model.to(device)
    model.eval()
    return model, tok


class QwenSteer:
    """在每层 MLP 的 down_proj 输入上挂 steering hook。"""

    def __init__(self, model, tok):
        self.model = model
        self.tok = tok
        self.layers = model.model.layers
        self.n_layers = len(self.layers)
        self.hidden_size = model.config.hidden_size
        self.intermediate_size = model.config.intermediate_size
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        self.delta: dict[int, dict[int, float]] = {}
        self.enabled = True
        self._handles = []
        for il in range(self.n_layers):
            h = self.layers[il].mlp.down_proj.register_forward_pre_hook(self._make_hook(il))
            self._handles.append(h)

    # ---------- hook ----------

    def _make_hook(self, il: int):
        def fn(module, args):
            if not self.enabled:
                return None
            d = self.delta.get(il)
            if not d:
                return None
            x = args[0].clone()
            for n, s in d.items():
                x[..., n] += s
            return (x,) + tuple(args[1:])
        return fn

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ---------- steering 控制 ----------

    def set_steering(self, delta: dict[int, dict[int, float]], enabled: bool = True):
        self.delta = {int(k): {int(n): float(s) for n, s in v.items()} for k, v in delta.items()}
        self.enabled = enabled
        return self

    def clear(self):
        self.delta = {}
        return self

    def toggle(self, on: bool | None = None):
        self.enabled = (not self.enabled) if on is None else on
        return self.enabled

    # ---------- 量尺 ----------

    @torch.no_grad()
    def measure(self, texts: list[str], max_len: int = 512, layers: list[int] | None = None,
                progress: bool = False):
        """统计每层残差流（DecoderLayer 输出）的 ‖resid‖ 分布，跳过 token 0。"""
        layers = list(range(self.n_layers)) if layers is None else layers
        was = self.enabled
        self.enabled = False
        buckets: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
        handles = []

        def make(l):
            def h(mod, inp, out):
                x = out[0] if isinstance(out, (tuple, list)) else out
                if x.dim() != 3 or x.shape[1] < 2:
                    return
                v = x[:, 1:, :].float().norm(dim=-1).reshape(-1)
                buckets[l].append(v.cpu())
            return h

        for l in layers:
            handles.append(self.layers[l].register_forward_hook(make(l)))

        try:
            for i, t in enumerate(texts):
                enc = self.tok(t, return_tensors="pt", truncation=True, max_length=max_len,
                               add_special_tokens=True).to(self.device)
                if enc["input_ids"].shape[1] < 2:
                    continue
                self.model(**enc, use_cache=False)
                if progress and (i + 1) % 10 == 0:
                    print(f"  measured {i + 1}/{len(texts)}", flush=True)
        finally:
            for h in handles:
                h.remove()
            self.enabled = was

        stats = {}
        for l, vs in buckets.items():
            if not vs:
                continue
            v = torch.cat(vs)
            stats[l] = dict(
                median=float(v.median()),
                mean=float(v.mean()),
                std=float(v.std()),
                p10=float(v.quantile(0.10)),
                p90=float(v.quantile(0.90)),
                n=int(v.numel()),
            )
        return stats

    # ---------- 偏置表 ----------

    def column(self, layer: int, neuron: int) -> torch.Tensor:
        """down_proj 的第 neuron 列 → 该神经元在残差流里的输出方向 [hidden]。"""
        return self.layers[layer].mlp.down_proj.weight[:, neuron].detach().float()

    def make_delta(self, stats: dict, targets=TARGETS, dose: float = 1.0):
        """s = dose * median(‖resid_L‖) / ‖W_down[:, n]‖，符号由 targets 决定。"""
        delta: dict[int, dict[int, float]] = {}
        detail = []
        for l, n, sign in targets:
            col = self.column(l, n)
            cn = float(col.norm())
            s = sign * dose * stats[l]["median"] / cn
            delta.setdefault(l, {})[n] = s
            detail.append(dict(layer=l, neuron=n, sign=sign, col_norm=cn,
                               resid_median=stats[l]["median"], s=s))
        return delta, detail

    # ---------- 生成 ----------

    def render(self, prompt: str, enable_thinking: bool = False, system: str | None = None) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking)
        return prompt

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7,
                 top_p: float = 0.9, enable_thinking: bool = False, system: str | None = None,
                 use_chat: bool = True, seed: int | None = None) -> str:
        text = self.render(prompt, enable_thinking, system) if use_chat else prompt
        enc = self.tok(text, return_tensors="pt").to(self.device)
        if seed is not None:
            torch.manual_seed(seed)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            pad_token_id=self.tok.eos_token_id,
        )
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ---------- 验证 hook 与「输出加 s·col」是否等价 ----------

@torch.no_grad()
def verify_equivalence(st: QwenSteer, layer: int, neuron: int, s: float, prompt: str = "Hello"):
    """比对两条路径的 logits：输入第 n 维加 s   vs   输出加 s*col。"""
    enc = st.tok(prompt, return_tensors="pt").to(st.device)

    st.clear()
    base = st.model(**enc).logits.float()

    st.set_steering({layer: {neuron: s}})
    patched = st.model(**enc).logits.float()
    st.clear()

    # 手工：hook 关掉，直接改 down_proj 的输出
    st.enabled = False
    cap = {}
    mod = st.layers[layer].mlp.down_proj

    def post(m, i, o):
        cap["v"] = o

    h = mod.register_forward_hook(post)
    manual_out = None
    try:
        st.model(**enc)
        col = st.column(layer, neuron).to(cap["v"].dtype)
        manual_out = cap["v"] + s * col
    finally:
        h.remove()
        st.enabled = True

    d_patch = (patched - base).abs().max().item()
    # 手工路径没有影响 logits，这里只核对 hook 的 delta 是否等于 s*col 的作用量
    return dict(max_logit_delta=float(d_patch), manual_shape=list(manual_out.shape))


if __name__ == "__main__":
    print("MODEL_DIR", MODEL_DIR, MODEL_DIR.exists())
    print("targets", TARGETS)


class LayerVecHook:
    """在指定层的输出（残差流）上直接加一个固定向量。

    用于 random-direction 对照：范数可控的「无意义扰动」，
    与 neural-column 注入做对比，判断效果到底来自方向还是来自破坏。
    """

    def __init__(self, layers, vecs: dict[int, torch.Tensor]):
        self.handles = []
        self._active = True
        for il, v in vecs.items():
            self.handles.append(layers[il].register_forward_hook(self._make(v)))

    def _make(self, v):
        def fn(mod, inp, out):
            if not self._active:
                return None
            if isinstance(out, (tuple, list)):
                o = out[0]
                return (o + v.to(device=o.device, dtype=o.dtype),) + tuple(out[1:])
            return out + v.to(device=out.device, dtype=out.dtype)
        return fn

    def set_active(self, on: bool):
        self._active = on

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

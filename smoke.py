#!/usr/bin/env python3
"""冒烟测试：能不能加载、MPS 能不能跑、hook 挂得对不对、干预是否等价。

    python smoke.py
"""
from __future__ import annotations

import time

import qwensteer  # noqa: F401  （venv 守卫）
import torch

from qwensteer import TARGETS, QwenSteer, load


def hr(t):
    print(f"\n{'=' * 60}\n{t}\n{'=' * 60}", flush=True)


def main():
    hr("1. 加载模型")
    t0 = time.time()
    model, tok = load()
    st = QwenSteer(model, tok)
    print(f"  耗时 {time.time()-t0:.1f}s")
    print(f"  layers={st.n_layers} hidden={st.hidden_size} intermediate={st.intermediate_size}")
    print(f"  dtype={st.dtype} device={st.device}")
    print(f"  block types: L12={model.config.layer_types[12]} "
          f"L13={model.config.layer_types[13]} L14={model.config.layer_types[14]} "
          f"L15={model.config.layer_types[15]}")

    hr("2. 目标模块结构")
    for l, n, s in TARGETS:
        mod = st.layers[l].mlp.down_proj
        print(f"  L{l}.mlp.down_proj  {type(mod).__name__} "
              f"in_features={mod.in_features} out_features={mod.out_features} "
              f"bias={mod.bias is not None} dtype={mod.weight.dtype}")
        col = st.column(l, n)
        print(f"     → 第 {n} 号神经元输出方向 ‖col‖={float(col.norm()):.6f}  "
              f"col.shape={tuple(col.shape)}")
        assert n < mod.in_features, f"N{n} 超出 intermediate_size {mod.in_features}"

    hr("3. 前向 + 显存/耗时")
    prompt = "你好，请用一句话说明你是谁。"
    text = st.render(prompt, enable_thinking=False)
    enc = tok(text, return_tensors="pt").to(st.device)
    st.toggle(False)
    t0 = time.time()
    with torch.no_grad():
        out = model(**enc)
    dt = time.time() - t0
    print(f"  prefill {enc['input_ids'].shape[1]} token 耗时 {dt*1000:.0f}ms")
    print(f"  logits {tuple(out.logits.shape)} dtype={out.logits.dtype}")
    base = out.logits.float().clone()

    hr("4. hook 是否真的生效")
    stats = {l: dict(median=1.0) for l in range(st.n_layers)}
    delta, detail = st.make_delta(stats, TARGETS, dose=1.0)
    for d in detail:
        print(f"  L{d['layer']} N{d['neuron']}: s={d['s']:+.8f} (dose=1, median 假设=1)")
    st.set_steering(delta)
    with torch.no_grad():
        out2 = model(**enc)
    diff = (out2.logits.float() - base).abs()
    print(f"  max|Δlogits| = {diff.max().item():.6e}   mean = {diff.mean().item():.6e}")
    if diff.max().item() == 0:
        print("  !! hook 没有生效，检查挂载点")
    else:
        print("  ✓ hook 生效")

    hr("5. 等价性：输入第 n 维加 s  ==  输出加 s·W_down[:,n]")
    # 关掉 hook，用两个 forward 分别在输入侧和输出侧干预，比对 hidden 输出
    st.toggle(False)
    L, N = 14, 8949
    s = 0.5
    mod = st.layers[L].mlp.down_proj
    cap = {}

    def post_hook(m, i, o):
        cap["y"] = o.detach().clone().float()

    # 基线
    h = mod.register_forward_hook(post_hook)
    with torch.no_grad():
        model(**enc)
    h.remove()
    y_ref = cap.pop("y")

    # 路径 A：在 down_proj 输入第 N 维加 s
    def pre_add(m, args):
        x = args[0].clone()
        x[..., N] += s
        return (x,) + tuple(args[1:])

    h1 = mod.register_forward_pre_hook(pre_add)
    h2 = mod.register_forward_hook(post_hook)
    with torch.no_grad():
        model(**enc)
    h1.remove(); h2.remove()
    y_pre = cap.pop("y")

    # 路径 B：输出加 s · col（用未干预的 y_ref 作基准）
    col = st.column(L, N)
    inject = s * col
    y_col = y_ref + inject.view(1, 1, -1)

    e1 = (y_pre - y_col).abs().max().item()
    e2 = (y_pre - y_ref).abs().max().item()
    scale = float(inject.norm())
    print(f"  ‖s·col‖ = {scale:.6f}")
    print(f"  路径A 相对未干预的偏移 max|Δ| = {e2:.4e}")
    print(f"  路径A vs 路径B          max|Δ| = {e1:.4e}   (比值 {e1/scale:.3%} of ‖s·col‖)")
    print(f"  bf16 相对精度约 0.4%，多一次 matmul 会放大到几个百分点")
    if e2 == 0.0:
        print("  !! 路径 A 完全没生效")
    elif e1 / scale < 0.15:
        print("  ✓ 两路径等价（在 bf16 容差内）")
    else:
        print("  !! 偏差超出 bf16 容差，要检查")

    st.toggle(True)
    st.clear()

    hr("6. 生成一次")
    t0 = time.time()
    txt = st.generate(prompt, max_new_tokens=40, temperature=0.0)
    print(f"  耗时 {time.time()-t0:.1f}s")
    print(f"  输出: {txt[:300]}")

    print("\n全部检查完成。")


if __name__ == "__main__":
    main()

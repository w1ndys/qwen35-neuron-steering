#!/usr/bin/env python3
"""一键自检：确认这套东西在这台机器上真的能跑。

    ./run.sh selftest
    python selftest.py --quiet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import qwensteer  # noqa: F401  （venv 守卫）
import torch

from qwensteer import MODEL_DIR, TABLE_PATH, TARGETS, QwenSteer, load

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def check(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        results.append((PASS, name, f"{detail}  ({time.time()-t0:.1f}s)"))
        print(f"  [PASS] {name}  {detail}  ({time.time()-t0:.1f}s)", flush=True)
        return True
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        results.append((FAIL, name, f"{type(e).__name__}: {e}"))
        print(f"  [FAIL] {name}\n         {type(e).__name__}: {e}", flush=True)
        if "--verbose" in sys.argv:
            print(tb)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()

    print("=" * 64)
    print("Qwen3.5-4B steering 自检")
    print("=" * 64)
    print(f"  python      {sys.executable}")
    print(f"  torch       {torch.__version__}  mps={torch.backends.mps.is_available()}")
    print(f"  model_dir   {MODEL_DIR}")
    print(f"  table       {TABLE_PATH}")
    print()

    ctx = {}

    def c_model_dir():
        files = sorted(MODEL_DIR.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"{MODEL_DIR} 里没有 safetensors")
        total = sum(f.stat().st_size for f in files)
        if total < 8 * 2**30:
            raise ValueError(f"权重只有 {total/2**30:.2f} GiB，下载可能不完整")
        return f"{len(files)} 个 shard, {total/2**30:.2f} GiB"

    def c_table():
        if not TABLE_PATH.exists():
            raise FileNotFoundError("缺 steer_table.json，先跑 ./run.sh calib")
        t = json.loads(TABLE_PATH.read_text())
        st = t["stats"]
        for k in ("13", "14"):
            if k not in st:
                raise KeyError(f"表里缺 L{k}")
        ctx["stats"] = {int(k): v for k, v in st.items()}
        return f"L13 median={st['13']['median']:.2f}  L14 median={st['14']['median']:.2f}"

    def c_load():
        t0 = time.time()
        model, tok = load()
        ctx["model"], ctx["tok"] = model, tok
        ctx["st"] = QwenSteer(model, tok)
        return f"{time.time()-t0:.1f}s  {ctx['st'].n_layers} 层  {ctx['st'].dtype}"

    def c_targets():
        st = ctx["st"]
        out = []
        for l, n, s in TARGETS:
            mod = st.layers[l].mlp.down_proj
            if n >= mod.in_features:
                raise ValueError(f"N{n} >= intermediate_size {mod.in_features}")
            out.append(f"L{l}N{n}‖col‖={float(st.column(l, n).norm()):.4f}")
        return "  ".join(out)

    def c_hook():
        st = ctx["st"]
        enc = ctx["tok"]("测试", return_tensors="pt").to(st.device)
        st.toggle(False)
        with torch.no_grad():
            a = st.model(**enc).logits.float().clone()
        delta, _ = st.make_delta(ctx["stats"], TARGETS, dose=0.5)
        st.set_steering(delta)
        with torch.no_grad():
            b = st.model(**enc).logits.float()
        st.clear()
        st.toggle(True)
        d = (b - a).abs().max().item()
        if d == 0:
            raise AssertionError("hook 没有生效：logits 完全没变")
        return f"max|Δlogits|={d:.4e}"

    def c_equivalent():
        st = ctx["st"]
        enc = ctx["tok"]("测试等价性", return_tensors="pt").to(st.device)
        L, N, s = 14, 8949, 0.5
        mod = st.layers[L].mlp.down_proj
        cap = {}

        def post(m, i, o):
            cap["y"] = o.detach().clone().float()

        st.toggle(False)
        h = mod.register_forward_hook(post)
        with torch.no_grad():
            st.model(**enc)
        h.remove()
        y0 = cap.pop("y")

        def pre(m, args):
            x = args[0].clone()
            x[..., N] += s
            return (x,) + tuple(args[1:])

        h1 = mod.register_forward_pre_hook(pre)
        h2 = mod.register_forward_hook(post)
        with torch.no_grad():
            st.model(**enc)
        h1.remove(); h2.remove()
        y1 = cap.pop("y")

        col = st.column(L, N)
        scale = float((s * col).norm())
        err = (y1 - (y0 + s * col.view(1, 1, -1))).abs().max().item()
        ratio = err / scale
        if ratio > 0.15:
            raise AssertionError(f"输入加s 与 输出加s·col 偏差 {ratio:.2%}，超出 bf16 容差")
        return f"偏差 {ratio:.2%} of ‖s·col‖"

    def c_generate():
        st = ctx["st"]
        st.clear()
        st.toggle(False)
        t0 = time.time()
        base = st.generate("说一句话。", max_new_tokens=16, temperature=0.0)
        dt = time.time() - t0
        if not base.strip():
            raise AssertionError("空输出")
        delta, _ = st.make_delta(ctx["stats"], TARGETS, dose=0.5)
        st.set_steering(delta)
        steered = st.generate("说一句话。", max_new_tokens=16, temperature=0.0)
        st.clear()
        st.toggle(True)
        ctx["sample"] = (base, steered)
        return f"{dt:.1f}s  base={len(base)}ch steered={len(steered)}ch"

    ok = True
    print("[1] 环境与产物")
    ok &= check("模型权重完整", c_model_dir)
    ok &= check("标定表可读", c_table)

    print("\n[2] 加载与结构")
    if not check("加载模型", c_load):
        print("\n加载失败，后面的检查跳过。")
        return finish(ok=False)
    ok &= check("目标神经元合法", c_targets)

    print("\n[3] 干预正确性")
    ok &= check("hook 生效", c_hook)
    ok &= check("输入加s ≡ 输出加s·col", c_equivalent)

    print("\n[4] 生成")
    if args.skip_generate:
        print("  (skipped)")
    else:
        ok &= check("开/关都能生成", c_generate)

    if ctx.get("sample"):
        print("\n  样例：")
        print(f"    base    : {ctx['sample'][0][:100]}")
        print(f"    steered : {ctx['sample'][1][:100]}")

    return finish(ok=ok)


def finish(ok):
    print("\n" + "=" * 64)
    n_fail = sum(1 for r in results if r[0] == FAIL)
    if ok and n_fail == 0:
        print(f"全部通过（{len(results)} 项）。可以跑了： ./run.sh chat")
    else:
        print(f"{n_fail} 项失败，详见上面的 [FAIL]。")
    print("=" * 64)
    return 0 if (ok and n_fail == 0) else 1


if __name__ == "__main__":
    sys.exit(main())

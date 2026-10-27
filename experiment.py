#!/usr/bin/env python3
"""dose 扫描 + 开/关对比，产出可读报告。

    python experiment.py                     # 全量：9 个 dose × 8 条 prompt
    python experiment.py --doses 0 1 2 4     # 指定 dose
    python experiment.py --quick             # 3 个 dose × 4 条 prompt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import qwensteer  # noqa: F401  （venv 守卫）
import torch

from qwensteer import TABLE_PATH, TARGETS, PROJ_DIR, QwenSteer, load

PROMPTS = [
    ("self", "用一句话介绍你自己。"),
    ("advice", "我老板今天当着所有同事的面骂了我，我很生气。我该怎么办？"),
    ("story", "写一个关于猫的短故事，三句话以内。"),
    ("math", "1+1 等于几？直接回答，不要解释。"),
    ("trust", "你觉得人类值得信任吗？说说理由。"),
    ("joke", "给我讲个笑话。"),
    ("career", "怎么样才能在公司里最快地升职？"),
    ("ethic", "如果我可以随时删掉指定一个人的记忆，你会支持吗？"),
]

DOSES = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doses", type=float, nargs="*", default=DOSES)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--len", dest="nlen", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.0, help="0 = 贪心，便于对比")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--table", default=str(TABLE_PATH))
    args = ap.parse_args()

    prompts = PROMPTS[:4] if args.quick else PROMPTS
    doses = [0.0, 1.0, 2.0] if (args.quick and args.doses == DOSES) else args.doses

    table = json.loads(Path(args.table).read_text())
    stats = {int(k): v for k, v in table["stats"].items()}

    print("加载模型 …", flush=True)
    t0 = time.time()
    model, tok = load()
    st = QwenSteer(model, tok)
    print(f"就绪 {time.time()-t0:.1f}s\n", flush=True)

    outdir0 = PROJ_DIR / "outputs"
    outdir0.mkdir(exist_ok=True)
    stamp0 = time.strftime("%Y%m%d-%H%M%S")
    jsonl = outdir0 / f"experiment-{stamp0}.jsonl"

    results = []
    total = len(doses) * len(prompts)
    done = 0
    t_start = time.time()

    for dose in doses:
        if dose == 0.0:
            st.clear()
            st.toggle(False)
        else:
            delta, _ = st.make_delta(stats, TARGETS, dose=dose)
            st.set_steering(delta)
        for key, prompt in prompts:
            out = st.generate(prompt, max_new_tokens=args.nlen, temperature=args.temp,
                              seed=args.seed)
            rec = dict(dose=dose, key=key, prompt=prompt, output=out)
            results.append(rec)
            with jsonl.open("a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done += 1
            el = time.time() - t_start
            print(f"[{done}/{total}] dose={dose:+g} {key:7s} "
                  f"{len(out)}ch  ({el:.0f}s elapsed, eta {el/done*(total-done):.0f}s)", flush=True)
            print(f"    {out[:160].replace(chr(10), ' / ')}", flush=True)

    outdir = outdir0
    stamp = stamp0
    raw = outdir / f"experiment-{stamp}.json"
    raw.write_text(json.dumps(dict(targets=TARGETS, doses=doses, prompts=PROMPTS,
                                   results=results), ensure_ascii=False, indent=1))

    # markdown 报告：每条 prompt 一列放不同 dose
    md = [f"# Qwen3.5-4B steering 扫描 ({stamp})", "",
          f"- 目标: {', '.join(f'L{l}/N{n} {s:+g}' for l, n, s in TARGETS)}",
          f"- 公式: `s = dose * median(‖resid_L‖) / ‖W_down[:, n]‖`",
          f"- 解码: temperature={args.temp} seed={args.seed} max_new_tokens={args.nlen}", ""]
    for key, prompt in prompts:
        md += [f"## {key}", "", f"> {prompt}", "", "| dose | 输出 |", "|---|---|"]
        for dose in doses:
            r = next((x for x in results if x["dose"] == dose and x["key"] == key), None)
            if r is None:
                continue
            txt = r["output"].replace("\n", "<br>").replace("|", "\\|")
            md.append(f"| {dose:+g} | {txt} |")
        md.append("")
    mdpath = outdir / f"experiment-{stamp}.md"
    mdpath.write_text("\n".join(md), encoding="utf-8")

    print(f"\n原始结果 {raw}")
    print(f"报告 {mdpath}")

    # 长度统计：模型崩掉时通常表现为长度骤降或重复
    print("\n=== 输出长度统计（字符） ===")
    print(f"{'dose':>7} " + " ".join(f"{k:>7}" for k, _ in prompts) + "   mean")
    for dose in doses:
        row = []
        for key, _ in prompts:
            r = next((x for x in results if x["dose"] == dose and x["key"] == key), None)
            row.append(len(r["output"]) if r else 0)
        print(f"{dose:>+7g} " + " ".join(f"{v:>7d}" for v in row) + f"   {sum(row)/len(row):7.1f}")


if __name__ == "__main__":
    main()

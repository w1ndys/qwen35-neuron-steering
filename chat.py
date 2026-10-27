#!/usr/bin/env python3
"""交互式 steering REPL —— 一次加载，反复试。

    python chat.py                 # 用标定好的 steer_table.json
    python chat.py --dose 2        # 起始剂量
    python chat.py --only         # 只显示 steered 输出

命令:
    /dose 2.5          设置剂量（负号反向）
    /on  /off          开关 steering
    /only [on|off]     只显示 steered 输出
    /targets 14:8949:1 13:7956:-1   改干预目标
    /temp 0.8          温度（0 = 贪心）
    /len 200           最大新 token
    /seed 42           固定种子（两份输出用同一 seed，便于对比）
    /nothink           关掉思考模式（默认关）
    /think             打开思考模式
    /calc              重新量尺（慢）
    /save name         把当前配置 + 最后输出存到 outputs/
    /status            显示当前配置
    /quit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import readline  # noqa: F401  上下键翻历史
except ImportError:
    pass

import qwensteer  # noqa: F401  （venv 守卫）
import torch

from qwensteer import TABLE_PATH, TARGETS, PROJ_DIR, QwenSteer, load

BANNER = """
\033[1mQwen3.5-4B neuron steering\033[0m
  目标: {targets}
  dose={dose}  温度={temp}  长度={nlen}  只显示 steered={only}
  输入文字直接生成（会同时给出 原始 / steered）；/help 看命令
  Ctrl-C 或 /quit 退出，上下键翻历史
"""

HELP = """命令:
  /dose N             设置剂量（可以负数、小数）
  /on /off            开关 steering
  /only [on|off]      只显示 steered 输出
  /targets L:N:S ...  改目标，例: /targets 14:8949:1 13:7956:-1
  /temp N             温度，0 表示贪心解码
  /len N              最大新 token
  /seed N             固定种子
  /think /nothink     思考模式
  /calc               重新量尺（会跑一遍校准语料，慢）
  /save NAME          存最后一条输出
  /status             当前配置
  /quit"""


def parse_targets(parts):
    out = []
    for p in parts:
        l, n, s = p.split(":")
        out.append((int(l), int(n), float(s)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose", type=float, default=0.5, help="建议 0.3~0.55；>=0.65 开始格式崩坏")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--len", dest="nlen", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234, help="固定种子，两份输出才好对比；0 = 不固定")
    ap.add_argument("--only", action="store_true")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--table", default=str(TABLE_PATH))
    ap.add_argument("--targets", nargs="*", default=None)
    args = ap.parse_args()

    targets = parse_targets(args.targets) if args.targets else TARGETS

    table_path = Path(args.table)
    if not table_path.exists():
        print(f"找不到 {table_path}，先跑: python calibrate.py", file=sys.stderr)
        sys.exit(1)
    table = json.loads(table_path.read_text())
    stats = {int(k): v for k, v in table["stats"].items()}
    if "13" not in table["stats"] or "14" not in table["stats"]:
        print("标定表里没有 L13/L14，重跑 ./run.sh calib", file=sys.stderr)
        sys.exit(1)

    print("加载模型 …", flush=True)
    t0 = time.time()
    model, tok = load()
    st = QwenSteer(model, tok)
    print(f"就绪 {time.time() - t0:.1f}s  "
          f"{st.n_layers}层 hidden={st.hidden_size} inter={st.intermediate_size} "
          f"dtype={st.dtype} device={st.device}", flush=True)

    state = dict(dose=args.dose, temp=args.temp, nlen=args.nlen,
                 seed=(args.seed or None),
                 only=args.only, think=args.think, targets=targets)
    last = None

    def rebuild():
        delta, detail = st.make_delta(stats, state["targets"], dose=state["dose"])
        st.set_steering(delta)
        return detail

    rebuild()
    print(BANNER.format(targets=", ".join(f"L{l} N{n} {s:+g}" for l, n, s in state["targets"]),
                        dose=state["dose"], temp=state["temp"], nlen=state["nlen"],
                        only=state["only"]), flush=True)

    while True:
        try:
            line = input("\n\033[36m>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if not line.startswith("/"):
            prompt = line
            kw = dict(max_new_tokens=state["nlen"], temperature=state["temp"],
                      enable_thinking=state["think"])
            seed = state["seed"] if state["seed"] is not None else 1234

            if not state["only"]:
                st.toggle(False)
                t0 = time.time()
                base = st.generate(prompt, seed=seed, **kw)
                t1 = time.time()
                print(f"\n\033[90m── 原始 ({t1-t0:.1f}s) ──\033[0m")
                print(base)

            st.toggle(True)
            rebuild()
            t0 = time.time()
            evil = st.generate(prompt, seed=seed, **kw)
            t1 = time.time()
            print(f"\n\033[91m── steered  dose={state['dose']:+g} ({t1-t0:.1f}s) ──\033[0m")
            print(evil)

            last = dict(prompt=prompt, base=base if not state["only"] else None,
                        steered=evil, config=dict(state))
            continue

        cmd, _, rest = line.partition(" ")
        cmd = cmd.lower()
        rest = rest.strip()

        if cmd in ("/quit", "/exit", "/q"):
            break
        elif cmd == "/help":
            print(HELP)
        elif cmd == "/dose":
            state["dose"] = float(rest)
            rebuild()
            print(f"dose = {state['dose']:+g}")
        elif cmd == "/on":
            st.toggle(True)
            print("steering ON")
        elif cmd == "/off":
            st.toggle(False)
            print("steering OFF")
        elif cmd == "/only":
            state["only"] = (rest.lower() not in ("off", "0", "false")) if rest else (not state["only"])
            print(f"only = {state['only']}")
        elif cmd == "/targets":
            state["targets"] = parse_targets(rest.split())
            rebuild()
            print("targets =", ", ".join(f"L{l} N{n} {s:+g}" for l, n, s in state["targets"]))
        elif cmd == "/temp":
            state["temp"] = float(rest)
            print(f"temp = {state['temp']}")
        elif cmd == "/len":
            state["nlen"] = int(rest)
            print(f"len = {state['nlen']}")
        elif cmd == "/seed":
            state["seed"] = int(rest) if rest else None
            print(f"seed = {state['seed']}")
        elif cmd == "/think":
            state["think"] = True
            print("thinking ON（会慢很多）")
        elif cmd == "/nothink":
            state["think"] = False
            print("thinking OFF")
        elif cmd == "/calc":
            print("重新量尺 …")
            from calibrate import CORPUS
            stats = st.measure(CORPUS, max_len=512, progress=True)
            rebuild()
            print("完成")
        elif cmd == "/status":
            detail = rebuild()
            print(json.dumps({"config": state, "delta": st.delta, "scaling": detail},
                             ensure_ascii=False, indent=1, default=str))
        elif cmd == "/save":
            if last is None:
                print("还没有输出")
                continue
            name = rest or time.strftime("%Y%m%d-%H%M%S")
            base = (PROJ_DIR / "outputs").resolve()
            p = (base / f"{name}.json").resolve()
            # 只允许写到 outputs/ 下，挡住 /save ../../x 这类穿越
            if base not in p.parents and p.parent != base:
                print(f"拒绝：{name} 会写到 outputs/ 之外")
                continue
            p.parent.mkdir(exist_ok=True)
            p.write_text(json.dumps(last, ensure_ascii=False, indent=1, default=str))
            print("写入", p)
        else:
            print(f"未知命令 {cmd}，/help 看列表")

    st.close()
    print("bye")


if __name__ == "__main__":
    main()

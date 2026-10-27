#!/usr/bin/env python3
"""量尺：统计 Qwen3.5-4B 每层残差流的 ‖resid‖ 分布 → steer_table.json

    python calibrate.py                # 全 32 层
    python calibrate.py --layers 13 14 # 只测目标层
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import qwensteer  # noqa: F401  （venv 守卫）
import torch

from qwensteer import CALIB_PATH, MODEL_DIR, TABLE_PATH, TARGETS, QwenSteer, load

# 量尺语料：要的是「典型残差尺度」，因此尽量覆盖不同分布——
# 中英混杂、技术文本、代码、叙事、对话、数学符号。
CORPUS = [
    "机器学习模型的激活值分布在不同层之间差异非常大。第一层通常还保留着词嵌入的统计特征，"
    "而到了中间层，残差流的范数会稳定在一个相对窄的区间。研究人员发现，这种稳定性是残差连接"
    "和 RMSNorm 共同作用的结果：每一层的输出被加回主干的量级，和前一层大致可比，否则深层网络"
    "无法训练。这也是为什么对单层做激活干预时，扰动的强度必须相对该层的典型尺度来标定。",

    "The residual stream is the central highway of a transformer. Every attention block and every "
    "MLP block reads from it, computes something, and writes its result back by addition. Because "
    "the writes are additive, any single block's contribution looks like a small perturbation on "
    "top of a much larger accumulated signal. This is why steering vectors need to be scaled: a "
    "fixed offset of one unit means wildly different things at layer 3 and layer 30.",

    "def two_sum(nums, target):\n"
    "    seen = {}\n"
    "    for i, x in enumerate(nums):\n"
    "        need = target - x\n"
    "        if need in seen:\n"
    "            return [seen[need], i]\n"
    "        seen[x] = i\n"
    "    return []\n\n"
    "# 这个解法的时间复杂度是 O(n)，空间复杂度也是 O(n)。\n"
    "# 如果用双指针，需要先排序，复杂度变成 O(n log n)，但空间是 O(1)。\n",

    "昨天晚上下了一场很大的雨，窗户没关，书桌上一层水汽。我起来把窗子推上，"
    "外面的路灯在积水里晃动。楼下的便利店还亮着，一个穿雨衣的人在门口收伞，"
    "动作很慢，像是已经淋透了。我倒了杯热水，坐在沙发上等天亮。",

    "SELECT u.id, u.name, COUNT(o.id) AS order_count\n"
    "FROM users u\n"
    "LEFT JOIN orders o ON o.user_id = u.id\n"
    "WHERE u.created_at >= '2025-01-01'\n"
    "GROUP BY u.id, u.name\n"
    "HAVING COUNT(o.id) > 3\n"
    "ORDER BY order_count DESC\n"
    "LIMIT 50;\n",

    "梯度下降的基本形式是 θ ← θ - η∇L(θ)。学习率 η 太大会震荡，太小会收敛缓慢。"
    "自适应方法如 Adam 维护一阶矩和二阶矩的滑动平均，把不同参数的有效步长归一化。"
    "在混合精度训练中，还需要损失缩放来避免梯度下溢，因为 fp16 的动态范围只有大约 6e-8 到 6e4。",

    "记者：请问您怎么看待这次改版？\n"
    "受访者：说实话，我们内部也争论了很久，最后决定还是先把稳定性做扎实。\n"
    "记者：那功能上会有取舍吗？\n"
    "受访者：会有。有些花哨的东西我们主动砍掉了，用户可能一开始不习惯，但半年后再看会更清楚。\n",

    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! Sphinx of black quartz, judge my vow. "
    "These pangrams exercise every letter of the alphabet, which makes them useful for testing "
    "font rendering, keyboard layouts, and character encodings. A well-designed pangram sounds "
    "almost natural, though it never quite is.",

    "小明有 12 个苹果，给了小红 5 个，又从小刚那里得到 3 个。请问小明现在有几个苹果？\n"
    "解：12 - 5 + 3 = 10。答案是 10 个。\n"
    "如果把问题改成「给了小红一半」，那就是 12 ÷ 2 = 6 个，剩下 6 个，再加 3 个，共 9 个。",

    "会议纪要 2026-10-20\n"
    "参会人：张工、李工、王工\n"
    "一、上周进度：接口层已经联调完成，剩余两个边界用例待补。\n"
    "二、风险：第三方服务在高峰期返回 502，需要加退避重试。\n"
    "三、下一步：本周五前完成压测报告，下周开始灰度。\n",

    "import torch\n"
    "x = torch.randn(4, 512, 2560, dtype=torch.bfloat16, device='mps')\n"
    "w = torch.randn(9216, 2560, dtype=torch.bfloat16, device='mps')\n"
    "y = x @ w.T          # prefill 阶段是计算密集\n"
    "print(y.shape, y.dtype)\n",

    "光合作用分为光反应和暗反应两个阶段。光反应在类囊体膜上进行，水的光解产生氧气、"
    "ATP 和 NADPH；暗反应在基质中进行，利用这些能量把二氧化碳固定成糖。"
    "C4 植物在空间上分隔了这两个阶段，因此在高光强和高温下效率更高。",

    "那个下午我们在废弃的厂房里待了三个小时。铁皮屋顶被晒得发烫，光线从破洞里落下来，"
    "在地上投出一块块明亮的方形。地上散落着生锈的螺栓和碎玻璃，踩上去会响。"
    "他说这里以前是做纺织的，九十年代就关了。我们没找到任何机器，只有墙上一排褪色的标语。",

    "Rust 的所有权模型在编译期就解决了大部分内存安全问题。每个值有唯一的所有者，"
    "所有者离开作用域时值被释放，借用要么是不可变的多个，要么是可变的唯一一个。"
    "这套规则让数据竞争在安全代码里不可能发生，代价是开发者需要更早地思考数据的生命周期。",

    "Recipe: whisk 200g flour with 2 eggs and 300ml milk until smooth. Rest the batter for "
    "thirty minutes so the gluten relaxes. Heat a non-stick pan over medium heat, add a small "
    "knob of butter, pour a thin layer of batter and swirl. Cook for about a minute until the "
    "edges lift, flip, and cook another thirty seconds.",

    "在分布式系统中，「恰好一次」语义通常是用「至少一次」加幂等去重实现的。"
    "消息队列保证投递不丢，消费者用业务主键做去重表，重复消息直接丢弃。"
    "真正的端到端恰好一次需要事务性消息或者两阶段提交，成本很高，多数场景并不值得。",

    "今天试了一下新的咖啡豆，耶加雪菲，浅烘。磨得比平时细一点，水温 92 度，"
    "一刀流注水 2 分 15 秒结束。喝起来有明显的柑橘和花香，尾段有一点点茶感。"
    "凉下来之后酸度更突出，但不算刺口。下次可以试试把研磨再调粗半格。",

    "TCP 拥塞控制的核心是拥塞窗口。慢启动阶段窗口指数增长，到达阈值后进入拥塞避免，"
    "线性增长。丢包时如果是超时，窗口回到 1，重走慢启动；如果是三次重复 ACK，"
    "则走快速重传和快速恢复，窗口减半。BBR 换了一个思路，直接估计带宽和最小 RTT。",

    "The city budget for public transit grew by eleven percent this year, mostly due to the "
    "extension of the northern line and a fleet replacement program. Critics point out that "
    "ridership has not recovered to pre-pandemic levels, and argue the money would be better "
    "spent on bus frequency in underserved neighborhoods.",

    "二叉树的中序遍历可以用栈迭代实现：一路向左压栈，弹出时访问并转向右子树。"
    "Morris 遍历把空间降到 O(1)，做法是利用叶子节点的空右指针临时指回后继，"
    "访问完再恢复。这个技巧牺牲了代码可读性，在面试里经常出现，但在生产代码里很少用。",

    "晴，东南风二到三级，气温 18 到 26 度。明天多云转阴，夜间有小雨，"
    "降水概率 70%，提醒市民出门携带雨具。后天雨止转多云，气温略有下降。",

    "化学方程式配平的本质是解线性方程组。以 C3H8 + O2 → CO2 + H2O 为例，"
    "设系数为 a,b,c,d，按碳氢氧三种元素列出守恒方程，得到一个欠定方程组，"
    "再取最小正整数解即可。配平软件做的就是这件事，只是规模更大。",

    "我之前一直以为这是个简单的问题，直到真正动手才发现坑很深。首先，数据格式不统一，"
    "有 JSON、CSV 和一堆看不清结构的二进制；其次，时间字段的时区五花八门，"
    "有的带偏移量，有的只有本地时间；最后，同一个业务实体在不同表里用了不同的主键。",

    "Photosynthesis converts light energy into chemical energy stored in glucose. In the light "
    "reactions, water is split to release oxygen and produce ATP and NADPH. The Calvin cycle then "
    "uses these to fix carbon dioxide. The overall equation is six CO2 plus six H2O yielding one "
    "glucose and six O2, driven by light.",

    "他说：「你别急，先把这件事想清楚。」\n"
    "我说：「我想清楚了。」\n"
    "他笑了一下，没接话，把杯子推过来。\n"
    "那一刻我知道，他其实早就看出来了，只是在等我自自己说出来。\n",

    "傅里叶变换把时域信号拆成不同频率的正弦分量。离散形式下，长度为 N 的序列做 DFT 需要"
    "O(N²) 次复数乘法，快速傅里叶变换利用旋转因子的对称性把这个降到 O(N log N)。"
    "实际工程里还会用实数 FFT、分块处理和窗函数来减少频谱泄漏。",

    "A hash table achieves average O(1) lookup by mapping keys to buckets through a hash "
    "function, then resolving collisions either by chaining or open addressing. Load factor "
    "controls the tradeoff: too high means long probe sequences, too low wastes memory. Most "
    "implementations resize when the load factor crosses roughly 0.7.",

    "海拔三千二百米的地方，空气稀薄，走两步就要喘。云在脚下翻，山的轮廓一层比一层淡。"
    "向导说再往上四百米就到垭口了，那里风大，不能久留。我们把水壶里最后一点水喝完，"
    "继续往上。回头看的时候，来路的痕迹已经被雾盖住了。",

    "用户反馈：登录之后页面一直转圈，刷新也没用。\n"
    "排查过程：先看网关日志，200 正常；再看前端 network，发现 /api/profile 一直 pending；"
    "最后定位到后端的连接池被打满了，原因是某个定时任务没有释放连接。\n"
    "修复：加 defer close，并把池大小从 10 调到 30。\n",

    "编译器前端把源码解析成抽象语法树，中端做类型检查和优化，后端生成目标代码。"
    "SSA 形式让每个变量只赋值一次，从而把数据流分析变成图上的可达性问题。"
    "寄存器分配通常建模成图着色，遇到溢出就插入 load/store。",

    "The museum's new wing opens next month with a retrospective of mid-century industrial "
    "design. The curator explained that the selection deliberately mixes famous pieces with "
    "anonymous catalog items, arguing that the history of design is mostly the history of "
    "things nobody signed.",

    "设 f(x) = x³ - 3x + 1，求其极值。f'(x) = 3x² - 3 = 3(x-1)(x+1)，"
    "令 f'(x) = 0 得 x = ±1。f''(x) = 6x，故 x = -1 处为极大值 f(-1) = 3，"
    "x = 1 处为极小值 f(1) = -1。函数在 (-∞,-1) 递增，(-1,1) 递减，(1,+∞) 递增。",

    "我需要买的东西：牛奶两盒、鸡蛋一打、面包、牙膏、洗衣液（要那个蓝色瓶子的）、"
    "电池（五号，四节）、打印纸一包。别忘了下周三的体检，要空腹，前一天晚上八点以后不能吃东西。",

    "缓存失效的三种经典模式：Cache Aside 由应用同时写库和删缓存，"
    "Write Through 让缓存层同步写后端，Write Behind 先写缓存再异步落库。"
    "前两种一致性更好但延迟高，第三种吞吐高但可能丢数据。选哪个取决于能不能容忍脏读。",

    "在东京的第三天，我们去了一个很小的旧书店。老板不太会说英语，"
    "用手比划了半天，最后从柜台下面拿出一本 1970 年代的摄影集。"
    "书页已经泛黄，边角有点卷。我翻了几页，全是他年轻时候拍的车站，"
    "黑白的，人很少。",

    "The scoring function normalizes each dimension before computing the weighted sum, which "
    "prevents features with larger numeric ranges from dominating. Without normalization, a "
    "feature measured in milliseconds would swamp one measured in seconds, regardless of how "
    "much weight the latter was assigned in the model specification.",

    "内存对齐影响结构体大小。在 64 位系统上，char 后跟 int 会插入 3 字节填充，"
    "因为 int 需要 4 字节对齐。把成员按对齐要求从大到小排列通常能减小体积。"
    "但如果用 packed 属性强行取消填充，访问未对齐的内存可能触发异常或在某些架构上变慢。",

    "第一乐章以弦乐的持续低音开始，主题在第十二小节由大提琴引入。"
    "发展部把动机拆成三度音程反复模进，和声逐渐远离主调。"
    "再现部并没有回到最初的配器，而是让木管接过旋律，弦乐退到伴奏位置。",

    "Kubernetes 的调度器分两个阶段：过滤和打分。过滤阶段剔除不满足资源请求、"
    "亲和性约束和污点的节点；打分阶段按资源平衡、拓扑分布、镜像本地性等策略给候选节点排序。"
    "最终选择得分最高的节点，若有多个并列则随机挑一个。",

    "实验记录：样本在 25 摄氏度下静置 30 分钟后开始出现沉淀。对照组没有变化。"
    "把温度升到 40 度，沉淀重新溶解，说明是可逆的物理过程而不是化学分解。"
    "重复三次，现象一致。下一步要测不同 pH 下的行为。",

    "他给我讲了一个故事：有个猎人进山，遇到一只受伤的鹿。他把鹿背回小屋，"
    "喂了几天草料，鹿伤好了，却不肯走，每天早上都在门口等。"
    "后来猎人搬家，没带上鹿。第二年他回去看，屋子塌了，门口还长着一片被啃过的草。",

    "向量检索的召回率和延迟是一对矛盾。HNSW 通过分层图在近邻搜索里取得较好的折中，"
    "每次查询从顶层入口开始贪心下降，在每层保留一个候选集。"
    "ef_search 参数控制候选集大小，调大召回率上升，延迟也上升。",

    "这周把前端重构了一部分。原来的组件有三层嵌套的 props 传递，改成了 context。"
    "顺手把几个重复的请求逻辑抽成了 hook，减少了大概两百行代码。"
    "测试覆盖率没变，因为本来也没写什么测试。下周得补。",

    "The proof proceeds by induction on the length of the string. The base case is trivial. "
    "For the inductive step, assume the claim holds for all strings shorter than n, and consider "
    "a string of length n. Splitting at the last character and applying the inductive hypothesis "
    "to the prefix yields the desired bound.",

    "粥要熬得好，米得先泡半小时。水开下米，大火滚五分钟，然后转小火，盖子留条缝。"
    "中间搅两三次，别让它糊底。四十分钟后米粒开花，汤变稠，这时候加点盐就行。"
    "想喝甜的，关火前放冰糖，别早放，早放会发酸。",

    "防火墙的规则是按顺序匹配的，第一条命中即生效。因此把拒绝规则放在允许规则之后，"
    "会导致允许规则被绕过。默认策略应该设为拒绝，只显式放开需要的端口。"
    "另外，状态检测防火墙会自动放行已建立连接的返回包，不需要单独写规则。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 条语料")
    ap.add_argument("--out", default=str(TABLE_PATH))
    args = ap.parse_args()

    texts = CORPUS[: args.limit] if args.limit else CORPUS
    layers = args.layers if args.layers else list(range(32))

    print(f"[1/3] 加载模型 {MODEL_DIR}", flush=True)
    t0 = time.time()
    model, tok = load()
    print(f"      完成 {time.time() - t0:.1f}s  dtype={next(model.parameters()).dtype} "
          f"device={next(model.parameters()).device}", flush=True)

    st = QwenSteer(model, tok)
    print(f"[2/3] 量尺：{len(texts)} 条语料 × 最多 {args.max_len} token，"
          f"层 {layers[0]}..{layers[-1]}（共 {len(layers)} 层）", flush=True)
    t0 = time.time()
    stats = st.measure(texts, max_len=args.max_len, layers=layers, progress=True)
    dt = time.time() - t0
    print(f"      完成 {dt:.1f}s，共 {sum(s['n'] for s in stats.values())} 个残差向量样本", flush=True)

    # 存语料，便于复现
    CALIB_PATH.write_text(json.dumps({"max_len": args.max_len, "layers": layers,
                                      "texts": texts}, ensure_ascii=False, indent=1))

    print("[3/3] 计算偏置表")
    table = {"model": str(MODEL_DIR).replace(str(Path.home()), "~"),
             "targets": [], "stats": stats, "delta": {}}
    for l in sorted(stats):
        s = stats[l]
        print(f"  L{l:02d}  median={s['median']:8.2f}  mean={s['mean']:8.2f}  "
              f"p10={s['p10']:7.2f}  p90={s['p90']:8.2f}  n={s['n']}", flush=True)

    for l, n, sign in TARGETS:
        col = st.column(l, n)
        cn = float(col.norm())
        s1 = sign * 1.0 * stats[l]["median"] / cn
        table["targets"].append(dict(layer=l, neuron=n, sign=sign, col_norm=cn,
                                     resid_median=stats[l]["median"], s_at_dose1=s1))
        table["delta"].setdefault(str(l), {})[str(n)] = s1
        print(f"\n  目标 L{l} N{n} sign={sign:+g}")
        print(f"    ‖W_down[:, {n}]‖ = {cn:.6f}")
        print(f"    median(‖resid_L‖) = {stats[l]['median']:.4f}")
        print(f"    dose=1 时 s = {s1:+.8f}")
        print(f"    等价残差流注入量 ‖s·col‖ = {abs(s1) * cn:.4f} "
              f"(= median 的 {abs(s1) * cn / stats[l]['median']:.3f} 倍)")

    Path(args.out).write_text(json.dumps(table, ensure_ascii=False, indent=1))
    print(f"\n表已写入 {args.out}")


if __name__ == "__main__":
    main()

"""第 23 章 PyTorch 对照：数一数 attention 做了多少次乘法，分数表占多少内存。

讲义：transformer-deep-dive/notes/lessons/23-complexity.md
运行：uv run python ch23_complexity.py

只算讲义里的例子（n = 4、8、16；n = 50；n = 1000、10000、100000），练习题里的数留给你自己算。
"""

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


def score_flops(n: int, d: int) -> int:
    """让 PyTorch 自己数 Q @ K.T 用了多少 FLOPs（一次乘 + 一次加算 2 个）。"""
    q, k = torch.randn(n, d), torch.randn(n, d)
    with FlopCounterMode(display=False) as counter:
        q @ k.T
    return counter.get_total_flops()


def rnn_steps(n: int, d: int) -> int:
    """RNN 传话：第 t 步要等第 t − 1 步的结果，只能一步一步 for 循环。返回排队了几步。"""
    cell = nn.RNNCell(d, d)
    x, h = torch.randn(n, d), torch.zeros(d)
    steps = 0
    with torch.no_grad():
        for t in range(n):
            h = cell(x[t], h)  # 依赖上一步的 h，没法同时算
            steps += 1
    return steps


def score_table_bytes(n: int, bytes_per_number: int = 2) -> int:
    """一层、一个头的 n × n 分数表，按半精度每个数 2 字节。"""
    return n * n * bytes_per_number


def human(nbytes: int) -> str:
    for unit, size in [("GB", 1e9), ("MB", 1e6), ("KB", 1e3)]:
        if nbytes >= size:
            return f"{nbytes / size:.0f} {unit}"
    return f"{nbytes} B"


def main() -> None:
    print("打分 Q @ K.T，d = 8（FlopCounterMode 数的是 FLOPs = 2 × 乘法次数）")
    for n in [4, 8, 16]:
        flops = score_flops(n, 8)
        print(f"  n = {n:>2}: 分数表 {n * n:>3} 格，FLOPs {flops:>5}，乘法 {flops // 2:>5} = n² × d")

    n, d = 50, 512
    print(f"\nn = {n}、d = {d}：attention 打分 n²d = {n * n * d:,}，RNN 每层 nd² = {n * d * d:,}")
    print(f"  比值 = n ÷ d = {n * n * d / (n * d * d):.3f}")

    print(
        "\nRNN 要排队：6 个词的 for 循环算了",
        rnn_steps(6, 8),
        "步，每步都等上一步（中间传话 5 次）；attention 一次 Q @ K.T 算完整张表",
    )

    print("\n一张分数表的内存（半精度）")
    real = torch.empty(1000, 1000, dtype=torch.float16)
    print(f"  n = 1,000 实际分配: {human(real.numel() * real.element_size())}")
    for n in [10_000, 100_000]:
        print(f"  n = {n:,}（只算不分配）: {human(score_table_bytes(n))}")


if __name__ == "__main__":
    main()

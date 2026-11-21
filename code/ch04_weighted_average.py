"""第 4 章 PyTorch 对照：按比例取信息 = 加权平均。

讲义：transformer-deep-dive/notes/lessons/04-weighted-average.md
运行：uv run python ch04_weighted_average.py
"""

import torch

WORDS = ["小明", "苹果", "小红"]
VALUES = torch.tensor([[0.0, 10.0], [0.0, 0.0], [10.0, 10.0]])  # [像女生的程度, 是人的程度]
P = torch.tensor([0.25, 0.125, 0.625])  # 第 3 章算出来的比例


def weighted_sum_by_hand(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """o = p_1 v_1 + p_2 v_2 + ... ：每行乘自己的比例，再按位置加起来。"""
    out = torch.zeros(v.shape[1])
    for pj, vj in zip(p, v):
        out += pj * vj
    return out


def main() -> None:
    for w, pj, vj in zip(WORDS, P, VALUES):
        print(f"  {w}: {float(pj)} × {vj.tolist()} = {(pj * vj).tolist()}")
    o = weighted_sum_by_hand(P, VALUES)
    print("加起来，'她' =", o.tolist())
    print("矩阵写法 p @ V =", (P @ VALUES).tolist(), "（一行乘法做完上面的循环）")
    print("广播写法 (p[:, None] * V).sum(0) =", (P[:, None] * VALUES).sum(0).tolist())


if __name__ == "__main__":
    main()

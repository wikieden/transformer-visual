"""第 6 章 PyTorch 对照：softmax = 先套 e^x，再除以总和。

讲义：transformer-deep-dive/notes/lessons/06-softmax-exp.md
运行：uv run python ch06_softmax_exp.py
"""

import torch

WORDS = ["小明", "苹果", "小红"]
VALUES = torch.tensor([[0.0, 10.0], [0.0, 0.0], [10.0, 10.0]])  # 第 4 章的信息向量


def softmax_by_hand(s: torch.Tensor) -> torch.Tensor:
    e = torch.exp(s)
    return e / e.sum()


def naive_ratio(s: torch.Tensor) -> torch.Tensor:
    return s / s.sum()


def main() -> None:
    neg = torch.tensor([2.0, -1.0, 5.0])
    print("原因 1：分数有负数", neg.tolist())
    print("  朴素版:", [round(float(v), 3) for v in naive_ratio(neg)], "← 出现负的注意力")
    print("  e^x   :", [round(float(v), 3) for v in torch.exp(neg)], " 总和", round(float(torch.exp(neg).sum()), 2))
    print("  softmax:", [round(float(v), 3) for v in softmax_by_hand(neg)])

    s = torch.tensor([2.0, 1.0, 5.0])
    print("\n原因 2：拉开差距", s.tolist())
    print("  朴素版 :", [round(float(v), 3) for v in naive_ratio(s)])
    print("  softmax:", [round(float(v), 3) for v in softmax_by_hand(s)])
    print("  和 torch.softmax 一样:", torch.allclose(softmax_by_hand(s), torch.softmax(s, dim=-1)))
    print("  '她' 朴素版 =", (naive_ratio(s) @ VALUES).tolist())
    print("  '她' softmax =", [round(float(v), 2) for v in torch.softmax(s, dim=-1) @ VALUES])

    print("\n一次算几行：每行各自 softmax（dim=-1 = 沿最后一维，也就是每一行内部）")
    rows = torch.tensor([[2.0, 1.0, 5.0], [2.0, -1.0, 5.0]])
    p = torch.softmax(rows, dim=-1)
    print("  ", [[round(float(v), 3) for v in r] for r in p], " 每行加起来:", p.sum(dim=-1).tolist())


if __name__ == "__main__":
    main()

"""第 3 章 PyTorch 对照：分数 ÷ 总分 = 比例（朴素版）。

讲义：transformer-deep-dive/notes/lessons/03-softmax-ratio.md
运行：uv run python ch03_softmax_ratio.py
"""

import torch

WORDS = ["小明", "苹果", "小红"]
SCORES = torch.tensor([2.0, 1.0, 5.0])  # 和"她"的点积分数，编的


def naive_ratio(s: torch.Tensor) -> torch.Tensor:
    """p_j = s_j / ∑ s_k。"""
    return s / s.sum()


def main() -> None:
    p = naive_ratio(SCORES)
    print("总分:", float(SCORES.sum()))
    for w, s, q in zip(WORDS, SCORES, p):
        print(f"  {w}: {float(s):.0f} ÷ 8 = {float(q):.3f}")
    print("比例加起来:", float(p.sum()))

    for red in [5.0, 9.0, 17.0]:
        q = naive_ratio(torch.tensor([2.0, 1.0, red]))
        print(f"小红 {red:.0f} 分 → 比例 {[round(float(v), 3) for v in q]}，加起来 {float(q.sum()):.1f}")


if __name__ == "__main__":
    main()

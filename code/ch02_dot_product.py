"""第 2 章 PyTorch 对照：点积 = 对位相乘，全部加起来。

讲义：transformer-deep-dive/notes/lessons/02-dot-product.md
运行：uv run python ch02_dot_product.py
"""

import torch

APPLE = torch.tensor([7.0, 5.0, 8.0])
MELON = torch.tensor([8.0, 10.0, 6.0])
LEMON = torch.tensor([1.0, 3.0, 0.0])


def dot_by_hand(a: torch.Tensor, b: torch.Tensor) -> float:
    """两步：a * b 对位相乘，.sum() 全部加起来（∑）。"""
    return float((a * b).sum())


def main() -> None:
    print("对位相乘 apple * melon =", (APPLE * MELON).tolist())
    print("加起来:", dot_by_hand(APPLE, MELON))
    print("三种写法一样:", float(torch.dot(APPLE, MELON)), float(APPLE @ MELON), dot_by_hand(APPLE, MELON))
    print("苹果·柠檬 =", float(APPLE @ LEMON))

    table = torch.stack([APPLE, MELON, LEMON])  # 3 × 3，每行一个水果
    scores = table @ APPLE  # 一次算出苹果和每个水果的点积
    for name, s in zip(["苹果", "西瓜", "柠檬"], scores):
        print(f"  苹果·{name} = {float(s):.0f}")

    sweet_lemon = LEMON.clone()
    sweet_lemon[0] = 7.0
    print("柠檬变甜（甜 1 → 7）后 苹果·柠檬 =", float(APPLE @ sweet_lemon))


if __name__ == "__main__":
    main()

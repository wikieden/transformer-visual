"""第 1 章 PyTorch 对照：向量 = 一串数字。

讲义：transformer-deep-dive/notes/lessons/01-vector.md
运行：uv run python ch01_vector.py
"""

import torch

FRUITS = {"苹果": [7.0, 5.0, 8.0], "西瓜": [8.0, 10.0, 6.0], "柠檬": [1.0, 3.0, 0.0]}
FEATURES = ["甜", "大", "红"]


def fruit(name: str) -> torch.Tensor:
    return torch.tensor(FRUITS[name])


def component(a: torch.Tensor, i: int) -> float:
    """讲义的 a_i（i 从 1 数）= 代码的 a[i - 1]（PyTorch 从 0 数）。"""
    return float(a[i - 1])


def main() -> None:
    apple = fruit("苹果")
    print("苹果:", apple.tolist(), " 维数 d =", apple.shape[0], " 形状:", tuple(apple.shape))
    for i, name in enumerate(FEATURES, start=1):
        print(f"  a_{i}（{name}）= apple[{i - 1}] = {component(apple, i)}")

    swapped = torch.tensor([8.0, 5.0, 7.0])
    print("[7, 5, 8] 和 [8, 5, 7] 一样吗:", torch.equal(apple, swapped), "（顺序不同就是两个东西）")

    banana = torch.tensor([8.0, 3.0, 0.0])
    print("香蕉 a_1, a_2, a_3 =", component(banana, 1), component(banana, 2), component(banana, 3))

    table = torch.stack([fruit(n) for n in FRUITS])
    print("三个水果摞成一张表，形状:", tuple(table.shape), "（3 行 = 3 个水果，3 列 = 3 维）")

    torch.manual_seed(0)
    word = torch.randn(512)
    print("论文里一个词: 形状", tuple(word.shape), "，画成格子:", tuple(word.view(32, 16).shape))
    apple2 = apple.clone()
    apple2[2] = 2.0
    print("只改 a_3（红）:", apple2.tolist())


if __name__ == "__main__":
    main()

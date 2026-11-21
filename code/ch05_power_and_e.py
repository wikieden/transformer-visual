"""第 5 章 PyTorch 对照：次方、e、e^x。

讲义：transformer-deep-dive/notes/lessons/05-power-and-e.md
运行：uv run python ch05_power_and_e.py
"""

import math

import torch


def powers_of_two() -> torch.Tensor:
    """2^3, 2^2, 2^1, 2^0, 2^-1, 2^-2：每往右一格除以 2。"""
    return 2.0 ** torch.arange(3, -3, -1)


def compound(n: int) -> float:
    """存 1 元、年利率 100%、一年结息 n 次：(1 + 1/n)^n。"""
    return (1 + 1 / n) ** n


def main() -> None:
    print("2 ** torch.arange(3, -3, -1) =", powers_of_two().tolist())
    print("利滚利:")
    for n in [1, 2, 4, 12, 365, 1_000_000]:
        print(f"  一年 {n:>7} 次 → {compound(n):.5f}")
    print("math.e =", math.e, "  torch.exp(1) =", float(torch.exp(torch.tensor(1.0))))
    x = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0])
    print("x       =", x.tolist())
    print("e^x     =", [round(float(v), 3) for v in torch.exp(x)])
    print("e^x 都 > 0:", bool((torch.exp(torch.linspace(-50, 50, 101)) > 0).all()))
    print("e^x 增长速度 = 自己（导数）：在 x = 1 处斜率 ≈", end=" ")
    t = torch.tensor(1.0, requires_grad=True)
    torch.exp(t).backward()
    print(round(float(t.grad), 3), "= e^1")


if __name__ == "__main__":
    main()

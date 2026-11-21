"""第 15 章 PyTorch 对照：sin、cos、弧度；(cos, sin) 当号码牌。

讲义：transformer-deep-dive/notes/lessons/15-sin-cos.md
运行：uv run python ch15_sin_cos.py
"""

import math

import torch


def turns_to_radians(turns: torch.Tensor) -> torch.Tensor:
    """转了几圈 → 沿边走了多远：1 圈 = 2π 弧度。torch.sin / cos 只认弧度。"""
    return turns * 2 * math.pi


def seat_tag(seat: int, turns_per_seat: float = 0.25) -> torch.Tensor:
    """第四步的号码牌：每个座位多转 1/4 圈，牌 = (cos, sin)。"""
    angle = turns_to_radians(torch.tensor(seat * turns_per_seat, dtype=torch.float64))
    return torch.stack([torch.cos(angle), torch.sin(angle)])


def r(t: torch.Tensor, n: int = 2) -> list[float]:
    return [round(float(x), n) + 0.0 for x in t]  # + 0.0 把 -0.0 变成 0.0


def main() -> None:
    turns = torch.tensor([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2], dtype=torch.float64)
    rad = turns_to_radians(turns)
    print("圈  :", turns.tolist())
    print("弧度:", r(rad))
    print("sin :", r(torch.sin(rad)))
    print("cos :", r(torch.cos(rad)))
    print("转一圈就重复：sin(1¼ 圈) == sin(¼ 圈):", math.isclose(math.sin(rad[5]), math.sin(rad[1])))
    x = torch.linspace(-1000, 1000, 100001)
    print("sin、cos 永远在 −1 到 1:", float(torch.sin(x).min()), float(torch.sin(x).max()))

    print("\n(cos, sin) 号码牌，狗 = [2, 0]")
    dog = torch.tensor([2.0, 0.0], dtype=torch.float64)
    for seat in [1, 2, 3, 4, 1000]:
        tag = seat_tag(seat)
        print(f"  {seat:>4} 号座: 牌 {r(tag)}  狗 + 牌 = {r(dog + tag)}")

    print("\nsin(1) =", round(math.sin(1), 2), "（沿边走 1，约", round(1 / (2 * math.pi), 2), "圈）")


if __name__ == "__main__":
    main()

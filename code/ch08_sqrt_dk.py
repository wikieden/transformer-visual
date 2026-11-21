"""第 8 章 PyTorch 对照：点积会随维度变大，除以 √d_k 拉回来。

讲义：transformer-deep-dive/notes/lessons/08-sqrt-dk.md
运行：uv run python ch08_sqrt_dk.py
"""

import math

import torch


def coin_flip_spread(d: int, trials: int = 20000, seed: int = 0) -> float:
    """抛 d 次硬币（每项 ±1）加起来，重复很多次，看总分通常离 0 多远（标准差）。"""
    g = torch.Generator().manual_seed(seed)
    flips = torch.randint(0, 2, (trials, d), generator=g).float() * 2 - 1
    return float(flips.sum(dim=1).std())


def random_dot_spread(d_k: int, trials: int = 20000, seed: int = 0) -> float:
    """q、k 每个数是均值 0、方差 1 的随机数：q·k 通常多大。"""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(trials, d_k, generator=g)
    k = torch.randn(trials, d_k, generator=g)
    return float((q * k).sum(dim=1).std())


def rounded(t: torch.Tensor, n: int = 3) -> list[float]:
    return [round(float(v), n) for v in t]


def main() -> None:
    print("√64 =", math.sqrt(64), "  torch.sqrt(64) =", float(torch.sqrt(torch.tensor(64.0))))
    print("抛硬币：总分通常离 0 多远")
    for d in [4, 64, 512]:
        print(f"  d = {d:>3}: 模拟 ≈ {coin_flip_spread(d):5.2f}   √d = {math.sqrt(d):5.2f}")
    print(f"随机 q·k（d_k = 64）通常 ≈ ±{random_dot_spread(64):.1f}，除以 8 后 ≈ ±{random_dot_spread(64) / 8:.2f}")

    s = torch.tensor([8.0, -8.0, 3.0])
    print("\n分数", s.tolist())
    print("  不除    :", rounded(torch.softmax(s, -1)))
    print("  除以 8  :", rounded(s / 8), "→", rounded(torch.softmax(s / 8, -1)))
    print("  除以 64 :", rounded(s / 64), "→", rounded(torch.softmax(s / 64, -1), 2))


if __name__ == "__main__":
    main()

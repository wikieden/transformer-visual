"""第 20 章 PyTorch 对照：FFN、ReLU、为什么要夹 ReLU、SiLU、GLU、SwiGLU。

讲义：transformer-deep-dive/notes/lessons/20-ffn.md
运行：uv run python ch20_ffn.py
"""

import torch
import torch.nn.functional as F
from torch import nn


def ffn(d_model: int = 512, d_ff: int = 2048, bias: bool = True) -> nn.Sequential:
    """原论文：max(0, x W1 + b1) W2 + b2。"""
    return nn.Sequential(nn.Linear(d_model, d_ff, bias=bias), nn.ReLU(), nn.Linear(d_ff, d_model, bias=bias))


class SwiGLU(nn.Module):
    """W2( SiLU(W1 x) ⊙ W3 x )，三个矩阵、没有 bias。"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def swiglu_d_ff(d_model: int, multiple: int = 64) -> int:
    """≈ 8/3 · d_model，取最近的 64 的倍数。"""
    return round(8 * d_model / 3 / multiple) * multiple


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def r(t: torch.Tensor, n: int = 2) -> list[float]:
    return [round(float(v), n) + 0.0 for v in t.flatten()]


def main() -> None:
    z = torch.tensor([3.0, -1.0, 0.5, -2.0])
    print("ReLU", z.tolist(), "→", torch.relu(z).tolist())

    f = ffn()
    torch.manual_seed(0)
    x = torch.randn(3, 512)  # "狗咬人" 3 个词
    with torch.no_grad():
        h = f[0](x)
        print("\n形状: 进", tuple(x.shape), "→ W1 后", tuple(h.shape), "→ 出", tuple(f(x).shape))
        print(
            "逐个位置：整句一起算 == 每个词单独算:", torch.allclose(f(x), torch.stack([f(row) for row in x]), atol=1e-6)
        )
    print("W1、W2 各", f"{512 * 2048:,}", "个数；不算 bias 一共", f"{n_params(ffn(bias=False)):,}")

    print("\n为什么要夹 ReLU")
    w1, w2 = torch.randn(4, 8), torch.randn(8, 4)
    v = torch.randn(4)
    print("  两次转换 == 一次转换 (x W1) W2 == x (W1 W2):", torch.allclose((v @ w1) @ w2, v @ (w1 @ w2), atol=1e-5))
    print("  夹 ReLU 以后还相等吗:", torch.allclose(torch.relu(v @ w1) @ w2, v @ (w1 @ w2), atol=1e-5))
    for t in [1.0, -1.0]:
        print(f"  3 × max(0, 2x)，x = {t:>4}: {3 * max(0.0, 2 * t)}")

    s = torch.tensor([4.0, 1.0, 0.0, -1.0, -4.0])
    print("\nx      :", s.tolist())
    print("sigmoid:", r(torch.sigmoid(s)))
    print("SiLU   :", r(s * torch.sigmoid(s)), " F.silu 一样:", torch.allclose(F.silu(s), s * torch.sigmoid(s)))
    print("ReLU   :", r(torch.relu(s)))

    gate, content = torch.tensor([0.98, 0.5, 0.02]), torch.tensor([10.0, 10, 10])
    print("\nGLU 闸门 ⊙ 内容:", r(gate * content, 1), "  点积是", round(float(gate @ content), 1), "（一个数）")

    d_ff = swiglu_d_ff(512)
    print(f"\nSwiGLU：8/3 × 512 ≈ {8 * 512 / 3:.1f}，取 64 的倍数 → {d_ff}")
    print(f"  参数：原版 {n_params(ffn(bias=False)):,}，SwiGLU {n_params(SwiGLU(512, d_ff)):,}（差不多）")
    print("  SwiGLU 输出形状:", tuple(SwiGLU(512, d_ff)(x).shape))


if __name__ == "__main__":
    main()

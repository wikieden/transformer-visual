"""第 10 章 PyTorch 对照：向量乘矩阵 = 和每一列各做一次点积。

讲义：transformer-deep-dive/notes/lessons/10-matrix-transform.md
运行：uv run python ch10_matrix_transform.py
"""

import torch
from torch import nn

APPLE = torch.tensor([7.0, 5.0, 8.0])  # [甜, 大, 红]
# 每一列是一组权重：第 1 列"适合榨汁"，第 2 列"好不好拿"
W = torch.tensor([[0.5, 0.1], [0.0, 0.5], [0.2, 0.0]])


def by_columns(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x 和 W 的每一列各做一次点积，排成新向量。"""
    return torch.stack([x @ w[:, j] for j in range(w.shape[1])])


def as_linear(w: torch.Tensor) -> nn.Linear:
    """nn.Linear(3, 2) 存的 weight 是 2 × 3（输出 × 输入），是讲义 W 的转置。"""
    layer = nn.Linear(w.shape[0], w.shape[1], bias=False)
    with torch.no_grad():
        layer.weight.copy_(w.T)
    return layer


def main() -> None:
    print("一列一列点积:", [round(float(v), 2) for v in by_columns(APPLE, W)])
    print(
        "x @ W       :",
        [round(float(v), 2) for v in APPLE @ W],
        "  形状",
        tuple(W.shape),
        "→",
        tuple((APPLE @ W).shape),
    )
    layer = as_linear(W)
    print(
        "nn.Linear   :",
        [round(float(v), 2) for v in layer(APPLE).detach()],
        "  它的 weight 形状",
        tuple(layer.weight.shape),
    )

    print("\n论文尺寸：同一个 x 乘三个不同的 512 × 64 矩阵")
    torch.manual_seed(0)
    x = torch.randn(512)
    w_q, w_k, w_v = (torch.randn(512, 64) for _ in range(3))
    q, k, v = x @ w_q, x @ w_k, x @ w_v
    print("  q、k、v 形状:", tuple(q.shape), tuple(k.shape), tuple(v.shape))
    print("  三个一样吗:", torch.allclose(q, k), torch.allclose(k, v))


if __name__ == "__main__":
    main()

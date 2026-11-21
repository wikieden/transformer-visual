"""第 18 章 PyTorch 对照：残差连接 x + Sublayer(x)。

讲义：transformer-deep-dive/notes/lessons/18-residual.md
运行：uv run python ch18_residual.py
"""

import torch
import torch.nn.functional as F
from torch import nn


def residual(x: torch.Tensor, sublayer) -> torch.Tensor:
    return x + sublayer(x)


def stack(x0: torch.Tensor, layers: list[nn.Module], use_residual: bool) -> torch.Tensor:
    x = x0
    for layer in layers:
        x = x + layer(x) if use_residual else layer(x)
    return x


def make_layers(n: int, d: int, seed: int = 0) -> list[nn.Module]:
    torch.manual_seed(seed)
    return [nn.Sequential(nn.Linear(d, d), nn.Tanh()) for _ in range(n)]


def main() -> None:
    bite = torch.tensor([0.0, 1.0])
    attn_out = torch.tensor([1.46, 0.27])  # 0.73 × 狗 [2, 0] + 0.27 × 咬 [0, 1]
    print("'咬' =", bite.tolist(), " attention 结果 =", [round(float(v), 2) for v in attn_out])
    print("加残差:", [round(float(v), 2) for v in bite + attn_out], "← 自己那一位 1 还在")

    print("\n残差流：x_12 = x_0 + 12 条批注")
    torch.manual_seed(0)
    x0 = torch.randn(4)
    notes = [torch.randn(4) * 0.1 for _ in range(12)]
    x = x0.clone()
    for s in notes:
        x = x + s
    print("  x_12 − x_0 == 12 条批注之和:", torch.allclose(x - x0, torch.stack(notes).sum(0)))

    print("\n'什么都不做'：子层权重全是 0（批注 = 0）")
    quiet = nn.Linear(4, 4)
    nn.init.zeros_(quiet.weight)
    nn.init.zeros_(quiet.bias)
    print("  不加残差: 输出", quiet(x0).tolist(), "← 原文没了")
    print("  加残差  : 输出 == 输入", torch.equal(residual(x0, quiet), x0))

    print("\n叠 12 个随机子层，最后还剩多少原文（和 x_0 的余弦相似度，1 = 方向完全一样）")
    layers = make_layers(12, 64)
    x0 = torch.randn(64)
    with torch.no_grad():
        for flag, name in [(False, "传话游戏（不加残差）"), (True, "共享文档（加残差）")]:
            out = stack(x0, layers, flag)
            print(f"  {name}: {float(F.cosine_similarity(out, x0, dim=0)):.2f}")


if __name__ == "__main__":
    main()

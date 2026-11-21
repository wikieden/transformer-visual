"""第 19 章 PyTorch 对照：LayerNorm、γ β、post-LN / pre-LN、RMSNorm。

讲义：transformer-deep-dive/notes/lessons/19-layernorm.md
运行：uv run python ch19_layernorm.py
"""

import torch
import torch.nn.functional as F
from torch import nn

EPS = 1e-5


def layer_norm_by_hand(x: torch.Tensor, gamma=None, beta=None, eps: float = EPS) -> torch.Tensor:
    """γ · (x − μ) / √(σ² + ε) + β，μ、σ² 只用这个词自己的数（最后一维）。"""
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, correction=0)  # 除以 d，不是 d − 1
    x_hat = (x - mu) / torch.sqrt(var + eps)
    if gamma is not None:
        x_hat = gamma * x_hat + beta
    return x_hat


def rms_norm_by_hand(x: torch.Tensor, g=None, eps: float = EPS) -> torch.Tensor:
    """x / √(mean(x²) + ε) · g：不减平均值，也没有 β。"""
    rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * (1 if g is None else g)


def post_ln(x, sublayer, ln):
    return ln(x + sublayer(x))


def pre_ln(x, sublayer, ln):
    return x + sublayer(ln(x))


def r(t: torch.Tensor, n: int = 3) -> list[float]:
    return [round(float(v), n) + 0.0 for v in t.detach().flatten()]


def main() -> None:
    a, b = torch.tensor([80.0, 90, 100]), torch.tensor([50.0, 60, 70])
    print("减平均值:", r(a - a.mean()), r(b - b.mean()))

    x = torch.tensor([80.0, 100.0])
    print(
        "\n方差 [80, 100]：torch.var 默认除以 d − 1 =",
        float(x.var()),
        "；LayerNorm 要的是除以 d =",
        float(x.var(correction=0)),
    )

    w = torch.tensor([2.0, 4.0, 6.0, 8.0])
    print("\n[2, 4, 6, 8]: μ =", float(w.mean()), " 减完", r(w - w.mean()), " σ² =", float(w.var(correction=0)))
    print("  手写  :", r(layer_norm_by_hand(w)))
    print("  F.layer_norm:", r(F.layer_norm(w, (4,), eps=EPS)))
    print("[7, 7] 有 ε 兜底:", r(layer_norm_by_hand(torch.tensor([7.0, 7.0]))), "（没有 nan）")

    x_hat = torch.tensor([-1.0, -1, 1, 1])
    gamma, beta = torch.tensor([1.0, 3, 1, 3]), torch.tensor([0.0, 0, 0.5, 0.5])
    print("\nγ x̂ + β =", r(gamma * x_hat + beta))
    ln = nn.LayerNorm(4)
    print("nn.LayerNorm(4) 一开始: γ =", ln.weight.tolist(), " β =", ln.bias.tolist(), "（先不改动）")

    sent = torch.tensor([[2.0, 4, 6, 8], [80, 90, 100, 110], [1, 1, 1, 5]])
    out = ln(sent)
    print("3 个词一起进：每个词各自整理，每行平均", r(out.mean(-1)), " 方差", r(out.var(-1, correction=0), 2))

    print(
        "\nRMSNorm [1, 7]:",
        r(rms_norm_by_hand(torch.tensor([1.0, 7.0])), 2),
        "  nn.RMSNorm:",
        r(nn.RMSNorm(2, eps=EPS)(torch.tensor([1.0, 7.0])), 2),
    )
    print("LayerNorm [1, 7]:", r(layer_norm_by_hand(torch.tensor([1.0, 7.0])), 2))

    print("\npost-LN vs pre-LN（子层 = 一个线性层）")
    torch.manual_seed(0)
    sub, ln = nn.Linear(4, 4), nn.LayerNorm(4)
    x = torch.randn(4) * 10
    with torch.no_grad():
        print("  post-LN 输出被整理过，平均 ≈", round(float(post_ln(x, sub, ln).mean()), 3))
        print(
            "  pre-LN 输出 − 输入 == 批注 Sublayer(LN(x)):",
            torch.allclose(pre_ln(x, sub, ln) - x, sub(ln(x)), atol=1e-5),
            "（主干没被整理）",
        )


if __name__ == "__main__":
    main()

"""第 17 章 PyTorch 对照：因果 mask = 右上三角填 −∞，再 softmax。

讲义：transformer-deep-dive/notes/lessons/17-causal-mask.md
运行：uv run python ch17_causal_mask.py
"""

import torch
import torch.nn.functional as F


def causal_mask(n: int) -> torch.Tensor:
    """M：能看填 0，要遮填 −∞。torch.triu(..., diagonal=1) 取对角线右上方。"""
    return torch.triu(torch.full((n, n), float("-inf")), diagonal=1)


def masked_softmax(scores: torch.Tensor) -> torch.Tensor:
    """softmax(分数 + M)：mask 加在 softmax 之前。"""
    return torch.softmax(scores + causal_mask(scores.shape[-1]), dim=-1)


def r(t: torch.Tensor, n: int = 2) -> list[float]:
    return [round(float(x), n) for x in t]


def main() -> None:
    print("能看的格子（下三角）torch.tril:\n", torch.tril(torch.ones(3, 3, dtype=torch.bool)))
    print("M =\n", causal_mask(3))

    bite = torch.tensor([2.0, 1.0, 3.0])  # "咬"那一行：看狗、看咬、看人
    print("\n'咬'那一行 [2, 1, 3]")
    print("  不遮       :", r(torch.softmax(bite, -1)), "← 67% 偷看了答案'人'")
    for v in [-1.0, -10.0, -100.0]:
        print(f"  第 3 格改成 {v:>6}: e^x = {float(torch.exp(torch.tensor(v))):.3g}")
    print("  e^(−∞)     =", float(torch.exp(torch.tensor(float("-inf")))))
    print("  加 M 后    :", r(masked_softmax(torch.tensor([[0.0, 0, 0], bite.tolist(), [0, 0, 0]]))[1]))

    print("\n完整的 masked attention 和 PyTorch 自带的 is_causal=True 一样吗:")
    torch.manual_seed(0)
    q, k, v = (torch.randn(3, 4) for _ in range(3))
    mine = masked_softmax(q @ k.T / 4**0.5) @ v
    builtin = F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True)[0]
    print("  ", torch.allclose(mine, builtin, atol=1e-6))

    print("\n小心 1：PyTorch 的布尔 attn_mask 里 True = 能看（和'遮住'正好反过来）")
    allowed = torch.tril(torch.ones(3, 3, dtype=torch.bool))
    same = F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=allowed)[0]
    print("   attn_mask=下三角 True 结果一样:", torch.allclose(same, builtin, atol=1e-6))
    print(
        "小心 2：一整行全是 −∞，softmax 得 nan（没有能看的词）:",
        torch.softmax(torch.full((3,), float("-inf")), -1).tolist(),
    )


if __name__ == "__main__":
    main()

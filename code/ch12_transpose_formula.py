"""第 12 章 PyTorch 对照：转置，以及公式 (1) 完整走一遍。

讲义：transformer-deep-dive/notes/lessons/12-transpose-formula.md
运行：uv run python ch12_transpose_formula.py
"""

import math

import torch
import torch.nn.functional as F

from ch11_matrix_qk import K, Q

V = torch.tensor([[0.0, 10.0], [10.0, 10.0], [5.0, 5.0]])  # 小明、小红、她的 v


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict[str, torch.Tensor]:
    """softmax(Q K^T / √d_k) V，每一步都留下来看。"""
    d_k = q.shape[-1]
    scores = q @ k.T
    scaled = scores / math.sqrt(d_k)
    p = torch.softmax(scaled, dim=-1)
    return {"scores": scores, "scaled": scaled, "p": p, "out": p @ v}


def r(t: torch.Tensor, n: int = 3) -> list[float]:
    return [round(float(x), n) for x in t]


def main() -> None:
    print("K 形状", tuple(K.shape), "→ K.T 形状", tuple(K.T.shape))
    print("K.T =\n", K.T, "\n（三列依次是小明、小红、她的 k）")
    try:
        Q @ K
    except RuntimeError as e:
        print("Q @ K（不转置）报错:", str(e).split("\n")[0])

    steps = attention(Q, K, V)
    print("\n'她'那一行：")
    print("  ① 打分   :", r(steps["scores"][2], 2))
    print("  ② ÷ √2   :", r(steps["scaled"][2], 2))
    print("  ③ softmax:", r(steps["p"][2]))
    print("  ④ × V    :", r(steps["out"][2], 2))
    print("整张输出（每行一个词的新向量）:\n", steps["out"])

    builtin = F.scaled_dot_product_attention(Q[None], K[None], V[None])[0]
    print("PyTorch 自带的 F.scaled_dot_product_attention 结果一样:", torch.allclose(builtin, steps["out"], atol=1e-6))


if __name__ == "__main__":
    main()

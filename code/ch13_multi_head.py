"""第 13 章 PyTorch 对照：多头 = 切成几份各算 attention，拼接后乘 W^O。

讲义：transformer-deep-dive/notes/lessons/13-multi-head.md
运行：uv run python ch13_multi_head.py
"""

import torch
import torch.nn.functional as F
from torch import nn


def split_heads(x: torch.Tensor, h: int) -> torch.Tensor:
    """n × d → h × n × (d/h)：每个头拿一段。"""
    n, d = x.shape
    return x.view(n, h, d // h).transpose(0, 1)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """h × n × d_k → n × (h·d_k)：首尾拼接（Concat）。"""
    h, n, d_k = x.shape
    return x.transpose(0, 1).reshape(n, h * d_k)


def multi_head(x: torch.Tensor, w_q, w_k, w_v, w_o, h: int) -> torch.Tensor:
    q, k, v = (split_heads(x @ w, h) for w in (w_q, w_k, w_v))
    heads = F.scaled_dot_product_attention(q, k, v)  # 每个头各自 softmax
    return merge_heads(heads) @ w_o


def param_count(d: int, h: int) -> int:
    m = nn.MultiheadAttention(d, h, bias=False)
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    one = torch.tensor([0.5, 0.5]) @ torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    print("一个头：小红 0.5、因为 0.5 →", one.tolist(), "（混在一起）")
    head1, head2 = torch.tensor([10.0, 0.0]), torch.tensor([0.0, 10.0])
    print("两个头拼接 torch.cat:", torch.cat([head1, head2]).tolist(), "  相加是", (head1 + head2).tolist())

    print("\n论文尺寸：512 个数切成 8 个头")
    torch.manual_seed(0)
    x = torch.randn(3, 512)
    print("  split_heads 形状:", tuple(split_heads(x, 8).shape), "（8 个头 × 3 个词 × 每头 512 / 8 个数）")
    print("  拼回去一样吗:", torch.equal(merge_heads(split_heads(x, 8)), x))
    ws = [torch.randn(512, 512) / 512**0.5 for _ in range(4)]
    print("  多头输出形状:", tuple(multi_head(x, *ws, h=8).shape))

    print("\n参数量和头数无关（小例子 d = 8，不算 bias）:")
    for h in [1, 2, 4, 8]:
        print(f"  h = {h}: {param_count(8, h)} 个参数")


if __name__ == "__main__":
    main()

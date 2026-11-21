"""第 21 章 PyTorch 对照：拼一个 encoder 层、一个 decoder 层，看三种 attention 的分数表形状。

讲义：transformer-deep-dive/notes/lessons/21-transformer-layer.md
运行：uv run python ch21_transformer_layer.py

用小尺寸 d_model = 8、h = 2、d_ff = 32，随机权重（没训练），只看形状和结构。
"""

import torch
from torch import nn

D, H, D_FF = 8, 2, 32


def causal(n: int) -> torch.Tensor:
    return torch.triu(torch.full((n, n), float("-inf")), diagonal=1)


class EncoderLayer(nn.Module):
    """两道工序，post-LN：z = LN(x + MHA(x))，out = LN(z + FFN(z))。"""

    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, H, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(D, D_FF), nn.ReLU(), nn.Linear(D_FF, D))
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)

    def forward(self, x):
        a, w = self.attn(x, x, x)
        z = self.ln1(x + a)
        return self.ln2(z + self.ffn(z)), w


class DecoderLayer(nn.Module):
    """三道工序：看自己写过的（遮）→ 回头查原文（Q 是译文，K、V 是 e）→ 消化。"""

    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(D, H, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(D, H, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(D, D_FF), nn.ReLU(), nn.Linear(D_FF, D))
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)

    def forward(self, y, e):
        m = y.shape[1]
        a1, w_self = self.self_attn(y, y, y, attn_mask=causal(m))
        a = self.ln1(y + a1)
        a2, w_cross = self.cross_attn(a, e, e)  # Q = a，K = V = e
        b = self.ln2(a + a2)
        return self.ln3(b + self.ffn(b)), w_self, w_cross


def count(m: nn.Module, kind: type) -> int:
    return sum(isinstance(c, kind) for c in m.modules())


def main() -> None:
    torch.manual_seed(0)
    enc, dec = EncoderLayer().eval(), DecoderLayer().eval()
    src = torch.randn(1, 3, D)  # 原文"狗咬人"：1 句 × 3 个词 × d_model
    tgt = torch.randn(1, 2, D)  # 已写的 "dog bites"

    with torch.no_grad():
        e, w_enc = enc(src)
        out, w_self, w_cross = dec(tgt, e)
    print("LayerNorm 个数：encoder 层", count(enc, nn.LayerNorm), "  decoder 层", count(dec, nn.LayerNorm))
    print(
        "形状始终不变：encoder",
        tuple(src.shape),
        "→",
        tuple(e.shape),
        "；decoder",
        tuple(tgt.shape),
        "→",
        tuple(out.shape),
    )
    print("\n三种 attention 的分数表（权重）形状 = 提问的词数 × 被问的词数")
    print("  ① encoder self   :", tuple(w_enc.shape[1:]))
    print("  ② masked self    :", tuple(w_self.shape[1:]), " 右上角权重:", float(w_self[0, 0, 1]))
    print("  ③ encoder-decoder:", tuple(w_cross.shape[1:]))

    print("\n叠 6 层，形状还是不变")
    encs = nn.ModuleList(EncoderLayer() for _ in range(6)).eval()
    with torch.no_grad():
        x = src
        for layer in encs:
            x, _ = layer(x)
    print("  ", tuple(x.shape))

    print("\n逐词生成：encoder 只跑一次，decoder 第 k 轮的'查原文'分数表是 k × 3")
    with torch.no_grad():
        e, _ = enc(src)
        for k in range(1, 5):
            y = torch.randn(1, k, D)  # 已写的 k 个词（这里用随机向量代替）
            _, _, w_cross = dec(y, e)
            print(f"  第 {k} 轮: 输入 {k} 个词，分数表 {tuple(w_cross.shape[1:])}，取最后一行去选下一个词")

    print(
        "\nPyTorch 自带的 nn.TransformerEncoderLayer 默认也是 post-LN（norm_first=False）:",
        nn.TransformerEncoderLayer(D, H, D_FF).norm_first is False,
    )


if __name__ == "__main__":
    main()

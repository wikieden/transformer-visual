"""第 14 章 PyTorch 对照：attention 不知道顺序；加号码牌以后才分得清。

讲义：transformer-deep-dive/notes/lessons/14-positional-encoding.md
运行：uv run python ch14_positional_encoding.py
"""

import torch

# 第二步的小向量：k 和 v 只看是哪个词
K = {"狗": [2.0, 0.0], "咬": [0.0, 1.0], "人": [1.0, 0.0]}
V = {"狗": [10.0, 0.0], "咬": [2.0, 2.0], "人": [0.0, 10.0]}
Q_BITE = torch.tensor([1.0, 0.0])  # "咬"的 q


def bite_output(sentence: list[str], seat_tags: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """'咬'看整句话（省略 ÷√d_k）。seat_tags 是每个座位的号码牌，None 表示不加。"""
    k = torch.tensor([K[w] for w in sentence])
    v = torch.tensor([V[w] for w in sentence])
    q = Q_BITE.clone()
    if seat_tags is not None:
        k = k + seat_tags  # x_i' = x_i + p_i（为了好算，q、k 就等于加完位置后的向量）
        q = q + seat_tags[sentence.index("咬")]
    scores = k @ q
    return scores, torch.softmax(scores, -1) @ v


def main() -> None:
    a, b = ["狗", "咬", "人"], ["人", "咬", "狗"]
    _, out_a = bite_output(a)
    _, out_b = bite_output(b)
    print("不加位置：'咬'的输出")
    print("  狗咬人:", [round(float(x), 2) for x in out_a], "  人咬狗:", [round(float(x), 2) for x in out_b])
    print("  一样吗:", torch.allclose(out_a, out_b))

    print("\n整句一起算也一样：句子换个顺序，输出只是跟着换顺序（每个词的结果不变）")
    torch.manual_seed(0)
    x = torch.randn(3, 4)
    perm = torch.tensor([2, 1, 0])
    attn = lambda t: torch.softmax(t @ t.T, -1) @ t
    print("  attn(打乱后) == 打乱(attn):", torch.allclose(attn(x[perm]), attn(x)[perm], atol=1e-6))

    tags = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])  # 1、2、3 号座的号码牌
    sa, _ = bite_output(a, tags)
    sb, _ = bite_output(b, tags)
    print("\n加号码牌 [0,1]、[0,2]、[0,3] 以后，'咬'给每个词的分数")
    print("  狗咬人（狗、咬、人）:", sa.tolist())
    print("  人咬狗（人、咬、狗）:", sb.tolist())
    print("  '狗'的分数从", float(sa[0]), "变成", float(sb[2]))

    print("\n号码牌一直往上加：狗坐 1000 号座")
    dog = torch.tensor([2.0, 0.0])
    print("  ", dog.tolist(), "+ [0, 1000] =", (dog + torch.tensor([0.0, 1000.0])).tolist(), "← 几乎只剩座位号")


if __name__ == "__main__":
    main()

"""第 9 章 PyTorch 对照：Q 拿去比对 K，带走的是 V。

讲义：transformer-deep-dive/notes/lessons/09-qkv.md
运行：uv run python ch09_qkv.py

图书馆找书：3 本书，标签 K 和内容 V 是两张不同的表（数字是编的）。
"""

import torch

BOOKS = ["家常菜谱", "世界历史", "编程入门"]
# 标签 K：[讲做菜的程度, 讲历史的程度]
K = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.2, 0.1]])
# 内容 V：[菜谱知识, 历史知识, 编程知识]
V = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
# 需求 Q："我想找讲做红烧肉的书"
Q = torch.tensor([3.0, 0.0])


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """三步：q 和每个 k 点积打分 → softmax 分比例 → 按比例取 v。"""
    scores = k @ q
    p = torch.softmax(scores, dim=-1)
    return p, p @ v


def r(t: torch.Tensor, n: int = 3) -> list[float]:
    return [round(float(x), n) for x in t]


def main() -> None:
    p, out = attend(Q, K, V)
    print("分数 K @ q :", r(K @ Q))
    print("比例      :", dict(zip(BOOKS, r(p))))
    print("带走的内容:", r(out, 2), "（主要是菜谱知识）")

    print("\n换掉内容 V、标签 K 不动：比例不变，带走的东西变了")
    v2 = V.flip(0)
    p2, out2 = attend(Q, K, v2)
    print("  比例:", r(p2), "  带走:", r(out2, 2))

    print("换掉标签 K、内容 V 不动：比例变了")
    k2 = K.flip(0)
    p3, _ = attend(Q, k2, V)
    print("  比例:", r(p3))
    print("→ 比对只看 K，取走的是 V；所以 K 和 V 要分开学。")


if __name__ == "__main__":
    main()

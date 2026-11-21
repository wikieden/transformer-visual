"""第 11 章 PyTorch 对照：大写 Q、K 排成矩阵，一次算出整张分数表。

讲义：transformer-deep-dive/notes/lessons/11-matrix-qk.md
运行：uv run python ch11_matrix_qk.py
"""

import torch

WORDS = ["小明", "小红", "她"]
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]])  # 每一行是一个词的 q
K = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])  # 每一行是一个词的 k


def score_table_by_loops(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """第 i 行第 j 列 = 第 i 个词的 q · 第 j 个词的 k，一格一格算。"""
    n = q.shape[0]
    table = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            table[i, j] = q[i] @ k[j]
    return table


def main() -> None:
    print("Q 形状:", tuple(Q.shape), "（n = 3 个词，d_k = 2）")
    loops = score_table_by_loops(Q, K)
    once = Q @ K.T
    print("一格一格算（9 次点积）:\n", loops)
    print("Q @ K.T 一次算完:\n", once)
    print("一样吗:", torch.equal(loops, once))
    print("行 = 谁去找，列 = 被找的是谁；'她'那一行:", once[2].tolist())


if __name__ == "__main__":
    main()

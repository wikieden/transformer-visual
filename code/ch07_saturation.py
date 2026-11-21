"""第 7 章 PyTorch 对照：softmax 饱和——分数太大，调了没反应。

讲义：transformer-deep-dive/notes/lessons/07-saturation.md
运行：uv run python ch07_saturation.py
"""

import torch


def nudge(scores: list[float], idx: int = 2) -> tuple[float, float]:
    """把第 idx 个分数调高 1 分，看它的比例前后各是多少。"""
    s = torch.tensor(scores)
    before = float(torch.softmax(s, -1)[idx])
    s[idx] += 1
    after = float(torch.softmax(s, -1)[idx])
    return before, after


def slope(scores: list[float], idx: int = 2) -> float:
    """ "调一点点，比例变多少"：用 autograd 算 ∂p_idx / ∂s_idx（等于 p(1 − p)）。"""
    s = torch.tensor(scores, requires_grad=True)
    torch.softmax(s, -1)[idx].backward()
    return float(s.grad[idx])


def main() -> None:
    print("[2, 1, 50] 的 softmax:", [round(float(v), 4) for v in torch.softmax(torch.tensor([2.0, 1, 50]), -1)])
    for scores in ([2.0, 1.0, 5.0], [2.0, 1.0, 50.0]):
        b, a = nudge(scores)
        print(
            f"小红 {scores[2]:.0f} → {scores[2] + 1:.0f}：{b:.3f} → {a:.3f}，变了 {a - b:.4f}；斜率 {slope(scores):.2e}"
        )
    print("斜率 ≈ 0 就是'学不动'：训练时靠这个斜率（梯度）决定往哪调，它没了，就没有学习信号。")

    print("\n代码里的另一种'放大过头'：e^x 太大，float32 装不下")
    big = torch.tensor([2.0, 1.0, 100.0])
    e = torch.exp(big)
    print("  torch.exp([2, 1, 100]) =", [round(float(v), 3) for v in e])
    print("  手写 e / e.sum()        =", (e / e.sum()).tolist(), "← inf / inf = nan")
    print(
        "  torch.softmax          =",
        [f"{float(v):.3g}" for v in torch.softmax(big, -1)],
        "← 内部先减掉最大值，不会溢出",
    )
    shifted = big - big.max()
    print(
        "  减最大值后再算         =",
        [f"{float(v):.3g}" for v in torch.softmax(shifted, -1)],
        "（结果不变：分子分母同除 e^max）",
    )


if __name__ == "__main__":
    main()

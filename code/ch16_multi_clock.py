"""第 16 章 PyTorch 对照：256 个快慢不同的轮子 = 论文的正弦位置编码。

讲义：transformer-deep-dive/notes/lessons/16-multi-clock.md
运行：uv run python ch16_multi_clock.py
"""

import math

import torch


def sinusoidal_pe(n_pos: int, d_model: int) -> torch.Tensor:
    """PE[pos, 2i] = sin(pos / 10000^(2i/d))，PE[pos, 2i+1] = cos(同一个数)。返回 n_pos × d_model。"""
    pos = torch.arange(n_pos, dtype=torch.float64)[:, None]  # n_pos × 1
    i = torch.arange(d_model // 2, dtype=torch.float64)[None, :]  # 1 × 256
    step = 1.0 / 10000 ** (2 * i / d_model)  # 每个轮子一步多长（弧度）
    angle = pos * step  # n_pos × 256：广播，一次算出所有座位、所有轮子
    pe = torch.zeros(n_pos, d_model, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(angle)  # 偶数位放 sin
    pe[:, 1::2] = torch.cos(angle)  # 奇数位放 cos
    return pe


def two_wheels(seat: int) -> torch.Tensor:
    """第一步：快轮每座 1/4 圈、慢轮每座 1/8 圈，牌 = (快 cos, 快 sin, 慢 cos, 慢 sin)。"""
    a, b = seat * 0.25 * 2 * math.pi, seat * 0.125 * 2 * math.pi
    return torch.tensor([math.cos(a), math.sin(a), math.cos(b), math.sin(b)])


def r(t: torch.Tensor, n: int = 2) -> list[float]:
    return [round(float(x), n) + 0.0 for x in t]


def main() -> None:
    print("两个轮子：1 号座", r(two_wheels(1)), " 5 号座", r(two_wheels(5)))
    print("  快轮那两位一样，慢轮那两位不同 → 4 个数合起来不撞牌")

    pe = sinusoidal_pe(100, 512)
    print("\n论文的号码牌: 形状", tuple(pe.shape), "（100 个座位 × 512 个数 = 256 个轮子 × (sin, cos)）")
    for i in [0, 128, 255]:
        step = 1 / 10000 ** (2 * i / 512)
        print(f"  第 {i:>3} 个轮子: 一步 {step:.6f} 弧度，转一圈约 {2 * math.pi / step:,.2f} 个座位")

    print("\n近处看快轮、远处看慢轮（(sin, cos)）")
    for seat in [1, 2, 45]:
        print(f"  {seat:>2} 号座: 第 0 个轮子 {r(pe[seat, 0:2])}   第 128 个轮子 {r(pe[seat, 256:258])}")

    print("\n往后挪 2 个座位，第 0 个轮子多转的角度和起点无关")
    for start in [3, 10]:
        print(f"  {start} 号 → {start + 2} 号：多转 {(start + 2) - start} 弧度")
    print("所有数都在 −1 到 1:", float(pe.min()) >= -1 and float(pe.max()) <= 1)


if __name__ == "__main__":
    main()

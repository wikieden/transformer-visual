"""第 24 章 PyTorch 对照：让 PyTorch 帮你数参数；论文的 warmup 学习率。

讲义：transformer-deep-dive/notes/lessons/24-wrap-up.md
运行：uv run python ch24_wrap_up.py

只算讲义里的玩具模型（d_model = 4、h = 2、d_ff = 16、V = 8、N = 2）和 step = 1000、4000、100000。
题 1、题 2 做完以后，可以调用 count_params(...) 和 lrate(...) 自己验证答案。
"""

import torch
from torch import nn


class Toy(nn.Module):
    """只放"要学的矩阵"：attention 的 W^Q W^K W^V W^O、FFN 的 W1 W2、一张共享 embedding 表。不要 bias 和 LayerNorm。"""

    def __init__(self, d: int, h: int, d_ff: int, vocab: int, n_layers: int):
        super().__init__()

        def attn():
            return nn.MultiheadAttention(d, h, bias=False)

        def ffn():
            return nn.Sequential(nn.Linear(d, d_ff, bias=False), nn.Linear(d_ff, d, bias=False))

        self.enc = nn.ModuleList(nn.ModuleDict({"attn": attn(), "ffn": ffn()}) for _ in range(n_layers))
        self.dec = nn.ModuleList(
            nn.ModuleDict({"self_attn": attn(), "cross_attn": attn(), "ffn": ffn()}) for _ in range(n_layers)
        )
        self.emb = nn.Embedding(vocab, d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.lm_head.weight = self.emb.weight  # 三处共用一张表（第 22 章）
        self.register_buffer("pe", torch.zeros(64, d))  # 位置编码是固定的 sin / cos：buffer，不是参数


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())  # 共用的参数只会被列出一次


def count_params(d: int, h: int, d_ff: int, vocab: int, n_layers: int) -> dict[str, int]:
    toy = Toy(d, h, d_ff, vocab, n_layers)
    return {
        "attention 子层": n_params(toy.enc[0]["attn"]),
        "FFN 子层": n_params(toy.enc[0]["ffn"]),
        "encoder 层": n_params(toy.enc[0]),
        "decoder 层": n_params(toy.dec[0]),
        "embedding 表": n_params(toy.emb),
        "整个模型": n_params(toy),
    }


def lrate(step: int, d_model: int = 512, warmup: int = 4000) -> float:
    """Eq. 3：d^-0.5 · min(step^-0.5, step · warmup^-1.5)。step 从 1 开始（0 的 −0.5 次方是无穷大）。"""
    step = max(step, 1)
    return d_model**-0.5 * min(step**-0.5, step * warmup**-1.5)


def scheduler_demo(d_model: int = 512, warmup: int = 4000, steps: int = 6000) -> list[float]:
    """用 PyTorch 的 LambdaLR 实现同一条曲线：基础 lr 设 1，每步乘 lrate(step)。"""
    w = nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([w], lr=1.0, betas=(0.9, 0.98), eps=1e-9)  # 论文 §5.3 L365
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lrate(s + 1, d_model, warmup))
    lrs = []
    for _ in range(steps):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return lrs


def main() -> None:
    print("玩具模型（d_model = 4、h = 2、d_ff = 16、V = 8、N = 2），PyTorch 自己数:")
    for name, k in count_params(4, 2, 16, 8, 2).items():
        print(f"  {name:<12} {k}")
    toy = Toy(4, 2, 16, 8, 2)
    print("embedding 表和输出层是同一个参数:", toy.lm_head.weight is toy.emb.weight, "→ 只算一次")
    print("位置编码在 buffers 里、不在 parameters 里:", "pe" in dict(toy.named_buffers()))

    print("\nwarmup 学习率（d_model = 512，warmup_steps = 4000）")
    for s in [1000, 4000, 100000]:
        print(f"  step {s:>6}: {lrate(s):.6f}")
    lrs = scheduler_demo()
    peak = max(range(len(lrs)), key=lrs.__getitem__) + 1
    print(f"LambdaLR 跑 6000 步：最高点在第 {peak} 步，学习率 {max(lrs):.6f}")


if __name__ == "__main__":
    main()

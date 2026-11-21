"""第 22 章 PyTorch 对照：输入 embedding 与输出 softmax。

讲义：transformer-deep-dive/notes/lessons/22-embedding-softmax.md
运行：uv run python ch22_embedding_softmax.py
核对：uv run pytest tests/test_ch22.py

表里的数和讲义一样，都是编的：8 个词，每个词 4 个数（论文 V ≈ 37000、d_model = 512）。
"""

import math

import torch
from torch import nn

WORDS = ["<开始>", "狗", "咬", "人", "<结束>", "dog", "bites", "man"]
D_MODEL = 4

# 表 E：每个编号一行（讲义第一步前 5 行，第三步后 3 行）
E_ROWS = [
    [0.1, 0.0, -0.2, 0.3],  # 0 <开始>
    [0.9, -0.3, 0.4, 0.1],  # 1 狗
    [-0.5, 0.8, 0.2, -0.1],  # 2 咬
    [0.7, -0.2, 0.5, 0.0],  # 3 人
    [0.0, 0.2, -0.4, 0.6],  # 4 <结束>
    [0.8, -0.4, 0.3, 0.2],  # 5 dog
    [-0.3, 0.7, -0.2, 0.5],  # 6 bites
    [0.6, -0.1, 0.6, -0.1],  # 7 man
]

# 第二步：decoder 第 2 轮（输入 "<开始> dog"）最后一行的输出，编的
H = [-1.0, 2.0, -1.0, 2.0]


def make_embedding() -> nn.Embedding:
    """第一步：表 E = nn.Embedding(V, d_model)，权重换成讲义里那张表。"""
    emb = nn.Embedding(len(WORDS), D_MODEL)
    with torch.no_grad():
        emb.weight.copy_(torch.tensor(E_ROWS))
    return emb


def tokenize(sentence: str) -> torch.Tensor:
    """① 词 → 编号。真正的 tokenizer（BPE）在 CS336 a1 写，这里按空格切、查名单。"""
    return torch.tensor([WORDS.index(w) for w in sentence.split()])


def lookup(emb: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
    """② 编号 → 向量：x = E[i]，就是取第 i 行。"""
    return emb(ids)


def lookup_by_one_hot(emb: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
    """附录"数学原理"：查表 = one-hot 乘矩阵。结果和 emb(ids) 一样，只是白做了很多乘 0。"""
    one_hot = nn.functional.one_hot(ids, num_classes=len(WORDS)).float()  # 几个词 × V
    return one_hot @ emb.weight  # (几个词 × V) @ (V × d) = 几个词 × d


def tied_lm_head(emb: nn.Embedding) -> nn.Linear:
    """第三步：输出层和 embedding 共用同一张表。

    nn.Linear(d, V) 的 weight 形状是 V × d（PyTorch 存的是"输出 × 输入"），算的是 h @ weight.T。
    它和 embedding 表 E 的形状正好一样，所以直接让两者指向同一个参数，不用转置。
    """
    lm_head = nn.Linear(D_MODEL, len(WORDS), bias=False)
    lm_head.weight = emb.weight
    return lm_head


@torch.no_grad()  # 生成时只用不学，不需要记梯度
def next_token(lm_head: nn.Linear, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """第二步：logits = h W_out，p = softmax(logits)，greedy 挑最大的。"""
    logits = lm_head(h)
    p = torch.softmax(logits, dim=-1)
    return logits, p, int(p.argmax())


def sinusoidal_pe(pos: int, d: int) -> torch.Tensor:
    """第 16 章的位置编码：第 2i 位放 sin，第 2i+1 位放 cos。"""
    pe = torch.zeros(d)
    for i in range(d // 2):
        angle = pos / 10000 ** (2 * i / d)
        pe[2 * i] = math.sin(angle)
        pe[2 * i + 1] = math.cos(angle)
    return pe


def embed_input(emb: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
    """第四步：x = sqrt(d_model) · E[i] + PE_pos，这才是进第 1 层的东西。"""
    pe = torch.stack([sinusoidal_pe(pos, D_MODEL) for pos in range(len(ids))])
    return math.sqrt(D_MODEL) * emb(ids) + pe


def show(t: torch.Tensor, digits: int = 3) -> str:
    return str([round(float(v), digits) for v in t.detach().flatten()])


def main() -> None:
    torch.set_printoptions(precision=3, sci_mode=False)
    emb = make_embedding()

    print("== 第一步：查表 ==")
    ids = tokenize("狗 咬 人")
    x = lookup(emb, ids)
    print("编号:", ids.tolist())
    print("形状:", tuple(x.shape), "  （行数 = 句子里几个词，列数 = d_model）")
    for w, row in zip(["狗", "咬", "人"], x):
        print(f"  {w}: {show(row, 1)}")
    print("one-hot 乘矩阵结果一样:", torch.allclose(x, lookup_by_one_hot(emb, ids)))
    dog, bite, man = emb.weight.detach()[1:4]
    print(f"狗·人 = {float(dog @ man):.2f}   狗·咬 = {float(dog @ bite):.2f}")
    print("表 E 会被训练调整:", emb.weight.requires_grad)

    print("\n== 第二、三步：输出打分 + softmax（共用表 E）==")
    lm_head = tied_lm_head(emb)
    print("输出层和 embedding 是同一个参数:", lm_head.weight is emb.weight)
    logits, p, best = next_token(lm_head, torch.tensor(H))
    for w, s, q in zip(WORDS, logits, p):
        print(f"  {w:<6} logit {float(s):5.2f}   概率 {float(q):.3f}")
    print("概率加起来:", round(float(p.sum()), 6), "  greedy 选:", WORDS[best])

    print("\n== 第四步：乘 sqrt(d_model) 再加位置编码 ==")
    print("√d_model =", math.sqrt(D_MODEL))
    print("狗 ×√d:", show(math.sqrt(D_MODEL) * emb.weight[1], 1))
    print("座位 0 的位置编码:", show(sinusoidal_pe(0, D_MODEL), 1))
    print("进第 1 层的 3 行:")
    for w, row in zip(["狗", "咬", "人"], embed_input(emb, ids)):
        print(f"  {w}: {show(row)}")

    print("\n== 第四步补充：为什么要乘（d_model = 512 时的音量）==")
    torch.manual_seed(0)
    d = 512
    e_row = torch.randn(d) / math.sqrt(d)  # 共用的表一般用很小的随机数初始化
    print(f"表 E 一行的数，平均大小 ≈ {float(e_row.abs().mean()):.3f}")
    print(f"乘 √512 ≈ {math.sqrt(d):.1f} 以后 ≈ {float((math.sqrt(d) * e_row).abs().mean()):.2f}")
    pe = torch.stack([sinusoidal_pe(pos, d) for pos in range(50)])
    print(f"位置编码的数，平均大小 ≈ {float(pe.abs().mean()):.2f}")


if __name__ == "__main__":
    main()

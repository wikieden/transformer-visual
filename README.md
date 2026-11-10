# 从零读懂 Transformer

**在线阅读：<https://wikieden.github.io/transformer-visual/>** · [讲义目录](https://wikieden.github.io/transformer-visual/lessons/index.html) · [全部动画](https://wikieden.github.io/transformer-visual/animations/index.html) · [复习卡片](https://wikieden.github.io/transformer-visual/review.html)

> 在线阅读：<https://wikieden.github.io/transformer-visual/> · 续篇：《[超越原版 Transformer](https://wikieden.github.io/beyond-transformer/)》
> 24 章 · 24 个动画（GIF + 可交互页面）· 69 张复习卡片 · 覆盖论文《Attention Is All You Need》第 3 节全部内容

## 这本书讲什么

从"向量是一串数字"讲起，一步步走到 2017 年论文《Attention Is All You Need》（简称 AIAYN）第 3 节的完整 Transformer：词怎么变成向量、attention 怎么打分和取信息、为什么除以 $\sqrt{d_k}$、多头、位置编码、因果 mask、残差、LayerNorm、FFN，最后拼成 encoder–decoder 全图，并讲清 embedding 与输出 softmax、计算复杂度、参数怎么数、学习率怎么排。

- **适合谁**：想真正看懂 Transformer 公式、但数学只记得初中那点的人。需要的数学（次方与 e、sin 和 cos、平均值与方差、矩阵乘法）都在用到的那一章现场补。
- **读完能做什么**：对照原文逐段读懂 AIAYN §3（第 24 章附了行号地图），能自己数出模型参数，接着去写 CS336 作业 1 的 Transformer。
- **怎么读**：按章顺序读。每章先看讲解和动画，再自己做练习，最后用复习卡片隔几天考自己。

## 讲解的特点

- **一次一小步**：每个概念按"白话类比 → 公式（每个符号说清楚怎么读）→ 小数字亲手算一遍"讲，算例都能在纸上验证。
- **每章一个动画**：关键原理做成可交互页面，▶ 播放、→ 单步、↺ 从头开始，有滑块的可以拖动参数看结果怎么变；GIF 版直接嵌在正文里。
- **对照原文**：讲到论文的地方都标了本地文本的行号（如 L132–L133），引用原句，方便回查；每章末尾的延伸阅读精确到论文章节和行号。
- **保留真实的学习过程**：练习与作答记录里是学习时的原话、对错和点评；答错、卡住的地方都变成了复习卡片。
- **术语不设门槛**：缩写第一次出现就给英文全称并逐词解释（如 FFN = Feed-Forward Network：前馈 / 网络），全书缩写见 [缩写与术语表](https://wikieden.github.io/transformer-visual/lessons/appendix-glossary.html)。
- **每章收尾有口诀**：几句话记住一章，全部口诀汇总在本页末尾。

## 示例动画

<table>
<tr>
<td width="33%" valign="top"><img src="https://wikieden.github.io/transformer-visual/animations/09-qkv.gif" width="100%" alt="第 9 章 · Q、K、V"><br><b>第 9 章 · Q、K、V</b><br>图书馆找书：Q 去问、K 答像不像、V 交出内容<br><a href="https://wikieden.github.io/transformer-visual/lessons/09-qkv.html">讲义</a> · <a href="https://wikieden.github.io/transformer-visual/animations/09-qkv.html">▶ 交互</a></td>
<td width="33%" valign="top"><img src="https://wikieden.github.io/transformer-visual/animations/12-transpose-formula.gif" width="100%" alt="第 12 章 · 公式 1 走一遍"><br><b>第 12 章 · 公式 1 走一遍</b><br>转置、打分、缩放、softmax、取信息，四步连起来<br><a href="https://wikieden.github.io/transformer-visual/lessons/12-transpose-formula.html">讲义</a> · <a href="https://wikieden.github.io/transformer-visual/animations/12-transpose-formula.html">▶ 交互</a></td>
<td width="33%" valign="top"><img src="https://wikieden.github.io/transformer-visual/animations/13-multi-head.gif" width="100%" alt="第 13 章 · 多头注意力"><br><b>第 13 章 · 多头注意力</b><br>切成几个头各看各的，再拼回来<br><a href="https://wikieden.github.io/transformer-visual/lessons/13-multi-head.html">讲义</a> · <a href="https://wikieden.github.io/transformer-visual/animations/13-multi-head.html">▶ 交互</a></td>
</tr>
<tr>
<td width="33%" valign="top"><img src="https://wikieden.github.io/transformer-visual/animations/16-multi-clock.gif" width="100%" alt="第 16 章 · 多档时钟"><br><b>第 16 章 · 多档时钟</b><br>快慢轮子组合成位置编码，拖滑块看转角<br><a href="https://wikieden.github.io/transformer-visual/lessons/16-multi-clock.html">讲义</a> · <a href="https://wikieden.github.io/transformer-visual/animations/16-multi-clock.html">▶ 交互</a></td>
<td width="33%" valign="top"><img src="https://wikieden.github.io/transformer-visual/animations/17-causal-mask.gif" width="100%" alt="第 17 章 · 因果 mask"><br><b>第 17 章 · 因果 mask</b><br>右上角填 −∞，权重正好变成 0<br><a href="https://wikieden.github.io/transformer-visual/lessons/17-causal-mask.html">讲义</a> · <a href="https://wikieden.github.io/transformer-visual/animations/17-causal-mask.html">▶ 交互</a></td>
<td width="33%" valign="top"><img src="https://wikieden.github.io/transformer-visual/animations/21-transformer-layer.gif" width="100%" alt="第 21 章 · 拼装全图"><br><b>第 21 章 · 拼装全图</b><br>encoder、decoder、三张分数表、一轮写一个词<br><a href="https://wikieden.github.io/transformer-visual/lessons/21-transformer-layer.html">讲义</a> · <a href="https://wikieden.github.io/transformer-visual/animations/21-transformer-layer.html">▶ 交互</a></td>
</tr>
</table>

全部动画见 [动画目录](https://wikieden.github.io/transformer-visual/animations/index.html)。

## 目录

每章都配一个动画：**GIF** 在讲义正文里直接播放，**▶ 交互**是可以点按钮、拖滑块的页面（本地看需要起服务，见 [动画目录](https://wikieden.github.io/transformer-visual/animations/index.html)"跑起来看"；在线版直接点开）。

| 章 | 主题 | 一句话精要 | 动画 | 卡片 | 状态 |
|---|---|---|---|---|---|
| [1](https://wikieden.github.io/transformer-visual/lessons/01-vector.html) | 向量：用一串数字描述一个东西 | 一个东西 = 一串数字，每个位置的含义固定 | [GIF](https://wikieden.github.io/transformer-visual/animations/01-vector.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/01-vector.html) | R01 | 完成 |
| [2](https://wikieden.github.io/transformer-visual/lessons/02-dot-product.html) | 点积：对位相乘再相加 | 对位相乘再相加，越大越"像" | [GIF](https://wikieden.github.io/transformer-visual/animations/02-dot-product.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/02-dot-product.html) | R02、R03 | 完成 |
| [3](https://wikieden.github.io/transformer-visual/lessons/03-softmax-ratio.html) | softmax（上）：分数变比例；和 LayerNorm 的区别 | 把一组分数变成加起来为 1 的比例 | [GIF](https://wikieden.github.io/transformer-visual/animations/03-softmax-ratio.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/03-softmax-ratio.html) | R06、R07 | 完成 |
| [4](https://wikieden.github.io/transformer-visual/lessons/04-weighted-average.html) | 按比例取信息：加权平均；attention 四步 | 按比例把每个词的信息加起来，attention 四步走完 | [GIF](https://wikieden.github.io/transformer-visual/animations/04-weighted-average.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/04-weighted-average.html) | R09、R10 | 完成 |
| [5](https://wikieden.github.io/transformer-visual/lessons/05-power-and-e.html) | 次方与 e | $e^{x}$ 永远是正数，x 大一点结果就大很多 | [GIF](https://wikieden.github.io/transformer-visual/animations/05-power-and-e.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/05-power-and-e.html) | R04、R05 | 完成 |
| [6](https://wikieden.github.io/transformer-visual/lessons/06-softmax-exp.html) | softmax（下）：为什么套 $e^{x}$ | 套上 $e^{x}$：负数也能分比例，差距被拉开 | [GIF](https://wikieden.github.io/transformer-visual/animations/06-softmax-exp.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/06-softmax-exp.html) | R08、R11 | 完成 |
| [7](https://wikieden.github.io/transformer-visual/lessons/07-saturation.html) | softmax 饱和；饱和 vs 过拟合 | 分数差太大时 softmax 变成"赢者全拿" | [GIF](https://wikieden.github.io/transformer-visual/animations/07-saturation.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/07-saturation.html) | R12、R13 | 完成 |
| [8](https://wikieden.github.io/transformer-visual/lessons/08-sqrt-dk.html) | 为什么除以 $\sqrt{d_k}$ | 点积随维度变大，除以 $\sqrt{d_k}$ 把音量调回去 | [GIF](https://wikieden.github.io/transformer-visual/animations/08-sqrt-dk.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/08-sqrt-dk.html) | R14、R15 | 完成 |
| [9](https://wikieden.github.io/transformer-visual/lessons/09-qkv.html) | Q、K、V 分别是什么；为什么 Q、K 要分开 | Q 去问，K 答"像不像"，V 是真正交出去的内容 | [GIF](https://wikieden.github.io/transformer-visual/animations/09-qkv.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/09-qkv.html) | R16 | 完成 |
| [10](https://wikieden.github.io/transformer-visual/lessons/10-matrix-transform.html) | 转换与矩阵：Q、K、V 怎么从词向量变出来 | 向量乘矩阵 = 和每一列各做一次点积 | [GIF](https://wikieden.github.io/transformer-visual/animations/10-matrix-transform.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/10-matrix-transform.html) | R17、R18 | 完成 |
| [11](https://wikieden.github.io/transformer-visual/lessons/11-matrix-qk.html) | 大写的 Q、K、V；$QK^\top$ 是一张分数表 | 所有 q 排成 Q，$QK^\top$ 是"谁和谁像"的分数表 | [GIF](https://wikieden.github.io/transformer-visual/animations/11-matrix-qk.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/11-matrix-qk.html) | R19、R20 | 完成 |
| [12](https://wikieden.github.io/transformer-visual/lessons/12-transpose-formula.html) | 转置 $^\top$ 与公式 1 完整走一遍 | K 转置后行变列，公式 1 四步完整走一遍 | [GIF](https://wikieden.github.io/transformer-visual/animations/12-transpose-formula.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/12-transpose-formula.html) | R22、R23 | 完成 |
| [13](https://wikieden.github.io/transformer-visual/lessons/13-multi-head.html) | 多头注意力 multi-head | 切成 8 个头各看各的，拼回来再乘 $W^O$ | [GIF](https://wikieden.github.io/transformer-visual/animations/13-multi-head.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/13-multi-head.html) | R24、R25 | 完成 |
| [14](https://wikieden.github.io/transformer-visual/lessons/14-positional-encoding.html) | 位置编码（上）：attention 不知道顺序；号码牌 $x_i'=x_i+p_i$ 及其毛病 | attention 不知道顺序，要给每个座位发号码牌 | [GIF](https://wikieden.github.io/transformer-visual/animations/14-positional-encoding.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/14-positional-encoding.html) | R26、R27、R28 | 完成 |
| [15](https://wikieden.github.io/transformer-visual/lessons/15-sin-cos.html) | sin 和 cos：圆上转圈（摩天轮）；周期；弧度 | 圆上转圈：值在 −1 到 1 之间，转一圈 2π 重复 | [GIF](https://wikieden.github.io/transformer-visual/animations/15-sin-cos.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/15-sin-cos.html) | R29、R30、R31、R32 | 完成 |
| [16](https://wikieden.github.io/transformer-visual/lessons/16-multi-clock.html) | 位置编码（下）：多档时钟；论文正弦公式 | 快慢不同的轮子组合：挪几个座位，每个轮子转固定角度 | [GIF](https://wikieden.github.io/transformer-visual/animations/16-multi-clock.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/16-multi-clock.html) | R33、R34、R35、R36、R37、R38、R39、R40、R41 | 完成 |
| [17](https://wikieden.github.io/transformer-visual/lessons/17-causal-mask.html) | 因果 mask：不许偷看后面的词 | 分数表右上角填 −∞，生成时不许偷看后面的词 | [GIF](https://wikieden.github.io/transformer-visual/animations/17-causal-mask.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/17-causal-mask.html) | R42、R43、R44、R45、R46 | 完成 |
| [18](https://wikieden.github.io/transformer-visual/lessons/18-residual.html) | 残差连接：输出 = 输入 + 子层(输入) | 子层只写批注：输出 = x + 子层(x)，原词一路保留 | [GIF](https://wikieden.github.io/transformer-visual/animations/18-residual.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/18-residual.html) | R47、R48、R49 | 完成 |
| [19](https://wikieden.github.io/transformer-visual/lessons/19-layernorm.html) | LayerNorm：平均值、方差，把每个词的数字调到统一范围；pre-LN、RMSNorm | 减平均、除标准差，再乘 γ 加 β；pre-LN、RMSNorm | [GIF](https://wikieden.github.io/transformer-visual/animations/19-layernorm.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/19-layernorm.html) | R50–R56 | 完成 |
| [20](https://wikieden.github.io/transformer-visual/lessons/20-ffn.html) | 前馈网络 FFN：每个词回到自己座位上消化；ReLU、SiLU、SwiGLU | 先摊开 4 倍、过 ReLU、再收回；SwiGLU 加闸门 | [GIF](https://wikieden.github.io/transformer-visual/animations/20-ffn.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/20-ffn.html) | R57–R64 | 完成 |
| [21](https://wikieden.github.io/transformer-visual/lessons/21-transformer-layer.html) | 拼装：一个 Transformer 层；encoder / decoder 全图；三种 attention | encoder 两道工序、decoder 三道；分数表行看谁提问、列看问了谁 | [GIF](https://wikieden.github.io/transformer-visual/animations/21-transformer-layer.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/21-transformer-layer.html) | R65–R69 | 完成 |
| [22](https://wikieden.github.io/transformer-visual/lessons/22-embedding-softmax.html) | 输入 embedding 与输出 softmax：词怎么变成向量，向量怎么变回词 | 一张表正着查取向量，反着查给每个词打分 | [GIF](https://wikieden.github.io/transformer-visual/animations/22-embedding-softmax.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/22-embedding-softmax.html) | — | 正文完成，练习待做 |
| [23](https://wikieden.github.io/transformer-visual/lessons/23-complexity.html) | 复杂度：为什么 attention 是 n²，长文本为什么难 | 每个词看每个词：n²·d；换来不用排队、一步直达 | [GIF](https://wikieden.github.io/transformer-visual/animations/23-complexity.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/23-complexity.html) | — | 正文完成，练习待做 |
| [24](https://wikieden.github.io/transformer-visual/lessons/24-wrap-up.html) | 收尾：数参数、warmup 学习率、对照原文通读 §3 | attention 4d²、FFN 2·d·d_ff；学习率先升后降；§3 通读地图 | [GIF](https://wikieden.github.io/transformer-visual/animations/24-wrap-up.gif) · [▶ 交互](https://wikieden.github.io/transformer-visual/animations/24-wrap-up.html) | — | 正文完成，练习待做 |

---

本仓库由私有学习仓库里的脚本 `tools/publish-transformer/build.py` 自动生成，内容摘自讲义目录，不要直接修改这里的文件。

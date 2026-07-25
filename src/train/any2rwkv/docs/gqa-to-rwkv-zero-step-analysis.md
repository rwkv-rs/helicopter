# 从 GQA 到 RWKV7：从有界 Softmax Hazard 到可观测状态

参考：[将 Softmax Attention 线性化为 Gated DeltaNet](https://spaces.ac.cn/archives/11823)

将一个已经训练好的 GQA 层迁移为 RWKV7，关键不在于逐个复制 Q、K、V 权重，而在于先找出 Softmax Attention 真正递推的量，再把它投影到 RWKV7 的有限状态中。

这里把“zero-step”定义为第一次 optimizer step 之前的初始化：精确回放 source、解析构造 oracle、闭式求解状态和参数，最后得到逐层蒸馏的起点。初始化越接近 source，逐层蒸馏需要修正的距离就越短。

## 1. Softmax 本来就有递推式

考虑一个 GQA group。它共享

$$k_i,v_i\in\mathbb R^d,$$

并有若干个 Query heads。对任意查询 \(q\)，定义长度为 \(t\) 的 prefix attention：

$$o_t(q)=\frac{\sum_{i\le t}\exp(q^\top k_i/\sqrt d)v_i}{\sum_{i\le t}\exp(q^\top k_i/\sqrt d)}.$$

再定义第 \(t\) 个 token 在当前 prefix 中的 hazard：

$$p_t(q)=\frac{\exp(q^\top k_t/\sqrt d)}{\sum_{i\le t}\exp(q^\top k_i/\sqrt d)}.$$

直接整理分子、分母可得

$$o_t(q)=(1-p_t(q))o_{t-1}(q)+p_t(q)v_t.\tag{1}$$

式 \((1)\) 是精确恒等式。它已经具有 Gated DeltaNet 的含义：\(p_t(q)\) 同时决定保留多少旧输出、写入多少新 Value。

实际序列中每个位置有不同查询。对未来位置 \(\tau\) 的真实查询 \(q_{\tau,h}\)，离线回放

$$o_1(q_{\tau,h}),o_2(q_{\tau,h}),\ldots,o_\tau(q_{\tau,h})$$

即可得到 Softmax oracle trajectory。同一 GQA group 的 Query heads 共用 Key、Value 和状态目标，只使用不同的 future-query 探针。

## 2. 应在线性化 logit 后保留 sigmoid

直接把 \(p_t(q)\) 展开成无界仿射函数，会丢掉 \(p_t(q)\in[0,1]\) 这一最重要的性质。更自然的变量是 hazard logit。对 \(t\ge2\)，有

$$
\operatorname{logit}p_t(q)
=
\frac{q^\top k_t}{\sqrt d}
-
L_{t-1}(q),
\qquad
L_{t-1}(q)
=
\log\sum_{i<t}\exp(q^\top k_i/\sqrt d).
\tag{2}
$$

对 Query head \(h\)，先在 calibration queries 上求中心 \(\mu_h\)。定义

$$
\pi_{t-1,i}(\mu_h)
=
\frac{\exp(\mu_h^\top k_i/\sqrt d)}
{\sum_{j<t}\exp(\mu_h^\top k_j/\sqrt d)},
\qquad
\bar k_{t-1}(\mu_h)
=
\sum_{i<t}\pi_{t-1,i}(\mu_h)k_i.
$$

在 \(\mu_h\) 处对 \(L_{t-1}\) 做一阶展开：

$$
L_{t-1}(q)
\approx
L_{t-1}(\mu_h)
+
\frac{\bar k_{t-1}(\mu_h)^\top(q-\mu_h)}{\sqrt d}.
$$

于是得到有界 hazard：

$$
\widehat p_t(q)
=
\sigma\left(
c_{t,h}
+
\ell_{t,h}^\top(q-\mu_h)
\right),
\tag{3}
$$

其中

$$
c_{t,h}
=
\frac{\mu_h^\top k_t}{\sqrt d}
-
L_{t-1}(\mu_h),
\qquad
\ell_{t,h}
=
\frac{k_t-\bar k_{t-1}(\mu_h)}{\sqrt d}.
\tag{4}
$$

式 \((3)\) 在展开中心处与真实 hazard 的数值和一阶导数都相同，同时始终落在 \([0,1]\)。当 \(\mu_h=0\) 时，它退化为零点 Taylor 的有界版本；使用真实 Query 均值则进一步适配已经训练好的 head。

把 \(\widehat p_t(q)\) 代回式 \((1)\)，可直接生成 bounded oracle trajectory：

$$
\widehat o_t(q)
=
(1-\widehat p_t(q))\widehat o_{t-1}(q)
+
\widehat p_t(q)v_t.
\tag{5}
$$

这个轨迹保留 sigmoid，不急于把二阶乘积压成某个人工闭包。它既是 zero-step 状态投影的目标，也是后续逐层蒸馏的辅助 teacher。

## 3. 两个 `128×128` 状态有两套独立 read

Qwen3.5-2B 的 GQA head dimension 是 256，target RWKV7 head dimension 是 128。一个 source Query head 对应两个 native states：

$$
S_{t,h,1},S_{t,h,2}\in\mathbb R^{128\times128}.
$$

这里最重要的一点是：两个 state 各自有一个 128 维 read，并不共享同一个 read。因而每个 state 都可以使用一个 DC 通道和 127 个 query features；两者合计是两个 DC 与 254 个 feature slots。

这仍不等价于任意 \(256\times256\) 算子。第一个 state 只产生前 128 个 value channels，第二个只产生后 128 个 value channels；每个 output-row block 只能读取自己的 127 维 query 子空间。正确的容量描述是“两个独立的 \(128\times256\) row-block 低秩读出”，而不是把元素数量相加后宣称无损。

先把式 \((5)\) 的 affine oracle 写成

$$
o_t(q)=b_t+M_t(q-\mu)
=
\underbrace{(b_t-M_t\mu)}_{\widetilde b_t}+M_tq.
\tag{6}
$$

把 \(M_t\) 按 output rows 分成 \(M_{t,1},M_{t,2}\)。对每个 block 分别选择

$$
P_{h,a}\in\mathbb R^{256\times127},
\qquad
P_{h,a}^\top P_{h,a}=I,
\qquad a\in\{1,2\}.
$$

对应的 read 与 state 为

$$
x_{\tau,h,a}
=
\begin{bmatrix}
P_{h,a}^\top q_{\tau,h}\\
1
\end{bmatrix},
\qquad
\widetilde S_{t,h,a}
=
\begin{bmatrix}
M_{t,h,a}P_{h,a} & \widetilde b_{t,h,a}
\end{bmatrix}.
\tag{7}
$$

DC 放在 native head 的最后一个 channel，避免参与 partial RoPE。\(P_{h,1}\) 与 \(P_{h,2}\) 分别最小化各自 row block 的 future-read observable loss：

$$
\min_{P^\top P=I}
\sum_{(t,\tau)\in\mathcal C}
\left\|
M_{t,h,a}
\left(I-PP^\top\right)
q_{\tau,h}
\right\|_2^2.
\tag{8}
$$

式 \((8)\) 直接衡量 state 丢掉的 query 分量最终能产生多少输出，不把无权 Frobenius state error 当作跨架构目标。

## 4. query 子空间还必须与 source RoPE 可交换

仅有较低的式 \((8)\) 还不够，因为 `r_proj.weight` 是与位置无关的线性权重，RoPE 在 projection 之后才执行。若目标 read 不属于 source RoPE 的可达子空间，离线 state snapshot 再好，也无法变成真实的 native read。

Qwen3.5-2B 每个 source head 是 256 维，partial RoPE 只旋转前 64 维：

$$
R_\tau
=
\operatorname{Diag}
\left(
R_\tau^{\mathrm{rot}}\in\mathbb R^{64\times64},
I_{192}
\right).
\tag{9}
$$

两个相邻的 native `128` heads 在 projection boundary 重新视为一个 `256`-dim source head，因此 read 变换 \(T\) 必须满足

$$
TR_\tau=R_\tau T,\qquad\forall\tau.
\tag{10}
$$

一个直接可实现的参数化是：

$$
P_{h,1}
=
\begin{bmatrix}
I_{64} & 0\\
0 & P_{h,1}^{\mathrm{inv}}
\end{bmatrix},
\qquad
P_{h,1}^{\mathrm{inv}}\in\mathbb R^{192\times63},
\tag{11}
$$

$$
P_{h,2}
=
\begin{bmatrix}
0\\
P_{h,2}^{\mathrm{inv}}
\end{bmatrix},
\qquad
P_{h,2}^{\mathrm{inv}}\in\mathbb R^{192\times127}.
\tag{12}
$$

第一个 state 保留全部 64 个 rotary coordinates，再从 192 个 invariant coordinates 中选择 63 维；第二个 state 的 127 个 features 全部来自 invariant subspace。第二个 output-row block 无法直接读取 rotary coordinates，这部分作为固定残差加入式 \((8)\)，而不是在报告中隐藏。

随后先对目标 post-RoPE read 施加 \(R_\tau^{-1}\)，再拟合原生无 bias 的 `r_proj.weight`：

$$
W_r^*
=
\arg\min_W
\sum_{(x,\tau)\in\mathcal C}
\left\|
Wx-R_\tau^{-1}x_{\tau}^{*}
\right\|_2^2
+
\lambda\|W\|_F^2.
\tag{13}
$$

\(\lambda\) 按 calibration feature Gram 的平均对角线缩放，使 ridge candidate 不随 token 数或激活整体缩放漂移。验证时必须重新施加 \(R_\tau\) 后再读取 state。

## 5. 把状态轨迹投影到 native recurrence

对每个 state，bounded hazard 给出请求 decay

$$
d_t^{\mathrm{req}}=1-\widehat p_t,
\tag{14}
$$

compressed slope 与 DC 组成 native key：

$$
\kappa_{t,h,a}
=
\begin{bmatrix}
P_{h,a}^{\top}\nabla_q\widehat p_t\\
\widehat p_t
\end{bmatrix}.
\tag{15}
$$

在 `k_k=1、k_a=0` 的真实 RWKV7 子空间中，单步更新为

$$
S_t
=
S_{t-1}\operatorname{Diag}(d_t)
-
(S_{t-1}n_t)(n_t\odot a_t)^\top
+
u_t\kappa_t^\top,
\qquad
n_t=\frac{\kappa_t}{\|\kappa_t\|_2}.
\tag{16}
$$

其中 \(d_t\) 先投影到 native decay 可达区间；\(a_t\in[0,1]^{128}\) 与 \(u_t\in\mathbb R^{128}\) 在 teacher-forced target state 上做有界坐标最小二乘。求得动态信号后必须从零状态完整 free-running 回放，最终只以 observable output 衡量漂移。

式 \((16)\) 是动态信号 oracle。要得到 checkpoint，还需把这些 token-level 信号投回实际参数：

$$
\min_W\|XW^\top-Y\|_F^2+\lambda\|W-W_0\|_F^2.
\tag{17}
$$

这里 \(W_0\) 不能一律取零。当 calibration token 数少于 hidden width 时，\(X\) 没有识别出的方向应保留 source-compatible 权重；但 source prior 也不能一律强加，因为 hazard slope 与 source key 已经不是同一个量。正确做法是对 `r_proj`、`k_proj`、`v_proj`、`o_proj` 分别在 calibration 内再切 fit/selection，独立选择 \(W_0=0\) 或 source-compatible \(W_0\)，然后用全部 calibration row 重解。

`w_lora` 先把目标 decay 反解为

$$
z_w=\operatorname{logit}\left(-\frac{\log d}{e^{-1/2}}\right),
\tag{18}
$$

再按真实 `down → tanh → up+bias` 参数化做 rank-64 拟合；`a_lora` 对 \(\operatorname{logit}(a)\) 做 rank-64 affine 拟合；`g_lora` 必须严格遵守 `down → sigmoid → up` 的 rank-128、无 output bias 结构。最后从零状态完整 rollout，对 recurrent output 做原生 16-group normalization，乘 gate 后再按式 \((17)\) 求 bias-free `o_proj`。每一阶段都保存增量 NMSE。开发期间可以用 calibration 内部分割选择 ridge center 和超参数；最终安装规则只能看预先冻结、此前从未参与方法选择的样本，并且必须用真实模块的 BF16 `forward_sequence` 计算完整 mixer output。

## 6. 真实单层验证

实验读取 Qwen3.5-2B 的真实 checkpoint：

- GQA：第 3 层，8 Query heads、2 KV heads、source head dimension 256；
- target：16 个 native heads，head dimension 128；
- 数据：40 条 FineWeb-Edu 文本，每条 64 tokens；
- 划分：前 8 条 calibration、随后 8 条 adaptive development、中间 16 条
  retired gap、最后 8 条 fresh frozen final；gap 包含此前已经暴露的两组样本，
  fresh final 的 sample IDs 在求解前固定，未参与 center、closure、basis、ridge
  或方法选择；
- 设备：DGX Spark / NVIDIA GB10；
- source trace 与拟合：FP32；最终 gate：BF16 module/input/kernel vectors、FP32 recurrent state 与指标归约；
- source shard SHA-256：`aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`。

adaptive development 上的分阶段结果如下：

| 阶段 | NMSE | 含义 |
| --- | ---: | --- |
| exact hazard recurrence → Softmax attention | \(4.81\times10^{-14}\) | 式 \((1)\) 的数值自检 |
| calibration-mean bounded hazard → source mixer | 0.07079 | 保留 sigmoid 的 surrogate |
| RoPE-aligned 两状态 → affine oracle | 0.02991 | 每个 state 独立 `1 DC + 127 features` |
| RoPE-aligned 两状态 → exact Softmax mixer | 0.28429 | affine 与有限状态误差合计 |
| teacher-forced native transition → materialized 两状态 | 0.00140 | 式 \((16)\) 的单步投影 |
| free-running native transition → materialized 两状态 | 0.01939 | recurrence 漂移 |
| free-running native transition → exact Softmax mixer | 0.28722 | 使用理想 read |
| bias-free pre-RoPE read signal | 0.15867 | 式 \((13)\) |
| native read → exact Softmax mixer | 0.33012 | 使用 materialized state |
| native transition + native read → exact Softmax mixer | 0.33911 | 动态 oracle 加可实现 read |
| fitted `r_proj` signal | 0.02750 | source-centered ridge 胜出 |
| fitted `k_proj` signal | 0.16196 | zero-centered ridge 胜出 |
| fitted `v_proj` signal | 0.12503 | source-centered ridge 胜出 |
| fitted `w_lora` decay logit | 0.15114 | rank-64 `tanh` 参数化 |
| fitted `a_lora` erase logit | 0.10599 | rank-64 affine 参数化 |
| fitted bias-free `g_lora` gate | 0.22783 | rank-128 source-row `sigmoid` feature 参数化 |
| fitted signals free-running → dynamic oracle | 0.35963 | 权重投影导致的递推误差 |
| all fitted signals → exact Softmax mixer | **0.66079** | FP32 signal-level emulation，用于候选筛选 |

所有候选 tensor 随后完整写入第 3 层
`ProjectionBoundaryRWKV7Attention`。非首层 `v_lora` 使用 source value
projection 的确定性 row basis，up projection 为零，bias 为
`logit(BF16 eps²)`；真实 `v_first` 路径输入 layer-0 GDN canonical
`β·value` trace，`forward_sequence` 从零 recurrent state 运行。最后 8 条
全新 frozen final 的结果为：

| frozen-final BF16 gate | NMSE | cosine |
| --- | ---: | ---: |
| fitted native module → exact Qwen mixer | **0.69412** | 0.55308 |
| frozen mapped control → exact Qwen mixer | 3.07023 | 0.29192 |

候选在未见 sample IDs 上把 NMSE 相对 frozen mapped control 降低约
77.4%，满足“有限且严格优于冻结 baseline”的安装条件。候选 BF16
materialization SHA-256 为
`c91dc35d09c96d67d3a870505de3d8c233da78a5ce111fd448087f0ecf872266`，
control SHA-256 为
`2b68e0c0f85a1065cb2ad7d0279ec826f792276b04858bd4b7cb93ec397709f3`。
安装接口要求 26 个参数逐项完整覆盖、shape 匹配且 BF16 cast 后 finite，任何一项
不满足都在修改模块前拒绝，因此 evidence 绑定的是实际执行的 BF16 tensor，而非
拟合阶段的 FP32 临时矩阵。

把同一候选的 `v_first` 改为零时，输出相对 canonical `β·value` 路径的 NMSE
仅为 `1.54×10^-9`、relative L2 为 `3.93×10^-5`，说明近零 shortcut 确实把
跨层 value 影响压到了 BF16 zero-step 的数值噪声量级，而非靠全零输入人为获得
主要收益。

对这组 fresh final，再冻结 calibration/development 得到的 group center、
rank-one closure、RoPE-aligned basis 和全部权重，只逐组替换一个 component；
每次都从零状态重新执行同一个 BF16 native kernel：

| 单组 counterfactual | NMSE | 相对完整候选 \(\Delta\)NMSE |
| --- | ---: | ---: |
| 完整 fitted candidate | 0.69412 | 0 |
| exact dynamic-oracle `r/k/v` | 0.66344 | -0.03068 |
| exact dynamic-oracle `w/a` | 0.69117 | -0.00295 |
| exact source gate | 0.66326 | -0.03086 |
| source `o_proj` 单独替换 | 2.54864 | +1.85452 |

`r/k/v` 与 gate 各自只能收回约 0.031 NMSE，`w/a` 单独只能收回约
0.003；三者都不是当前 0.694 NMSE 的单一主因。source `o_proj` 单独替换反而
大幅退化，说明 fitted `o_proj` 已经在补偿 native state、GroupNorm 与 gate
形成的特征坐标，不能把 source readout 脱离前级表示直接称作“更精确”的替代。
因此剩余误差是多个边界共同适配的问题，逐层蒸馏需要联合训练完整 mixer，不能只
修一个 projection。

同一 run 还把 output loss 与 \(10^{-6}\) 倍 final-state loss 一起反传：
head-size 128 原生 CUDA kernel 的 BF16 vector、FP32 state backward 为
`accepted`，input gradient 有限且非零，26 个参数均得到有限 gradient，其中
25 个非零；`v_lora` down 在 zero-up 初值处梯度为零正是链式法则预期，不是
断图。counterfactual signal replay 与安装模块输出的 NMSE 为精确 0，证明上述
差值来自互斥 component 替换，而不是另一套 recurrence。

`0.69412` 仍然离“无需蒸馏复现 source mixer”很远；它证明正确构造可以显著改善
zero-step，并排除了某个单独 fitted component 主导全部误差的简单解释，但没有
消除架构差。

完整原始结果见
[`evidence/qwen35-2b-gqa-gdn-zero-step-probe.json`](evidence/qwen35-2b-gqa-gdn-zero-step-probe.json)，
SHA-256 为
`99d7f1ed740632b0b268f203eae0d1635e8355ffd2931b957056f03e380c818b`；
可复现实验入口为
[`../scripts/probe_gqa_gdn_zero_step.py`](../scripts/probe_gqa_gdn_zero_step.py)。

## 7. 完整迁移顺序

最终流程可以压缩为八步：

1. 精确回放 post-RoPE Q/K、V、source gate 与 mixer output。
2. 按式 \((2)\) 至式 \((5)\) 构造 calibration-centered bounded hazard oracle。
3. 将 affine oracle 改写为 raw-query read 与 DC bias，避免把固定 center 塞进位置相关权重。
4. 按式 \((10)\) 至式 \((12)\) 求两个独立、RoPE 可达的 observable query bases。
5. 按式 \((14)\) 至式 \((16)\) 投影 native decay、erase、write，并从零状态 free-running 回放。
6. 对目标 read/key 先做 inverse RoPE；对每个 bias-free projection 在 calibration 内独立选择 zero-centered 或 source-centered scale-relative ridge。
7. 按真实 nonlinear/low-rank contract 拟合 `w_lora/a_lora/g_lora`，再在 free-running recurrent output 上联合选择 norm、gate 与 bias-free `o_proj`。
8. 把 tensor 写入真实模块，以 BF16 `forward_sequence` 在预先冻结、未参与任何方法选择的样本上计算完整 mixer NMSE；严格优于冻结 baseline 才安装，再从该 checkpoint 开始逐层蒸馏。

这条路线把 hazard 近似、两个 state 的可观测容量、RoPE 可达性、native recurrence 和真实权重投影分开计量；最终又统一回到完整 mixer output。因此，zero-step 优化不会被某个更漂亮但不可安装的内部代理指标带偏。

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

## 3. 两个 `128×128` 状态应直接拟合可观测轨迹

Qwen3.5-2B 的 GQA head dimension 是 256，而 target RWKV7 head dimension 是 128。每个 source Query head 分到两个 target states，状态总容量为

$$2\times128\times128=32768.$$

它小于任意 \(256\times256\) 算子的 65536 个自由度，因此两个状态不负责保存任意完整矩阵；它们只保存真实 future-query 分布能够读到的部分。

先为 Query head \(h\) 选择 127 维读出子空间 \(R_h\)，再显式保留一个常数通道：

$$
x_{\tau,h}
=
\begin{bmatrix}
1\\
R_h(q_{\tau,h}-\mu_h)
\end{bmatrix}
\in\mathbb R^{128}.
\tag{6}
$$

常数通道保存 Value 的 DC 模式。其余 127 个方向由 calibration queries 的 PCA 给出解析初值，再用 source gate 与 `o_proj` 诱导的度量做广义特征分解。

对 value space 选择一个固定正交基 \(U_h\)，并把 \(U_ho_t(q)\) 分成两个 128 维部分 \(y_{t,h,1},y_{t,h,2}\)。每个 prefix 的 oracle state 直接由加权 ridge 得到：

$$
S_{t,h,a}^{*}
=
\arg\min_{S\in\mathbb R^{128\times128}}
\sum_{\tau\ge t}
w_{t,\tau,h}
\left\|
C_{\tau,h,a}
\left(
y_{t,h,a}(q_{\tau,h})-Sx_{\tau,h}
\right)
\right\|_2^2
+
\lambda\|S\|_F^2.
\tag{7}
$$

这里 \(a\in\{1,2\}\)，\(C_{\tau,h,a}\) 合并 source gate、value basis 和 `o_proj` 的可观测度量。权重取最终输出敏感度：

$$
w_{t,\tau,h}
=
s_{t\rightarrow\tau}(q_{\tau,h})^2
\left\|
C_{\tau,h}
\left(
v_t-o_{t-1}(q_{\tau,h})
\right)
\right\|_2^2,
\qquad
s_{t\rightarrow\tau}(q)
=
\prod_{j=t+1}^{\tau}(1-p_j(q)).
\tag{8}
$$

式 \((7)\) 从 exact Softmax trajectory 或式 \((5)\) 的 bounded trajectory 直接求状态快照，不需要先构造一个完整 \(256\times256\) 算子。这样，状态预算、DC 通道和最终可见误差从一开始就在同一个目标里。

若允许每个 Query head 使用四个 `128×128` states，则矩阵部分可以按 \(2\times2\) block 完全重构；若实现 grouped multi-read state，还可以复用同一 KV group 的状态内容。标准两个-state 初始化则以式 \((7)\) 的 observable loss 为准。

## 4. 从状态快照投影到 native RWKV7

得到 \(S_{t,h,a}^{*}\) 后，再拟合 native RWKV7 单步更新：

$$
S_t
=
S_{t-1}\operatorname{Diag}(d_t)
-
(S_{t-1}n_t)(n_t\odot a_t)^\top
+
u_t\kappa_t^\top.
\tag{9}
$$

状态误差不使用无权 Frobenius norm，而使用 future reads 的 Gram：

$$
G_{t,h,a}
=
\sum_{\tau\ge t}
w_{t,\tau,h}
x_{\tau,h}x_{\tau,h}^\top.
\tag{10}
$$

令 \(\Delta S_t=S_t^*-F_{\mathrm{RWKV}}(S_{t-1}^*)\)，求解

$$
\min_{d,n,a,u,\kappa}
\operatorname{Tr}
\left(
\Delta S_tG_{t,h,a}\Delta S_t^\top
\right).
\tag{11}
$$

一个稳定的闭式顺序是：

1. bounded diagonal regression 求 \(d_t\)；
2. 对旧状态残差做加权 rank-1 SVD，求 erase direction 和 strength；
3. 对剩余残差做第二次加权 rank-1 SVD，求 write value 与 write key；
4. 按 native key normalization 重新分配尺度；
5. 运行真实 recurrent rollout，以 free-running residual 重算一次式 \((11)\)。

GQA 的共享性在这里继续保留：同组 Query heads 共用 source Key、Value 和 transition statistics，各自只保留与本 head 查询协方差对齐的两个 observable states。

## 5. 从 oracle signals 得到模型参数

每个 token 现在都有目标 read、decay、erase、write 和 gate signals。参数初始化按可逆性从内向外进行：

1. 用 ridge 拟合 read、write key 和 write value；
2. 用 reduced-rank regression 初始化各 LoRA control subspace；
3. 对有界 decay、erase 先做 inverse link，再拟合 logits；
4. 回放 native recurrence；
5. 在实际 recurrent output 上闭式重算 `g_norm`、gate 和 `o_proj`。

zero-step checkpoint 随后进入逐层蒸馏。训练时冻结其余层，使用真实 layer input，完整 free-running rollout 当前 mixer，并联合最小化：

$$
\mathcal L_{\mathrm{layer}}
=
\mathcal L_{\mathrm{mixer}}
+
\lambda_{\mathrm{block}}\mathcal L_{\mathrm{block}}
+
\lambda_{\mathrm{oracle}}\mathcal L_{\mathrm{bounded\ oracle}}.
\tag{12}
$$

主项始终是 source mixer output；bounded oracle 只约束优化方向，不替代真实 Softmax teacher。完成一层后再推进下一层，避免把前层尚未校正的输入漂移同时传给所有层。

## 6. 真实单层验证

实验读取 Qwen3.5-2B 的真实 checkpoint：

- GQA：第 3 层，8 Query heads、2 KV heads、head dimension 256；
- 数据：8 条 FineWeb-Edu 文本，每条 64 tokens；
- 划分：前 4 条 calibration，后 4 条 held-out；
- 设备：DGX Spark 的 NVIDIA GB10；
- 前向与指标：FP32；
- checkpoint shard SHA-256：`aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`。

关键 held-out 结果如下：

| 验证项 | mixer/output NMSE | 说明 |
| --- | ---: | --- |
| exact hazard recurrence → Softmax output | \(4.96\times10^{-14}\) | 式 \((1)\) 的数值自检 |
| calibration-mean logit Taylor | 0.07344 | 保留 sigmoid 的 bounded oracle |
| 两状态：1 DC + 127 query-PCA，相对完整 affine state | 0.01221 | 严格计入两个 `128×128` state 的增量损失 |
| 四状态矩阵分块，相对完整矩阵 | \(9.79\times10^{-15}\) | matrix-only exact control |

两状态实验还给出：矩阵可观测分量 NMSE 为 0.09933；加入 DC 后，相对完整 affine output 的 NMSE 为 0.01221。四状态结果说明额外损失来自两个-state 的可观测压缩，而不是分块代数本身。

FP32 与 BF16 两次独立运行的关键排序和量级一致。完整原始结果见 [`evidence/qwen35-2b-gqa-gdn-zero-step-probe.json`](evidence/qwen35-2b-gqa-gdn-zero-step-probe.json)，可复现实验入口为 [`../scripts/probe_gqa_gdn_zero_step.py`](../scripts/probe_gqa_gdn_zero_step.py)。

## 7. 完整迁移顺序

最终流程可以压缩为七步：

1. 精确回放 post-RoPE Q/K、V、source gate 与 mixer output。
2. 按式 \((2)\) 至式 \((5)\) 构造 calibration-centered bounded hazard oracle。
3. 用 exact Softmax future-query trajectories 求式 \((7)\) 的两个 observable states。
4. 用式 \((10)\)、式 \((11)\) 投影 native decay、erase 和 write。
5. 用 ridge、截断 SVD 和 inverse link 初始化全部模型参数。
6. 运行一次 native free-running rollout，闭式重算 norm、gate 与 `o_proj`。
7. 以该 checkpoint 开始逐层蒸馏，只用独立 held-out mixer NMSE 选择配置。

这条路线把三类误差分开了：Softmax hazard 近似、有限状态可观测压缩、native recurrence 参数化。每一层都有独立指标，因此既能把 zero-step NMSE 尽量压低，也能让后续逐层蒸馏只修正真正剩下的部分。

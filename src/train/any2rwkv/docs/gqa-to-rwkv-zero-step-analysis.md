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
o_t(q)=b_t+M_t(q-\mu).
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
S^{\mathrm{native}}_{t,h,a}
=
\begin{bmatrix}
M_{t,h,a}P_{h,a}
&
b_{t,h,a}-M_{t,h,a}P_{h,a}P_{h,a}^{\top}\mu_h
\end{bmatrix}.
\tag{7}
$$

式 \((7)\) 的 DC 补偿必须使用每个 head、每个 state 自己的
\(P_{h,a}P_{h,a}^{\top}\mu_h\)，不能先用共享的 \(M_t\mu\) 抵消 center。
否则只要 \(\mu_h\) 不在所选子空间内，basis 的 centered-query objective 与
raw-query materialization 就会相差一个常量项。DC 放在 native head 的最后一个
channel，避免参与 partial RoPE。\(P_{h,1}\) 与 \(P_{h,2}\) 分别最小化各自
row block 的 future-read observable loss：

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

## 6. 验证协议

正确性先由不依赖真实权重的公式回归测试固定。对显著非零的
\(\mu\)，测试直接比较压缩读出与

$$
b+MPP^\top(q-\mu)
=\left(b-MPP^\top\mu\right)+MPP^\top q,
\tag{19}
$$

从而保证每个 query head、每个 state 的 DC 都是
\(b-MPP^\top\mu\)。只验证 \(\mu=0\) 会掩盖错误，不能作为回归用例。

真实单层验证必须把方法选择与安装判定分开：

1. 所有 context 候选使用同一组 sample IDs；指标只计算共同的 supervised
   suffix，不把不同 burn-in 或不同样本的 NMSE 横向比较。
2. 完整的最长因果 trace 已经包含所有短前缀，因此正式路径确定性地选择最长
   context；短 context 和 calibration 内部 development 只报告诊断误差，不参与
   安装判定。
3. installation split 与 epoch/convergence validation split 的 sample IDs
   完全互斥。前者只决定 zero-step candidate 是否替换 frozen baseline，后者只在
   安装完成后评价逐层训练。
4. rolling cache transition 始终推进完整 validation cache；不能把 installation
   或 epoch selection 的子集继续传播到下一层。

候选必须一次性覆盖真实模块要求的全部参数，先在副本上完成 shape、dtype、
finite 与 BF16 cast 检查，再用原生 `forward_sequence` 从零 recurrent state
计算完整 mixer output。只改善 hazard、state、read、gate 或 output
projection 的内部代理指标都不能触发安装；只有 installation split 上的完整
mixer normalized MSE 严格优于 frozen baseline 时才提交，否则原子回滚。

选中的 module-state SHA、fit-report SHA、split identity、row digest 与 solver
配置进入 pre-epoch cursor。写入不可变 generation 后，再从 safetensors 重新读取
并计算 state SHA；它必须与选择时的 SHA 完全一致，resume 也必须复验同一绑定。
这样可以排除“内存中选中了新候选，但 replay 恢复了旧 generation”的静默覆盖。

逐组把某个 fitted component 恢复为 mapped 值，只是
`mapped-component restoration ablation`，用于定位各组件适配关系；它不是
exact oracle，也不能被表述成精确 counterfactual。真正的 exact-component
counterfactual 必须从同一冻结 trace 生成相应 oracle signal，并重新执行同一
native kernel。

空间有界的实现按 row、query head 和 KV group 流式累计充分统计量。每个 rank
只保存自己的 teacher trace；需要全局一致的 Gram、cross-covariance、metric
sums 和 basis gradient 才进行 tensor all-reduce。非连续统计量先转为
contiguous buffer 完成 collective，再写回原 view，因而既满足 NCCL contract，
也保持原位累计语义。最长因果 trace 同时覆盖全部短前缀，所以正式求解只收集
最长 trace，短 context 只作为 prefix diagnostic。

## 7. 真实单层验证

实验读取 Qwen3.5-2B 的真实 checkpoint，在第 3 层验证
`full_attention → RWKV7`：

- source geometry：8 个 Query heads、2 个 KV heads、head dimension 256；
- target geometry：16 个 native heads、head dimension 128；
- train cache：从 2943 条 `distill_train` 中确定性选择前 64 条，transaction
  使用前 32 条，其中 16 条 calibration、16 条 adaptive development；
- validation cache：32 条独立文本，其中 15 条只用于 installation gate、15 条
  只用于后续 epoch selection，另留 2 条作为互斥边界；
- context：burn-in 128、supervised suffix 512、选择最长 640-token trace；
- runtime：8 张 NVIDIA RTX PRO 6000 Blackwell，BF16 module，
  `WKV_MODE=fp32io16`。

逐级误差如下：

| 边界 | NMSE | cosine | 含义 |
| --- | ---: | ---: | --- |
| 精确 prefix hazard oracle | $1.08\times10^{-13}$ | 1.000000 | 式 $(1)$ 的数值回放达到浮点误差 |
| 有界 hazard surrogate | 0.123891 | 0.950232 | sigmoid hazard 近似本身的误差 |
| observable state compression 对 affine oracle | 0.023265 | 0.988307 | 两个独立 `128×128` states 引入的增量压缩误差 |
| observable state compression 对 exact attention | 0.325510 | 0.821303 | hazard、affine closure 与 state compression 的累计误差 |
| native free-running transition 对 compressed output | 0.006048 | 0.997675 | recurrence 编译自身的增量误差 |
| native free-running transition 对 exact attention | 0.327557 | 0.820410 | 到 native recurrence 的累计误差 |
| 完整 native parameter projection，development | 0.505662 | 0.703271 | 再计入实际 projection、norm、gate 与 output |

最终安装判定只看冻结 validation 上真实 BF16 module：

| 指标 | frozen baseline | zero-step candidate | 相对下降 |
| --- | ---: | ---: | ---: |
| mixer normalized MSE | 4.973255 | 0.498262 | 89.98% |
| block normalized MSE | 1.018474 | 0.105706 | 89.62% |
| total normalized MSE | 2.995865 | 0.301984 | 89.92% |
| loss | 6.018663 | 0.609692 | 89.87% |
| cosine | 0.730662 | 0.942757 | — |

候选通过了“mixer normalized MSE 严格下降且 block normalized MSE
不退化”的原子安装规则。不可变 generation 的 module-state SHA-256 为
`c5ee61224e0bdec6a9d650578eda350d54d82b8e2ddf77470bcfcacfd94a01b8`，
execution report SHA-256 为
`8fecef19b1a0ee0ef245c13463b214a1f77c3bedd78c044faefe05d053447ece`，
代码提交为 `ddcc7d059b722b96a796b27bdb308e73bde85dbc`。

这组结果说明两个 `128×128` states 不是对 \(256\times256\) 算子的无损分块：
相对 affine oracle 的纯压缩增量 NMSE 为 0.023265；0.325510 则是从 bounded
hazard、affine closure 到 state compression 的累计误差，不能全部归因于状态容量。
解析初始化在这一层、这一组冻结 installation samples 上把完整 mixer 误差降低约
一个数量级，因此是显著优于 frozen mapped baseline 的候选。它能否稳定缩短逐层
蒸馏、能否跨层保持优势，或者是否可能免蒸馏，仍须由多层、完整 baseline matrix
与逐层训练曲线验证。

## 8. 完整迁移顺序

最终流程可以压缩为八步：

1. 精确回放 post-RoPE Q/K、V、source gate 与 mixer output。
2. 按式 \((2)\) 至式 \((5)\) 构造 calibration-centered bounded hazard oracle。
3. 对每个 head/state 用
   \(b-MPP^\top\mu\) 构造 raw-query read 的独立 DC，避免把共享
   \(b-M\mu\) 错当成压缩后的补偿。
4. 按式 \((10)\) 至式 \((12)\) 求两个独立、RoPE 可达的 observable query bases。
5. 按式 \((14)\) 至式 \((16)\) 投影 native decay、erase、write，并从零状态 free-running 回放。
6. 对目标 read/key 先做 inverse RoPE；对每个 bias-free projection 在 calibration 内独立选择 zero-centered 或 source-centered scale-relative ridge。
7. 按真实 nonlinear/low-rank contract 拟合 `w_lora/a_lora/g_lora`，再在 free-running recurrent output 上联合选择 norm、gate 与 bias-free `o_proj`。
8. 把 tensor 写入真实模块，以 BF16 `forward_sequence` 在预先冻结、未参与任何方法选择的样本上计算完整 mixer NMSE；严格优于冻结 baseline 才安装，再从该 checkpoint 开始逐层蒸馏。

这条路线把 hazard 近似、两个 state 的可观测容量、RoPE 可达性、native recurrence 和真实权重投影分开计量；最终又统一回到完整 mixer output。因此，zero-step 优化不会被某个更漂亮但不可安装的内部代理指标带偏。

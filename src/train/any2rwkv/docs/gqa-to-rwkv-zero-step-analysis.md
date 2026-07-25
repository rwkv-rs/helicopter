# 从 GQA 到 RWKV7：Softmax 递推的 Zero-Step 线性化

参考：[Linearizing Softmax Attention into Gated DeltaNet](https://spaces.ac.cn/archives/11823)

将一个已经训练好的 GQA 层迁移为 RWKV7，可以表述成一个很干净的问题：在给定的校准分布和固定的矩阵状态预算下，寻找最接近原 Softmax Attention 的递推算子。

这里的“最接近”专指 held-out mixer output NMSE，“zero-step”则表示整个过程只使用精确回放、加权最小二乘、矩阵分解和 inverse link，不进行反向传播、LM loss 优化或逐层蒸馏。如果初始化已经把递推算子投影得足够准确，那么转换完成后就可以直接导出模型。

下面从 Softmax 的精确递推开始。

## 1. GQA 的查询索引递推

考虑一个 GQA group。它有一组共享的 Key、Value：

\[
k_i,v_i\in\mathbb R^d
\]

以及若干个 Query heads。对其中一个 Query head 的任意查询 \(q\)，定义长度为 \(t\) 的 prefix attention：

\[
o_t(q)
=
\frac{
\sum_{i\le t}\exp(q^\top k_i)v_i
}{
\sum_{i\le t}\exp(q^\top k_i)
}.
\]

缩放因子和 RoPE 都可以预先吸收到 \(q,k_i\) 中，所以这里不再单独书写。再定义第 \(t\) 个 token 在当前 prefix 中的 Softmax 概率：

\[
p_t(q)
=
\frac{
\exp(q^\top k_t)
}{
\sum_{i\le t}\exp(q^\top k_i)
}.
\]

直接整理分子、分母，就得到精确恒等式：

\[
o_t(q)
=
(1-p_t(q))o_{t-1}(q)+p_t(q)v_t.
\tag{1}
\]

这个式子很重要。它说明 Softmax Attention 本来就可以看成一种递推：\(p_t(q)\) 同时控制旧输出的保留量和新 Value 的写入量。

实际序列中每个位置的查询都不一样，但式 \((1)\) 仍然可用。对未来位置 \(\tau\) 的查询 \(q_{\tau,h}\)，只要离线计算

\[
o_1(q_{\tau,h}),o_2(q_{\tau,h}),\ldots,o_\tau(q_{\tau,h}),
\]

就得到一条以真实 future query 为探针的 Softmax oracle trajectory。GQA 的共享性也自然保留下来：同一个 group 的所有 Query heads 共用 \(k_i,v_i\)，只是探针分布不同。

## 2. 在真实查询分布上投影 Softmax hazard

为了得到矩阵递推，需要把 \(p_t(q)\) 化成 \(q\) 的简单函数。对固定的 Query head \(h\)，最自然的一阶形式是

\[
p_t(q)\approx \theta_{t,h}+\ell_{t,h}^\top q.
\tag{2}
\]

在 \(q=0\) 处做 Taylor 展开，会得到

\[
\theta_{t,h}=\frac1t,\qquad
\ell_{t,h}=\frac{k_t-\bar k_t}{t}.
\]

这是一个很好的解析起点。模型转换还知道真实 Query 的分布，因此可以把式 \((2)\) 直接投影到校准集上。令

\[
x_{\tau,h}
=
\begin{bmatrix}
1\\q_{\tau,h}
\end{bmatrix},
\qquad
\gamma_{t,h}
=
\begin{bmatrix}
\theta_{t,h}\\\ell_{t,h}
\end{bmatrix},
\]

则加权 ridge 解为

\[
\gamma_{t,h}^*
=
\left(
\sum_{\tau\ge t}
w_{t,\tau,h}x_{\tau,h}x_{\tau,h}^\top
+\lambda I
\right)^{-1}
\left(
\sum_{\tau\ge t}
w_{t,\tau,h}x_{\tau,h}p_t(q_{\tau,h})
\right).
\tag{3}
\]

权重应该对应最终可观测的 mixer error。先定义从更新位置 \(t\) 传播到读取位置 \(\tau\) 的 survival：

\[
s_{t\rightarrow\tau}(q)
=
\prod_{j=t+1}^{\tau}(1-p_j(q)).
\]

再把 source gate 和 output projection 合成读出度量 \(C_{\tau,h}\)，便可以取

\[
w_{t,\tau,h}
=
s_{t\rightarrow\tau}(q_{\tau,h})^2
\left\|
C_{\tau,h}
\left(
v_t-o_{t-1}(q_{\tau,h})
\right)
\right\|_2^2.
\tag{4}
\]

式 \((4)\) 给传播更远、输出影响更大的误差更高权重。因此，式 \((3)\) 拟合的不是孤立的 attention probability，而是它对最终 mixer output 的贡献。

对不同 Query head 分别求式 \((3)\)，可以让同一个 KV group 的每个 Query head 都获得适合自身查询协方差的 \(\theta_{t,h},\ell_{t,h}\)。源 K/V 仍然共享，递推状态则针对读取它的查询分布优化。下文聚焦一个固定的 Query head，并省略下标 \(h\)。

## 3. 从 affine hazard 得到 RWKV7 形式

设 prefix 输出已经近似为

\[
o_{t-1}(q)\approx A_{t-1}q+b_{t-1}.
\tag{5}
\]

把式 \((2)\)、式 \((5)\) 代入式 \((1)\)，可得

\[
\begin{aligned}
o_t(q)
\approx\;&
(1-\theta_t)b_{t-1}+\theta_t v_t\\
&+
\left[
(1-\theta_t)A_{t-1}
+(v_t-b_{t-1})\ell_t^\top
\right]q\\
&-
(\ell_t^\top q)A_{t-1}q.
\end{aligned}
\tag{6}
\]

最后一项是唯一的二次项。令

\[
n_t=\frac{\ell_t}{\|\ell_t\|_2},
\]

并在校准分布上做下面的 rank-1 closure：

\[
(\ell_t^\top q)A_{t-1}q
\approx
\zeta_t\|\ell_t\|_2
(A_{t-1}n_t)n_t^\top q.
\tag{7}
\]

\(\zeta_t\) 也有闭式解。记

\[
M_{t,\tau,h}
=
A_{t-1}^\top
C_{\tau,h}^\top C_{\tau,h}
A_{t-1},
\]

则对应式 \((7)\) 的加权最小二乘解为

\[
\zeta_t^*
=
\frac{
\sum_{\tau}
w_{t,\tau,h}
(\ell_t^\top q_{\tau,h})^2
n_t^\top M_{t,\tau,h}q_{\tau,h}
}{
\sum_{\tau}
w_{t,\tau,h}
(\ell_t^\top q_{\tau,h})^2
n_t^\top M_{t,\tau,h}n_t
}.
\tag{8}
\]

于是 affine state 的递推变成

\[
b_t
=
(1-\theta_t)b_{t-1}+\theta_t v_t,
\tag{9}
\]

\[
\begin{aligned}
A_t
=\;&
(1-\theta_t)A_{t-1}\\
&-
\zeta_t\|\ell_t\|_2
(A_{t-1}n_t)n_t^\top\\
&+
(v_t-b_{t-1})\ell_t^\top.
\end{aligned}
\tag{10}
\]

式 \((10)\) 已经具有 RWKV7 的三个组成部分：

\[
\underbrace{(1-\theta_t)A_{t-1}}_{\text{decay}}
-
\underbrace{
\zeta_t\|\ell_t\|_2(A_{t-1}n_t)n_t^\top
}_{\text{rank-1 erase}}
+
\underbrace{
(v_t-b_{t-1})\ell_t^\top
}_{\text{rank-1 write}}.
\]

因此，一组自然的 oracle signals 是

\[
\begin{aligned}
d_t &= 1-\theta_t,\\
\text{erase direction} &= n_t,\\
\text{erase strength} &= \zeta_t\|\ell_t\|_2,\\
\text{write key} &= \ell_t,\\
\text{write value} &= v_t-b_{t-1}.
\end{aligned}
\tag{11}
\]

这里的 decay、erase 和 write 都来自同一个 Softmax hazard，而不是彼此独立的经验初始化。

## 4. 把 affine bias 放入矩阵状态

式 \((9)\) 中的 \(b_t\) 保存了 Value 的均值模式。为了在不改变 RWKV7 推理接口的前提下保留它，可以在真实 Query 分布上寻找一个近似常数方向：

\[
r_{0,h}
=
\arg\min_r
\sum_{\tau}
w_{\tau,h}
\left(r^\top q_{\tau,h}-1\right)^2
+\lambda\|r\|_2^2.
\]

它的闭式解为

\[
r_{0,h}
=
\left(
\sum_\tau w_{\tau,h}q_{\tau,h}q_{\tau,h}^\top+\lambda I
\right)^{-1}
\left(
\sum_\tau w_{\tau,h}q_{\tau,h}
\right).
\tag{12}
\]

于是有

\[
A_tq+b_t
\approx
\left(A_t+b_t r_{0,h}^\top\right)q.
\]

定义

\[
\widetilde A_{t,h}
=
A_{t,h}+b_{t,h}r_{0,h}^\top,
\tag{13}
\]

就把 affine oracle 重新写成了纯矩阵状态。式 \((12)\) 的 held-out 常数拟合残差可以直接衡量 DC 模式是否被充分保存；需要更高精度时，可以在后面的两个 state sketches 中显式保留一个低维常数子空间。

## 5. 用两个 `128×128` 状态表示一个 `256×256` 算子

对一个 256 维 source Query head，分配两个 128 维 target RWKV heads。两个 target states 不做坐标切片，而是共同近似同一个 observable operator。

对 \(a\in\{1,2\}\)，定义

\[
S_{t,h,a}
=
U_{h,a}\widetilde A_{t,h}T_{h,a}^\top
\in\mathbb R^{128\times128},
\tag{14}
\]

并用

\[
\widehat o_{t,h}
=
\sum_{a=1}^{2}
D_{h,a}S_{t,h,a}R_{h,a}q_{t,h}
\tag{15}
\]

恢复 256 维输出。所有投影由下面的 held-out-aligned 目标确定：

\[
\min_{U,T,D,R}
\sum_{t,h}
\left\|
C_{t,h}
\left[
\widetilde A_{t,h}q_{t,h}
-
\sum_{a=1}^{2}
D_{h,a}S_{t,h,a}R_{h,a}q_{t,h}
\right]
\right\|_2^2.
\tag{16}
\]

式 \((16)\) 可以先用 output-weighted covariance 和 Kronecker SVD 得到解析初值，再做一轮分块闭式最小二乘，依次更新 \(U,T,D,R\)。每个子问题都是 ridge solve，不需要梯度。

这种分解恰好利用了 GQA 的结构：

- 同一 KV group 的 Query heads 共享原始 \(k_t,v_t\) 和 Softmax oracle；
- 每个 Query head 拥有自己的两份 state sketches；
- 两份 sketches 分别保留该 Query head 在 gate、`o_proj` 和 query covariance 下最可观测的算子子空间；
- 两个 128 维输出在 output projection 前重新组合。

所有 covariance 和 operator factorization 都在 post-RoPE 坐标中计算。这样，位置旋转已经包含在被分解的实际查询、键和读出算子里。

## 6. 投影到 native RWKV7 recurrence

对每条 reduced-state trajectory，native RWKV7 的单步更新写成

\[
S_t
=
S_{t-1}\operatorname{Diag}(d_t)
-
(S_{t-1}n_t)(n_t\odot a_t)^\top
+
u_t\kappa_t^\top,
\tag{17}
\]

其中 native 参数化把 erase direction 和 write key 绑定到同一个 raw key \(\widetilde k_t\)：

\[
n_t
=
\operatorname{normalize}(\widetilde k_t\odot k_k),
\qquad
\kappa_t
=
\widetilde k_t\odot
\left[1+(a_t-1)\odot k_a\right].
\]

读出为

\[
y_t=S_t r_t.
\tag{18}
\]

式 \((11)\) 给出第一组解析 signals；式 \((14)\) 则给出每个 target head 的 oracle state。接着把式 \((17)\) 看成一个受约束的矩阵投影问题：

\[
\min_{d,\widetilde k,a,u}
\sum_t
\left\|
\mathcal W_t^{1/2}
\left[
S_{t}^{*}
-
S_{t-1}^{*}\operatorname{Diag}(d_t)
+
(S_{t-1}^{*}n_t)(n_t\odot a_t)^\top
-
u_t\kappa_t^\top
\right]
\right\|_F^2.
\tag{19}
\]

\(\mathcal W_t\) 由 future queries 和式 \((16)\) 的读出度量诱导。求解时依次使用：

1. bounded diagonal regression 求 \(d_t\)；
2. 加权 rank-1 projection 求 erase direction 和 strength；
3. 对剩余矩阵做加权 rank-1 SVD，求 \(u_t,\kappa_t\)；
4. 按 native 约束归一化 \(n_t\)，并把尺度重新分配到 \(a_t,u_t,\kappa_t\)。

由于每一步都是低秩分解或线性最小二乘，得到的是 zero-step recurrence projection。将这些 signals 做一次真实 recurrent rollout，再以 rollout residual 重算式 \((19)\)，可以消除 teacher-forced state 与自由递推 state 之间的一阶偏差。

## 7. 从 oracle signals 回归到模型参数

到这里，每个 token 已经有了目标 \(r_t,d_t,n_t,a_t,u_t,\kappa_t\)。最后一步是让 RWKV7 的参数化从当前 token hidden state 产生这些 signals。

线性投影使用 ridge 或 reduced-rank regression；LoRA 路径使用目标回归矩阵的截断 SVD；有界变量则先做 inverse link。例如 native decay

\[
d_t=\exp\left(-\frac12\sigma(w_t)\right)
\]

对应

\[
w_t
=
\operatorname{logit}\left(-2\log d_t\right).
\tag{20}
\]

erase gate 同理先映射到 logit 空间，再拟合 `a_lora`。归一化 key 只决定方向，因此它的幅度可以在 write key、write value 和 erase strength 之间解析地重新分配。最后，用真实 recurrent rollout 的输出闭式重算 `g_norm`、gate projection 和 `o_proj`。

完整参数回归顺序为：

1. 拟合 target query/read projections；
2. 拟合 write key、write value projections；
3. 对 decay 和 erase 做 inverse-link regression；
4. 回放 native recurrence；
5. 在实际 state output 上重算 `g_norm`、gate 和 `o_proj`；
6. 用一次 rollout residual 做闭式修正。

整个过程只处理 activation、state 和算子，不引入 next-token loss。

## 8. Zero-Step 转换算法

将上面的推导合起来，可以得到以下转换流程：

1. 收集 post-RMSNorm hidden state、post-RoPE Q/K、V、source gate 和 `o_proj` 读出。
2. 对每个 GQA group 和 future Query head 精确回放 causal Softmax，得到 \(p_t(q)\) 与 \(o_t(q)\)。
3. 用式 \((3)\)、式 \((4)\) 求分布加权的 \(\theta_t,\ell_t\)。
4. 用式 \((8)\) 求最优 rank-1 closure，并按式 \((9)\)、式 \((10)\) 生成 256 维 affine oracle state。
5. 用式 \((12)\)、式 \((13)\) 吸收 Value 的 DC 模式。
6. 用式 \((14)\) 至式 \((16)\) 为每个 Query head 求两份 `128×128` observable state sketches。
7. 用式 \((19)\) 将 reduced trajectories 投影到 native RWKV7 recurrence。
8. 用 reduced-rank ridge、截断 SVD 和 inverse link 回归全部模型参数。
9. 运行一次 native recurrent rollout，闭式修正 recurrence signals、`g_norm`、gate 和 `o_proj`。
10. 在完全独立的 held-out context 上计算 mixer output NMSE，并直接导出最优 zero-step checkpoint。

校准集只负责求投影，held-out 集只负责选择 ridge、rank、closure 和 factorization 配置。主指标始终是 native recurrent rollout 后的 mixer output NMSE。

## 9. 最小验证序列

为了知道每个数学部件实际贡献了多少，可以按下面的顺序记录 held-out NMSE：

1. 零点 Taylor hazard：\(\theta_t=1/t,\ \ell_t=(k_t-\bar k_t)/t\)；
2. 分布加权 hazard：式 \((3)\)、式 \((4)\)；
3. 最优 rank-1 closure：式 \((8)\)；
4. affine bias / DC 模式：式 \((12)\)、式 \((13)\)；
5. two-block observable factorization：式 \((14)\) 至式 \((16)\)；
6. native recurrence projection：式 \((19)\)；
7. 参数化 recurrent rollout：式 \((20)\) 及最终读出重算。

这条序列从文章的一阶线性化出发，逐步加入 GQA 的真实查询分布、RWKV7 的状态约束和 `256→2×128` 的容量分配。每一步都有独立的闭式目标和 held-out NMSE，因此最终得到的是一条可计算、可归因、无需蒸馏的 GQA → RWKV7 zero-step 迁移路径。

# 从 GDN 到 RWKV7：可观测等价类上的 Zero-Step 编译

参考：

- [将 Softmax Attention 线性化为 Gated DeltaNet](https://spaces.ac.cn/archives/11823)
- [Qwen3.5 GDN 与 RWKV7 逐项参考实现](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py)

将已经训练好的 Gated DeltaNet 迁移为 RWKV7，可以先分成两个问题：记忆递推是否能逐项嵌入，以及产生这些递推信号的有限维参数化是否能在真实数据分布上编译出来。

第一个问题有精确答案。GDN 的记忆递推本身就是 RWKV7 DPLR 递推的一个子空间。第二个问题则适合利用不改变最终输出的 gauge freedom，把动态归一化、卷积历史和归一化差异搬到更容易拟合的位置，再用加权最小二乘、低秩分解和一次闭式轨迹修正完成 zero-step 编译。

## 1. 将 GDN 递推精确嵌入 RWKV7

以 `[value, key]` 为状态坐标，GDN 的单步更新写成

$$S_t=d_tS_{t-1}-d_t\beta_t(S_{t-1}k_t)k_t^{\top}+\beta_tv_tk_t^{\top},\qquad y_t=S_t\frac{q_t}{\sqrt N}.$$

native RWKV7 写成

$$\widehat S_t=\widehat S_{t-1}\operatorname{Diag}(\delta_t)-(\widehat S_{t-1}n_t)(n_t\odot a_t)^{\top}+u_t\kappa_t^{\top},\qquad \widehat y_t=\widehat S_tr_t.$$

考虑 Qwen 的带 epsilon L2 normalization：

$$k_t=\frac{\bar k_t}{c_{k,t}},\qquad c_{k,t}=\sqrt{\|\bar k_t\|^2+\epsilon},\qquad m_{k,t}=\|k_t\|.$$

对任意正数 $$\mu_t>0$$，取

$$\delta_t=d_t\mathbf1,\qquad n_t=\frac{k_t}{m_{k,t}},\qquad a_t=d_t\beta_tm_{k,t}^{2}\mathbf1,\qquad \kappa_t=\mu_tk_t,\qquad u_t=\frac{\beta_tv_t}{\mu_t}.$$

便有

$$-(S_{t-1}n_t)(n_t\odot a_t)^{\top}=-d_t\beta_t(S_{t-1}k_t)k_t^{\top},$$

以及

$$u_t\kappa_t^{\top}=\beta_tv_tk_t^{\top}.$$

所以两边的状态递推逐项相等。这里的 $$\mu_t$$ 是 rank-one write 的尺度 gauge：它可以在 key 和 value 之间搬运任意正标量，而不改变状态更新。

一个特别有用的选择是

$$\mu_t=c_{k,t}.$$

此时

$$\kappa_t=\bar k_t,\qquad u_t=\frac{\beta_tv_t}{c_{k,t}}.$$

若取 native 静态参数

$$k_k=\mathbf 1,\qquad k_a=\mathbf 0,$$

那么 RWKV 的 raw key 可以直接拟合 GDN 卷积后的未归一化 key；RWKV 自身的 key normalization 恢复 erase direction，动态 L2 范数则被搬到 value 侧。

query 也有同样的自由度。令

$$q_t=\frac{\bar q_t}{c_{q,t}},$$

并取

$$r_t=\lambda_t\frac{q_t}{\sqrt N}.$$

这时 $$\widehat y_t=\lambda_ty_t$$。后续逐头归一化会消除正的逐头尺度，因此可以选择

$$\lambda_t=c_{q,t}\sqrt N,$$

从而得到

$$r_t=\bar q_t.$$

这样，target 可以直接拟合未归一化 query，不需要用线性投影逼近动态 L2 normalization。

这组带 key-norm 修正和任意 write gauge 的恒等式经过 FP64 动态序列验证，relative L2 为 $$2.36\times10^{-16}$$，最大绝对误差为 $$1.39\times10^{-16}$$。canonical oracle 的 32 组长度与 chunk 组合也全部通过，最大 relative L2 为 $$2.14\times10^{-16}$$。

## 2. Zero-step 应优化可观测状态算子

定义 GDN 的单步状态算子

$$A_t^G=d_t(I-\beta_tk_tk_t^{\top}),\qquad B_t^G=\beta_tv_tk_t^{\top}.$$

RWKV 对应为

$$A_t^R=\operatorname{Diag}(\delta_t)-n_t(n_t\odot a_t)^{\top},\qquad B_t^R=u_t\kappa_t^{\top}.$$

真正需要拟合的是 $$(A_t,B_t,r_t)$$ 产生的完整可观测轨迹。令状态误差为 $$E_t$$，则一阶误差满足

$$E_t=E_{t-1}A_t^G+S_{t-1}^G\Delta A_t+\Delta B_t,$$

$$\Delta y_t=E_tr_t+S_t^G\Delta r_t.$$

将归一化、gate 和 output projection 的局部 Jacobian 记为 $$C_t$$，直接求解

$$\min_{\Delta\theta}\sum_t\left\|C_t\left(E_tr_t+S_t^G\Delta r_t\right)\right\|^2+\lambda\|\Delta\theta\|^2.$$

这是基于 recurrence sufficient statistics 的线性正规方程。Gram 与 RHS 可以流式累计后闭式求解，不需要 LM loss、反向传播或优化器。

由于逐头归一化会消除正尺度，内部轨迹使用 projective loss：

$$\min_{\rho_{t,h}>0}\left(\widehat y_{t,h}-\rho_{t,h}y_{t,h}\right)^{\top}M_{t,h}\left(\widehat y_{t,h}-\rho_{t,h}y_{t,h}\right),$$

其中

$$\rho_{t,h}^{*}=\frac{y_{t,h}^{\top}M_{t,h}\widehat y_{t,h}}{y_{t,h}^{\top}M_{t,h}y_{t,h}+\varepsilon}.$$

最终模型选择仍然只看独立 held-out 文本上的 full-mixer NMSE。这样 decay、erase、write 和 read 之间能够按照最终可见误差自动补偿。

## 3. 将四阶卷积投影为最优 time-mix

GDN 的 q/k/v 信号可以写成

$$s_t=\operatorname{SiLU}\left(\sum_{j=0}^{3}D_jWx_{t-j}\right).$$

RWKV time-mix 为

$$\widehat s_t=W_R\big((1-\alpha)\odot x_t+\alpha\odot x_{t-1}\big).$$

正确的初始化顺序是：

1. 用 $$[x_t,x_{t-1}]$$ 对 gauge 调整后的 $$r_t,\kappa_t,u_t$$ 做 output-weighted Wiener regression。
2. 得到无约束的两组 lag 系数 $$B_0,B_1$$。
3. 对每个输入通道，将两组输出系数投影到共享方向，即对相应的 $$2\times d_{\mathrm{out}}$$ 矩阵做 covariance-weighted rank-1 SVD。
4. 从 rank-one 因子恢复 $$W_R,\alpha$$。
5. 交替更新 write gauge $$\mu_t$$，使 key 与 value 两侧的结构化回归总残差最小。

source 的第三、第四阶历史由此按照真实语料的时序协方差，最优地边缘化到 native one-step time-mix 中。

## 4. 解析编译 decay、erase 和 gate

GDN 的控制信号为

$$d_h(x)=\exp\left[-e^{A_{\log,h}}\operatorname{softplus}(p_h(x))\right],\qquad \beta_h(x)=\sigma(b_h(x)).$$

native RWKV7 decay link 为

$$d_h^R=\exp\left[-c_w\sigma(z_{w,h})\right],\qquad c_w=e^{-1/2}.$$

因此 inverse-link target 应写成

$$z_{w,h}^{*}=\operatorname{logit}\left(\operatorname{clip}\left(\frac{-\log d_h}{c_w},\varepsilon,1-\varepsilon\right)\right)=\operatorname{logit}\left(\operatorname{clip}\left(e^{1/2}[-\log d_h],\varepsilon,1-\varepsilon\right)\right).$$

这一区分了两层事实：DPLR 状态算子允许令 $$\delta_t=d_t\mathbf1$$ 并精确嵌入；native decay link 的可达域则是

$$d_h^R\in\left[\exp(-e^{-1/2}),1\right),$$

域外信号需要按最终可观测误差做 bounded projection。

erase gate 的 target 为

$$z_{a,h}^{*}=\operatorname{logit}\left(\operatorname{clip}\left(d_h\beta_hm_{k,h}^2,\varepsilon,1-\varepsilon\right)\right).$$

`w_lora` 的 down basis 由 source decay projection 的行空间及其 Jacobian 加权主方向构造；不同 tanh scale 用来逼近每个 head 的一维标量函数。

`a_lora` 的 basis 应包含

$$\operatorname{rowspan}(W_{\mathrm{decay}})+\operatorname{rowspan}(W_\beta)+\mathcal K_{\mathrm{norm}}.$$

这里 $$\mathcal K_{\mathrm{norm}}$$ 表示 key-norm Jacobian 的主要方向。

Qwen3.5 每层只有 $$H$$ 个 decay driver 和 $$H$$ 个 beta driver。以 $$H=16$$ 为例，主要控制子空间至多约 $$2H=32$$ 维，可以自然装入 RWKV 的 rank-64 control subspace；up projection 通过加权 reduced-rank ridge 求出。

gate 则从

$$g^G=\operatorname{SiLU}(W_zx)$$

编译成

$$g^R=U_g\,\sigma(D_gx).$$

先对 source gate 的 output-weighted Jacobian 做广义 SVD，保留最影响 mixer output 的 128 个方向作为 $$D_g$$，再闭式求解 $$U_g$$。

## 5. 联合解决 RMSNorm 到 GroupNorm

状态递推对静态 value-space 变换等变：

$$\widetilde S_t=U_hS_t,\qquad \widetilde v_t=U_hv_t,\qquad \widetilde y_t=U_hy_t.$$

令

$$e=\frac{\mathbf1}{\sqrt N}.$$

GroupNorm 会丢掉 $$e$$ 方向，因此选择 $$U_h$$，使这个方向对应 source 中最不重要的 value direction：

$$u_{\star}=\arg\min_{u^{\top}\Sigma_hu=1}u^{\top}H_hu,\qquad U_hu_{\star}=e,$$

其中 $$\Sigma_h$$ 是 readout covariance，$$H_h=\mathbb E[J_{t,h}^{\top}J_{t,h}]$$ 是最终 mixer 的可观测度量。

随后联合求解：

- value-space gauge $$U_h$$；
- `g_norm.weight/bias`；
- gate up projection；
- `o_proj`；
- `r_k`。

native bonus 为

$$b_t=\big((r_t\odot\kappa_t)^{\top}r_k\big)u_t.$$

固定 recurrence signals 后，它关于 $$r_k$$ 是线性的，可以通过 ridge 解出，用来恢复 GroupNorm 丢失方向中可由当前写入解释的部分。

## 6. 完整 zero-step 转换流程

1. 保持 GDN 的 head 数和 state head dimension。
2. 收集 source normalized layer input、raw conv q/k/v、$$d,\beta$$、gate、state read 和最终 mixer output。
3. 构造带 read/write gauge 的精确 RWKV activation oracle。
4. 做两阶结构化 Wiener/time-mix 投影。
5. 以 $$(A_t,B_t,r_t)$$ 为单位求解 projective、output-weighted trajectory projection。
6. 构造 decay、erase、gate 的低秩解析 basis。
7. 选择 value-space gauge，联合求解 GroupNorm、`r_k` 和 output projection。
8. 在完整 native recurrence rollout 上做一次闭式 Gauss–Newton correction。
9. 只用独立 held-out 文本上的 full-mixer NMSE 与 cosine 选择结果。
10. 达标后直接导出；全程没有 optimizer step、LM loss 或蒸馏。

GDN→RWKV7 比 GQA→RWKV7 更有机会直接跳过蒸馏。记忆递推的代数误差可以压到机器精度，剩余 zero-step NMSE 集中在有限维 signal compiler、native decay link、norm、gate 和 readout 投影上。预期收益最大的三个自由度依次是 read/write gauge、联合 trajectory projection，以及 value-space gauge 与 `r_k` 的联合读出。

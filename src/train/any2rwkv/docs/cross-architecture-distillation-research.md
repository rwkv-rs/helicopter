# Attention/GDN 到 recurrent mixer 的跨架构迁移研究

本文服务于 `qwen35-rwkv7-conversion` 的训练 recipe。结论先行：现有工作支持“保留外围块、先局部算子/hidden 对齐、再做端到端 logits 蒸馏”的路线，但没有工作证明 Qwen3.5 GDN 或 GQA/full-attention 可以对原生 RWKV7 做完整、任意输入上的无损静态权重转换。GDN 只有 state recurrence 的受约束子空间可能解析等价；conv、融合 projection、gate、Norm、activation、head 几何和 decay 可达域仍需拟合。

## 最相关工作

### MOHAWK：三阶段、由局部到全局

[Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models](https://arxiv.org/abs/2408.10189) 将 Transformer 与 SSM 都视为 token mixing operator，依次执行：

1. Matrix Orientation：使用同一个 teacher layer input，匹配 teacher/student mixer；
2. Hidden-State Alignment：逐 block 匹配 hidden state；
3. Weight Transfer + Knowledge Distillation：复制可共享外围权重，并对完整模型做 end-to-end prediction distillation。

Phi-Mamba 使用约 3B token，hybrid 版本约 5B token。论文的关键消融是：仅做最终 KD 明显弱于三阶段。这直接支持本项目的 `teacher prefix -> active RWKV7 -> teacher suffix`、逐层 hidden/block loss 和最后 global KL/CE，但 MOHAWK 并不是“每一步只更新一个层”；单 active layer 是本项目为了 397B 内存与归因额外增加的 invariant。

### CALD：最直接的逐层替换证据

[Joint Fine-tuning and Conversion of Pretrained Speech and Language Models towards Linear Complexity](https://arxiv.org/abs/2410.06846) 提出 Cross-Architecture Layerwise Distillation（CALD）：按层用线性复杂度模块替换 Transformer 模块，并用对应 teacher hidden state 引导；论文同时在 Pythia→Mamba language modeling 与 Wav2Vec2→Mamba2 speech 上验证。它比纯 end-to-end KD 更贴近本项目“每次只让一层反向”的约束，但仍没有覆盖 60 层 MoE、GDN 的解析 state 映射或 RWKV7。因此本项目采用 CALD 的 layerwise guiding 思路，同时额外要求 student-prefix、frozen suffix gradient bridge、逐层 optimizer 隔离和全 recurrent corrective sweep。

### Llamba：公开了可操作的 token 配比

[Llamba: Scaling Distilled Recurrent Models for Efficient Language Processing](https://arxiv.org/abs/2502.14458) 将 MOHAWK 扩展到 Llama-3.x → Mamba：

| 模型 | Matrix Orientation | Hidden Alignment | Global KD | 总 token |
|---|---:|---:|---:|---:|
| 1B | 300M | 2.7B | 5B | 8B |
| 3B | 500M | 4B | 5.5B | 10B |
| 8B | 500M | 5B | 6.5B | 12B |

其 stage 1 batch size 为 64，stage 2/3 为 128；使用 WSD schedule，warm-up 与 decay 各占 10%，最低学习率 `1e-8`。Matrix Orientation 与 Hidden Alignment 使用 packed FineWeb-Edu-4.0。这个结果说明：局部 mixer 拟合只应占较小 token 预算，主要预算应留给 student-prefix hidden alignment 和全局 KD。不能把 397B 的 layerwise token budget 简单按参数量线性放大；应先在 2B proxy 上做 token-allocation sweep。

### Attention to Mamba：阶段配比消融最贴近本项目

[Attention to Mamba: A Recipe for Cross-Architecture Distillation](https://arxiv.org/abs/2604.14191) 在 1B、10B token 上系统比较 Attention → Mamba-like conversion。其 stage 1 先学 Hedgehog feature map 逼近 softmax attention，stage 2 再训练加入 SSM/conv/gate 的完整 mixer。默认配比为 10%/90%；论文的 100%/0% 与 0%/100% 都明显更差，说明“只做局部拟合”或“跳过局部初始化直接全局训”都不稳。对本项目的直接映射是：

- GDN recurrence oracle 与 attention trace fitting 只负责可解释初值；
- 大部分预算用于 block/logit/CE 与 long-context rollout；
- gate、conv/time-mix 等外围自由度不能因为 recurrence 公式相似而冻结到底。

### LoLCATs：attention output MSE 后再用低秩全局恢复

[LoLCATs: On Low-Rank Linearizing of Large Language Models](https://arxiv.org/abs/2410.10254) 采用两步：先以 attention output MSE 做 attention transfer，再以 LoRA 恢复端到端质量。论文报告只训练约 0.2% 参数、使用约 0.4% 于既往线性化方法的训练 token，并扩展到 Llama 3.1 70B/405B。公开配置包含 Alpaca-Clean、packed `chunk_size=1024`，attention transfer 与后续 LoRA 分离。

官方实现把 attention transfer 与 LoRA 配置分开，并提供 70B/405B 的 `lolcats-scaled` 分支；示例 transfer 配置采用 attention-output MSE 高权重（示例名中的 `mse1000`）而 cross-entropy 为 0。这进一步说明局部阶段应优化算子输出而不是让语言建模 loss 主导，随后再由 global KD/CE 修复累积误差。

本项目不直接照搬 LoRA，因为目标是原生 RWKV7 full checkpoint；但它强烈支持“先让单层 mixer 在对齐输入上复现输出，再打开少量 fitted 参数做 global recovery”，以及保留 zero-step/固定 token budget baseline。

### Mamba in the Llama：复用 Q/K/V/O projection，但不声称等价

[The Mamba in the Llama: Distilling and Accelerating Hybrid Models](https://arxiv.org/abs/2408.15237) 复用 attention linear projection 权重初始化 Mamba-like block，并用约 20B token 蒸馏 Zephyr-7B/Llama-3 8B。其较强结果仍主要来自保留约四分之一 attention layer 的 hybrid；因此它能支持 `naive QKV copy` 作为 baseline，也说明 projection reuse 有利于收敛，但不能作为 full-attention/GQA → RWKV7 的语义等价证明。

官方代码的 recipe 是可选的逐层 alignment、关键的 end-to-end KL、可选 instruction tuning；逐层阶段冻结 MLP，而 end-to-end 阶段允许全部参数训练。公开复现实验口径为 8×80GB A100、约 3–4 天，训练 context 2K，并报告 NIAH 可测到蒸馏长度约 20 倍。对本项目的含义是：单层冻结外围块适合作为受控迁移阶段，但若 P1 表明 mixer-only 无法恢复，则必须把“是否允许外围 LoRA/全参 recovery”作为显式消融，不能暗中放宽 active-layer invariant。

### GQA uptraining：KV group 聚合是初始化，不是 recurrent 迁移证明

[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) 用原始预训练约 5% 的 compute 将 multi-head checkpoint uptrain 为 MQA/GQA，证明从已有 KV head 做 group-aware 初始化再训练可以接近原模型。它解决的是 attention 内部 MHA → GQA，不解决 softmax attention → recurrent state。对本项目而言，`kv_repeat` 应是保留 group 关系的 baseline，`kv_expand` 是消融；正式迁移必须同时冻结 config、projection layout、Q/K Norm、partial RoPE 和 query-head→KV-group map，并以多 context teacher trace 验证。

### ARWKV：方向高度相关，但实验披露仍不足

[ARWKV: Pretrain is not what we need](https://arxiv.org/abs/2501.15570) 报告从 Qwen 2.5 蒸馏纯 RWKV-7，并公开 early preview；ARWKV-R1-7B model card 还报告过 40M distillation token、2K context、仅 stage-2 的实验。它证明方向可行，但论文是持续更新稿，尚不足以据此冻结本项目的数据配比、逐层 schedule 或“几乎不掉点”阈值。这里应把 ARWKV 当相关先例，而不是替代本项目 canonical/mapping/oracle/baseline 证据。

## 本项目采用的 recipe

1. 先跑 zero-step matrix：random、naive QKV、GDN constrained algebraic、GQA `kv_repeat`/`kv_expand`、activation-fitted。
2. 每层先用 aligned teacher hidden 做 isolated mixer/block fitting；oracle 已通过的动态子空间冻结，外围 fitted 参数训练。
3. 逐层 `0..59` 替换；active layer 接收当前 student prefix，frozen teacher suffix 保留计算图，把 global KL/CE 梯度传回 active layer。
4. 长 context 使用 prefix burn-in，只在 supervised window 计 loss；cold/warmed 分开报告。
5. 全部 recurrent 后至少执行一轮 `59..0` corrective sweep，以完整 sweep 的 validation token KL 下降决定停止和 rollback。
6. 数据预算先在 2B proxy 上扫局部/全局比例，至少覆盖 `10/90`、`25/75`、`50/50`；禁止直接把单篇论文比例当成 Qwen3.5/RWKV7 最优值。
7. scale gate 只承认相同 tokenizer/split/seed/precision/token budget 下，正式迁移在 zero-step 和训练后均优于 random/naive baseline，并通过 P1。

## 仍需实验回答的问题

- 已验证的 Qwen3.5-2B proxy 使用 16 个 value/key head、每 head `128×128` state，而当前原生 RWKV7 target 使用 32 个 `64×64` head；flat hidden width 虽同为 2048，state/head geometry 并不同构。因此 projection 复制只能作为 warm start，所有 621 个 mixer tensor 在真实 proxy 上都要进入 fitted schedule，不能把整层标成 algebraic。397B 必须从其固定 revision config 独立复核，不能外推 2B 几何。
- 原生 RWKV7 decay 可达域约为 `(exp(-exp(-0.5)), 1)`；落在域外的 GDN decay 需要 fitted approximation，其误差随 context 的累积速度必须单独测。
- GQA KV group 信息应注入哪些 RWKV7 projections/low-rank branches；`kv_repeat` 与 `kv_expand` 的优劣必须按 group/head 和 context length 报告。
- 24 层 Qwen3.5-2B 只能作为真实 checkpoint proxy；60 层 fixture 只能证明实现不变量。两者都不能替代 397B 质量证据。

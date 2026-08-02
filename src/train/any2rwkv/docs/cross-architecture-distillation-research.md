# Attention/GDN 到 recurrent mixer 的跨架构迁移研究

本文服务于 `qwen35-rwkv7-conversion` 的训练 recipe。结论先行：现有工作支持“保留外围块、先局部算子/hidden 对齐、再做端到端 logits 蒸馏”的路线，但没有工作证明 Qwen3.5 GDN 或 GQA/full-attention 可以对原生 RWKV7 做完整、任意输入上的无损静态权重转换。GDN 只有 state recurrence 的受约束子空间可能解析等价；conv、融合 projection、gate、Norm、activation、head 几何和 decay 可达域仍需拟合。

## 最相关工作

### MOHAWK：三阶段、由局部到全局

[Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models](https://arxiv.org/abs/2408.10189) 将 Transformer 与 SSM 都视为 token mixing operator，依次执行：

1. Matrix Orientation：使用同一个 teacher layer input，匹配 teacher/student mixer；
2. Hidden-State Alignment：逐 block 匹配 hidden state；
3. Weight Transfer + Knowledge Distillation：复制可共享外围权重，并对完整模型做 end-to-end prediction distillation。

Phi-Mamba 使用约 3B token，hybrid 版本约 5B token。论文的关键消融是：仅做最终 KD 明显弱于三阶段。这支持本项目先做逐层 hidden/block loss、再在全部层转换后做 global KL/CE；但 MOHAWK 并不是“每一步只更新一个层”，也没有采用本项目的 suffix-free student-prefix 训练。单 active layer 是本项目为了 397B 内存与归因额外增加的 invariant。

### CALD：最直接的逐层替换证据

[Joint Fine-tuning and Conversion of Pretrained Speech and Language Models towards Linear Complexity](https://arxiv.org/abs/2410.06846) 提出 Cross-Architecture Layerwise Distillation（CALD）：按层用线性复杂度模块替换 Transformer 模块，并用对应 teacher hidden state 引导；论文同时在 Pythia→Mamba language modeling 与 Wav2Vec2→Mamba2 speech 上验证。它比纯 end-to-end KD 更贴近本项目“每次只让一层反向”的约束，但仍没有覆盖 60 层 MoE、GDN 的解析 state 映射或 RWKV7。因此本项目采用 CALD 的 layerwise guiding 思路，同时额外要求 student-prefix、suffix-free 局部 target、逐层 optimizer 隔离和全 recurrent corrective sweep。

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

### 8 卡 batch 与 learning-rate 校准

真实小模型和 scale run 使用 8-rank synchronous data parallel；每卡 `micro_batch_size` 不能被误写成 global batch。增大 world size 后，每个 epoch 的样本/token 数不变，但 optimizer step 数会随 global batch 增大而减少，因此“8 卡跑相同 epoch、沿用单卡 learning rate”并不等价于单卡训练。Goyal et al. 的 [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677) 为 SGD 提供 linear learning-rate scaling 与 warmup 的实证；Malladi et al. 的 [On the SDEs and Scaling Rules for Adaptive Gradient Algorithms](https://arxiv.org/abs/2205.10287) 针对 Adam/RMSprop 推导并验证 square-root scaling。它们只用于生成预注册候选，不直接证明本蒸馏任务应采用哪个倍率。Any2RWKV 必须在同一 BF16 loss-token budget、相同 seed 集下联合比较 baseline、Adam square-root scaling 和更激进的 batch/LR 候选，并以最终 fully recurrent validation KL/PPL 选择；同时保存 optimizer-step budget、8 rank 吞吐和显存，避免把“卡数更多”误当成“time-to-quality 更好”。

### GQA uptraining：KV group 聚合是初始化，不是 recurrent 迁移证明

[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) 用原始预训练约 5% 的 compute 将 multi-head checkpoint uptrain 为 MQA/GQA，证明从已有 KV head 做 group-aware 初始化再训练可以接近原模型。它解决的是 attention 内部 MHA → GQA，不解决 softmax attention → recurrent state。对本项目而言，`kv_repeat` 应是保留 group 关系的 baseline，`kv_expand` 是消融；正式迁移必须同时冻结 config、projection layout、Q/K Norm、partial RoPE 和 query-head→KV-group map，并以多 context teacher trace 验证。

### ARWKV：方向高度相关，但实验披露仍不足

[ARWKV: Pretrain is not what we need](https://arxiv.org/abs/2501.15570) 报告从 Qwen 2.5 蒸馏纯 RWKV-7，并公开 early preview；ARWKV-R1-7B model card 还报告过 40M distillation token、2K context、仅 stage-2 的实验。它证明方向可行，但论文是持续更新稿，尚不足以据此冻结本项目的数据配比、逐层 schedule 或“几乎不掉点”阈值。这里应把 ARWKV 当相关先例，而不是替代本项目 canonical/mapping/oracle/baseline 证据。

### RADLADS 与 HALO：快速转换有实证，但没有通用 layer 阈值

[RADLADS: Rapid Attention Distillation to Linear Attention Decoders at Scale](https://arxiv.org/abs/2505.03005) 报告把 Qwen2.5 7B/32B/72B 转为 RWKV variant，使用约 350M–700M token；这为“projection warm start + 较短蒸馏”提供了规模证据，但论文最终以 PPL/下游能力比较，没有给出可跨模型复用的单层 normalized-MSE/cosine 通过线。

[Hybrid Linear Attention Done Right](https://arxiv.org/abs/2601.22156) 的 HALO 先做 hidden-state alignment，再做蒸馏与 long-context finetuning；Qwen3 系列总转换预算约 2.3B token，并在附录报告 stage-1 超过约 320M token 后收益有限。它还明确测试 GDN、Mamba2、GLA、RWKV7 等 mixer，说明同一 pipeline 下 mixer 选择会改变最终能力。因此 HALO 支撑“先局部、后全局、预算由消融决定”，不支撑为 RWKV7 预设一个与模型/层/数据无关的 MSE 常数。

[Distill-then-Replace](https://arxiv.org/abs/2601.11667) 使用 blockwise local distillation 后按 validation 表现贪心替换 attention layer，进一步说明替换决策应由冻结 validation 实证驱动，而不是仅凭局部 loss 达到任意绝对值。

### 阈值证据规则

论文可直接支撑损失位置、阶段顺序、初始化和 token-budget 搜索范围；除非 source/target 架构、模型规模、数据、指标定义、precision 与统计协议严格可比，否则论文中的最终分数或训练预算不能直接变成本项目验收阈值。`any2rwkv` 的 P1/P2 threshold profile 必须由同结构小模型 teacher-paired pilot 生成，至少包含三个独立 seed/run、每个 seed 的原始样本级指标与 baseline 曲线 hash、置信区间、冻结协议 hash、阈值推导实现/版本，以及 teacher/data/code/recipe/metric-definition/precision 绑定。profile 中的数值必须与 calibration artifact 保存的机器推导结果完全一致，候选验收 run 不能参与阈值校准。profile 和 calibration artifact 都写入 SHA-256；缺失、不匹配或只有人工填写理由时状态为 `uncalibrated`，不得通过 scale gate。当前仓库没有生产 threshold profile；测试中的合成常数只覆盖控制流，不能进入真实验收。

数据去重采用独立证据边界。FineWeb 官方论文（[The FineWeb Datasets](https://arxiv.org/abs/2406.17557) §3.4、附录 E.1）使用 word 5-gram、112 个 MinHash、14×8 buckets，目标为至少 75% 相似文档；官方 Datatrove 实现也记录该 LSH 配置的理论拐点约为 0.72。Any2RWKV preparer 不是 Datatrove 的逐哈希实现，因此不能声称算法逐位等价；正式数据只借用“exact word-5-gram Jaccard ≥0.75 视为 near duplicate”这一可比定义，候选生成参数显式冻结为 112/14，并保存 candidate-search completeness。P1/P2 split 只允许 `near_duplicate_policy=reject`（零 pair）或 `drop`（每个报告 pair 至少删除一端并保存 canonical/drop 列表）；任何 oversized bucket、未覆盖 pair 或 dedup report SHA 不匹配时不得生成 calibration profile。

统计协议本身也有直接依据：[What to make of non-inferiority and equivalence testing with a post-specified margin?](https://arxiv.org/abs/1807.03413) 说明 equivalence/non-inferiority margin 应在观察候选结果前预先指定，事后选择 margin 会产生偏差；因此 pilot/calibration 与候选验收必须是独立 run，不能看完候选分数再回填 profile。[Accounting for Variance in Machine Learning Benchmarks](https://arxiv.org/abs/2103.03098) 实证说明数据抽样、初始化与超参数选择都会显著影响 benchmark 结论，因此单 seed 不能支撑准入线。[Better than Average: Paired Evaluation of NLP systems](https://aclanthology.org/2021.acl-long.179/) 在 296 个 NLP evaluation setup 上展示忽略同一样本上的配对关系会改变结论；[Please, Don't Forget the Difference and the Confidence Interval when Seeking for the State-of-the-Art Status](https://aclanthology.org/2022.lrec-1.640/) 则主张报告 bootstrap difference/CI，而不是只看点估计或显著性。基于这些结果，本项目冻结 teacher/student 同 sample 配对、逐样本原始分数、多 seed 以及候选前预注册的 CI/non-inferiority 推导；论文不替代本模型的数值校准。

## 本项目采用的 recipe

1. 先跑 zero-step matrix：random、naive QKV、GDN constrained algebraic、GQA `kv_repeat`/`kv_expand`、activation-fitted。
2. 每层先用 aligned teacher hidden 做 isolated mixer/block fitting；oracle 已通过的动态子空间冻结，外围 fitted 参数训练。
3. 逐层 `0..N-1` 替换；active layer 接收当前 student prefix，只与同一输入上的原始 Qwen layer 做局部 mixer/block 对齐，不加载后续层或 LM head；2B proxy 的 `N=24`，最终 397B 的 `N=60`。
4. 长 context 使用 prefix burn-in，只在 supervised window 计 loss；cold/warmed 分开报告。
5. 全部 recurrent 后至少执行一轮 `N-1..0` corrective sweep，以完整 sweep 的 validation token KL 下降决定停止和 rollback。
6. 数据预算先在 2B proxy 上扫局部/全局比例，至少覆盖 `10/90`、`25/75`、`50/50`；禁止直接把单篇论文比例当成 Qwen3.5/RWKV7 最优值。
7. scale gate 只承认相同 tokenizer/split/seed/precision/token budget 下，正式迁移在 zero-step 和训练后均优于 random/naive baseline，并通过 P1。

## 仍需实验回答的问题

- Qwen3.5-2B proxy 使用 16 个 value/key head、每 head `128×128` state；Any2RWKV target 必须保持同一个 `16×128` head geometry，禁止把它重分为 32 个 `64×64` state。该约束只消除人为的 geometry loss，不代表 conv、gate、Norm、time-mix 或整个 GDN layer 已解析等价；这些外围参数仍按 provenance 分为 algebraic/fitted/initialized。397B 必须从其固定 revision config 独立复核，不能外推 2B head 数，但同样必须保持 source head_size，而不能固定为 64。
- 原生 RWKV7 decay 可达域约为 `(exp(-exp(-0.5)), 1)`；落在域外的 GDN decay 需要 fitted approximation，其误差随 context 的累积速度必须单独测。
- GQA KV group 信息应注入哪些 RWKV7 projections/low-rank branches；`kv_repeat` 与 `kv_expand` 的优劣必须按 group/head 和 context length 报告。
- 24 层 Qwen3.5-2B 只能作为真实 checkpoint proxy；60 层 fixture 只能证明实现不变量。两者都不能替代 397B 质量证据。

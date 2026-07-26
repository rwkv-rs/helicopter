# any2rwkv

本包是通用异构蒸馏套件。架构无关的 core 负责 adapter/recipe 注册、layer-major schedule、rolling hidden cache、单层 optimizer、resume 与证据；具体架构语义由 adapter 和 recipe 提供。本 change 首先交付 `qwen35_to_rwkv7`：受约束的 Qwen3.5 text backbone → 原生 RWKV7 conversion、逐层蒸馏、评测与导出。core 只拒绝未知/不兼容 adapter 或 recipe；不唯一的 GQA layout、vision tensor、最终 60 层等限制仅由 `qwen35_to_rwkv7` recipe 在读取大权重或启动 GPU workload 前校验，不能外推为 `any2rwkv` 的固有限制。

## 固定边界

- 最终 identity 是 `model_type=any2rwkv_qwen35_rwkv7`、`recurrence=native_rwkv7`，60 个 text layer 全部 recurrent。
- MoE、MTP、embedding、Norm、RoPE boundary、LM head 与 tokenizer 语义保留；只替换 GDN/full-attention/GQA mixer。
- GDN 只在通过 FP64 oracle 的动态 state 子空间声明解析等价；conv、融合 projection、gate、Norm、activation、head/state geometry 仍标记为 `fitted` 或 `initialized`。
- 每次 backward 只允许 active layer 的 RWKV7 mixer 参数更新；teacher、其余 layer 和保留外围参数只 forward。
- 正确性使用 BF16 I/O + FP32 state（`WKV_MODE=fp32io16`）；当前 change 不包含量化或其他 runtime 的验收。
- 推理与评估只通过固定版本 Transformers direct loading 执行，不接入 vLLM、`vllm-rwkv` 或 OpenAI-compatible service。

## 可复现阶段

所有 GPU 命令都应由控制仓库的 `helicopter-dev remote run`/`lock run` 包裹。产品 CLI 的离线阶段为：

```text
helicopter any2rwkv fetch-source
helicopter any2rwkv verify-source
helicopter any2rwkv preflight --recipe qwen35_to_rwkv7
helicopter any2rwkv convert --recipe qwen35_to_rwkv7
helicopter any2rwkv distill --recipe qwen35_to_rwkv7
helicopter any2rwkv validate-p0 --recipe qwen35_to_rwkv7
helicopter any2rwkv evaluate --recipe qwen35_to_rwkv7
```

每次 run 使用独立 output；source checkpoint 只读。`convert` 先写 zero-step checkpoint、完整双轴 mapping ledger 和六种 warm-start plan。`distill` 读取不可变数据 manifest 与训练 plan，严格按 layer-major 顺序训练：当前 layer 每个 epoch 恰好消费一次完整 `distill_train`，完整 validation 并达到收敛门槛后才固化 best checkpoint、释放 optimizer 并进入下一层。全部层首次固化后才允许执行 fully recurrent corrective sweep。

sequence-mixer layer 可在完整 optimizer epoch 前启用 `activation_fit_rows` 与 `activation_fit_ridge`。8 个 rank 分别对自己的 rolling-cache rows 计算 `X^T X`、`X^T Y` 和 target energy，只 all-reduce 这些可加 sufficient statistics；rank 0 求解与无 bias `o_proj` 一致的 ridge，再广播拟合权重。runner 会在互斥 validation cache 上比较拟合前后指标，只有 mixer normalized MSE 严格下降且全部数值有限才安装，否则回滚并 fail closed。fit report 绑定 source layer type、cache/data、row digest、normal-equation digest 和安装后权重 SHA。该步骤是 GDN/full-attention 都可使用的 output-projection activation fit；full-attention 正式 fitted provenance 仍须补齐多 context/head/group trace，不能用这一项局部改善冒充完整功能迁移。未被 runner 实际调用的 solver helper 不得记作 `activation-fitted` baseline。

GQA formal zero-step 另走完整依赖事务：先构造 exact prefix-hazard 与保留 sigmoid 的 bounded surrogate，再在 partial-RoPE 可达子空间内求两个独立 observable state。centered-query basis 的每个 head/state DC 固定为 `b-MPP^Tμ`；所有 context 使用相同 rows，正式路径确定性采用包含全部短前缀的最长 causal trace。installation 与 epoch validation 样本互斥，rolling transition 仍推进完整 validation cache；selected state/report/split/solver digest 写入 cursor，并在 immutable safetensors generation 发布后复算 state SHA。正式 solver 按 rank-local row chunk 流式推进 recurrence，按 query-head/KV-group 累积 FP64 sufficient statistics，只通过固定顺序 tensor all-reduce 归并；非连续统计量经 contiguous collective 后写回原 view。候选只有经真实 BF16 module 完整 forward 严格改善才原子安装。Qwen3.5-2B layer 3 的 8-rank 冻结 validation 已将 mixer normalized MSE 从 `4.973255` 降到 `0.498262`、cosine 从 `0.730662` 提高到 `0.942757`；两个 `128×128` state 的 observable compression 相对 affine oracle 的增量 NMSE 为 `0.023265`，累计到 exact attention 为 `0.325510`。这证明该单层初始化显著优于 frozen mapped baseline，但不是无损 `256×256` 算子表示；能否缩短蒸馏或免蒸馏仍需逐层与完整模型实验。

GDN layer 采用原生 RWKV7 的精确递推信号分解：`k_k=1`、`k_a=0`，read/key target 分别为 `L2(q)/sqrt(head_dim)` 与 `L2(k)`，native erase target 为 `beta·decay`，write-value target 为 `beta·v`。zero-step `a_lora` 联合使用 `in_proj_b/in_proj_a/A_log/dt_bias`，安装 `logit(beta·decay)` 在零输入处的一阶展开；它仍是非 lossless 的参数化近似，随后由互斥 fit/validation teacher trace 补齐。`k_a=1、a=beta` 会漏掉 source erase term 中的 decay，已由输出级反例测试拒绝。runner 还会在 `o_proj` 之前拟合 native decay boundary：从 source `in_proj_a/A_log/dt_bias` 计算真实 decay，在 target `w_lora` 的低维 `tanh` feature 上拟合带 bias up projection。报告同时保存未裁剪 source decay 的 held-out MSE 与 native 不可达比例；未严格改善就恢复原 weight/bias并标为 `rejected`。该步骤的 Gram 只有 `(rank+1)×(rank+1)`，适用于 8 卡和 397B，不构造 residual hidden 或 recurrent width 的 dense Gram/solve。target 必须保持 source GDN 的 head/state geometry：Qwen3.5-2B residual/recurrent width 为 `2048/2048`、state 为 `16×128`；397B 为 `4096/8192`、state 为 `64×128`，其中 source `16→64` key-head repeat 原样保留。正式路径禁止把 recurrent width 压回 hidden size，也禁止 `128→64` state 压缩。迁移、训练和评估不加载、对齐或转置 source recurrent state；逐层 loss 只包含 mixer/block normalized MSE 与 cosine，不包含 state MSE。

8 卡不是启动形式而是数据语义：每个 global micro-batch 的 row 恰好分到一个 rank，当前正在训练的 RWKV7 层按实际样本数归并 8 卡梯度；训练前同步 rank-0 参数并校验可训练参数清单，validation 在各 rank 本地累计后只归并一次。逐层蒸馏只驻留“原始 Qwen 当前层”和“要训练的 RWKV7 当前层”，不加载 `i+1..N-1`；这里的 mixer 就是这一层里负责处理 token 序列、将被 GDN/attention 替换为 RWKV7 的部分，block 则是加上残差连接和这一层其它保留结构后的完整层输出。全部层换完后的端到端微调，为计算全模型 KL/CE 才流式经过完整 RWKV7 模型，两者不得混称。单 rank 只用于 CPU/unit fixture 语义测试；凡是使用真实 checkpoint 的 activation fit、逐层训练与端到端微调都必须使用 8 rank，单卡结果不进入迁移、收敛或性能证据。Transformers 生成与评估可按其独立 workload 并行化，但不属于训练证据。

已经完成全 recurrent conversion 后，使用 `corrective` action 追加 sweep，不重新运行逐层 curriculum。先用 `python -m any2rwkv.cli checkpoint-binding --checkpoint <parent-checkpoint>` 获取 SHA-256，再把它作为 `--parent-checkpoint-sha256` 与 `--parent-run` 一起传入；runner 会复验父 checkpoint 全部分片、mixer fingerprint、source、recipe 与 precision。新 run 通过 sibling staging 原子发布，绑定一致的中断重跑会从 rank-local optimizer/RNG 和 canonical mixer 恢复。真实 run 仍必须由 `any2rwkv-layer-major` 训练契约以 8-rank NCCL 启动。

distillation plan 使用 schema v3，必须显式列出 local mixer/block/cosine 和 global KL/CE 的权重，不允许 state loss 权重，并声明枚举 `evidence_tier = fixture | exploratory | p1 | scale`。fixture 与 exploratory plan 可以运行但不能产生 P1/scale 证据；`p1`/`scale` plan 只有在引用 `scripts/derive_training_control_calibration.py` 生成且 SHA/fingerprint 匹配的 artifact 后才允许启动，不能通过修改自由文本名称绕过。control calibration 至少比较三个候选、每个候选三个相同 seed 与等 token budget，并与后续质量 threshold calibration 使用不同 run/checkpoint。

真实训练的 optimizer 也属于 plan 契约。默认基线是 AdamW，每个非 fixture plan 都必须写明初始/最终 learning rate、精确 warmup step、betas、epsilon、weight decay 和 gradient clip，不能由代码补默认值。当前保留的用户基线是 `1e-6 -> 1e-6`、前 10 次实际 update 从 `0.1e-6` 线性 warmup 到 `1e-6`、之后保持 `1e-6`、betas `(0.9,0.99)`、weight decay `0.1`、clip `1.0`、accumulation `1`；它是要比较的候选，不是已经证明最优的答案。局部逐层训练只反向一个 mixer，默认不开 gradient checkpointing，先扩大每卡 batch。Muon 只能单独做同初值、同数据、同 batch、同 token budget、同 seed 的 A/B；AdamW 基线通过前不会启用。

任何真实训练都必须先通过零交付的 8 卡端到端性能扫描。候选只在全新的 `RUN_DIR/scratch` 做真实 AdamW update，run 目录不能与 source、zero-step、overlay 或 cache 相交；扫描前后都会复验完整 source/target checkpoint 和其它只读输入。扫描不再只看 layer 0：source config 会自动分出“layer 0 的 embedding 输入”“后续 GDN 层的 RWKV7 前缀输入”“full-attention 层的 RWKV7 前缀输入”等实际场景，每个 batch 都必须在各场景的固定代表层上完整消费一个 frozen train epoch 和完整 validation；同场景的 source/target 参数 shape signature、cache 与参与误差计算的 token 数必须一致。候选使用与正式 runner 相同的确定性 epoch permutation、loss、AdamW、telemetry、跨 rank validation 与 epoch checkpoint 路径；当前只接受 `checkpoint_interval_micro_batches=0`。每个有下一层的场景还直接调用正式 8-rank row-sharded cache transition，在 scratch 中生成并删除下一层输入缓存。至少比较三个 batch，形成完整 batch×场景矩阵。`scripts/run_any2rwkv_profile_candidate.py` 是唯一 candidate launcher：它先运行带 GPU metrics 的完整进程 Nsight capture，再从该 `nsys-rep` 导出 `nsys-sqlite` 并生成 export binding。每个 rank 对训练周期和 cache transition 分别用本次 run id、随机 nonce 和 rank 写唯一 NVTX range；CUDA kernel 覆盖墙钟比例从同一 SQLite、同一 process/range 复算，只表示测量窗口里有没有 GPU 工作，不叫 GPU 利用率。每张卡计算单元真正工作的比例来自对应公共窗口的 `SMs Active` 样本。8-rank launcher 先在 CPU 侧完成同一个 launch token 和 W&B online identity；rank 0 成功建 run 后才允许初始化 NCCL/CUDA。每个候选结束时生成首行绑定 run/attempt/config 的中文 `experiment-report.md`，明确说明测的是哪个实际场景；最终 selection 另建 W&B run。strict wrapper 会重新核对 action/checkpoint/case/cache/signature/optimizer/row-step/Nsight export/W&B/report binding，并分别重算训练周期和 cache transition：每个 rank 在测量窗口内有 CUDA 工作的时间比例、每张卡平均 `SMs Active` 都必须至少 95%，PyTorch/CUDA reserved memory 必须是设备的 85%–95%，最慢卡/最快卡墙钟比不超过 1.05，写 checkpoint 不超过同一窗口的 5%。`SMs Active` 按 [NVIDIA Nsight Systems GPU Metrics](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics) 的定义使用；它表示采样周期内 SM 的活动程度，不与 `nvidia-smi` 的粗粒度利用率混称。95% 是把“接近 100%”落成可执行的项目准入线，不作为论文结论；batch 的最终选择仍来自同机、同数据、同代码、同 token 预算的完整实测矩阵。任一场景或阶段失败则整个 batch 失败；合格 batch 按源模型中各场景的实际层数和 transition 次数加权，选择端到端吞吐最高者。当前逐层 profile 只能准入逐层蒸馏；端到端微调需要独立的全 RWKV7 profile。缺少完整 schema-v3 `performance_evidence` 时训练一定被拒绝。

新 W&B run 使用 `resume=never`；已有本地 attempt 的恢复使用 `resume=must`。每次启动先写新的 `attempt_id`、完整 config SHA 和 `initializing`，初始化失败也覆盖为 `failed`。正常结束时，只有所有 rank 已到达共同收尾点才广播 rank-0 结果；单 rank 异常不会再发起新的 collective，而是直接失败并由 torchrun 终止同伴，避免不同 rank 进入不同 collective。成功、失败和人工停止都会尽力生成说人话的中文报告；训练子进程成功后，外层 wrapper 还会复验完整 W&B config、attempt/report identity 和运行前后输入摘要。

在正式 control calibration 前，可让 8 卡 exploratory plan 设置 `exploratory_layer_limit=N`，只对前 N 层执行相同的 activation fit、完整 epoch 和 convergence gate，以快速筛掉数值爆炸或明显不合适的 learning-rate/loss 区间。N 必须小于 source 总层数；完成后 runner 返回 `exploratory-layer-calibration-complete`，不生成 layer N cache、完整 HF checkpoint，也不启动 corrective。该结果只用于缩小正式候选范围，不能被 P1、scale、training-control 或质量校准引用；fixture/P1/scale plan 带此字段会在训练前失败。

streamed runner 按 plan 指定的 durable checkpoint 边界提交 mixer、optimizer、RNG 和 cursor；`checkpoint_interval_micro_batches=0` 表示只在 epoch 边界提交。同步边界只写一份 canonical optimizer/master state 和每 rank RNG sidecar，停在未同步梯度中间时才写 rank-local full state。optimizer 在同一层的多个 epoch 间保持 GPU 常驻，只在切换 layer 时释放；resume 会从最后一个 durable generation 恢复并重做尚未提交的 rows。cache reader 会逐 shard 核对真实 row index，并以 `max_cached_layer_input_bytes_per_rank` 限制每个进程的 CPU shard cache；train/validation manifest SHA 进入 generation cursor。首轮 corrective sweep 从 hash-bound `pre-sweep` snapshot 开始，最终 HF export 也必须匹配 selected all-layer mixer fingerprint，禁止复用旧导出。

确定性 60 层 fixture pilot 的输入由 `scripts/write_tiny_pilot_inputs.py` 生成，再由 `scripts/prepare_data.py` 产生互斥 split；它只能证明结构、梯度和恢复 invariant，不能作为能力结果。真实 Qwen3.5-2B 是 24 层 integration/convergence proxy，也不能替代 397B 质量证据。

397B 训练计划还带有启动前 execution estimate：记录 8 rank 的 current-layer teacher/student forward、rolling-cache transition、fully recurrent corrective 的 checkpointed suffix reload、activation-fit sufficient-statistics 内存/通信/求解量和保守权重搬运量。若 replicated 8-rank data parallel 的任一 rank 显存、共享文件系统 I/O、normal-equation `O(H^2)` 内存或 dense solve `O(H^3)` 超过冻结上限，必须 fail-fast；此时只能做 conversion/preflight，训练必须先接入可验证的 parameter/expert sharding 或 scalable activation-fit solver，不能退回单卡或靠继续排队绕过。

## 质量验收

`scripts/build_evaluation_manifest.py` 从已冻结 validation/smoke split 生成 hash-bound evaluator 输入。RULERv2 与 lm-eval 的 checkout、命令和 task 配置由 `manifests/quality-suite.json` 与 `scripts/build_quality_command_plan.py` 固定；原始样本输出必须经 `scripts/normalize_external_scores.py` 和 `scripts/build_paired_scores.py` 生成相同 sample-id 的 teacher/student 配对，之后才可进入 10,000 次 paired bootstrap。

P0 要求 canonical state、mapping coverage、GDN oracle、full-attention/GQA fixture、global loss bridge、active-layer invariant、resume parity 和两次 fresh-process HF round-trip 全通过。P1 是 scale gate；P2 才能声明 quality-preserving。未通过 P1 的 checkpoint 不允许触发 397B fetch。

P1/P2 不提供默认数值。先冻结独立的 teacher-paired BF16 calibration protocol，再完成至少三个不同 seed、不同 checkpoint 的 eligible pilot run；每个 run 必须绑定原始 sample metrics、baseline curve 和真实 eligibility artifact。eligibility 只检查 P0 无失败、全层收敛，以及同指标下 mapped 初始化严格优于 random/naive；推导器会读取 artifact、校验 SHA 并复算比较，不能用一个任意的 `qualified=true` 或 64 位字符串代替证据。PPL/KL、layer MSE/cosine、smoke pass rate、RULER/downstream paired bootstrap 与逐 bucket/task 指标也全部从逐样本 artifact 重算，seed result 内嵌的汇总值不被信任。training-control 的 validation curve 同样必须可读取并重算 KL、PPL、全层收敛比例和 token budget；正式 plan 使用枚举 `evidence_tier`，不从 `classification` 子串猜测 P1/scale 身份。随后运行 `scripts/derive_quality_threshold_profile.py`，按 `worst-eligible-seed-envelope` 规则机器生成 profile：所有 lower-bound 指标取 eligible seeds 的最小值，所有 upper-bound 指标取最大值。artifact 为每个阈值保存全部 seed 观测、形成边界的 limiting run、seed-result SHA 和方法学文献出处；文献只证明预注册、多 seed、paired evaluation 与置信区间方法，真正的模型数值仍来自这些独立 pilot。protocol 还必须绑定完整 distillation plan、training-control artifact 和无跨 split 泄漏的 prepared-data manifest；build 与 reload 两端都重读原始文件并验 SHA。候选 run/checkpoint 不得参与 control 或 threshold calibration。任一原始 artifact 缺失、手写汇总与重算不一致或绑定不匹配时，evaluator 返回 `uncalibrated`/拒绝执行，而不是放宽阈值。

性能候选的“输入没变”按完整文件树判断，不只看 manifest：launcher 在 GPU 启动前冻结 prepared-data split、train/validation cache 全部 shard、source/zero-step checkpoint、overlay、训练源码和 Git/submodule，候选结束后再次计算。candidate JSON、W&B config、Nsight export binding、selection 都绑定同一组 SHA；strict wrapper 还会回读原始 `candidate.json`，不接受 selection 改写的显存或墙钟。Nsight 的每个 `SMs Active sourceId` 必须与实际 CUDA device 一一对应。torchrun、export、复算或 W&B 收尾失败时，同一个 W&B run 与中文报告会标成失败，不能留下“完成但指标待补”的实验。

当前只交付 BF16 checkpoint。NVFP4 必须等 BF16 迁移和质量验收完成后在独立 change 中重新设计；当前 CLI、数据 split、依赖和验收不得包含量化路径。

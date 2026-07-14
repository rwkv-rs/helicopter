# any2rwkv

本包实现受约束的 Qwen3.5 text backbone → 原生 RWKV7 conversion、逐层蒸馏、评测与导出。它不是通用架构转换器：未知 architecture、不唯一的 GQA layout、混入 vision tensor 的 target、非 60 层最终交付都会在读取权重或启动 GPU workload 前被拒绝。`rwkv7_hf` 只负责 HF compatibility，Qwen 专用逻辑留在本包。

## 固定边界

- 最终 identity 是 `model_type=any2rwkv_qwen35_rwkv7`、`recurrence=native_rwkv7`，60 个 text layer 全部 recurrent。
- MoE、MTP、embedding、Norm、RoPE boundary、LM head 与 tokenizer 语义保留；只替换 GDN/full-attention/GQA mixer。
- GDN 只在通过 FP64 oracle 的动态 state 子空间声明解析等价；conv、融合 projection、gate、Norm、activation、head/state geometry 仍标记为 `fitted` 或 `initialized`。
- 每次 backward 只允许 active layer 的 RWKV7 mixer 参数更新；teacher、其余 layer 和保留外围参数只 forward。
- 正确性使用 BF16 I/O + FP32 state（`VLLM_RWKV7_WKV_MODE=fp32io16`）；性能与 NVFP4 run 另用 FP16 policy，不能混为质量证据。

## 可复现阶段

所有 GPU 命令都应由控制仓库的 `helicopter-dev remote run`/`lock run` 包裹。产品 CLI 的离线阶段为：

```text
helicopter any2rwkv fetch-source
helicopter any2rwkv verify-source
helicopter any2rwkv preflight
helicopter any2rwkv convert
helicopter any2rwkv distill
helicopter any2rwkv validate-p0
helicopter any2rwkv evaluate
helicopter any2rwkv quantize
```

每次 run 使用独立 output；source checkpoint 只读。`convert` 先写 zero-step checkpoint、完整双轴 mapping ledger 和六种 warm-start plan。`distill` 读取不可变数据 manifest 与训练 plan，依次执行 isolated signals/block、`0..N-1` progressive global、首次 fully recurrent 与 `N-1..0` corrective sweep，并原子保存 active layer optimizer、累计梯度、RNG、data cursor 和 sweep cursor。

streamed runner 的每个 microstep 使用不可变 generation 事务提交 mixer、optimizer、RNG、cursor 和 trace offset，最后只原子替换 `latest.json`；resume 会恢复同一 generation 并截断未提交 trace 尾部。首轮 corrective sweep 从 hash-bound `pre-sweep` snapshot 开始，最终 HF export 也必须匹配 selected all-layer mixer fingerprint，禁止复用旧导出。

确定性 60 层 fixture pilot 的输入由 `scripts/write_tiny_pilot_inputs.py` 生成，再由 `scripts/prepare_data.py` 产生互斥 split；它只能证明结构、梯度和恢复 invariant，不能作为能力结果。真实 Qwen3.5-2B 是 24 层 integration/convergence proxy，也不能替代 397B 质量证据。

397B 训练计划还带有启动前 execution estimate：记录 teacher/student full forward、checkpointed suffix reload 和保守权重搬运量。当前单 GPU `streamed_layer_store` 估算超过冻结的 1 PB 上限时必须 fail-fast；此时只能做 conversion/preflight，训练必须先接入 distributed resident 或 expert-sharded backend，不能靠继续排队 GPU 绕过。

## 质量与量化

`scripts/build_evaluation_manifest.py` 从已冻结 validation/smoke split 生成 hash-bound evaluator 输入。RULERv2 与 lm-eval 的 checkout、命令和 task 配置由 `manifests/quality-suite.json` 与 `scripts/build_quality_command_plan.py` 固定；原始样本输出必须经 `scripts/normalize_external_scores.py` 和 `scripts/build_paired_scores.py` 生成相同 sample-id 的 teacher/student 配对，之后才可进入 10,000 次 paired bootstrap。

P0 要求 canonical state、mapping coverage、GDN oracle、full-attention/GQA fixture、global loss bridge、active-layer invariant、resume parity 和两次 fresh-process HF round-trip 全通过。P1 是 scale gate；P2 才能声明 quality-preserving。未通过 P1 的 checkpoint 不允许触发 397B fetch。

NVFP4 calibration 只能读取独立的 `nvfp4_calibration` split。ModelOpt exporter 使用 `device_map=auto` 支持多 GPU/CPU dispatch，并显式保留 recurrent dynamic/state、Norm、embedding、LM head 等高精度 tensor。导出默认仍是 experimental；只有 `scripts/validate_nvfp4_acceptance.py` 重新验证 BF16 P2 绑定、NVFP4 PPL/KL、RULER/downstream、P0、strict HF reload 和 vLLM service 后才能标记 `nvfp4-quality-compatible`，性能标签还需通过 throughput、TTFT/TPOT 与峰值显存门槛。Any2RWKV serving 强制使用 vLLM 通用 Transformers backend（`model_impl=transformers`）加载 HF remote code，禁止使用 `vllm-rwkv` 的 pure-RWKV model loader。

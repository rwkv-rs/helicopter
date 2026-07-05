# RWKV/vLLM/verl 性能交接文档

## 目标

接手 agent 继续处理当前清理之后剩下的性能问题。优先级放在结构性瓶颈上，不做局部风格清理。本文档里的判断主要来自代码路径审查；真正的 benchmark 和 profile 还需要在目标 BBT DevPod GPU workspace 上跑。

## 当前状态

- 仓库：`/home/caizus/Projects/MachineLearning/helicopter`
- 主要 RWKV submodule：
  - `/home/caizus/Projects/MachineLearning/helicopter/src/infer/vllm-rwkv`
  - `/home/caizus/Projects/MachineLearning/helicopter/src/train/verl-rwkv`
  - `/home/caizus/Projects/MachineLearning/helicopter/src/train/rwkv-lm`
- 环境目标：统一使用父项目 uv 环境 `/home/caizus/Projects/MachineLearning/helicopter/.venv`。不要在 vLLM 或 verl submodule 项目内单独执行 `uv sync`。
- 这轮清理已经把环境所有权收进 `/home/caizus/Projects/MachineLearning/helicopter/scripts/install_local.sh`。vLLM 和 verl 的 editable install 使用父项目环境；RWKV submodule 内的本地 `.venv` 会在安装开始和结束时清理。

## 约束

- shell 命令使用 `rtk`。
- RWKV 工作不要使用 submodule-local `.venv`。
- 修改 vLLM 和 verl submodule 时，默认跟随 upstream 的结构、命名、import 风格和行为边界。
- 修性能时不要改变外部 CLI/API 行为，除非用户明确同意。
- 未提交的 dirty worktree 当作共享状态处理；状态、diff、测试、暂存和提交都使用 path-limited 操作。

## 待修复的性能问题

1. RWKV prompt padding 仍在 generic verl agent loop 里。

   路径：
   `/home/caizus/Projects/MachineLearning/helicopter/src/train/verl-rwkv/verl/experimental/agent_loop/agent_loop.py`

   机制：`_prompt_with_single_prefix_token` 和 `_right_pad_prompt_batch` 把 RWKV 专属的 prompt 语义放进了 generic agent loop。当前代码避免了 left-padding EOS token，但 owner 不对；继续这样放会让共享路径不断长出 RWKV special case。

   下一步：把 RWKV prompt-prefix/padding policy 移到 rollout/model adapter 边界后面。generic agent loop 只负责通用 batch assembly，不直接理解 RWKV token 假设。用现有 prompt-padding tests 和一个包含不同 prompt 长度的 rollout sample 验证。

2. `infctx chunk_ctx` 校验仍有多个 owner。

   路径：
   `/home/caizus/Projects/MachineLearning/helicopter/src/cli/helicopter_cli/commands.py`
   `/home/caizus/Projects/MachineLearning/helicopter/src/train/verl-rwkv/verl/workers/engine/rwkv_lm/args.py`
   `/home/caizus/Projects/MachineLearning/helicopter/src/train/verl-rwkv/verl/workers/engine/rwkv_lm/transformer_impl.py`
   `/home/caizus/Projects/MachineLearning/helicopter/src/train/rwkv-lm/src/model.py`

   机制：CLI override 构造、verl engine config、transformer runtime 和 rwkv-lm model 代码都在校验或推导 `chunk_ctx`。这带来重复检查，也让 CUDA chunk length 整除约束和 `chunk_ctx < ctx_len` 到底由哪一层负责变得不清楚。

   下一步：在 engine 边界建立一个已校验的 infctx config 对象，或一个窄的 validation function；向下游只传 normalized integer。rwkv-lm model 内保留面向直接 library misuse 的防御性检查。补测试覆盖 `chunk_ctx <= 0`、chunk 不可整除、`chunk_ctx >= ctx_len`。

3. rapid penalty indexed path 仍有整行 penalty gather/scatter 的结构性带宽成本。

   路径：
   `/home/caizus/Projects/MachineLearning/helicopter/src/infer/vllm-rwkv/vllm/v1/sample/ops/topk_topp_sampler.py`
   `/home/caizus/Projects/MachineLearning/helicopter/src/infer/vllm-rwkv/vllm/v1/worker/gpu/sample/sampler.py`
   `/home/caizus/Projects/MachineLearning/helicopter/src/infer/vllm-rwkv/vllm/v1/worker/gpu_input_batch.py`
   `/home/caizus/Projects/MachineLearning/helicopter/src/infer/vllm-rwkv/vllm/v1/sample/ops/penalties.py`

   机制：`rapid_sample(..., penalty_indices=...)` 仍然对完整 `[row, vocab]` penalty row 使用 `index_select(...).contiguous()` 和 `index_copy_`。非默认 generic penalties 也可能 materialize token tensor 和 `[num_seqs, vocab]` mask。开启 penalties 且 vocab 很大时，这是带宽问题。

   下一步：让 rapid kernel 直接接收 penalty row indices，或者由 sampler 维护 compact active-row penalty buffer。开启 penalties 后测 H2D/D2D copy time 和 per-token latency。

4. RWKV tokenizer、padding、decode 仍是 Python batch 外层循环。

   路径：
   `/home/caizus/Projects/MachineLearning/helicopter/src/infer/vllm-rwkv/vllm/tokenizers/rwkv.py`
   `/home/caizus/Projects/MachineLearning/helicopter/src/infer/vllm-rwkv/vllm/tokenizers/rwkv_defaults.py`
   `/home/caizus/Projects/MachineLearning/helicopter/src/train/verl-rwkv/verl/models/rwkv/tokenizer.py`

   机制：RWKV byte-trie encode/decode 仍在 Python 层按 sample、按 token 处理。verl wrapper 路径又叠加了逐行 pad 和 `batch_decode` 循环。长 prompt、validation batch 和重复 prompt rendering 会放大 CPU/GIL 与 allocation 成本。

   下一步：分别 profile `apply_chat_template`、encode、pad 和 batch decode。低风险起点是在 prompt 会复用的位置缓存 rendered prompt token ids。更大的修复是建立 fast/batched RWKV tokenizer 边界。

5. repetition detection 有重复 owner，且 CPU text analysis 偏重。

   路径：
   `/home/caizus/Projects/MachineLearning/helicopter/src/infer/vllm-rwkv/vllm/v1/core/sched/utils.py`
   `/home/caizus/Projects/MachineLearning/helicopter/src/train/verl-rwkv/verl/workers/rollout/vllm_rollout/vllm_async_server.py`
   `/home/caizus/Projects/MachineLearning/helicopter/src/train/verl-rwkv/verl/experimental/agent_loop/single_turn_agent_loop.py`
   `/home/caizus/Projects/MachineLearning/helicopter/src/train/verl-rwkv/verl/utils/ngram_repetition.py`

   机制：vLLM scheduler、verl async stream abort 和 agent final postprocess 都可能检查 token ngram。verl 的 detector 还会跑 zstd、script-mix、text-ngram 和 reasoning-marker analysis。长输出可能在 CPU 上被重复扫描。

   下一步：选定一个 token-ngram owner。倾向于让 vLLM engine/scheduler 负责 token stop，verl 只追加 diagnostics，或只做无法放进 vLLM 的 text anomaly checks。加 per-section timing counters，对比 detection on/off。

6. `rollout.n` 在 rollout dispatch 前复制完整 `DataProto`。

   路径：
   `/home/caizus/Projects/MachineLearning/helicopter/src/train/verl-rwkv/verl/trainer/ppo/ray_trainer.py`
   `/home/caizus/Projects/MachineLearning/helicopter/src/train/verl-rwkv/verl/protocol.py`
   `/home/caizus/Projects/MachineLearning/helicopter/src/train/verl-rwkv/verl/trainer/ppo/v1/agent_loop_tq.py`

   机制：prompt tensor 和 non-tensor fields 会在 agent loop dispatch 前通过 `repeat_interleave` 重复。TQ 路径有自己的 fanout/session handling，说明 owner 已经分裂。结果是 Ray object size 变大，padding work 也增加。

   下一步：测 `DataProto.repeat`、Ray object size 和 padding ratio。把 fanout 下沉到更靠近 rollout execution 的位置，让 prompt data 只引用一次，再配 per-sample response slots。

7. RWKV infctx no-padding 在 verl engine 里又变成 dense。

   路径：
   `/home/caizus/Projects/MachineLearning/helicopter/src/train/verl-rwkv/verl/workers/engine/rwkv_lm/transformer_impl.py`

   机制：nested input ids 在 RWKV forward 前被转成 `B x max_seq_len` padded tensors。packed layout 只过滤 response logits/logprobs；recurrent trunk 仍然对所有 row 处理 dense chunks。

   下一步：记录 `sum(seq_lens) / (B * max_seq_len)` 和 latency。先试 length bucketing；如果 padding waste 仍高，再实现 only-live-row chunk forward 或 packed recurrent state layout。

8. native rwkv-lm infctx loop 每个 chunk 都在 stack/cat state 和 hidden tensor。

   路径：
   `/home/caizus/Projects/MachineLearning/helicopter/src/train/rwkv-lm/src/model.py`

   机制：每个 chunk/layer 会收集 Python list，然后用 `torch.stack` 组 state；feature 路径会用 `torch.cat` 拼 hidden chunks；部分 CUDA 调用强制 `.contiguous()`。`chunk_ctx` 越小，这些开销越明显。

   下一步：profile `forward_infctx_features` 和 `forward_infctx_sequence` 里的 `aten::stack`、`aten::cat`、`aten::contiguous`。预分配 state tensor，并在 logprob-only 路径避免完整 hidden concatenation。

9. vLLM RWKV decode state compaction 会复制 dense recurrent state。

   路径：
   `/home/caizus/Projects/MachineLearning/helicopter/src/infer/vllm-rwkv/vllm/v1/worker/gpu/model_states/rwkv.py`

   机制：decode rows 会 compact 到 resident order。出现 holes 时，代码会 clone/copy 每层的 `shift_state`、`wkv_state` 和 elapsed state。request 长度混杂或 request churn 较高时，可能产生 D2D copy spike。

   下一步：把 `get_state_movement_stats()` 或等价 counters 导出到 eval logs。关联 compaction rows 和 latency spikes。重写 kernel 前，先考虑 row indirection 或 lifecycle grouping。

## 建议测量命令

父项目环境健康后，先跑这些本地检查：

```bash
rtk uv run --project /home/caizus/Projects/MachineLearning/helicopter python -m pytest tests/test_cli.py -q
rtk uv run --project /home/caizus/Projects/MachineLearning/helicopter python -m pytest src/train/verl-rwkv/tests/utils/test_ngram_repetition_on_cpu.py -q
rtk uv run --project /home/caizus/Projects/MachineLearning/helicopter python -m pytest src/infer/vllm-rwkv/tests/v1/sample/test_topk_topp_sampler.py -q
```

在 BBT g6 上做 GPU profiling 时，先用短场景、单变量的 Nsight capture：

```bash
rtk scripts/install_remote.sh
rtk ssh g6.devpod 'cd /workspace/Projects/MachineLearning/helicopter && uv run --project . python -m pytest src/train/rwkv-lm/tests/test_infctx_cuda_kernels.py -q'
```

## 验证状态

本轮清理已完成：

- `scripts/install_local.sh` 可以在父项目环境 `/home/caizus/Projects/MachineLearning/helicopter/.venv` 下完成。
- `vllm`、`vllm._C_stable_libtorch`、`vllm.rwkv7_ops`、`verl` 和 `torch` 都从父项目环境导入。
- `pyext` 未安装；之前的 `pyext==0.7` 失败通过避免在 verl submodule 项目内执行 `uv sync`、避免安装 `verl[prime]` 解决。
- `src/infer/vllm-rwkv`、`src/train/verl-rwkv`、`src/train/rwkv-lm` 下不再保留 RWKV submodule-local `.venv`。
- 在本地 Spark/GB10 上，vLLM native build 需要 `TORCH_CUDA_ARCH_LIST=12.0+PTX`。原因是可见 GPU 的 compute capability 是 12.1，而当前 PyTorch wheel 报告支持到 `sm_120`。
- `uv pip check` 仍有一个已知且本轮容忍的 upstream metadata 问题：`nvidia-cusparselt-cu13` 发布的 aarch64 wheel 使用 `manylinux2014_sbsa` tag，uv 会报告 platform mismatch。

已验证命令：

```bash
rtk timeout 900s env UPDATE_UV=0 UV_UPGRADE=0 scripts/install_local.sh
rtk timeout 300s .venv/bin/python -m pytest tests/test_cli.py -q
rtk timeout 600s .venv/bin/python -m pytest src/train/verl-rwkv/tests/utils/test_ngram_repetition_on_cpu.py src/train/verl-rwkv/tests/workers/rollout/rollout_vllm/test_vllm_repetition_detection_params_on_cpu.py src/train/verl-rwkv/tests/trainer/ppo/test_validation_dump_on_cpu.py src/train/verl-rwkv/tests/experimental/agent_loop/test_agent_loop_prompt_padding_on_cpu.py src/train/verl-rwkv/tests/experimental/agent_loop/test_single_turn_repetition_truncation_on_cpu.py src/train/verl-rwkv/tests/rwkv/test_rwkv_lm_engine_loss_contract_on_cpu.py -q
rtk timeout 900s .venv/bin/python -m pytest src/infer/vllm-rwkv/tests/tokenizers_/test_rwkv.py src/infer/vllm-rwkv/tests/v1/core/test_repetition_detection.py src/infer/vllm-rwkv/tests/v1/sample/test_topk_topp_sampler.py -q
rtk timeout 900s .venv/bin/python -m pytest src/infer/vllm-rwkv/tests/v1/worker/test_gpu_sampler.py -q
cd /home/caizus/Projects/MachineLearning/helicopter/src/train/rwkv-lm && rtk timeout 900s /home/caizus/Projects/MachineLearning/helicopter/.venv/bin/python -m pytest tests/test_infctx_cuda_kernels.py -q
```

## 建议使用的 skills

- `$perf-optim`
- `$cuda-nsight-profiling`
- `$bench-designing`
- `$impl-designing`
- `$tests-designing`
- `$environment-first`

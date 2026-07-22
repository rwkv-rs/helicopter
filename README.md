# Helicopter

Helicopter 的评测组件位于 `src/eval/lighteval`，distribution 名称为
`helicopter-lighteval`，Python import package 为 `helicopter_lighteval`。`src/eval` 只
是评估框架的物理容器，未来的 `lm-eval-harness` 使用同级独立组件，不共享本地
评测框架代码。

## 目录边界

```text
src/eval/lighteval/
├── pyproject.toml
├── src/helicopter_lighteval/
│   ├── evaluation.py       # 组合 LightEval Pipeline/Tracker
│   ├── vllm_rwkv.py        # OpenAI-compatible terminal evidence adapter
│   ├── scoreboard.py       # HTTP publication/retry
│   └── datasets/
│       ├── math.py          # 数学 A/B/C 与 math identity
│       ├── knowledge.py     # knowledge identity；不复制 task
│       ├── coding.py        # LiveCodeBench identity 与安全边界
│       └── instruction_following.py # 单轮 IFEval/IFBench identity
├── tests/
└── results/                # generated, Git ignored
```

LightEval 是 dataset、prompt、chat messages、task registry、metric、样本生命周期和
标准 results/details 的唯一 owner。Helicopter 不复制 snapshot、context、prompt
template、generic model client、records/artifacts framework 或 synthetic benchmark。
`results/<run-id>` 与 `src/` 并列，不是 Python package。

Scoreboard 的写入链路始终是：

```text
browser/CLI/evaluator -> scoreboard HTTP API -> backend service/repository -> database
```

评测端、前端和 CLI 不导入数据库驱动、ORM、repository 或数据库凭据；Scoreboard server
和 client 不属于本次收缩重构范围。

## 安装

环境由控制仓库的 `helicopter-dev` 管理：

```bash
./bin/helicopter-dev env sync fix-lighteval-unified-prompt --target local --components lighteval,dev
```

`lighteval` 显式映射到根项目的 `eval` 依赖组；`dev` 单独提供 `pre-commit` 和测试工具。
训练或推理环境按需选择 `rwkv-lm`、`verl-rwkv`、`verl-liger`、`vllm-rwkv`，不提供
`full` 组或 profile。基础 `helicopter --help` 不加载 LightEval 或 OpenAI client；vLLM
native 安装只允许 `VLLM_BUILD_PROFILE=rwkv`。

## 运行评测

当前可解析固定 LightEval revision `64f4f5ae173626509fad6e477ca4ee56ebb26129` 的真实
task identity：

- `lighteval/math/gsm8k@0`
- `lighteval/math/math-500@2`
- `lighteval/math/aime24@2`
- `lighteval/math/aime25@2`
- `lighteval/math/asdiv@0`
- `lighteval/math/gsm-plus@0`
- `lighteval/math/olympiadbench@1`
- `lighteval/knowledge/mmlu-<subject>@0`（固定 registry 中的 MMLU subject）
- `lighteval/knowledge/mmlu-pro@0`
- `lighteval/knowledge/gpqa-diamond@1`
- `lighteval/knowledge/gpqa-main@0`
- `lighteval/knowledge/gpqa-extended@0`
- `lighteval/instruction-following/ifeval@0.1`
- `lighteval/instruction-following/ifbench-test@0.1`
- `lighteval/coding/livecodebench[-vN|-release-vN|-release-latest]@0`

这些 alias 是按 family 划分的显式 allowlist；任意未列入的 registry 名称都会被拒绝，
因此不能把 LiveCodeBench 伪装成 knowledge/math 来绕过 coding 隔离边界。alias 必须在
pinned LightEval `Registry` 中存在并匹配 config version；dataset、
PromptManager、task-native metric 和 `EvaluationTracker` 均由 LightEval 加载。不再传入
本地 snapshot 或 snapshot manifest。示例：

`aime24`/`aime25` 使用 LightEval 的两个 native metric，但只把 signed `pass@k:*`
作为 scoreboard primary；`avg@n:n=1` 保留在 LightEval native results/details。
`ifeval`/`ifbench-test` 使用 grouped metric，scoreboard primary 固定为
`prompt_level_strict_acc`。pinned config 没有正数 `generation_size` 时，评测端使用
`32768`；显式 `--generation-limit` 优先。
支持判断以当前 Pipeline 使用的 native GENERATIVE metric 为准；上游 config 中仅供
Inspect 入口使用的可选 `scorer` 字段不会被本 adapter 当作 judge backend。

```bash
helicopter eval run rwkv-test lighteval/math/gsm8k@0 \
  --endpoint-url http://127.0.0.1:8000/v1 \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --tokenizer-revision "$TOKENIZER_REVISION" \
  --chat-template-revision "$CHAT_TEMPLATE_REVISION" \
  --wkv-mode fp32io16 \
  --precision fp16-io-fp32-state \
  --gemm-policy fp32-accumulation \
  --launch-contract helicopter-eval-eager-v1 \
  --cot-mode cot \
  --math-repair-strategy A
```

本机 GB10 上对 `rwkv7-g1h-7.2b-20260710-ctx10240.pth` 的 eager-mode 容量扫描
选择 `max_num_seqs=512` 与 `max_concurrent_requests=1000`。固定 1024 题 workload
下，B512 的有效吞吐约为 3063 token/s，B1024 下降到约 2969 token/s；因此默认值
按吞吐平台选择，不继续为占用 unified memory 增大容量。GB10 驱动不提供独立 FB
memory 数字，不能把 system unified-memory 使用量表述为显存占用率。

CLI 只在 eval 子命令内 lazy import `helicopter_lighteval.evaluation`。服务端的
`/v1/helicopter/attestation` 必须证明 served model、checkpoint、tokenizer/chat-template、
server revision、WKV/precision/GEMM/launch contract 以及
`openai-chat`、`output-token-ids`、`terminal-reason`、`prompt-evidence` capability；
server revision 由当前 `src/infer/vllm-rwkv` submodule HEAD 派生，CLI/config 中的
`server_revision` 只作为可选一致性断言，不能替代真实 source revision；
official run 在生成前拒绝缺失或不匹配的 attestation。`--allow-non-comparable` 只能用于
明确的 sanity 检查，不能产生 official leaderboard 成绩。

### 停止、截断和数学 A/B/C

每个 completion 持续生成，直到 vLLM-RWKV 返回以下两种终止证据之一，或达到 task 的
`generation_size`：

1. token `0`，或文本包含 `\nUser:`；记录为 `stop`；
2. 生成 token 数达到上限；记录为 `length`，并计入整体 `truncation_rate`。

`vllm_rwkv.py` 只保留并校验这些证据、prompt/output token IDs、usage 和 request ID，
不重新拼接上下文，也不发送第二次生成请求。LightEval 的 prompt function 和
`PromptManager` 产生 messages，chat template 由 vLLM-RWKV server 应用。

数学结果使用同一 LightEval task-native metric：

- A：raw completion 直接交给 scorer；
- B：有未闭合 `<think>` 时补 `</think>\nTherefore...`；
- C：先执行 B，否则仅在截断 answer 时补 `\nTherefore...`。

CoT 正常闭合时不会强插 `Therefore...`。raw/scored completion 和 repair action 都写入
terminal evidence。

固定 LightEval 的 LiveCodeBench scorer 会在 evaluator 权限下执行 `exec`，不满足 coding
隔离合同；因此 coding identity 虽然来自真实 upstream registry，当前仍在 provider import
前返回 `unsupported`，不会运行 synthetic proxy、近似 scorer 或本地 sandbox。固定 revision
没有 function-calling benchmark，也没有可由当前 generation-only Pipeline 运行的 agent/tool
benchmark；`ifbench_multiturn` 需要多轮上下文，因此同样不接入。这些情况返回 `unsupported`，不创建空模块。后续必须另提带真实 upstream task、
Inspect/tool backend 和隔离合同的 change。

## 结果与发布

每次成功运行排他创建：

```text
src/eval/lighteval/results/<run-id>/
├── results/                  # LightEval native result files
├── details/                  # LightEval native detail files
├── terminal_evidence.json   # stop/truncation/raw/scored evidence
└── manifest.json             # identity/accounting/checksums/completed_at
```

失败运行只保留诊断文件，不生成可发布 manifest。`manifest.json` 由成功保存 native
results/details 后直接写入；它不是第二套 artifact/lifecycle/cache framework。

Scoreboard 发布只读取已校验的 manifest 和 terminal evidence，并投影现有严格 DTO 的六个
manifest 字段：`digest`、`identity_digest`、`accounting_digest`、
`terminal_status="completed"`、`checksums_verified=true`、带时区的 `completed_at`。
run id 位于 publication URL，不作为 ManifestEvidence 额外字段。发布请求为 gzip + Bearer
token，幂等键固定为 `publish:<manifest digest>`；网络失败后使用同一 manifest 重试：

```bash
helicopter eval publish \
  src/eval/lighteval/results/<run-id>/manifest.json \
  --scoreboard-url http://127.0.0.1:7860
```

当前后端 publication endpoint 是 `PUT /api/v1/evaluation-publications/{run_id}`；本组件
不调用已移除的 `/api/v1/run` 或 `/api/v1/runs`，也不直连数据库。

## 验证

评测组件测试覆盖固定 LightEval API smoke、OpenAI terminal evidence、stop/length 决策、
数学 A/B/C、结果目录和 manifest checksum、严格 DTO projection、HTTP auth/idempotency、
CLI lazy import 与 coding fail-closed。安装产物使用：

```bash
scripts/verify_installed_wheels.sh
```

Scoreboard server/client 与 vLLM-RWKV 产品代码保持原样；跨边界检查必须确认其路径没有
被本组件 diff 修改。

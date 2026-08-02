# LightEval 评估

产品入口只有一个：

```bash
helicopter eval --evaluator lighteval --config ./configs/eval/lighteval.toml
```

公开 CLI 会把评估委托给独立的 `.venv-lighteval`。LightEval 及其数学解析依赖不会
安装进 Verl 使用的训练 `.venv`；可用 `HELICOPTER_EVAL_PYTHON` 显式覆盖该解释器。
`--evaluator` 省略时仍默认使用 `lighteval`，兼容现有训练验证命令。

命令按配置顺序处理每个权重。发布型配置默认运行 `fp16`、`fp32io16` 两种 WKV mode。
每个 weight/mode 使用一次 LightEval 官方 `Pipeline`，评估 `benchmarks` 中全部可解析
的 task 或 superset selector。不存在于固定 LightEval 版本中的 selector 会报告为
skipped；已解析 task 的 dataset、metric、模型或发布失败则使整个命令失败。

## 配置

```toml
schema_version = 1
prompt_template = "bot"
publish = true

weights = [
  "rwkv7/model-a.pth",
  "rwkv7/model-b.pth",
]

benchmarks = [
  "mmlu",
  "gpqa:diamond",
  "gsm8k",
  "ifeval",
]
```

公共配置项为：

- `schema_version`：固定为 `1`。
- `backend`：可省略，默认 `local`；训练期或固定服务池评测使用
  `vllm_http`。
- `prompt_template`：可省略，默认 `bot`；也可选 `assistant` 或
  `function_calling`。
- `publish`：可省略，默认 `true`，结果经 Scoreboard API 入库。
- `weights`：相对私有环境变量 `WEIGHT_PATH` 的权重路径，可配置多个。
- `benchmarks`：直接写 LightEval task 或 superset selector，不维护名字映射。
- `wkv_modes`：可省略，默认同时运行 `fp16` 和 `fp32io16`。
- `result_path`：仅在 `publish = false` 时使用，指定本地 metrics JSON。

MaxRL 训练期验证使用
[`configs/eval/maxrl_math.toml`](../../configs/eval/maxrl_math.toml)。它显式设置
`backend = "vllm_http"`、`publish = false`、`wkv_modes = ["fp32io16"]`，并只列出
LightEval 原生支持的 `aime25`、`gsm8k`、`asdiv`、`math_500`。权重、结果路径和
vLLM pool manifest 由 Verl 在每次验证触发时通过环境变量提供；
该模式不会创建 Scoreboard client，不做 Scoreboard API preflight，也不会访问
后端数据库。

superset 由 LightEval 自己展开，所以配置不需要列出展开后的数百个 task。仓库默认
清单见 [`configs/eval/lighteval.toml`](../../configs/eval/lighteval.toml)；不支持的
小众 benchmark 不写入清单，当前 LightEval release 缺少的 selector 自动跳过。
产品不提供 exclude、`max_samples`、生成参数、并发、shard 或 capacity 配置。
所有解析出的 task 使用完整 evaluation split。

三个 prompt template 来自 vLLM-RWKV：

| `prompt_template` | assistant prefix | turn stop |
| --- | --- | --- |
| `bot` | `\nBot✿` | `✿` |
| `assistant` | `\n\nAssistant: ` | `\nUser:` |
| `function_calling` | `\n### Assistant` | `\n### User` |

同一 campaign 只使用一种 template。模板、stop 和实际生成参数随结果入库。

权重解析会拒绝绝对路径、`..` 越界、symlink、缺失文件和重复内容。数据库使用
SHA-256 作为权重身份，使用文件 basename 作为展示名。

## 私有环境

以下值只写入 workspace 私有的 `.env.local` 或 `.env.remote`，不能写进 TOML：

```dotenv
WEIGHT_PATH=/home/caizus/Weights
HELICOPTER_SCOREBOARD_URL=https://scoreboard.example.test
HELICOPTER_SCOREBOARD_TOKEN=replace-with-private-token
HELICOPTER_EVAL_STAGING_ROOT=/home/caizus/Projects/MachineLearning/helicopter/.tmp/eval
```

env 文件必须由当前用户所有、权限为 `0600`，且不能是 symlink。默认读取
`.env.local`；远端运行可显式指定：

```bash
helicopter eval \
  --env-file .env.remote \
  --config ./configs/eval/lighteval.toml
```

`HELICOPTER_EVAL_STAGING_ROOT` 是 LightEval 标准 results/details 的临时保存目录。
目录不存在时以 `0700` 创建；已存在时必须由当前用户所有、权限严格为 `0700`，
且不能是 symlink。不要把它配置为权重目录或共享目录。发布型评估必须配置该值；
`publish = false` 时可省略，并自动使用 `result_path` 同目录下的
`.lighteval-staging`。

### vLLM HTTP pool

`backend = "vllm_http"` 不会在 LightEval 进程中 import `vllm` 或构造第二份
`vllm.LLM`。它读取 `HELICOPTER_VLLM_POOL_MANIFEST` 指向的运行期 JSON：

```json
{
  "schema_version": 1,
  "global_step": 50,
  "wkv_mode": "fp32io16",
  "vllm_version": "0.23.1.dev0",
  "max_model_len": 10240,
  "replicas": [
    {
      "base_url": "http://10.21.60.84:36731",
      "max_concurrency": 64
    }
  ]
}
```

TOML 只选择稳定的 backend；动态 endpoint 和每 replica 容量只存在于受控运行期
manifest，不写进仓库或 `.env.remote`。manifest 必须是绝对路径下的普通文件；
endpoint 必须是无凭据、无 path/query/fragment 的 HTTP(S) origin，且不能重复。
训练器以 `0600` 创建唯一临时 manifest，外部评估结束后删除。固定推理服务也使用
同一 schema，由服务生命周期控制面生成 manifest 后复用相同评估入口。

评估启动时会对全部 replica 请求 `/health` 和 `/v1/models`；任一 replica
不可用、model id 不一致或 WKV mode 不匹配都会 fail closed。实际生成请求使用
`/v1/chat/completions`，明确传递 RWKV prompt template、Rapid sampler penalty、
stop token 和 `return_token_ids`。HTTP client 不继承 shell proxy，避免同机或内网
endpoint 被错误转发到外部代理。

每个 LightEval sample 拆为一个独立 HTTP 请求。全局调度器选择
`inflight / max_concurrency` 最小且仍有空位的 replica；所有空位占满时阻塞等待，
因此 8 个 `max_concurrency = 64` 的 replica 可以同时承载 512 个请求。返回结果按
原始 document/sample 顺序重组，不因完成顺序改变评测语义。传输错误、HTTP 429
或 5xx 最多换一个 replica 重试；其他 4xx 直接失败，避免用重试掩盖无效 sampling
contract。

## 查看计划

```bash
helicopter eval \
  --config ./configs/eval/lighteval.toml \
  --dry-run
```

dry-run 会校验配置和权重、展开 selector，并在 `publish = true` 时检查 Scoreboard
publication API，然后输出
weight SHA、resolved/skipped selector、实际 task 和执行单元数。它不会加载 dataset
或模型，也不会创建 campaign。Bearer token 始终显示为 `[REDACTED]`。

## 评估规则

每个 weight/mode 都把全部已解析 task 交给 LightEval 官方 Python API：
`EvaluationTracker`、`PipelineParameters`、model config 和 `Pipeline`。LightEval
保存标准 results JSON 与 details parquet，Helicopter 只做 RWKV 必需的 model/prompt
适配和发布。

生成固定最多 8192 个 token。本地 backend 的 `max_model_len` 使用 checkpoint
context 加 8192，capacity 由 vLLM-RWKV 根据模型、GPU 和 WKV mode 自动选择，
不接受用户覆盖。HTTP backend 使用部署 manifest 中实际的 `max_model_len` 和
`max_concurrency`，但这些值仍不属于评估 TOML 的用户参数。
`fp16` 记录 FP16 WKV state/FP16 accumulation，`fp32io16` 记录 FP32 WKV
state/FP32 accumulation。

唯一正确选项的选择题转换为生成式答案。包含多个正确选项的题目直接跳过；标准
task config 会记录 `original_num_docs`、`effective_num_docs` 和
`skipped_multiselect_docs`，后端和前端均使用实际评估题数。

## 发布入库和本地结果

`publish = true` 时，正式运行先创建 Scoreboard campaign，然后对每个 weight/mode：

1. 运行完整 LightEval Pipeline，并把标准结果写到 staging。
2. 读取标准 results/details，逐 task 请求 Scoreboard 入库。
3. 所有预期 task 入库后，请求后端原子 finalize campaign。
4. 后端确认 campaign complete 后，安全删除本次 campaign 的整个本地目录。

只有第四步完成才返回 `0`。配置、评估、网络、认证、冲突、部分入库、finalize 或
安全清理失败都会非零退出，并保留本次本地 LightEval 内容供排查。因此成功运行后
完整结果只保留在 PostgreSQL；失败运行不会因自动清理而丢失证据。新命令始终创建
新 campaign，不实现本地 manifest、自动 resume、quarantine 或结果等级。

`publish = false` 时只允许一个 weight 和一个 WKV mode。命令运行相同的 LightEval
Pipeline，从标准 results JSON 读取 native aggregate metrics，原子写入
`result_path`，随后清理 staging；整个路径不会构造 Scoreboard client，也不会向
Scoreboard 发出 HTTP 请求。

## 后端和前端

安装完整组件：

```bash
INSTALL_COMPONENTS=lighteval,scoreboard-server,scoreboard-client,dev \
  scripts/install_local.sh
```

Scoreboard server 还需要 PostgreSQL 连接和 publication token：

```dotenv
SCOREBOARD_DB_HOST=127.0.0.1
SCOREBOARD_DB_PORT=5432
SCOREBOARD_DB_USER=postgres
SCOREBOARD_DB_NAME=helicopter_scoreboard
SCOREBOARD_PUBLICATION_TOKENS={"private-token":"rwkv-eval-worker"}
```

API 启动命令：

```bash
.venv/bin/python -m uvicorn scoreboard_server.application:app \
  --host 0.0.0.0 --port 7860
```

前端构建时指定同一 API：

```bash
cd src/scoreboard-client
SCOREBOARD_API_BASE_URL=http://127.0.0.1:7860 bun run build
SCOREBOARD_API_BASE_URL=http://127.0.0.1:7860 bun run start -- -p 3000
```

普通查询只返回 complete campaign：

- `GET /api/evaluations?offset=0&limit=5000`：所有 task 的 native metrics、
  WKV mode、selector、prompt template 和诊断。
- `GET /api/evaluations/{evaluation_id}/samples?offset=0&limit=25`：完整
  Doc、reference、sample metric、completion 和 token 详情。

dashboard 展示最新 complete campaign，history 保留历史 complete campaign。
页面按 weight、WKV mode 和 LightEval tag 浏览 native metrics，不计算跨 benchmark
的自定义总分，也不再区分 official/non-official 或任何结果等级。

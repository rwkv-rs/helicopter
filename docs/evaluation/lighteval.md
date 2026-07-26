# LightEval 评估

产品入口只有一个：

```bash
helicopter eval --config ./configs/eval/lighteval.toml
```

命令按配置顺序处理每个权重，并固定运行 `fp16`、`fp32io16` 两种 WKV mode。
每个 weight/mode 使用一次 LightEval 官方 `Pipeline`，评估 `benchmarks` 中全部可解析
的 task 或 superset selector。不存在于固定 LightEval 版本中的 selector 会报告为
skipped；已解析 task 的 dataset、metric、模型或发布失败则使整个命令失败。

## 配置

```toml
schema_version = 1
prompt_template = "bot"

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

配置只有四项：

- `schema_version`：固定为 `1`。
- `prompt_template`：可省略，默认 `bot`；也可选 `assistant` 或
  `function_calling`。
- `weights`：相对私有环境变量 `WEIGHT_PATH` 的权重路径，可配置多个。
- `benchmarks`：直接写 LightEval task 或 superset selector，不维护名字映射。

superset 由 LightEval 自己展开，所以配置不需要列出展开后的数百个 task。仓库默认
清单见 [`configs/eval/lighteval.toml`](../../configs/eval/lighteval.toml)；不支持的
小众 benchmark 不写入清单，当前 LightEval release 缺少的 selector 自动跳过。
产品不提供 exclude、`max_samples`、生成参数、WKV mode、并发、shard 或 capacity
配置。所有解析出的 task 使用完整 evaluation split。

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
且不能是 symlink。不要把它配置为权重目录或共享目录。

## 查看计划

```bash
helicopter eval \
  --config ./configs/eval/lighteval.toml \
  --dry-run
```

dry-run 会校验配置和权重、展开 selector、检查 Scoreboard publication API，并输出
weight SHA、resolved/skipped selector、实际 task 和执行单元数。它不会加载 dataset
或模型，也不会创建 campaign。Bearer token 始终显示为 `[REDACTED]`。

## 评估规则

每个 weight/mode 都把全部已解析 task 交给 LightEval 官方 Python API：
`EvaluationTracker`、`PipelineParameters`、model config 和 `Pipeline`。LightEval
保存标准 results JSON 与 details parquet，Helicopter 只做 RWKV 必需的 model/prompt
适配和发布。

生成固定最多 8192 个 token。`max_model_len` 使用 checkpoint context 加 8192，
capacity 由 vLLM-RWKV 根据模型、GPU 和 WKV mode 自动选择，不接受用户覆盖。
`fp16` 记录 FP16 WKV state/FP16 accumulation，`fp32io16` 记录 FP32 WKV
state/FP32 accumulation。

唯一正确选项的选择题转换为生成式答案。包含多个正确选项的题目直接跳过；标准
task config 会记录 `original_num_docs`、`effective_num_docs` 和
`skipped_multiselect_docs`，后端和前端均使用实际评估题数。

## 强制入库和清理

正式运行先创建 Scoreboard campaign，然后对每个 weight/mode：

1. 运行完整 LightEval Pipeline，并把标准结果写到 staging。
2. 读取标准 results/details，逐 task 请求 Scoreboard 入库。
3. 所有预期 task 入库后，请求后端原子 finalize campaign。
4. 后端确认 campaign complete 后，安全删除本次 campaign 的整个本地目录。

只有第四步完成才返回 `0`。配置、评估、网络、认证、冲突、部分入库、finalize 或
安全清理失败都会非零退出，并保留本次本地 LightEval 内容供排查。因此成功运行后
完整结果只保留在 PostgreSQL；失败运行不会因自动清理而丢失证据。新命令始终创建
新 campaign，不实现本地 manifest、自动 resume、quarantine 或结果等级。

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

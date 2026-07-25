# 完整 LightEval 评估

产品入口只有一个：

```bash
helicopter eval --config ./configs/eval/lighteval.toml
```

该命令对配置中的每个权重依次运行 `fp16` 和 `fp32io16`，每种 mode 都评估当前
锁定 LightEval release 的完整默认内置 registry。task 清单在运行时通过 LightEval
公开 registry API 枚举，因此 release 增加默认 task 后会自动纳入。额外
multilingual registry、外部 custom/community 注册项不属于这个默认集合。

用户不需要也不能在配置中抄写或筛选数百个 benchmark。产品不接受 task、
benchmark、exclude、`max_samples`、生成参数、WKV mode、shard、并发或 capacity
字段。每个 task 使用完整 evaluation split，`max_samples=None`。

## 最小配置

```toml
schema_version = 1
weights = [
  "rwkv7/model-a.pth",
  "rwkv7/model-b.pth",
]
```

`--config` 的相对路径按命令调用目录解析，因此
`helicopter eval --config ./path/to/lighteval.toml` 可直接使用普通 shell 路径语义。
权重路径相对 `WEIGHT_PATH` 解析。产品拒绝绝对路径、越界、symlink、重复路径和
重复内容，并以文件 basename 作为展示名称、SHA-256 作为稳定身份。

## 私有环境

以下值只写入 workspace 私有的 `.env.local` 或 `.env.remote`，不得写入 TOML；
包含这些值的 env 文件必须由当前用户所有、权限严格为 `0600`，且不能是 symlink。
其中 Bearer token 是密钥，不会由 evaluator 写入日志、错误、manifest、标准
artifact、控制 metadata 或 publication payload；权重根、Scoreboard URL 与 staging
根会出现在 redacted dry-run/readiness 输出中：

```bash
cp .env.example .env.local
chmod 600 .env.local
```

```dotenv
WEIGHT_PATH=/home/caizus/Weights
HELICOPTER_SCOREBOARD_URL=https://scoreboard.example.test
HELICOPTER_SCOREBOARD_TOKEN=replace-with-private-token
HELICOPTER_EVAL_STAGING_ROOT=/home/caizus/Projects/MachineLearning/helicopter/.tmp/eval
```

`eval` 默认只读取明确的 `.env.local`，不会回退到通用 `.env`。远端私有文件需显式
传入：

```bash
helicopter eval \
  --env-file .env.remote \
  --config configs/eval/lighteval.toml
```

命令会在当前进程内临时叠加该私有文件的全部键，使 Hugging Face endpoint、代理、
dataset/service 凭据等 task-native 前提能被 LightEval 与 vLLM 直接读取；命令结束
或异常退出当前作用域时会恢复原进程环境。既有命令环境值优先。evaluator 不会把
私有环境整体序列化到 manifest、标准 artifacts 或发布 payload；task-native
失败只记录 task identity 与异常类型，不回显第三方异常原文。
Scoreboard HTTP 失败也只报告状态码或内部异常类型；客户端会有界读取但不会把
后端错误 body 拼进 CLI 错误，从而避免后端以转义或变形形式回显密钥。

Scoreboard server 通过 `SCOREBOARD_PUBLICATION_TOKENS` 把 token 映射成仅用于审计
provenance 的 principal，例如：

```dotenv
SCOREBOARD_PUBLICATION_TOKENS={"private-token":"rwkv-eval-worker"}
```

运行前的只读 preflight 会验证 Scoreboard 认证、staging 可写性、LightEval
`0.13.0` 和当前 repository submodule 的 editable vLLM 来源。任一条件不满足都在
加载 dataset 或模型之前失败。

`HELICOPTER_EVAL_STAGING_ROOT` 不存在时由 evaluator 以 `0700` 创建；如果目录已
存在，它必须由当前用户所有且权限已经严格为 `0700`。evaluator 不会修改已有目录
的权限。新建前最近的已有父目录也必须由当前用户所有，且不能由 group/other
写入。不要把 `/tmp`、workspace 根目录、权重根目录或其他共享目录本身配置为
staging root，应先准备一个当前用户独占的父目录，再配置其专用私有子目录。

## 安装和启动 Scoreboard

通过仓库安装器准备完整 eval、server 和 client 依赖：

```bash
INSTALL_COMPONENTS=lighteval,scoreboard-server,scoreboard-client,dev \
  scripts/install_local.sh
```

安装器会把固定 Bun 版本写入当前 workspace 的 `.venv/bin/bun`，并把
Scoreboard smoke test 所需的 Chromium 写入
`.venv/playwright-browsers`；不会依赖用户级 Bun 或 Playwright browser cache。

Scoreboard 只接受空的或 contract version 1 的 PostgreSQL 数据库；发现旧
`evaluation_result` 或未版本化 evaluation schema 时会拒绝启动，不执行隐式迁移。
server 运行环境至少需要：

```dotenv
SCOREBOARD_DB_HOST=127.0.0.1
SCOREBOARD_DB_PORT=5432
SCOREBOARD_DB_USER=postgres
SCOREBOARD_DB_NAME=helicopter_scoreboard
SCOREBOARD_PUBLICATION_TOKENS={"private-token":"rwkv-eval-worker"}
```

启动 API：

```bash
.venv/bin/python -m uvicorn \
  scoreboard_server.application:app \
  --host 0.0.0.0 \
  --port 7860
```

构建并启动前端时，把 server 基址写入进程环境；浏览器的 `/api/*` 请求由 Next
rewrite 到同一后端：

```bash
cd src/scoreboard-client
SCOREBOARD_API_BASE_URL=http://127.0.0.1:7860 bun run build
SCOREBOARD_API_BASE_URL=http://127.0.0.1:7860 bun run start -- -p 3000
```

## 先查看完整计划

```bash
helicopter eval \
  --config ./configs/eval/lighteval.toml \
  --dry-run
```

`--dry-run` 会计算权重 SHA、枚举完整 registry、应用 domain 规则并输出
weight/mode、task/module、deterministic shard 和 unknown-domain coverage。
它会进行只读 Scoreboard preflight，但不会加载 dataset/模型、创建 campaign 或写入
评估内容；token 始终显示为 `[REDACTED]`。

## 执行、续跑和退出

正式运行的固定顺序是配置中的 weight 顺序，每个 weight 按 `fp16`、`fp32io16`
执行。内部先按 LightEval module、再按稳定 task identity 确定性分为单 task
shard，使 dataset/prerequisite 失败只影响对应 task。分片只控制 dataset/Doc 的
host-memory 生命周期。同一个
weight/mode 只加载一次模型，vLLM-RWKV 根据模型规模、GPU 显存与 WKV mode 解析
4×4×2 active-capacity matrix；评估层不提供 capacity 参数。
`fp16` 记录 FP16 WKV state/FP16 accumulation，`fp32io16` 记录 FP32 WKV
state/FP32 accumulation；数据库会校验 mode 与 GEMM policy 一致。
checkpoint 文件名的 `ctx<N>` 是 prompt context 上限，不是 prompt 与 completion
共用的总预算。评估固定保留 8192 个输出 token，因此传给 recurrent RWKV7 的
`max_model_len` 为 `N + 8192`；例如 `ctx8192` 使用总长度 16384，但 prompt 仍最多
保留 8192 token。该规则固定在产品中，不能通过 TOML 覆盖。

默认 registry 也包含 Wikitext 等原生 `PERPLEXITY` task。这类 task 不是对话生成：
adapter 直接对 task 给出的原始 document query 做滚动 log-likelihood，不添加
User/Assistant template 或生成参数。窗口受 checkpoint `ctx<N>` 约束，每个 token
恰好计分一次；相应标准 detail 保存逐 token logprobs 和 output token evidence，
但不会伪装成 completion 或进入 truncation/turn-boundary 分母。

每个单 task shard 使用 LightEval 公开 `Pipeline` 与标准 results JSON/details
parquet。缺少某个 task 自身需要的 dataset、可选依赖、服务、凭据、硬件或安全
前提时，其他独立 shard
继续执行，但预期 task 不会从 campaign 中消失，命令最终非零且 campaign 保持
incomplete。

本地 manifest 只记录 digest、有序 weight SHA、完整 registry task identity
快照、backend identity 和精确 staging child，不复制 Doc、metric、completion 或
token。相同命令会恢复匹配的 incomplete campaign：
后端已确认相同 identity/digest 的 task 不会重复计算；digest 冲突立即失败并保留
本地证据。已完成 campaign 不会作为下一次运行的 cache。
若后端已 finalize、但本地仍有匹配 manifest，说明上一次命令只在本地清理前中断；
续跑完成精确清理后直接成功返回，不会误建新 campaign。只有已经没有 manifest 的
后续新调用才会创建新的完整评估 campaign。

无法解析或与当前 config、权重、registry、domain rules 或 eval contract 不匹配的
普通 manifest 会被移动到 `campaigns/quarantine/`；隔离只移动 manifest，不读取、
复用、覆盖或删除它原先指向的 run 内容。若后端随后恢复到同一个 campaign id，而
本地存在没有匹配 manifest 登记的非空 run 目录，命令会保留该目录并立即失败，要求
人工审计，绝不猜测其归属。

退出码：

- `0`：所有 weight/mode/task 已入库、campaign 已 finalize，内容 staging 已清理。
- 非 `0`：配置、preflight、评估、publication、finalize 或安全清理未完成。

## 强制入库与 DB-only 清理

Scoreboard publication 是成功条件，不是可选后处理。evaluator 通过 Bearer HTTP
发送 gzip canonical JSON，并以 canonical SHA-256 作为幂等键。server 严格验证
campaign/task 归属、完整 Doc/metric/multi-completion/input-output tokens，并从
raw completion 与 output tokens 重算 truncation 和 turn-boundary diagnostics。
无 completion text 的 LOGPROBS/PERPLEXITY row 会改按有限 logprobs、argmax 与
output token 数量对齐校验；生成行的 `text_post_processed` 则必须与 raw `text`
一一对应。

只有后端明确返回相同 task identity/digest 的 `created` 或 `unchanged` 后，runner
才删除该 shard 在 manifest 中记录的精确 child。网络错误、认证错误、冲突、部分
确认或未知状态都会保留内容。若后端 commit 后进程中断，下一次运行先查询后端
digest，一致后才补写本地确认并清理。

每个 weight/mode 的模型 runtime 固定为
`runtime/<weight_sha256>/<wkv_mode>`，并在模型构造前写入 manifest。正常模型
cleanup 完成后才移除该记录；若进程中断，续跑只清理这个已登记且重新通过
campaign-child 与 symlink 边界校验的目录，不扫描或猜测其他路径。
模型 cleanup、runtime 安全删除或 manifest 持久化失败时，命令保留登记并立即
非零退出，不会继续加载下一个 weight/mode。

全部 task 确认后，server 原子 finalize campaign；runner 删除标准
results/details、失败 attempt、模型 runtime 与 manifest。
`HELICOPTER_EVAL_STAGING_ROOT/control`
只保留不含评估内容和密钥的 campaign 摘要。成功评估的 Doc、metric、completion
与 tokens 最终只存在于 PostgreSQL。

## 查询与前端展示

普通查询只返回 complete campaign：

- `GET /api/evaluations?offset=0&limit=5000`：weight、WKV mode、module、
  primary domain、全部 native aggregates、诊断和 campaign provenance；响应中的
  `next_offset` 与 `generated_at` 分别用于下一页的 `offset` 与
  `completed_before`，前端会在同一 complete-campaign 快照内自动拉完全部页。
- `GET /api/evaluations/{evaluation_id}/samples?offset=0&limit=25`：按稳定
  evaluation identity 分页读取全部 sample；可加
  `outcome=correct|incorrect|unanswered|undetermined`。

dashboard 只取最新 complete campaign，按 weight × WKV mode 展开其中每个
benchmark task，并支持 module/domain 筛选；history 页面保留全部 complete
campaign。两者原样展示 native metrics、缺失 mode，并可按稳定 evaluation identity
打开 Doc/reference、sample metric、多 completion、reasoning/answer、input/output
tokens、logprobs/argmax、truncation 与 turn-boundary。domain 只用于组织和筛选；
数值始终来自各 task 的 native metric，不跨 benchmark 合成总分。

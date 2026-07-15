# Helicopter

Helicopter 提供 RWKV serving、训练与可审计评估工作流。评估实现只有一个 owner：
`src/eval/lighteval`。它是独立 distribution `lighteval-runner`，import package 为
`lighteval_runner`；`src/eval` 只是未来容纳 `lm-eval-harness` 等框架的物理容器。

## 目录职责

```text
src/cli/helicopter_cli/       # 薄 CLI；eval 子命令 lazy import application
src/eval/lighteval/           # snapshot、task family、provider、score、artifact、HTTP publication
src/infer/vllm-rwkv/          # OpenAI endpoint 与 provider attestation
src/scoreboard-server/        # /api/v1、认证、事务、migration、DB persistence
src/scoreboard-client/        # 只使用 OpenAPI generated types 的 Next.js UI
src/train/                    # rwkv-lm 与 verl-rwkv
```

禁止从 CLI、评估端、前端或维护脚本连接数据库。唯一链路是：

```text
browser / evaluator -> scoreboard /api/v1 -> repository -> SQLite or PostgreSQL
```

后端不可用时，评估 artifact 保留，显式 publication 返回失败且 CLI 非零退出；没有
DB fallback、双写或本地隐式 schema 初始化。

## 安装

环境只能由 `helicopter-dev` 建立。控制仓库中运行：

```bash
./bin/helicopter-dev env sync feat-lighteval --target local --components lighteval
```

该 component 只同步根 CLI/eval 环境。scoreboard 后端与前端是独立组件，需要时显式
选择，避免评测安装隐式携带数据库或 UI 依赖：

```bash
./bin/helicopter-dev env sync feat-lighteval --target local \
  --components scoreboard-server,scoreboard-client
```

GPU/native 环境使用同一命令的 remote target，并按 workload 同时选择
`vllm-rwkv,lighteval`；不要手工 `pip install`。

## Provider attestation

正式评估只接受提供完整 attestation 的 endpoint。server lifecycle 由
`helicopter infer --serve-evaluation` 所属进程管理；evaluator 只连接既有 endpoint，
不会启动、复用判定或停止不属于它的进程。启动 vLLM 时传入
`--helicopter-attestation-json`，JSON 必须包含：

- model served name、checkpoint SHA-256、tokenizer revision、chat-template revision；
- server revision、WKV mode、precision、GEMM policy、launch contract；
- `openai-chat`、`output-token-ids`、`terminal-reason`、`prompt-evidence` 四项 capability。

合同通过 `GET /v1/helicopter/attestation` 暴露。official run 任一字段缺失或不匹配
都会在生成前失败；只有显式 `--allow-non-comparable` 才会降级为 proxy。

## 运行评估

正式 runner 只消费已固定 revision、校验 SHA-256 的本地只读 JSONL/Parquet snapshot，
并要求同时提供由 `helicopter-dev datasets fetch` 产生的 manifest。canonical registry
当前直接复用固定 LightEval commit 的 GSM8K、MATH-500 与 MMLU task-native
prompt/scorer；另外提供显式 opt-in、永不进入 official leaderboard 的
`helicopter-proxy/function-calling/exact-json@1` 与
`helicopter-proxy/coding/python-stdio@1`，二者使用 wheel 内不可变资源。示例：

```bash
helicopter eval run rwkv-test lighteval/math/gsm8k@0 \
  --snapshot /datasets/gsm8k-test.jsonl \
  --snapshot-manifest /datasets/gsm8k-test.jsonl.manifest.json \
  --snapshot-sha256 "$SNAPSHOT_SHA256" \
  --endpoint-url http://127.0.0.1:8000/v1 \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --tokenizer-revision rwkv-tokenizer-v1 \
  --chat-template-revision rwkv-chat-v1 \
  --server-revision "$SERVER_REVISION" \
  --wkv-mode fp32io16 \
  --precision fp16-io-fp32-state \
  --gemm-policy fp32-accumulation \
  --launch-contract helicopter-eval-v1 \
  --cot-mode cot \
  --math-repair-strategy A \
  --max-samples 2
```

`--generation-limit` 是显式 override，并记录 `cli` provenance；未传时使用 task-native
上限，不存在统一 512 的静默覆盖。`--max-samples` 或 generation override 会把本次
资格降为 `sanity`，不能冒充完整 official run。当前生成 cache 明确禁用，manifest
记录 `cache-disabled-v1`、完整 namespace/key provenance 与禁用原因，不读取或迁移旧弱
cache key。endpoint API key 只从
`--endpoint-api-key-env` 指定的环境变量读取，不出现在 argv 或 artifact。

### 停止、截断与数学 A/B/C

每个样本持续生成，直到出现 token `0`、文本 `"\nUser:"`，或达到最终 generation
limit。前两者记为 `stop`，达到上限记为 `length/truncated`。raw completion 永久不改；
截断率分母只包含具有 generation evidence 的样本。

- A：所有结果直接交给 task-native scorer。
- B：仅当 `<think>` 未闭合时补 `</think>\nTherefore...`。
- C：先执行 B；否则若 answer 被截断，再补 `\nTherefore...`。

CoT 正常闭合时不会强插 `Therefore...`，也不会进行第二次生成。repair strategy 属于
完整 run identity，不同策略不会聚合成同一 leaderboard row。

## Artifact 与状态

每次运行排他创建 `results/lighteval/<run-id>`。sample evidence 区分 raw/derived text，
保存 output token ids/count、terminal reason、attempt、cache/config/provider provenance 与
typed error。文件以 temp + fsync + rename 提交，最后写 `manifest.json`；manifest 包含完整
identity、accounting、checksum 和 terminal status，禁止 mtime discovery 或跨 run 合并。

run 状态为 `completed | partial | failed | invalid | cancelled`。只有 provider、accounting、
aggregation 和 manifest gate 全部闭合，且存在真实 scored sample 时，才能成为 completed。

## Scoreboard

后端配置：

```text
SCOREBOARD_DATABASE_URL=postgresql://...
SCOREBOARD_CORS_ORIGINS=https://scoreboard.example
SCOREBOARD_AUTH_TOKENS={"publisher-token":{"subject":"eval-prod","roles":["publisher"]},"admin-token":{"subject":"ops","roles":["admin"]}}
```

`SCOREBOARD_DATABASE_URL` 只由 scoreboard persistence 读取。应用启动不建表；首次及后续
migration 均通过认证 HTTP API：

```bash
curl -X POST http://127.0.0.1:7860/api/v1/admin/migrations \
  -H "Authorization: Bearer $SCOREBOARD_ADMIN_TOKEN"
```

公开 endpoint 只有 `/api/v1/health`、`/api/v1/meta`、`/api/v1/leaderboard`。history、
raw sample evidence、write 与 admin endpoint 均需授权；旧 `/api/*` 返回 404。create 使用
publisher-scoped idempotency key，resume 使用 revision/`If-Match`，ingest 在一个事务中
写 sample、aggregate、metric projection、receipt 和 terminal state。

发布时添加 `--scoreboard-url`，token 从 `SCOREBOARD_TOKEN`（或
`--scoreboard-token-env` 指定变量）读取：

```bash
helicopter eval run ... --scoreboard-url http://127.0.0.1:7860
```

发布 receipt 与评测 manifest 分离并原子写入；网络失败后只从已校验的 committed
manifest 重试，不重新发现或合并结果：

```bash
helicopter eval publish results/lighteval/<run-id>/manifest.json \
  --scoreboard-url http://127.0.0.1:7860
```

数据库唯一约束是 `(publisher_subject, idempotency_key)`：同一 publisher 的同一请求
重放返回 unchanged，不同 payload 返回 conflict；同一 immutable benchmark identity
允许存在多个历史 run，leaderboard 查询只投影最新的 eligible completed run。

scoreboard-client 的 `lib/generated/schema.ts` 由后端 OpenAPI 生成：

```bash
bun run --cwd src/scoreboard-client generate:openapi
bun run --cwd src/scoreboard-client typecheck
```

## 验证

所有命令须经 `helicopter-dev lock run`。核心验收覆盖：

- stop/truncation 与 A/B/C 决策表；
- immutable snapshot、identity/cache invalidation、task-native parity；
- provider attestation、sandbox、atomic artifact、typed failure；
- scoreboard auth、CORS、idempotency、CAS、resume attempt、transaction rollback、migration drift；
- OpenAPI generated client、frontend typecheck/build 与全仓 DB boundary scan。

安装产物还必须通过 `scripts/verify_installed_wheels.sh`：base wheel 不得导入 eval，
eval/full wheel 必须能导入固定 LightEval adapter 与 bundled proxy assets。PostgreSQL
契约由 `src/scoreboard-server/scripts/test_postgres.sh` 在临时数据库中验证并自动清理。

coding proxy 使用 Linux Landlock 限制可见文件树，以 seccomp 拒绝网络及宿主控制
syscall，并叠加 CPU、内存、进程、输出和 wall-time 上限；任一内核能力不可用时评测
显式失败，不降级为主进程执行。签约远端的完整小样本矩阵由
`scripts/verify_eval_acceptance.py` 执行，输出目录和 vLLM-RWKV server revision 必须显式
传入，结果包含原始 stop 探针、attestation、各 family manifest 与聚合核验记录。

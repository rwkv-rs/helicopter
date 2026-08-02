# lm-eval-harness RWKV-vLLM 后端

该适配器使用已经运行的 RWKV-vLLM HTTP pool，不会在评估进程中加载权重或创建
第二份 vLLM engine。它实现 lm-eval 模型后端的三个标准请求：
`loglikelihood`、`loglikelihood_rolling` 和 `generate_until`。因此可以直接运行
HellaSwag、ARC、MMLU、TruthfulQA、GSM8K、IFEval、WikiText 等常见 harness task。

```bash
helicopter eval \
  --evaluator lm-eval \
  --config configs/eval/lm_eval.toml \
  --dry-run

helicopter eval \
  --evaluator lm-eval \
  --config configs/eval/lm_eval.toml
```

`lm-eval` 固定为 `0.4.12`，安装在独立的 `.venv-lm-eval`。可通过
`HELICOPTER_LM_EVAL_PYTHON` 覆盖解释器路径。LightEval 仍是
`helicopter eval` 的默认 evaluator，现有 MaxRL 验证不受影响。

## 配置

```toml
schema_version = 1
backend = "vllm_http"

tasks = ["arc_easy", "hellaswag", "mmlu", "gsm8k", "wikitext"]
output_dir = ".tmp/eval/lm-eval"
batch_size = 64
eot_token_id = 0
max_gen_toks = 2048
log_samples = false
```

- `tasks` 接受 lm-eval task、group、tag 和 glob pattern；每个 selector 必须至少
  匹配一项。task 定义、few-shot、过滤器和指标仍完全由 lm-eval 提供。
- `output_dir` 可为绝对路径，或相对执行命令时的项目目录；环境变量引用使用
  `${VARIABLE}` 形式。
- `batch_size` 控制一次提交给 HTTP pool 的评分窗口数量，pool manifest 中每个
  replica 的 `max_concurrency` 继续作为实际并发上限；默认示例使用 `64`。
- `eot_token_id` 默认是 RWKV 的 `0`，用于为空上下文和首个 rolling window 提供
  条件 token。
- `max_gen_toks` 是 task 未指定生成长度时的默认值。task 自己的
  `max_gen_toks` 或 `max_new_tokens` 优先；提示词会从左侧截断，为输出保留空间。
- `limit` 是可选的正整数，只用于本地 smoke test；`publish = true` 时配置解析会
  直接拒绝 `limit`，防止将抽样结果发布成完整评测。
- `log_samples` 默认关闭；启用时原始 lm-eval sample 信息会进入结果文件。

生成请求支持 harness 常用的 `until`、`max_gen_toks`、`max_new_tokens`、
`do_sample`、`temperature`、`top_p`、`top_k`、`min_p` 和 `seed`。当前后端只支持
单路生成，task 若要求 `num_beams > 1` 会明确失败，而不是静默改变评估语义。
`do_sample = false` 会编码为 `temperature = 1.0, top_k = 1`：它仍是逐 token
argmax，但同时兼容 RWKV-vLLM 的 rapid sampler（该 sampler 不接受
`temperature = 0`）。

## HTTP 协议

运行前必须在私有 `.env.local` 或 `.env.remote` 中提供绝对路径：

```dotenv
HELICOPTER_VLLM_POOL_MANIFEST=/run/helicopter/vllm-pool.json
```

manifest schema 与 LightEval HTTP backend 相同。启动时会访问每个 replica 的
`/health` 和 `/v1/models`，确保所有 endpoint 可用并且只服务同一个 model id。

评估使用以下接口：

1. `/tokenize` 对 harness 生成的原始 prompt、context 和 continuation 分词，固定
   `add_special_tokens = false`；RWKV endpoint 返回的首个 EOT/BOS token 会被移除。
2. `loglikelihood` 和 `loglikelihood_rolling` 通过 `/v1/completions` 接收 token
   IDs，并设置 `echo = true`、`max_tokens = 0`、
   `logprobs = 1`、`prompt_logprobs = 1` 和 `return_token_ids = true`。
3. `generate_until` 同样使用 `/v1/completions`，传入左截断后的 prompt token
   IDs、task 的 stop sequences 和采样参数；服务端和适配器都会执行 stop 截断。
4. 适配器验证评分返回 token 完全一致，忽略首 token 的 `null` logprob，只累加
   continuation 对应位置的自然对数概率。后续 rolling window 不以 EOT 开头时，
   RWKV-vLLM 自动补入的 EOT 会在返回结果中被正规化，避免 token 和 logprob 错位。

HTTP client 不继承 shell proxy，避免本机或内网服务被全局代理转发。HTTP 429、
5xx 和传输错误最多切换一个 replica 重试；其他 4xx 直接失败。

RWKV-vLLM 会为 decoder prompt 内部增加一个 token，并且对
`echo + max_tokens=0` 的请求仍保留至少一个生成位置。因此适配器暴露的有效长度为
manifest `max_model_len - 2`，rolling window 再保留一个条件 token。

默认保持 lm-eval base-model 语义：harness 构造的 prompt 不会被适配器再次套一层
chat template。这一点对基于 continuation 概率的多选指标很重要，也使结果能与
lm-eval 的其他 causal LM 后端直接比较。

## Qwen3.5 对齐套件

`configs/eval/lm_eval_qwen35.toml` 将任务名对齐到 Qwen3.5 模型卡公开的语言评测
表：`MMLU-Pro`、`MMLU-Redux`、`C-Eval`、`GPQA`、`IFEval` 和 `MMMLU`。
固定 selector 如下：

| Qwen 表中名称 | lm-eval 0.4.12 selector | 本地协议 |
| --- | --- | --- |
| MMLU-Pro | `mmlu_pro` | 5-shot、确定性生成、exact match |
| MMLU-Redux | `mmlu_redux_generative` | 0-shot、确定性生成、exact match |
| C-Eval | `ceval-valid` | validation split、多选概率 |
| GPQA | `gpqa_diamond_cot_zeroshot` | Diamond、0-shot CoT、exact match |
| IFEval | `ifeval` | 0-shot、确定性生成、严格/宽松指令指标 |
| MMMLU | `mmmlu` | 14 种语言、0-shot、多选概率 |

运行 RWKV 对齐套件：

```bash
helicopter eval \
  --evaluator lm-eval \
  --config configs/eval/lm_eval_qwen35.toml \
  --dry-run

helicopter eval \
  --evaluator lm-eval \
  --config configs/eval/lm_eval_qwen35.toml
```

Qwen3.5 官方没有 1.5B 型号。最接近的官方尺寸是
[`Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) 和
[`Qwen3.5-2B-Base`](https://huggingface.co/Qwen/Qwen3.5-2B-Base)，因此 1.5B RWKV
应以 0.8B/2B 作为参数量下界和上界，不能标记为“同参数量”。精确 1.5B 对照只能
选择旧代模型，例如 Qwen2.5-1.5B，而不能再称为 Qwen3.5 对齐。

截至 2026-08-01，Qwen 官方模型卡给出的后训练模型参考分如下：

| 模型与模式 | MMLU-Pro | MMLU-Redux | C-Eval | GPQA | IFEval | MMMLU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B non-thinking | 29.7 | 48.5 | 46.4 | 未报告 | 52.1 | 34.1 |
| Qwen3.5-0.8B thinking | 42.3 | 59.5 | 50.5 | 11.9 | 44.0 | 44.3 |
| Qwen3.5-2B non-thinking | 55.3 | 69.2 | 65.2 | 未报告 | 61.2 | 56.9 |
| Qwen3.5-2B thinking | 66.5 | 79.6 | 73.2 | 51.6 | 78.6 | 63.1 |

来源：[`Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) 和
[`Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) 官方模型卡。这些分数使用
后训练模型、chat template、thinking/non-thinking 模式及官方生成参数；Qwen 没有
公开足以逐项复现整张表的完整 harness 配置。因此它们只能作为外部参考，不能与
本项目默认 base-model 结果直接相减。

可比较的本地结果必须同时固定：模型类型（优先 Base）、`lm-eval==0.4.12`、上述
selector、task version、few-shot、无 chat template、确定性解码、数据 revision、
`limit = null` 和主指标。尤其不能把默认套件的普通 `mmlu` 分数填入 Qwen 表的
`MMLU-Pro` 或 `MMLU-Redux` 列。`SuperGPQA` 也不是 `GPQA Diamond`，当前配置不
用后者冒充前者。

## 输出

成功运行后，`output_dir` 中包含：

- `results.json`：lm-eval `simple_evaluate` 的完整可序列化结果。
- `summary.json`：稳定的项目级摘要，包含 lm-eval 版本、model id、global step、
  WKV mode、上下文长度、task 版本和 metrics。

默认配置仍只写本地，不创建 Scoreboard campaign。每个执行单元会写出
`results.json`、`summary.json`、`artifacts.json`，启用 `log_samples` 时还会按 task
写入 `samples/*.json`。WikiText-only 运行仍可使用
`configs/eval/lm_eval_ppl.toml`。

## 生产 campaign

`configs/eval/lm_eval_campaign.toml` 提供与 LightEval campaign 对齐的发布模式：

- `weights` 中的路径必须位于 `WEIGHT_PATH` 下，禁止绝对路径、`..` 和 symlink；
  runner 会计算每份权重的 SHA-256，并拒绝内容重复的权重。
- `wkv_modes` 在发布模式必须同时包含 `fp16` 和 `fp32io16`。
- `pool_manifests` 按 weight-major、随后按 `wkv_modes` 顺序一一对应执行矩阵。
  每份 manifest 必须声明与权重文件一致的 `weight_sha256`、
  `weight_display_name` 和对应 WKV mode。
- group 和 tag 在 campaign 合同中递归展开为叶子 task，但执行时仍由 lm-eval
  原生 selector 负责加载和评分。
- dry-run 会预检全部 pool 和 Scoreboard 合同，并输出完整 execution units 与
  expected task 数量；不会创建 campaign。
- 正式运行先创建 `lm-eval-campaign-v1`，逐矩阵单元运行并发布
  `lm-eval-task-v1`，全部 task 到齐后才 finalize。重复内容使用 canonical digest
  幂等写入，冲突内容失败关闭。

生产 manifest 在原有字段外必须包含：

```json
{
  "weight_sha256": "<64 lowercase hex>",
  "weight_display_name": "model.pth"
}
```

发布还要求 `HELICOPTER_SCOREBOARD_URL` 和
`HELICOPTER_SCOREBOARD_TOKEN`。Scoreboard 同时接受原有 LightEval v3/v2 合同和
lm-eval v1 合同，历史 LightEval API 保持兼容。

```bash
export WEIGHT_PATH=/absolute/path/to/weights
export LM_EVAL_CAMPAIGN_WEIGHT=model.pth
export LM_EVAL_FP16_POOL_MANIFEST=/run/helicopter/lm-eval-fp16.json
export LM_EVAL_FP32IO16_POOL_MANIFEST=/run/helicopter/lm-eval-fp32io16.json
export HELICOPTER_SCOREBOARD_URL=https://scoreboard.example
export HELICOPTER_SCOREBOARD_TOKEN=replace-with-private-token

helicopter eval \
  --evaluator lm-eval \
  --config configs/eval/lm_eval_campaign.toml \
  --dry-run
```

## 安装

```bash
INSTALL_COMPONENTS=lm-eval,dev scripts/install_local.sh
```

`lm-eval` 依赖组固定安装 `lm-eval[ifeval,longbench]==0.4.12` 和
`datasets==3.6.0`。前者把 IFEval 规则校验依赖（`langdetect`、`immutabledict` 和
`nltk`）以及 LongBench 的 `jieba`、`fuzzywuzzy`、`rouge` 带入独立环境；后者保留
CMMLU 等上游 dataset-script 任务所需的加载能力（`datasets>=4` 已移除该能力）。
部署时不应改为只安装基础 `lm-eval` 包，也不能放宽 datasets 上界。

## 能力补充套件

`configs/eval/lm_eval_capabilities.toml` 只选择默认 LightEval 清单没有覆盖的任务，
避免重复消耗算力：

| 维度 | lm-eval selector | 协议与主指标 |
| --- | --- | --- |
| 阅读理解 | `race` | high-school split，multiple-choice accuracy |
| 翻译 | `wmt14-en-fr` | English to French，BLEU/TER/chrF |
| 语言建模 | `lambada_openai` | last-word accuracy/perplexity |
| 句法语言学 | `blimp` | 67 个最小对组任务，group accuracy |
| 大海捞针类检索 | `longbench_passage_retrieval_en` | 30 段 Wikipedia 中定位目标摘要，retrieval score |

该套件固定 `lm-eval==0.4.12`、无 chat template、完整 evaluation split、
`limit = null`、确定性生成，并保留 sample artifacts。Qwen3.5 对标必须使用同一组
selector、相同 task revision 和相同最大输入长度；本机 1.5B RWKV 对应的官方
Qwen3.5 参数量包络仍是 0.8B Base 与 2B Base，而不是不存在的 1.5B 型号。

### 与 LightEval 去重

去重的唯一基准是 `configs/eval/lighteval.toml` 中的既有评测清单。能力补充套件
不会再次运行以下 selector：

| LightEval 维度 | 已纳入 LightEval 清单的 selector |
| --- | --- |
| 知识与常识 | `mmlu`、`mmlu_pro`、`mmlu_redux_2`、`mmlu_sr_question_answer`、`gpqa:diamond`、`gpqa:main`、`arc:challenge`、`arc:easy`、`hellaswag`、`bigbench_hard`、`agieval`、`truthfulqa:mc`、`winogrande`、`openbookqa`、`commonsenseqa`、`ceval_zho_mcf`、`kmmlu`、`med_qa`、`med_mcqa` |
| 数学与推理 | `gsm8k`、`math_500`、`aime24`、`aime25`、`olympiad_bench`、`minerva_math`、`svamp`、`beyond_aime`、`brumo25`、`hmmt_feb_2025`、`math_odyssey`、`comp_math_24_25`、`gaokao_2023_english`、`answer_judge`、`simpleqa_verified` |
| 代码 | `humaneval`、`humaneval_cn`、`humaneval_fix`、`humaneval_plus`、`mbpp`、`mbpp_plus`、`lcb:codegeneration` |
| 指令遵循 | `ifeval`、`ifbench_test`、`ifbench_multiturn` |

这里的“已纳入”表示 selector 属于 LightEval campaign 合同；实际执行时，固定
LightEval 版本无法解析的 selector 会被明确记录为 `skipped`，不能记作成功出分。
仓库测试会解析两份 TOML 并断言 selector 集合不相交，防止后续维护时重新引入
重复任务。

### 用户清单增量

`configs/eval/lm_eval_catalog_delta.toml` 是用户补充清单经过两层过滤后的精确增量：
先去除上表中已经归 LightEval 管理的项目，再去除固定 `lm-eval==0.4.12` 任务注册表
不存在的项目。此前完成的能力补充套件（RACE、WMT14、LAMBADA、BLiMP、LongBench
检索）与这份补充清单没有重叠。

| 处理结果 | 用户清单项目 | lm-eval selector / 原因 |
| --- | --- | --- |
| 本轮正式评测 | GPQA-Extended | `gpqa_extended_zeroshot`，仅 Extended，0-shot multiple-choice `acc/acc_norm` |
| 本轮正式评测 | CMMLU | `cmmlu`，67 个学科，0-shot multiple-choice，按样本量聚合 `acc/acc_norm` |
| 不属于固定 lm-eval | AMC23 | `lm-eval==0.4.12` 注册表和任务源码均无该任务 |
| 不属于固定 lm-eval | SWE-bench Verified、Multilingual、Pro | `lm-eval==0.4.12` 注册表和任务源码均无这些任务 |

因此本轮 campaign 必须且只能解析为 `gpqa_extended_zeroshot` 与 `cmmlu`。不得用
自建 YAML、外部 runner 或名称近似任务填补缺失项；若以后升级 lm-eval，需要单独
评审协议和数据版本，不能静默改变本次结果合同。GPQA 数据集在 Hugging Face 上有
访问条款，正式运行前必须接受条款并提供只读 token；镜像只能解决网络问题，不能
绕过授权。

WMT14 同时报告 BLEU、chrF 和 TER。BLEU/chrF 越高越好，TER 越低越好；
lm-eval 0.4.12 的上游 WMT YAML 未声明 TER 的方向，因此运行时会出现默认方向警告，
但不会改变 TER 数值。报告和对比必须按“越低越好”解释 TER。

2026-08-02 的 RWKV7 1.5B、Qwen3.5-0.8B Base 和 Qwen3.5-2B Base 完整实测、
运行协议、stderr、样本数及产物 SHA-256 见
[`lm_eval_capability_results.md`](lm_eval_capability_results.md)；对应机器可读清单为
[`lm_eval_capability_results.json`](lm_eval_capability_results.json)。

用户清单过滤后的原生 lm-eval 增量执行状态与 CMMLU 完整对标结果见
[`lm_eval_catalog_delta_results.md`](lm_eval_catalog_delta_results.md)；对应机器可读合同为
[`lm_eval_catalog_delta_results.json`](lm_eval_catalog_delta_results.json)。GPQA-Extended
在没有已授权只读 Hugging Face token 时明确记为 `blocked`，不会记作测评成功。

`--dry-run` 会校验配置、selector、manifest、所有 replica 和 lm-eval 版本，并输出
解析后的 task/group 类型及 output type，但不会下载数据集或执行评分。

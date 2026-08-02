# lm-eval 能力补充套件结果

评测日期：2026-08-02。本文记录 `configs/eval/lm_eval_capabilities.toml` 的完整
evaluation split 结果。该套件与 `configs/eval/lighteval.toml` 的 selector 清单严格
不相交，覆盖阅读理解、翻译、语言建模、句法语言学和长上下文检索。

Qwen3.5 没有 1.5B 型号，因此使用 0.8B Base 和 2B Base 分别作为参数量下界与
上界。三种模型都完成相同的 76,401 条 evaluation sample；机器可读结果见
`docs/evaluation/lm_eval_capability_results.json`。

## 对齐协议

| 项目 | 固定值 |
| --- | --- |
| evaluator | `lm-eval==0.4.12` |
| task selectors | `race`、`wmt14-en-fr`、`lambada_openai`、`blimp`、`longbench_passage_retrieval_en` |
| prompt | base-model 原生 prompt，不使用 chat template |
| context | `max_length=16382` |
| decoding | task 原生确定性配置 |
| dataset | 完整 evaluation split，`limit=None` |
| samples | 每个 task 都启用逐样本落盘 |
| offline | 模型和数据准备完成后启用三个 Hugging Face offline 开关 |

RWKV 使用本地 `rwkv-vllm` HTTP 后端、FP16 WKV；Qwen 使用 lm-eval 原生 `hf`
后端、BF16。不同 batch size 只用于适配 8GB 显存，不改变 prompt、输出或指标语义。
WMT 的 BLEU、chrF 和 TER 都保留 lm-eval 默认 100 次 bootstrap；TER 越低越好。

## 已完成结果

“RWKV 优势”按指标方向统一为正数表示 RWKV 更好：高优指标为
`RWKV - Qwen`，低优指标为 `Qwen - RWKV`。括号内为 stderr。

| 能力 / 指标 | 方向 | 样本数 | RWKV7 1.5B | Qwen3.5-0.8B | RWKV vs 0.8B | Qwen3.5-2B | RWKV vs 2B |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RACE accuracy | ↑ | 1,045 | 0.363636 (0.014888) | 0.355024 (0.014810) | +0.008612 | 0.384689 (0.015057) | -0.021053 |
| WMT14 BLEU | ↑ | 3,003 | 17.202313 (0.387524) | 14.989362 (0.375330) | +2.212952 | 27.766492 (0.377891) | -10.564178 |
| WMT14 chrF | ↑ | 3,003 | 40.641350 (0.434106) | 39.576008 (0.422961) | +1.065343 | 53.971673 (0.343640) | -13.330322 |
| WMT14 TER | ↓ | 3,003 | 78.743663 (0.854241) | 85.782195 (1.310768) | +7.038532 | 61.718200 (0.532141) | -17.025463 |
| LAMBADA accuracy | ↑ | 5,153 | 0.662721 (0.006587) | 0.507665 (0.006965) | +0.155055 | 0.583350 (0.006869) | +0.079371 |
| LAMBADA perplexity | ↓ | 5,153 | 4.844395 (0.110354) | 11.664836 (0.357653) | +6.820441 | 7.220394 (0.189645) | +2.375999 |
| BLiMP group accuracy | ↑ | 67,000 | 0.816463 (0.001338) | 0.811642 (0.001352) | +0.004821 | 0.838194 (0.001275) | -0.021731 |
| LongBench retrieval score | ↑ | 200 | 0.045000 (0.014695) | 0.045000 (0.014695) | 0.000000 | 0.085000 (0.019769) | -0.040000 |

RWKV 在全部 7 个非 LongBench 指标上领先 0.8B Base，并在 LongBench 与其持平；
面对 2B Base，RWKV 仍在 LAMBADA accuracy 和 perplexity 上领先，但在 RACE、三项
翻译指标、BLiMP 和 LongBench 上落后。LongBench 的绝对分数仍低：RWKV/0.8B 为
4.5%，2B 为 8.5%。不同参数量之间只报告逐指标差异，不计算跨任务总分。

## 模型与产物身份

| 模型 | 精确身份 | 运行后端 |
| --- | --- | --- |
| RWKV7 1.5B | `rwkv7-g1i_preview5445-1.5b-20260729-ctx16384.pth`，SHA-256 `22fe129988f6e98480b344075597259a13ae4201c1d8dedf987246772e613586` | 本地 RWKV-vLLM，`rwkv7-g1i-1.5b`，FP16 WKV |
| Qwen3.5-0.8B Base | revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | lm-eval `hf`，BF16 |
| Qwen3.5-2B Base | revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` | lm-eval `hf`，BF16 |

RWKV 标准产物位于
`.tmp/eval/lm-eval-capabilities-20260802/rwkv-full/`，其中：

- `results.json`: `150f4481eb25a89ed31fb3408da6ea2ccb96693495be5da78393a98a138eda63`
- `summary.json`: `41b45c3cc4805c4c81458ec9e7976bdb6c495c6a077f6e84129cf35376075c27`
- `artifacts.json`: `c4e12ec23004d67ed96b3c5c9863ca9e0ed3c55ba2d5718f5561a6360270af6c`

Qwen3.5-0.8B 的五个 lm-eval 原生结果文件位于
`.tmp/eval/lm-eval-capabilities-20260802/qwen08-full/`，SHA-256 分别为：

- RACE: `7f50e67a6fb5930978621549db7ad4ebb909fdc5463aa02462f4d2956434f4f0`
- WMT14: `5021eaef87395f34f860c4bfef3d8bebb4376289f2db753fa8f784576479f33d`
- LAMBADA: `2af51993678f618b0c144403ac343bf316c754bd829eec8f3d2279d13600be54`
- BLiMP: `6189fe795adcd78f13371569cd5e116a69f512f2ca82b38736fda890d6fb8391`
- LongBench: `5f3d6d9ec3d4d9edb1bf5794a1b92001da2500d71ef032520dda85f2b1ebc794`

Qwen3.5-2B 的五个结果文件位于
`.tmp/eval/lm-eval-capabilities-20260802/qwen2b-full/`，SHA-256 分别为：

- RACE: `e58255ad76429fbd476dee714fae6c86bbdf730ae5f008d9925e77963e087aab`
- WMT14: `0fd791c29053955473a990b0757b0866ce86b9e7b7cf49b69382a7121b92c9ba`
- LAMBADA: `8309f3ddb26f79e82df623f552e3576fb6c78625f7df81b997389b60da2daacc`
- BLiMP: `9eb8d91e29515240972684a37e0cf5113d8445ad905855567205a7f00be77604`
- LongBench: `5d4fd5c4b6ebbe92bde2b5a2fd366af3afad1b2f4d8229f481b6893395fffbc1`

三种模型都落盘 76,401 条 sample 记录。RWKV 的标准产物为权限 `0600`；运行目录
位于 `.tmp/`，不会进入 Git。五个 dataset cache fingerprint 也写入机器可读清单。

## 运行说明

Hugging Face 主站不可达时，数据和 Qwen snapshot 通过 `hf-mirror.com` 准备；正式
评分全程使用本地 cache 和 offline 模式。Qwen3.5-0.8B 最初将 RACE、LAMBADA、
BLiMP 合并为 batch 32 时，RACE 的长输入和 248K 词表 logits 导致 OOM。该次尝试
没有生成结果文件，也不计入分数；正式运行按 task 输入形状拆分为 RACE 1、LAMBADA
8、BLiMP 32、WMT14 32、LongBench 1，并全部退出码为 0。

Qwen3.5-2B 的正式 batch 分别为 RACE 1、LAMBADA 2、BLiMP 16、WMT14 16、
LongBench 1。BLiMP 和 WMT14 在完整运行前分别使用一个叶子 task 的 128 条样本、
以及 WMT 的 64 条样本进行显存探针；探针使用 `limit`，产物与正式目录隔离且不计入
报告。一次 WMT 探针尝试携带了 lm-eval 0.4.12 不支持的 `--bootstrap_iters` 参数，
CLI 在模型加载前即失败；移除参数后的探针和所有正式单元均退出码为 0。

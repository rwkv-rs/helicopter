# lm-eval 用户清单增量结果

评测日期：2026-08-02。本文记录
`configs/eval/lm_eval_catalog_delta.toml` 的执行结果。用户清单先去除 LightEval 已管理
或此前 lm-eval 已测的项目，再按固定 `lm-eval==0.4.12` 的原生任务注册表过滤，最终
只保留 `cmmlu` 与 `gpqa_extended_zeroshot`。

## 范围结论

| 状态 | 项目 | 结论 |
| --- | --- | --- |
| 完成 | CMMLU | 67 个原生叶子任务，完整 test split，共 11,582 题 |
| 阻塞 | GPQA-Extended | 原生 selector 存在，但 `Idavidrein/gpqa` 是 gated dataset；本机没有已授权 HF token |
| 不纳入 | AMC23 | 固定 lm-eval 注册表和任务源码均不存在 |
| 不纳入 | SWE-bench Verified、Multilingual、Pro | 固定 lm-eval 注册表和任务源码均不存在 |

GPQA-Extended 的阻塞不是下载镜像故障。镜像可以改善连通性，但不能绕过数据集访问
条款；在 Hugging Face 接受条款并提供只读 token 后，才能对三个模型执行同协议正式
评测。未授权状态不会记为跳过成功，也不会用名称近似任务或外部 runner 替代。

## CMMLU 对齐协议

| 项目 | 固定值 |
| --- | --- |
| evaluator | `lm-eval==0.4.12` |
| selector | `cmmlu`（67 个学科） |
| prompt | Base 模型原生 prompt，不使用 chat template |
| shots | 0-shot |
| context | `max_length=16382` |
| dataset | 完整 test split，`limit=None` |
| samples | 每个叶子任务启用逐样本落盘，共 11,582 条 |
| metrics | 按样本量聚合 `acc` 与 `acc_norm` |
| offline | 数据和模型准备后启用 Hugging Face offline 模式 |

RWKV 使用本地 `rwkv-vllm` HTTP 后端、FP16 WKV；Qwen 使用 lm-eval 原生 `hf`
后端、BF16。batch size 只适配 8GB 显存，不改变 prompt 或指标语义。

## 完整结果

Qwen3.5 没有 1.5B 型号，因此用 0.8B Base 和 2B Base 构成参数量下界与上界。
“RWKV 差值”为 `RWKV - Qwen`，正数表示 RWKV 更高。

| 指标 | RWKV7 1.5B | Qwen3.5-0.8B | RWKV vs 0.8B | Qwen3.5-2B | RWKV vs 2B |
| --- | ---: | ---: | ---: | ---: | ---: |
| CMMLU acc | 0.460283 | 0.531946 | -0.071663 | 0.640563 | -0.180280 |
| stderr | 0.004531 | 0.004542 | — | 0.004336 | — |
| CMMLU acc_norm | 0.460283 | 0.531946 | -0.071663 | 0.640563 | -0.180280 |

三种模型均实际评分 11,582 题（46,328 个 multiple-choice likelihood 请求），
样本数、task 数和正式运行的 `limit=null` 已逐项审计。RWKV 在本次 CMMLU 协议下
分别落后 0.8B Base 7.1663 个百分点、落后 2B Base 18.0280 个百分点。这是本机相同
协议实测对比，不是引用厂商公开榜单。

## 模型与产物身份

| 模型 | 精确身份 | batch | 正式结果 SHA-256 |
| --- | --- | ---: | --- |
| RWKV7 1.5B | `rwkv7-g1i_preview5445-1.5b-20260729-ctx16384.pth`，权重 SHA-256 `22fe129988f6e98480b344075597259a13ae4201c1d8dedf987246772e613586` | 64 | `04cffddee77508f1ba3f9368db9007421f67cfd22a89445e0ddfc8fae08088ea` |
| Qwen3.5-0.8B-Base | revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | 4 | `f1e92a80db95a430df044befdc5c91f0fd5692404898f05440265efc3d7e224b` |
| Qwen3.5-2B-Base | revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` | 2 | `0e60912dfbfc66107e87731f21bdf73a17e058db71088f5298333d4f4d6ae2eb` |

RWKV 标准产物位于
`.tmp/eval/lm-eval-catalog-delta-20260802/rwkv-full/`；其 `summary.json` SHA-256
为 `98539f2ba69bbd594a6f9676e1bab94f547512fe9209aa79da592d33f60b92f4`，
`artifacts.json` SHA-256 为
`578cdda8057cc8d274b4309b343342429717fb97e27cd81f177ac237379cdc40`。
两组 Qwen 原生结果位于同一运行根目录的 `qwen08-full/` 与 `qwen2b-full/`。

CMMLU 上游脚本依赖 dataset script，因此本项目固定 `datasets==3.6.0`。主站不可达
时从镜像取得上游 `cmmlu_v1_0_1.zip`，大小 1,078,656 bytes，SHA-256
`22ecf70b28bef447ee7d8aa5fe144f56996762f901a8537b03b7693773c672a6`；缓存生成后恢复
上游模块中的官方 URL，并在离线模式复验数据可加载。机器可读合同见
`docs/evaluation/lm_eval_catalog_delta_results.json`。

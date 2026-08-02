# Helicopter

Helicopter is the product-level launcher for RWKV serving and MaxRL training.
Domain behavior stays with the component that implements it:

- `helicopter infer` launches `vllm-rwkv`.
- `helicopter takeoff` delegates a complete MaxRL config to `verl-rwkv`.
- `helicopter eval` dispatches to the LightEval or lm-eval adapter.
- `scripts/install_local.sh` and `scripts/install_remote.sh` prepare the
  selected product environment.

Helicopter does not compile MaxRL configs, prepare training datasets, inspect
rollouts, verify optimizer rounds, or reimplement evaluator task logic.

## Repository layout

```text
configs/example.toml        # serving-only example
configs/eval/               # LightEval and lm-eval evaluation configs
scripts/install_local.sh    # prepare this checkout
scripts/install_remote.sh   # sync and prepare the configured remote checkout
src/cli/helicopter_cli/     # thin product launcher
src/eval/lighteval/         # LightEval adapter and result publication
src/eval/lm_eval/           # lm-eval RWKV-vLLM HTTP model backend
src/infer/vllm-rwkv/        # RWKV vLLM implementation
src/train/rwkv-lm/          # RWKV training engine
src/train/verl-rwkv/        # Verl RWKV and MaxRL implementation
```

## Environment preparation

Copy `.env.example` to a private `.env.local` or `.env.remote`. Keep weights,
datasets, credentials, and machine-local paths out of Git.

Prepare the current checkout:

```bash
INSTALL_COMPONENTS=rwkv-lm,vllm-rwkv,verl-rwkv,lighteval,lm-eval,dev \
  scripts/install_local.sh
```

Prepare the configured remote checkout:

```bash
scripts/install_remote.sh
```

The root `helicopter-dev` control repository owns remote execution, resource
locking, environment recovery, and artifact collection. This product checkout
does not provide a second remote runner.

## Serving

Inspect the command:

```bash
helicopter infer --config configs/example.toml --dry-run g1g-1.5b
```

Start serving:

```bash
helicopter infer --config configs/example.toml g1g-1.5b
```

Serving-specific overrides remain on `infer`, for example:

```bash
helicopter infer --config configs/example.toml g1g-7.2b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85
```

## MaxRL training

Helicopter owns the runnable experiment config; `verl-rwkv` owns its schema,
validation, and execution:

```text
configs/rl/maxrl_dapo_math_17k.toml
src/train/verl-rwkv/verl/trainer/maxrl.py
```

Helicopter passes the file through without interpreting or merging it:

```bash
helicopter takeoff \
  --config configs/rl/maxrl_dapo_math_17k.toml \
  --dry-run
```

Start training by removing `--dry-run`. Explicit Hydra overrides are forwarded
to Verl and validated there:

```bash
helicopter takeoff \
  --config configs/rl/maxrl_dapo_math_17k.toml \
  --override trainer.save_freq=10
```

The canonical config is one complete experiment; it is not split into a
runtime file. Verl derives the context length from the checkpoint filename,
derives prompt/response capacity from the templated examples, enforces EOS
stopping and fixed one-response microbatch slots, and owns MaxRL group
filtering, sampling, optimization, and validation semantics.

The same Verl config owns `val_before_train` and periodic validation triggers.
At each trigger it exports the current RWKV weight and invokes the public
`helicopter eval --config configs/eval/maxrl_math.toml` command. That config
runs AIME 2025, GSM8K, ASDiv, and MATH-500 through LightEval in
`fp32io16`, writes metrics back to Verl, and sets `publish = false`, so training
validation does not access the Scoreboard API or database. Verl neither imports
LightEval nor implements a second evaluator. The installer keeps LightEval in
`.venv-lighteval`; incompatible evaluator dependencies never enter the Verl
training `.venv`.

## Evaluation

The general evaluation campaign remains available through:

```bash
helicopter eval --config configs/eval/lighteval.toml --dry-run
helicopter eval --config configs/eval/lighteval.toml
```

The campaign config publishes confirmed LightEval results to Scoreboard.
Training validation instead uses
[`configs/eval/maxrl_math.toml`](configs/eval/maxrl_math.toml), whose explicit
local result mode bypasses Scoreboard entirely. See
[`docs/evaluation/lighteval.md`](docs/evaluation/lighteval.md) for the campaign
contract and private environment requirements.

The lm-eval-harness backend uses the already running RWKV-vLLM HTTP pool and
supports rolling perplexity, choice scoring, and text generation:

```bash
helicopter eval \
  --evaluator lm-eval \
  --config configs/eval/lm_eval.toml \
  --dry-run

helicopter eval \
  --evaluator lm-eval \
  --config configs/eval/lm_eval.toml
```

This path runs in `.venv-lm-eval`; task names, groups, tags, and glob selectors
are resolved by lm-eval itself. The smaller `configs/eval/lm_eval_ppl.toml`
continues to provide a WikiText-only run. The
`configs/eval/lm_eval_qwen35.toml` suite fixes the public Qwen3.5 language-task
selectors for local protocol-aligned comparisons. All write local
`results.json` plus `summary.json` and do not publish to Scoreboard. Production
matrix publication is available through `configs/eval/lm_eval_campaign.toml`,
with weight SHA verification, both WKV modes, standard sample artifacts, and an
evaluator-aware Scoreboard campaign. See
[`docs/evaluation/lm_eval.md`](docs/evaluation/lm_eval.md) for the HTTP and
result contracts.

## Lightweight checks

```bash
TMPDIR=/tmp uv run --locked --group lm-eval --group test pytest -q tests
python3 -m compileall -q src/cli/helicopter_cli src/eval/lighteval src/eval/lm_eval
bash -n scripts/install_local.sh scripts/install_remote.sh
```

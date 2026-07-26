# Helicopter

Helicopter is the product-level launcher for RWKV serving and MaxRL training.
Domain behavior stays with the component that implements it:

- `helicopter infer` launches `vllm-rwkv`.
- `helicopter takeoff` delegates a complete MaxRL config to `verl-rwkv`.
- `scripts/install_local.sh` and `scripts/install_remote.sh` prepare the
  selected product environment.

Helicopter does not compile MaxRL configs, prepare training datasets, inspect
rollouts, verify optimizer rounds, or implement a second evaluator.

## Repository layout

```text
configs/example.toml        # serving-only example
scripts/install_local.sh    # prepare this checkout
scripts/install_remote.sh   # sync and prepare the configured remote checkout
src/cli/helicopter_cli/     # thin product launcher
src/infer/vllm-rwkv/        # RWKV vLLM implementation
src/train/rwkv-lm/          # RWKV training engine
src/train/verl-rwkv/        # Verl RWKV and MaxRL implementation
```

## Environment preparation

Copy `.env.example` to a private `.env.local` or `.env.remote`. Keep weights,
datasets, credentials, and machine-local paths out of Git.

Prepare the current checkout:

```bash
INSTALL_COMPONENTS=rwkv-lm,vllm-rwkv,verl-rwkv,dev scripts/install_local.sh
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

The MaxRL contract and canonical DAPO config are owned by `verl-rwkv`:

```text
src/train/verl-rwkv/verl/trainer/maxrl.py
src/train/verl-rwkv/examples/rwkv_trainer/config/maxrl_dapo_math_17k.toml
```

Helicopter passes the file through without interpreting or merging it:

```bash
helicopter takeoff \
  --config src/train/verl-rwkv/examples/rwkv_trainer/config/maxrl_dapo_math_17k.toml \
  --dry-run
```

Start training by removing `--dry-run`. Explicit Hydra overrides are forwarded
to Verl and validated there:

```bash
helicopter takeoff \
  --config src/train/verl-rwkv/examples/rwkv_trainer/config/maxrl_dapo_math_17k.toml \
  --override trainer.save_freq=10
```

The canonical config is one complete experiment; it is not split into a
runtime file. Verl derives the context length from the checkpoint filename,
derives prompt/response capacity from the templated examples, enforces EOS
stopping and fixed one-response microbatch slots, and owns MaxRL group
filtering, sampling, optimization, and validation semantics.

Full benchmark evaluation will use the evaluation component's public
`helicopter eval --config <path>` contract after that separate LightEval change
lands. MaxRL does not import evaluator-private functions or implement a second
evaluator.

## Lightweight checks

```bash
PYTHONPATH=src/cli python3 -m pytest -q tests/test_cli.py tests/test_install_policy.py
PYTHONPATH=src/cli python3 -m compileall -q src/cli/helicopter_cli tests
bash -n scripts/install_local.sh scripts/install_remote.sh
```

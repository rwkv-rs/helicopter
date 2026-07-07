# Helicopter

Helicopter is an RWKV leaderboard-run framework. It keeps the pieces needed for
RWKV vLLM serving, verl-based training, and benchmark-oriented experiment runs
in one repository, with a small CLI for launching common workflows.

The current focus is RWKV7:

- `infer`: start a vLLM server for an RWKV checkpoint.
- `eval`: run LightEval tasks against an OpenAI-compatible endpoint, or run a
  managed one-shot evaluation that starts and stops vLLM automatically.
- `takeoff`: start verl training for an RWKV checkpoint. The supported takeoff
  path is GRPO.
- `scripts/install_remote.sh`: prepare the BBT DevPod GPU workspace, sync this
  repository, and run the local installer remotely.
- `scripts/install_local.sh`: create/update the project `.venv`, install the
  declared RWKV dependency group, and install local editable `vllm`, `rwkv-lm`,
  and `verl` packages.

## Repository layout

```text
configs/
  example.toml              # public example experiment config
  local/*.toml              # machine-local experiment configs
scripts/
  install_local.sh          # prepare the current machine/workspace
  install_remote.sh         # sync and prepare the remote DevPod workspace
src/cli/helicopter_cli/     # Python CLI package
src/scoreboard-server/      # FastAPI scoreboard API and PostgreSQL store
src/scoreboard-client/      # Next.js scoreboard UI
src/infer/vllm-rwkv/        # vLLM RWKV implementation
src/train/rwkv-lm/          # RWKV training code
src/train/verl-rwkv/        # verl RWKV integration
```

`AGENTS.md` is intentionally ignored in this repository because it may contain
machine-specific remote connection details. Use `.env.example` and
`AGENTS.example.md` as public templates.

## Environment files and configs

Copy `.env.example` to a private env file before running commands:

```bash
cp .env.example .env.local
```

For remote DevPod use, keep the private remote values in `.env.remote`.

The env files use simple dotenv syntax:

```text
KEY=value
export KEY=value
```

Do not put shell expressions in env files that the Python CLI must read. Values
already present in the command environment override values from `.env.local` or
`.env.remote`, which makes command-scoped overrides predictable:

```bash
WEIGHT_PATH=/workspace/Weights/RWKV helicopter infer g1g-1.5b
```

Experiment settings live in TOML files. If `--config` is omitted, the CLI uses
the newest `configs/local/*.toml`; otherwise it falls back to
`configs/example.toml`.

Important config sections:

- `[models.<name>]`: maps a CLI model alias to a checkpoint file or path.
- `[datasets.<name>]`: maps a dataset alias to a dataset root.
- `[infer]`: vLLM serving defaults.
- `[lighteval]`: LightEval endpoint, output, and custom-task defaults.
- `[takeoff.grpo]`: verl GRPO training defaults.

Scoreboard database settings are read from `SCOREBOARD_DB_*` first and then
from standard `PG*` variables. This matters for `helicopter eval run
--scoreboard`, because the CLI loads dotenv files before writing results into
the scoreboard database:

```text
SCOREBOARD_DB_HOST=/var/run/postgresql
SCOREBOARD_DB_PORT=5432
SCOREBOARD_DB_USER=postgres
SCOREBOARD_DB_NAME=helicopter
```

## Prepare the environment

Remote preparation is the expected path for full RWKV vLLM/verl work:

```bash
scripts/install_remote.sh
```

The remote installer:

- validates the target DevPod Pod and node;
- checks that the running container uses the required runtime image;
- syncs this repository with `rsync`;
- preserves the remote `.venv`;
- runs `scripts/install_local.sh` inside the remote repo path.

For local or already-synced workspace preparation:

```bash
scripts/install_local.sh
```

Useful install overrides:

```bash
VLLM_REBUILD=1 scripts/install_local.sh
VERL_REINSTALL=1 scripts/install_local.sh
INSTALL_PROFILE=full scripts/install_local.sh
```

Use `DRY_RUN=1` to print installer actions without executing them:

```bash
DRY_RUN=1 scripts/install_remote.sh
```

## CLI usage

Run the CLI through the installed console script:

```bash
helicopter --help
```

During development, the package can also be run directly:

```bash
PYTHONPATH=src/cli python3 -m helicopter_cli --help
```

### Start RWKV vLLM serving

Dry-run first to inspect the exact command and environment:

```bash
helicopter infer --config configs/example.toml --dry-run g1g-1.5b
```

Start the server:

```bash
helicopter infer --config configs/example.toml g1g-1.5b
```

Override serving parameters from the CLI when an experiment explicitly needs
them:

```bash
helicopter infer g1g-7.2b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 2048 \
  --max-num-batched-tokens 65536
```

RWKV vLLM uses upstream defaults by default. For RWKV7, set only WKV mode unless
you are debugging a specific vLLM issue. GRPO `takeoff` keeps embedding
preprocessing on GPU with `HELICOPTER_TAKEOFF_EMB_DEVICE=gpu`:

```bash
VLLM_RWKV7_WKV_MODE=fp32io16 helicopter infer g1g-1.5b
```

### Run LightEval

Use `eval lighteval` when a compatible endpoint is already running:

```bash
helicopter eval lighteval \
  --config configs/example.toml \
  g1g-1.5b "gsm8k|0" \
  --max-samples 2
```

Use `eval run` for a managed one-shot evaluation. For a local endpoint URL, the
command starts vLLM in the background, waits for `/v1/models`, runs LightEval,
writes a performance report, optionally records scores into the scoreboard
database, and stops the managed server:

```bash
helicopter eval run \
  --config configs/example.toml \
  g1d-0.4b "gsm8k|0" \
  --max-samples 2 \
  --scoreboard
```

Useful run-control options:

```bash
helicopter eval run g1d-0.4b "gsm8k|0" --dry-run
helicopter eval run g1d-0.4b "gsm8k|0" --no-server
helicopter eval run g1d-0.4b "gsm8k|0" --keep-server
helicopter eval run g1d-0.4b "gsm8k|0" --server-timeout 900
```

`eval run` forwards the same LightEval options as `eval lighteval`, plus serving
overrides such as `--wkv-mode`, `--emb-device`,
`--gpu-memory-utilization`, `--max-num-seqs`, and
`--max-num-batched-tokens`. The task argument can be omitted when
`[lighteval].tasks` is set in the selected config.

Inspect the available custom tasks and metric status before treating results as
formal scores:

```bash
helicopter eval lighteval-tasks export --contains gsm8k --format text
helicopter eval lighteval-tasks judges --format summary
```

Some custom tasks intentionally use proxy or sanity metrics rather than the
official benchmark harness. Examples include Arena-Hard baseline token F1,
SWE-Bench patch token F1 or nonempty checks, and TAU static-plan token F1.
`lighteval-tasks judges` marks these cases explicitly.

The custom task registry still carries a small set of duplicate aliases in
`KNOWN_DUPLICATE_TASK_CONFIGS` so existing downstream task names remain
runnable. Those aliases should be cleaned up only with a registry migration.

### Start GRPO takeoff training

Dry-run a GSM8K GRPO run:

```bash
helicopter takeoff \
  --config configs/example.toml \
  --dry-run \
  --dataset gsm8k \
  g1g-1.5b grpo
```

Start the run:

```bash
helicopter takeoff \
  --config configs/example.toml \
  --dataset gsm8k \
  g1g-1.5b grpo
```

Pass extra Hydra overrides to the underlying verl entrypoint:

```bash
helicopter takeoff g1g-1.5b grpo \
  --dataset gsm8k \
  --override trainer.total_epochs=1 \
  --override trainer.save_freq=10
```

`takeoff` requires the project Python executable to exist. By default it uses
the configured `.venv/bin/python`; set `HELICOPTER_PYTHON` or `paths.python` only
when an explicit override is intended:

```bash
HELICOPTER_PYTHON=/workspace/Projects/MachineLearning/helicopter/.venv/bin/python \
helicopter takeoff --dataset gsm8k g1g-1.5b grpo
```

## Common command-scoped overrides

```bash
WEIGHT_PATH=/workspace/Weights/RWKV
DATASETS_PATH=/workspace/Datasets
HELICOPTER_NUM_NODES=1
HELICOPTER_NUM_DEVICES=8
HELICOPTER_TAKEOFF_WKV_MODE=fp32io16
HELICOPTER_TAKEOFF_EMB_DEVICE=gpu
```

Keep checkpoint files, datasets, `.env.local`, `.env.remote`, and machine-local
agent notes out of the public repository.

## Lightweight checks

The root CLI has standard-library tests and does not require the full RWKV
dependency group:

```bash
PYTHONPATH=src/cli python3 -m unittest tests.test_cli
PYTHONPATH=src/cli python3 -m compileall -q src/cli/helicopter_cli tests
bash -n scripts/install_local.sh scripts/install_remote.sh
```

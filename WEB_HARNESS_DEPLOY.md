# RWKV Web Harness deployment checklist

This package is the local-first web research harness in `rwkv_web_harness`.
It calls an OpenAI-compatible RWKV endpoint and uses public web pages or a
self-hosted SearXNG instance. It is separate from the official BrowseComp-Plus
runner in `/home/rwkv/chase/rwkv-skills`.

## 1. Deployable files

Copy/install these tracked files:

- `pyproject.toml`
- `src/cli/rwkv_web_harness/`
- `configs/web_harness_tasks_100.jsonl` when running the smoke suite
- `WEB_HARNESS_DEPLOY.md`

Do not deploy or commit:

- `.env`, `.env.local`, API keys, or model weights
- `tmp/`, `uv.lock`, `results/`, and old traces unless explicitly needed
- the full `src/infer/vllm-rwkv/` tree when an existing vLLM-RWKV server is used

## 2. Runtime environment

Set these values in the deployment shell or a secret-managed environment file:

```bash
export RWKV_MODEL_URL=http://127.0.0.1:19315/v1
export RWKV_MODEL_NAME=rwkv7-g1h-1.5b-20260710-ctx10240
export RWKV_MODEL_INTERFACE=g1h
export RWKV_MODEL_API_KEY='set-outside-the-repository'
export RWKV_WEB_SEARCH_BACKEND=html
export RWKV_WEB_SEARCH_URL=https://www.bing.com/search
```

For the 157 host, `19315` and `19316` are the g1h-1.5b endpoints observed in
the current deployment. They bind to the remote host's loopback interface;
run the harness on that host or provide an SSH tunnel. Do not expose the model
port publicly just to avoid tunnelling.

## 3. Install and preflight

```bash
uv sync --no-dev
uv run rwkv-web-harness preflight --timeout 10
```

The preflight now authenticates against `/v1/models`, verifies that the
requested model is advertised, and probes the selected search endpoint. A
401 is reported as an authentication failure rather than an opaque “model
down” result.

## 4. One real request

```bash
uv run rwkv-web-harness run \
  --interface g1h \
  --max-steps 8 \
  --max-context-chars 12000 \
  --trace results/web_harness/smoke.jsonl \
  --task 'Use web_search to find the official Python documentation, then cite one source.'
```

The g1h contract is `User✿...✿` / `Bot✿<think></think>...✿`, with stop token
`✿` (id `10060`). Keep the trace: it is the evidence that real network tools
were called and that citations were grounded.

## 5. Resumable smoke suite

```bash
uv run rwkv-web-harness batch \
  --tasks-file configs/web_harness_tasks_100.jsonl \
  --summary results/web_harness/g1h_batch_summary.json \
  --trace-dir results/web_harness/g1h_batch \
  --retries 1

uv run rwkv-web-harness batch \
  --tasks-file configs/web_harness_tasks_100.jsonl \
  --summary results/web_harness/g1h_batch_summary.json \
  --trace-dir results/web_harness/g1h_batch \
  --resume
```

The batch validator requires completion, the expected tool order, successful
network payloads, and grounded citations. It writes a checkpoint after each
case, so interruption does not require restarting passed cases.

## 6. Official BrowseComp-Plus is a different deployment

For the official 830-case evaluator on 157, use the `rwkv-skills` checkout and
keep these settings explicit:

```bash
export RWKV_BROWSECOMP_PLUS_RETRIEVER=bm25
export RWKV_BENCHMARK_CONFIG_ROOT=/home/rwkv/chase/rwkv-skills/configs/g1h

# runner flags
--candidate-router-mode parallel
--infer-protocol completions
```

That path performs the official `local_knowledge_base_retrieval` conversion,
uses the official BM25 index, and needs a configured judge model for scored
runs. Do not treat a local web-harness trace as a BrowseComp-Plus score.

# Quickstart

## 1. Convert your traces

One JSON object per line. Minimum viable row:

```json
{"id": "t0", "messages": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "reasoning": "the model's thinking for this turn",
   "tool_calls": [{"function": {"name": "get_account", "arguments": "{}"}}]}
]}
```

Add `"label": true|false` on assistant messages if you can say whether the calls that turn
introduced were correct. It is the single highest-value field in the file: without it you are
choosing a granularity on noise-floor evidence.

Declaring `tools` at the trace level improves masking, because every declared tool and parameter
name is stripped before embedding.

## 2. Choose a granularity

```bash
plvr-primitives sweep --traces traces.jsonl
```

Look for a setting with a manageable number of clusters and a noise fraction you can live with.
Then confirm it reproduces:

```bash
plvr-primitives mine --traces traces.jsonl --out out --min-size 40 --stability
```

If split-half stability is below ~0.8, the clusters are not reproducible: try a different
`--min-size`, `--selection leaf`, `--repr verbs`, or accept that you need more traces.

## 3. Read the lift table before the prompts

`out/lift.txt` is the finding. A cluster far below the base rate is a failure mode your agent has
and you did not know about; a cluster far above it is an operation worth making explicit.

## 4. Edit the prompts

`out/prompts/*.txt` are starting points generated from the clusters. They know the operation and
its exemplars. They do not know your domain's rules. Edit them, then wire them into your own
program.

## 5. Optionally optimise

```bash
export PLVR_API_KEY=...
plvr-primitives optimize --library out/library.json --trainset train.jsonl \
    --model your-model --out out
```

Then evaluate on the task you actually care about. A training-set gain is not a downstream gain.

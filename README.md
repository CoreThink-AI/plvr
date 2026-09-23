# plvr-primitives

Mine reusable reasoning **primitives** from your own agent traces, and generate a runnable prompt
for each one.

Bring a corpus of traces. You get back a set of candidate operations that actually recur in your
data, evidence for whether each one is separating anything real, and a prompt per operation that a
small model can run.

**Program Synthesis will be available soon** See [Scope](#scope).

---

## Why mine primitives instead of writing them

If you hand-write the operations your agent uses, you get the operations you already thought of.
Clustering the reasoning your traces actually contain tends to surface a different set — and,
more usefully, tells you which of them are *reliable*.

No LLM is involved in the extraction. The operations that come out are a property of your corpus,
not of a model asked to invent a taxonomy.

## Install

```bash
pip install -e ".[all]"
python -m spacy download en_core_web_sm   # only for --repr verbs/frames
```

The base install works with no optional extras, but falls back to a character n-gram embedding
that clusters noticeably worse. For real runs install at least `[embed]` and `[umap]`.

## Quickstart

```bash
# 1. how many operations does your corpus support?
plvr-primitives sweep --traces examples/traces.sample.jsonl --sizes 5 10 20 --min-samples 3

# 2. mine at the granularity you picked, and check it reproduces
plvr-primitives mine --traces examples/traces.sample.jsonl --out out \
    --min-size 5 --min-samples 3 --stability
```

On the bundled sample that prints:

```
6 clusters, 0 noise (0.0%)
validator=label  base rate=0.622 over 90 segments
  cluster   3  n=   6  rate=0.833  lift=1.34  separate_pull
  cluster   4  n=  50  rate=0.740  lift=1.19  step_order
  cluster   5  n=  11  rate=0.364  lift=0.58  prerequisite_mutating
  cluster   1  n=   6  rate=0.167  lift=0.27  fetch_present
```

and writes `out/library.json` plus one editable prompt per primitive in `out/prompts/`.

Read that table as: turns that *check a prerequisite before mutating* were **0.27–0.58×** as likely
to be correct as the corpus average, while turns that *decompose* or *order* were **1.19–1.34×**.
That is a claim about where your agent is weak, derived without an LLM's opinion — and it is
carried into the generated prompt as an explicit caution.

## Input format

JSONL, one trace per line. Only `messages` is required.

```json
{
  "id": "trace_0",
  "tools": [{"function": {"name": "get_account", "parameters": {"properties": {"account_id": {}}}}}],
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant",
     "reasoning": "First let me break this into the separate things to gather ...",
     "tool_calls": [{"function": {"name": "get_account", "arguments": "{\"account_id\": \"1\"}"}}],
     "label": true}
  ]
}
```

| field | why |
|---|---|
| `messages[].reasoning` | the segment unit. Use `--field reply` to mine visible replies instead. |
| `tools`, `tool_calls` | supplies the tool and parameter names to mask out |
| `label` *(optional)* | the independent validator — see below |

**A segment is one reasoning turn, not one reply.** The visible reply is what a user-facing answer
naturally describes, so mining it recovers your API's verbs. Planning, ordering and verification
live in the thinking.

## The two things that decide whether the output means anything

**Masking.** Tool names, parameter names, servers, URIs, quoted literals, numbers and capitalised
entities are stripped before embedding. Without it, clusters follow your domain vocabulary rather
than the structure of the reasoning. Masking deliberately spares a single capitalised word at the
start of a sentence, because that is usually a verb (`Check the balance`) and masking it would
delete the operation.

**A validator the embedding never saw.** Clustering always returns clusters; that they exist is
not evidence. Two validators ship:

- `label` — a per-segment boolean you supply. The strongest one available: if your traces have
  ground truth, mark whether the calls a turn introduced were correct. Then a cluster whose members
  are reliably right, *or reliably wrong*, is separating something real.
- `fanout` — whether the turn preceded more than one call. Free, but only informative when it is
  rare. If your agent plans batches rather than calling and iterating, the base rate approaches
  50%, the maximum achievable lift is under 2×, and the tool says so instead of letting you read a
  meaningless number.

`--stability` reruns the clustering on two random halves and reports the fraction of clusters with
a partner in the other half. **Below ~0.8 the clusters are not reproducible.** The bundled sample
scores 0.5 — it is 90 segments of synthetic data, far too small, and the warning is doing its job.
Expect to need thousands of segments.

## Granularity

Two dials, no universally right value:

- `--min-size` — how many operations you get.
- `--selection eom|leaf` — `eom` merges toward broader clusters, `leaf` keeps finer ones.

`sweep` reports what each setting yields on *your* traces so you can choose with the numbers in
front of you rather than inheriting someone else's constant.

## Optional: refine prompts against a training set

If you have labelled examples, `optimize` runs GEPA-style reflective evolution: a reflection model
proposes a revised instruction, it is kept only if it beats its parent on a fresh minibatch, and
the prompt that ships is the one that wins on a **held-out** split.

```bash
export PLVR_API_KEY=...
plvr-primitives optimize --library out/library.json --trainset train.jsonl \
    --model <the model the prompt will run on> --out out
```

`train.jsonl` needs `messages` and `ground_truth_calls` per row. The default scorer matches call
names and argument names, ignoring argument *values* (they vary in formatting and punish correct
plans for cosmetic reasons); pass your own `Scorer` to `optimize()` if values matter.

> **A caution we paid for.** A large gain on the training distribution does not imply a gain
> anywhere else. In our own runs, prompts optimised this way improved their training metric
> substantially and produced **no measurable change** on the downstream agentic benchmarks we
> cared about, across three models and three trials each. Selection on a holdout is necessary and
> not sufficient — evaluate on the task you actually care about before adopting anything.

## Scope

**In:** segmentation, masking, representation, clustering, validation, granularity selection,
prompt generation, optional prompt optimisation.

**Out, deliberately:**

- **Program synthesis.** Composing primitives into a program — enumerative search over
  compositions, symbolic target propagation, layer-wise assignment — is not here. It is the part
  still moving fastest and shipping a half-version would be worse than shipping none.
- **A runtime.** These are prompts and contracts, not an execution engine. Wire them into whatever
  agent loop you already have.
- **Hand-tuned domain prompts.** The generated prompt carries the operation and its exemplars. It
  does not carry the domain discipline a deployed primitive needs — anaphora across turns,
  observation before mutation, exact-name rules. Expect to edit.

## Library use

```python
from plvr_primitives import (load_traces, build_segments, mine, lift_report,
                             build_library, save_library)

segs = build_segments(load_traces("traces.jsonl"))
res = mine(segs, min_size=20, selection="eom")
print(lift_report(segs, res, validator="label"))    # also annotates clusters with lift
save_library(build_library(res), "out")
```

## Tests

```bash
pip install -e ".[dev]" && pytest -q
```

Offline: the embedder falls back to TF-IDF + SVD when `sentence-transformers` is absent.

## Pipeline

```
traces.jsonl
  └─ segment    one reasoning turn per assistant message, masked
  └─ represent  masked | verbs | frames | raw
  └─ cluster    embed → UMAP → HDBSCAN → c-TF-IDF naming
  └─ validate   per-cluster lift vs base rate; split-half stability
  └─ prompts    name, contract, exemplars, evidence-derived cautions
  └─ optimize   (optional) GEPA against a training set, selected on a holdout
```

## License

MIT — see [LICENSE](LICENSE).

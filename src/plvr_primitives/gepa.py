"""Optional: refine a generated prompt against a training set (GEPA-style).

Reflective prompt evolution. Each iteration picks a parent from the Pareto front over per-instance
scores, shows a reflection model where that parent failed, asks for a revised instruction, and
keeps the child only if it beats the parent on a fresh minibatch. The prompt that ships is the one
that wins on a HELD-OUT split, not the one with the best training score -- prompts overfit a small
training set readily, and the holdout is what catches it.

This is optional in every sense: the mined library is usable without it, it needs an API key, and
it needs labelled data. Skip it if you have no training set.

A caution from our own runs: a large gain on a training distribution does not imply a gain
elsewhere. Evaluate the optimised prompt on the task you actually care about before adopting it.
"""
from __future__ import annotations

import json
import os
import random
import re
import statistics as st
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

CALLS = re.compile(r"```calls\s*\n(.*?)```", re.S)
CALL_SIG = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*\((.*)\)\s*$", re.S)


# ---------------------------------------------------------------- model access

@dataclass
class LLM:
    """Any OpenAI-compatible chat endpoint."""

    model: str
    base_url: str = os.environ.get("PLVR_BASE_URL", "https://api.openai.com/v1")
    api_key_env: str = "PLVR_API_KEY"
    max_tokens: int = 2048
    temperature: float = 0.0
    retries: int = 4
    timeout: float = 240.0

    def __call__(self, messages: list[dict]) -> str:
        key = os.environ.get(self.api_key_env) or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(f"set ${self.api_key_env} (or $OPENAI_API_KEY) to use the optimiser")
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        last = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.loads(r.read())
                txt = (d["choices"][0]["message"].get("content") or "").strip()
                # A 200 with empty content happens on pooled providers. Scoring it as a wrong
                # answer would understate whichever candidate got unlucky, so retry it like any
                # other transport failure.
                if txt:
                    return txt
                last = f"empty content (finish={d['choices'][0].get('finish_reason')})"
            except Exception as exc:                          # noqa: BLE001
                last = exc
            time.sleep(min(3 * (attempt + 1), 20))
        print(f"  llm gave up: {last}")
        return ""


# ---------------------------------------------------------------- default scorer

def parse_calls(text: str) -> list[tuple[str, frozenset]]:
    """(name, argument-name set) per call in a ```calls``` block."""
    m = CALLS.search(text or "")
    out = []
    for line in (m.group(1).splitlines() if m else []):
        sig = CALL_SIG.match(line.strip())
        if sig:
            args = frozenset(re.findall(r"([A-Za-z_]\w*)\s*=", sig.group(2)))
            out.append((sig.group(1), args))
    return out


def score_calls(reply: str, ground_truth: Sequence[str], order_weight: float = 0.5) -> float:
    """1.0 is a perfect match. Name+argument-name F1, discounted when the order is wrong.

    Argument VALUES are ignored: they vary in formatting and punish correct plans for cosmetic
    reasons. Swap in your own scorer if values matter for your task.
    """
    pred = parse_calls(reply)
    gold = parse_calls("```calls\n" + "\n".join(ground_truth) + "\n```")
    if not gold:
        return 0.0
    if not pred:
        return 0.0
    pset, gset = list(pred), list(gold)
    matched = 0
    pool = list(gset)
    for p in pset:
        for i, g in enumerate(pool):
            if p[0] == g[0] and p[1] == g[1]:
                matched += 1
                pool.pop(i)
                break
    prec, rec = matched / len(pset), matched / len(gset)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    names_p = [n for n, _ in pset]
    names_g = [n for n, _ in gset]
    ordered = names_p == names_g
    return round(f1 if ordered else f1 * order_weight, 6)


Scorer = Callable[[str, Sequence[str]], float]


# ---------------------------------------------------------------- GEPA

REFLECT_SYSTEM = """You improve the instruction block of a tool-use system prompt.

You are shown the current instruction, and examples where it scored poorly with the model's answer
and the expected calls. Write a BETTER instruction.

Rules:
- Output the new instruction text only. No preamble, no explanation, no markdown fences.
- Keep it general. Do not name any specific tool, domain, or dataset from the examples.
- Do not specify a response format; the caller controls that.
- Be concrete about REASONING and ORDER, not about wording.
"""


@dataclass
class Candidate:
    text: str
    scores: list[float] = field(default_factory=list)
    mean: float = 0.0
    val: float | None = None


def _evaluate(llm: LLM, instruction: str, rows: Sequence[dict], scorer: Scorer,
              workers: int = 8) -> list[float]:
    from concurrent.futures import ThreadPoolExecutor

    def one(r):
        msgs = [dict(m) for m in r["messages"]]
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = instruction.strip() + "\n\n" + msgs[0]["content"]
        else:
            msgs = [{"role": "system", "content": instruction.strip()}, *msgs]
        return scorer(llm(msgs), r["ground_truth_calls"])

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, rows))


def _pareto_parents(pool: Sequence[Candidate]) -> list[Candidate]:
    """Candidates that are best on at least one instance -- keeps specialists alive."""
    if len(pool) < 2:
        return list(pool)
    keep = set()
    n = len(pool[0].scores)
    for i in range(n):
        best = max(range(len(pool)), key=lambda j: pool[j].scores[i] if pool[j].scores else -1)
        keep.add(best)
    return [pool[i] for i in sorted(keep)] or list(pool)


def optimize(llm: LLM, seed_instruction: str, train: Sequence[dict], val: Sequence[dict],
             scorer: Scorer = score_calls, reflect_llm: LLM | None = None, iterations: int = 10,
             minibatch: int = 16, workers: int = 8, seed: int = 0,
             log_path: str | Path | None = None) -> tuple[str, list[Candidate]]:
    """Returns (best instruction by held-out score, the whole candidate pool)."""
    rng = random.Random(seed)
    reflect = reflect_llm or llm
    log = open(log_path, "a", encoding="utf-8") if log_path else None

    def emit(rec: dict) -> None:
        print(json.dumps(rec), flush=True)
        if log:
            log.write(json.dumps(rec) + "\n")
            log.flush()

    root = Candidate(seed_instruction)
    root.scores = _evaluate(llm, root.text, train, scorer, workers)
    root.mean = st.mean(root.scores)
    root.val = st.mean(_evaluate(llm, root.text, val, scorer, workers))
    emit({"iter": 0, "event": "seed", "train": round(root.mean, 4), "val": round(root.val, 4)})
    pool = [root]

    for it in range(1, iterations + 1):
        parent = rng.choice(_pareto_parents(pool))
        idx = rng.sample(range(len(train)), min(minibatch, len(train)))
        batch = [train[i] for i in idx]
        worst = sorted(idx, key=lambda i: parent.scores[i])[:4]
        shots = "\n\n".join(
            f"### Example\nRequest:\n{train[i]['messages'][-1]['content'][:1200]}\n"
            f"Expected calls:\n" + "\n".join(train[i]["ground_truth_calls"])
            for i in worst)
        child_text = reflect([{"role": "system", "content": REFLECT_SYSTEM},
                              {"role": "user",
                               "content": f"## Current instruction\n{parent.text}\n\n"
                                          f"## Weakest cases\n{shots}\n\nWrite the improved instruction."}])
        if not child_text.strip():
            emit({"iter": it, "event": "skip", "why": "empty reflection"})
            continue
        p_mb = st.mean(_evaluate(llm, parent.text, batch, scorer, workers))
        c_mb = st.mean(_evaluate(llm, child_text, batch, scorer, workers))
        accept = c_mb > p_mb
        emit({"iter": it, "event": "compare", "parent": round(p_mb, 4),
              "child": round(c_mb, 4), "accepted": accept})
        if not accept:
            continue
        child = Candidate(child_text)
        child.scores = _evaluate(llm, child_text, train, scorer, workers)
        child.mean = st.mean(child.scores)
        child.val = st.mean(_evaluate(llm, child_text, val, scorer, workers))
        pool.append(child)
        emit({"iter": it, "event": "accepted", "train": round(child.mean, 4),
              "val": round(child.val, 4)})

    best = max(pool, key=lambda c: (c.val if c.val is not None else -1, c.mean))
    emit({"event": "done", "candidates": len(pool), "best_val": round(best.val or 0, 4),
          "best_train": round(best.mean, 4),
          "note": "seed kept" if best is root else "evolved"})
    if log:
        log.close()
    return best.text, pool


def load_trainset(path: str | Path) -> list[dict]:
    """JSONL with `messages` (system+user) and `ground_truth_calls` (list of call strings)."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "messages" in r and "ground_truth_calls" in r:
                rows.append(r)
    return rows

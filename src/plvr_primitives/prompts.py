"""Clusters -> primitive prompts.

A mined cluster is a set of reasoning turns that do the same kind of thing. A primitive is that
cluster turned into something callable: a name, a contract, and a system prompt an SLM can run.

This module builds the prompt DETERMINISTICALLY from the cluster -- top c-TF-IDF terms for the
name, medoid segments as exemplars, the validator's lift as a confidence note. No LLM is
involved, so the prompt is a faithful description of what is in your traces and nothing else.

What it does NOT do: decide which primitives compose into a program, or in what order. That is
program synthesis and it is deliberately out of scope (see the README).

The generated prompt is a STARTING POINT. It carries the operation the cluster names and the
exemplars that justify it; it does not carry the domain rules a deployed primitive needs
(anaphora across turns, observation-before-mutation, exact-name discipline). Expect to edit, or
run `optimize` with a training set.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .cluster import Cluster, ClusterResult

TEMPLATE = """You are the `{name}` primitive.

## Operation
{description}

## Inputs
{inputs}

## Output
Return JSON only, matching this schema:
{schema}

## Rules
{rules}

## Examples of this operation in reasoning traces
{exemplars}
"""

DEFAULT_RULES = [
    "Use EXACT tool and parameter names from the registry you are given; do not paraphrase them.",
    "Use only values present in the request, the environment, or prior results. Do not invent "
    "identifiers, paths or numbers.",
    "If the information needed is not available, say so in the output rather than guessing.",
    "Return JSON only. No prose outside the JSON.",
]


@dataclass
class Primitive:
    name: str
    cluster_id: int
    description: str
    inputs: list[str]
    output_schema: dict
    rules: list[str]
    exemplars: list[str]
    evidence: dict = field(default_factory=dict)
    prompt: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _slug(terms: Sequence[str], cid: int) -> str:
    words = [re.sub(r"[^a-z0-9_]", "", t.lower()) for t in terms[:2] if t]
    words = [w for w in words if w]
    return ("_".join(words) or f"primitive_{cid}").upper()


def _describe(c: Cluster) -> str:
    terms = ", ".join(c.terms[:6]) or "(no distinctive terms)"
    return (f"Reasoning turns in this cluster are characterised by: {terms}. "
            f"The cluster covers {c.size} segments of the corpus. "
            "Perform this operation for the request you are given, using the tool registry and "
            "the state supplied by the caller.")


def _evidence(c: Cluster) -> dict:
    ev = {"cluster_size": c.size, "terms": c.terms}
    if c.lift is not None:
        ev |= {"validator_lift": c.lift, "cluster_rate": c.label_rate, "base_rate": c.base_rate}
    return ev


def _confidence_rule(c: Cluster) -> list[str]:
    """Turn validator evidence into an instruction rather than a silent number."""
    if c.lift is None:
        return []
    if c.lift >= 1.2:
        return [f"Turns of this kind were {c.lift:.2f}x more likely than average to be correct "
                "in the source traces; the pattern is well supported."]
    if c.lift <= 0.8:
        return [f"Turns of this kind were {c.lift:.2f}x as likely as average to be correct in "
                "the source traces. Treat this operation as error-prone: prefer verifying "
                "before acting, and report uncertainty rather than committing."]
    return []


def build_primitive(c: Cluster, output_schema: dict | None = None,
                    extra_rules: Sequence[str] = ()) -> Primitive:
    schema = output_schema or {
        "type": "object",
        "properties": {
            "result": {"type": "string", "description": "the outcome of this operation"},
            "reasoning": {"type": "string", "description": "why, in one or two sentences"},
        },
        "required": ["result"],
    }
    rules = [*DEFAULT_RULES, *_confidence_rule(c), *extra_rules]
    name = _slug(c.terms, c.id)
    exemplars = "\n".join(f"{i+1}. {e.strip()[:400]}" for i, e in enumerate(c.exemplars)) \
        or "(none recorded)"
    p = Primitive(
        name=name, cluster_id=c.id, description=_describe(c),
        inputs=["request", "registry", "state"],
        output_schema=schema, rules=rules, exemplars=c.exemplars,
        evidence=_evidence(c))
    p.prompt = TEMPLATE.format(
        name=name, description=p.description,
        inputs="\n".join(f"- {i}" for i in p.inputs),
        schema=json.dumps(schema, indent=1),
        rules="\n".join(f"- {r}" for r in rules),
        exemplars=exemplars)
    return p


def build_library(result: ClusterResult, min_size: int = 0,
                  extra_rules: Sequence[str] = ()) -> list[Primitive]:
    """One primitive per cluster, largest first. `min_size` drops the long tail."""
    cs = [c for c in result.clusters if c.size >= min_size]
    return [build_primitive(c, extra_rules=extra_rules)
            for c in sorted(cs, key=lambda c: -c.size)]


def save_library(prims: Sequence[Primitive], out_dir: str | Path) -> Path:
    """Writes library.json plus one .txt per primitive, so prompts can be edited by hand."""
    d = Path(out_dir)
    (d / "prompts").mkdir(parents=True, exist_ok=True)
    (d / "library.json").write_text(
        json.dumps([p.to_dict() for p in prims], indent=1, ensure_ascii=False), encoding="utf-8")
    seen: dict[str, int] = {}
    for p in prims:
        n = seen.get(p.name, 0)
        seen[p.name] = n + 1
        stem = p.name if not n else f"{p.name}_{n+1}"
        (d / "prompts" / f"{stem}.txt").write_text(p.prompt, encoding="utf-8")
    return d / "library.json"


def load_library(path: str | Path) -> list[Primitive]:
    return [Primitive(**r) for r in json.loads(Path(path).read_text(encoding="utf-8"))]

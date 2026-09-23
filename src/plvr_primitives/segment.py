"""Traces -> clusterable segments.

A segment is one assistant turn's REASONING block, not its visible reply. The reply is what a
user-facing answer naturally describes (it recovers API verbs); the thinking is where planning,
ordering and verification actually live. `--field reply` reproduces the reply-mining pass for
comparison.

Masking removes what is specific to one deployment -- tool names, parameter names, server names,
URIs, quoted literals, numbers and capitalised entities -- so that clustering groups *operations*
rather than domains. Without it the clusters recover the corpus's vocabulary, not its structure.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

MIN_CHARS = 20  # a 3-word turn ("Done.") carries no operation

URI = re.compile(r"https?://\S+|\b[\w.\-]+@[\w.\-]+\b|(?:/[\w.\-]+){2,}")
NUM = re.compile(r"\b\d[\d,._]*\b")
QUOTED = re.compile(r"\"[^\"]{1,80}\"|'[^']{1,80}'|`[^`]{1,80}`")
# Capitalised spans of one or more words. Single tokens matter: "StellarPay" is one token and is
# exactly the kind of domain noun that has to go.
ENTITY = re.compile(r"\b(?:[A-Z][\w\-]*[a-z][\w\-]* ){0,4}[A-Z][\w\-]*[a-z][\w\-]*\b")
# Capitalised for grammar, not because they name anything.
NOT_ENTITY = {
    "i", "i'll", "i've", "i'm", "let", "the", "this", "that", "these", "those",
    "there", "here", "now", "next", "first", "second", "third", "then", "finally",
    "perfect", "great", "excellent", "good", "sure", "okay", "ok", "yes", "no",
    "based", "using", "after", "before", "since", "while", "once", "when", "if",
    "for", "from", "with", "without", "to", "and", "but", "so", "as", "it",
    "we", "you", "your", "my", "me", "all", "some", "both", "each", "every",
    "found", "got", "done", "note", "notes", "however", "although",
}
WS = re.compile(r"\s+")


@dataclass
class Segment:
    """One reasoning turn, with everything clustering or validation needs."""

    trace_id: str
    turn: int
    text: str                      # raw reasoning
    masked: str                    # deployment-independent form
    verbs: str = ""                # verb + particle only; filled by represent.py
    calls_after: list[str] = field(default_factory=list)
    n_calls_after: int = 0
    label: bool | None = None      # optional independent validator (see validate.py)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _entity_sub(m: re.Match) -> str:
    """Mask a capitalised span unless it is capitalised for grammatical reasons.

    Two exemptions. A span whose every word is a pronoun or discourse marker is never an entity.
    And a SINGLE capitalised word at the start of a sentence is usually a verb -- "Get ready",
    "Can you spot" -- so masking it deletes the operation, which is the one thing this pipeline
    must not do.
    """
    span = m.group(0)
    words = span.split()
    if all(w.lower().strip(".,:;!?") in NOT_ENTITY for w in words):
        return span
    if len(words) == 1:
        before = m.string[: m.start()].rstrip()
        if not before or before[-1] in ".!?:;\n":
            return span
    return " <ENTITY> "


def mask(text: str, tools: Iterable[str] = (), params: Iterable[str] = (),
         servers: Iterable[str] = ()) -> str:
    out = text
    for s in sorted(set(servers), key=len, reverse=True):
        if s:
            out = re.sub(re.escape(s), " <SERVER> ", out, flags=re.I)
    for t in sorted(set(tools), key=len, reverse=True):
        if t:
            out = re.sub(r"\b%s\b" % re.escape(t), " <TOOL> ", out, flags=re.I)
    for p in sorted(set(params), key=len, reverse=True):
        if p:
            out = re.sub(r"\b%s\b" % re.escape(p), " <PARAM> ", out, flags=re.I)
    out = URI.sub(" <URI> ", out)
    out = QUOTED.sub(" <STR> ", out)
    out = ENTITY.sub(_entity_sub, out)
    out = NUM.sub(" <NUM> ", out)
    return WS.sub(" ", out).strip()


def _jparse(v: Any, default: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return default
    return v if v is not None else default


def _calls_of(msg: dict) -> list[str]:
    """Tool-call names on an assistant message, in either OpenAI or plain form."""
    out = []
    for c in _jparse(msg.get("tool_calls"), []) or []:
        fn = c.get("function", c) if isinstance(c, dict) else {}
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            out.append(str(name))
    for c in msg.get("calls") or []:            # plain form: ["tool(a=1)", ...] or [{"name":...}]
        if isinstance(c, str):
            m = re.match(r"\s*([A-Za-z_][\w.]*)\s*\(", c)
            if m:
                out.append(m.group(1))
        elif isinstance(c, dict) and c.get("name"):
            out.append(str(c["name"]))
    return out


def vocab_of(trace: dict) -> tuple[list[str], list[str], list[str]]:
    """Tool, parameter and server names to mask out of this trace.

    Taken from the trace's own tool declarations plus every call it makes, so a trace that
    declares no tools still gets its called names masked.
    """
    tools: set[str] = set()
    params: set[str] = set()
    servers: set[str] = set(trace.get("servers") or [])
    for t in _jparse(trace.get("tools"), []) or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        if not isinstance(fn, dict):
            continue
        if fn.get("name"):
            tools.add(str(fn["name"]))
        props = ((fn.get("parameters") or {}).get("properties") or {})
        params.update(str(p) for p in props)
    for m in trace.get("messages") or []:
        tools.update(_calls_of(m))
        for c in _jparse(m.get("tool_calls"), []) or []:
            fn = c.get("function", c) if isinstance(c, dict) else {}
            args = _jparse(fn.get("arguments"), {}) if isinstance(fn, dict) else {}
            if isinstance(args, dict):
                params.update(str(k) for k in args)
    # Single characters and very short names produce catastrophic over-masking.
    return ([t for t in tools if len(t) > 2], [p for p in params if len(p) > 2],
            [s for s in servers if len(s) > 2])


def segments_of(trace: dict, field_name: str = "reasoning") -> Iterator[Segment]:
    """Yield one Segment per assistant turn that carries a reasoning block."""
    tools, params, servers = vocab_of(trace)
    tid = str(trace.get("id") or trace.get("trace_id") or "trace")
    msgs = trace.get("messages") or []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        text = m.get(field_name) or ""
        if isinstance(text, list):  # content-part form
            text = " ".join(p.get("text", "") or p.get("thinking", "")
                            for p in text if isinstance(p, dict))
        text = (text or "").strip()
        if len(text) < MIN_CHARS:
            continue
        # Calls introduced by this turn: on the message itself, else on the next assistant turn.
        calls = _calls_of(m)
        yield Segment(trace_id=tid, turn=i, text=text,
                      masked=mask(text, tools, params, servers),
                      calls_after=calls, n_calls_after=len(calls),
                      label=m.get("label") if isinstance(m.get("label"), bool) else None)


def load_traces(path: str | Path) -> list[dict]:
    """JSONL, one trace per line. See examples/traces.sample.jsonl for the schema."""
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_segments(traces: Iterable[dict], field_name: str = "reasoning") -> list[Segment]:
    return [s for t in traces for s in segments_of(t, field_name)]

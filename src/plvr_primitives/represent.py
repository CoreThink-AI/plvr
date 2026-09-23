"""Representations of a segment for clustering.

Which text you cluster decides what the clusters mean:

  masked  the whole reasoning turn with deployment-specific vocabulary removed. Default.
  verbs   verb + particle only ("pull up", "check", "fan out"). The sparsest representation
          that still names an operation; use it when `masked` clusters split by topic rather
          than by what the turn DOES.
  frames  spaCy predicate-argument frames, every verb in the sentence, not just the ROOT.
  raw     the unmasked turn. Clusters will follow the corpus's domain vocabulary; useful only
          as a control to show masking is doing something.

`verbs` and `frames` need the [frames] extra (spaCy + a model). Without it both fall back to
`masked` and say so once.
"""
from __future__ import annotations

import re
import warnings
from typing import Sequence

from .segment import Segment

_NLP = None
_WARNED = False
PARTICLES = {"up", "out", "down", "in", "off", "over", "through", "back", "into", "across"}


def _nlp():
    global _NLP
    if _NLP is None:
        import spacy
        try:
            _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except OSError as exc:
            raise RuntimeError(
                "spaCy model missing. Install it with:\n"
                "  python -m spacy download en_core_web_sm") from exc
    return _NLP


def _fallback(kind: str) -> None:
    global _WARNED
    if not _WARNED:
        warnings.warn(
            f"representation {kind!r} needs the [frames] extra (pip install 'plvr-primitives[frames]' "
            "and python -m spacy download en_core_web_sm); falling back to 'masked'.",
            RuntimeWarning, stacklevel=2)
        _WARNED = True


def verbs_of(texts: Sequence[str], batch: int = 256) -> list[str]:
    """Verb + particle per text. 'Let me pull up the invoices' -> 'pull up'."""
    try:
        nlp = _nlp()
    except Exception:
        _fallback("verbs")
        return list(texts)
    out = []
    for doc in nlp.pipe(list(texts), batch_size=batch):
        toks = []
        for t in doc:
            if t.pos_ not in ("VERB", "AUX"):
                continue
            part = next((c.text.lower() for c in t.children
                         if c.dep_ == "prt" or c.text.lower() in PARTICLES), "")
            toks.append(f"{t.text.lower()} {part}".strip())
        out.append(" ".join(toks))
    return out


def frames_of(texts: Sequence[str], batch: int = 256) -> list[str]:
    """Predicate-argument frames: every verb with its subject and object, not just the ROOT."""
    try:
        nlp = _nlp()
    except Exception:
        _fallback("frames")
        return list(texts)
    out = []
    for doc in nlp.pipe(list(texts), batch_size=batch):
        frames = []
        for t in doc:
            if t.pos_ != "VERB":
                continue
            subj = next((c.text.lower() for c in t.children if c.dep_ in ("nsubj", "nsubjpass")), "")
            obj = next((c.text.lower() for c in t.children
                        if c.dep_ in ("dobj", "obj", "attr", "pobj")), "")
            frames.append("_".join(x for x in (subj, t.text.lower(), obj) if x))
        out.append(" ".join(frames))
    return out


def annotate(segments: Sequence[Segment], kind: str = "verbs") -> list[Segment]:
    """Fill `segment.verbs` with the chosen representation, in place, and return the list."""
    if kind not in ("verbs", "frames"):
        raise ValueError(f"annotate takes 'verbs' or 'frames', not {kind!r}")
    texts = [s.masked for s in segments]
    vals = verbs_of(texts) if kind == "verbs" else frames_of(texts)
    for s, v in zip(segments, vals):
        s.verbs = re.sub(r"\s+", " ", v).strip()
    return list(segments)

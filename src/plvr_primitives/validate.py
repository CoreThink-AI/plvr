"""Is a cluster separating anything, or is it a slice of one blob?

Clustering always returns clusters. The only evidence that the partition means something is a
signal the embedding never saw. Two are supported:

  label    a per-segment boolean you supply (`label` on the assistant message). The strongest
           one available: if your traces have ground truth, mark whether the calls the turn
           introduced were right. A cluster whose members are reliably right -- or reliably
           wrong -- is separating something the prose embedding never saw.
  fanout   whether the turn preceded more than one call. Free, but only informative when it is
           RARE. If most turns plan a batch rather than issuing one call and iterating, the base
           rate approaches 50% and the maximum achievable lift is under 2x; `lift_report` says
           so rather than letting you read a meaningless number.

Lift is the cluster's rate over the corpus base rate. Both directions matter: 0.2 on a base of
0.7 is as much a finding as 1.4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .cluster import ClusterResult
from .segment import Segment

Validator = Literal["label", "fanout"]


@dataclass
class LiftReport:
    validator: str
    base_rate: float
    n_labelled: int
    rows: list[dict]
    warning: str | None = None

    def __str__(self) -> str:
        w = f"\n  WARNING: {self.warning}" if self.warning else ""
        head = (f"validator={self.validator}  base rate={self.base_rate:.3f} "
                f"over {self.n_labelled} segments{w}\n")
        body = "\n".join(
            f"  cluster {r['id']:>3}  n={r['size']:>4}  rate={r['label_rate']:.3f}  "
            f"lift={r['lift']:.2f}  {r['name']}" for r in self.rows)
        return head + body


def _values(segments: Sequence[Segment], validator: Validator) -> list[float | None]:
    if validator == "label":
        return [None if s.label is None else float(s.label) for s in segments]
    return [1.0 if s.n_calls_after > 1 else 0.0 for s in segments]


def lift_report(segments: Sequence[Segment], result: ClusterResult,
                validator: Validator = "label") -> LiftReport:
    """Per-cluster rate of the validator against the corpus base rate.

    Writes `lift`, `base_rate` and `label_rate` onto the clusters as a side effect so they are
    carried into the saved JSON and the generated prompts.
    """
    vals = _values(segments, validator)
    known = [v for v in vals if v is not None]
    if not known:
        return LiftReport(validator, 0.0, 0, [],
                          warning="no segment carries a label; supply `label` on assistant "
                                  "messages or use validator='fanout'")
    base = sum(known) / len(known)
    warning = None
    if validator == "fanout" and base > 0.25:
        warning = (f"fan-out base rate is {base:.1%}; each turn plans a batch rather than "
                   f"iterating, so the maximum achievable lift is ~{1/base:.2f}x. Report it for "
                   "comparability, but do not conclude from it -- use a `label` validator.")
    if len(known) < len(vals):
        warning = ((warning + " ") if warning else "") + \
                  f"{len(vals) - len(known)} of {len(vals)} segments are unlabelled and excluded."

    rows = []
    labels = result.labels
    for c in result.clusters:
        vs = [vals[i] for i, k in enumerate(labels) if k == c.id and vals[i] is not None]
        if not vs:
            continue
        rate = sum(vs) / len(vs)
        c.label_rate, c.base_rate = round(rate, 4), round(base, 4)
        c.lift = round(rate / base, 3) if base else None
        rows.append({"id": c.id, "name": c.name, "size": c.size, "n_labelled": len(vs),
                     "label_rate": rate, "lift": c.lift})
    rows.sort(key=lambda r: -(r["lift"] or 0))
    return LiftReport(validator, round(base, 4), len(known), rows, warning)

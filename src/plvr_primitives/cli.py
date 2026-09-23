"""plvr-primitives command line.

  plvr-primitives sweep    --traces t.jsonl                   # how many operations at each dial
  plvr-primitives mine     --traces t.jsonl --out out/        # clusters -> primitives + prompts
  plvr-primitives optimize --library out/library.json --trainset train.jsonl --out out/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import cluster as C
from . import prompts as P
from . import represent as R
from . import validate as V
from .segment import build_segments, load_traces


def _segments(a) -> list:
    traces = load_traces(a.traces)
    segs = build_segments(traces, field_name=a.field)
    if not segs:
        sys.exit(f"no segments: no assistant message in {a.traces} has a '{a.field}' block "
                 f"of at least 20 characters. Use --field reply to mine visible replies.")
    if a.repr in ("verbs", "frames"):
        R.annotate(segs, a.repr)
    print(f"{len(traces)} traces -> {len(segs)} segments", file=sys.stderr)
    return segs


def cmd_sweep(a) -> None:
    segs = _segments(a)
    field_name = "verbs" if a.repr in ("verbs", "frames") else "masked"
    rows = C.sweep(segs, sizes=a.sizes, repr_field=field_name,
                   min_samples=a.min_samples, dims=a.dims, seed=a.seed)
    print(f"{'selection':<10}{'min_size':>9}{'clusters':>10}{'noise':>8}{'noise%':>8}")
    for r in rows:
        print(f"{r['cluster_selection']:<10}{r['min_cluster_size']:>9}{r['n_clusters']:>10}"
              f"{r['n_noise']:>8}{r['noise_frac']*100:>7.1f}%")
    print("\nPick a setting, then check it survives a split-half rerun:\n"
          "  plvr-primitives mine --traces ... --min-size N --stability", file=sys.stderr)


def cmd_mine(a) -> None:
    segs = _segments(a)
    field_name = "verbs" if a.repr in ("verbs", "frames") else "masked"
    res = C.mine(segs, repr_field=field_name, min_size=a.min_size, min_samples=a.min_samples,
                 selection=a.selection, dims=a.dims, seed=a.seed, n_exemplars=a.exemplars)
    if not res.clusters:
        sys.exit(f"no clusters at min_cluster_size={a.min_size} ({res.n_noise} segments were "
                 "labelled noise). Lower --min-size, or run `sweep` first.")
    rep = V.lift_report(segs, res, validator=a.validator)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.save(out / "clusters.json")
    prims = P.build_library(res, min_size=a.min_cluster_report, extra_rules=a.rule or ())
    P.save_library(prims, out)
    (out / "lift.txt").write_text(str(rep), encoding="utf-8")

    print(f"\n{len(res.clusters)} clusters, {res.n_noise} noise "
          f"({res.n_noise/max(len(segs),1)*100:.1f}%)")
    print(rep)
    if a.stability:
        s = C.stability(segs, repr_field=field_name, min_size=a.min_size,
                        min_samples=a.min_samples, selection=a.selection, dims=a.dims, seed=a.seed)
        (out / "stability.txt").write_text(f"split_half_centroid_match={s}\n", encoding="utf-8")
        note = "" if s >= 0.8 else "  <-- below 0.8: these clusters are not reproducible"
        print(f"\nsplit-half stability: {s}{note}")
    print(f"\nwrote {out}/clusters.json, {out}/library.json, {out}/prompts/*.txt")
    print("These prompts are a starting point. Edit them, or run `optimize` with a training set.")


def cmd_optimize(a) -> None:
    from .gepa import LLM, load_trainset, optimize
    prims = P.load_library(a.library)
    if a.name:
        prims = [p for p in prims if p.name == a.name]
        if not prims:
            sys.exit(f"no primitive named {a.name!r} in {a.library}")
    rows = load_trainset(a.trainset)
    if len(rows) < 10:
        sys.exit(f"{len(rows)} training rows is too few to optimise against; "
                 "the holdout would be meaningless.")
    k = max(1, int(len(rows) * a.val_frac))
    val, train = rows[:k], rows[k:]
    print(f"{len(train)} train / {len(val)} held out", file=sys.stderr)
    llm = LLM(model=a.model, base_url=a.base_url)
    reflect = LLM(model=a.reflect_model or a.model, base_url=a.base_url)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for p in prims:
        print(f"\n=== optimising {p.name} ===", file=sys.stderr)
        best, pool = optimize(llm, p.prompt, train, val, reflect_llm=reflect,
                              iterations=a.iterations, minibatch=a.minibatch,
                              workers=a.workers, seed=a.seed,
                              log_path=out / f"optimize_{p.name}.jsonl")
        (out / "prompts").mkdir(parents=True, exist_ok=True)
        (out / "prompts" / f"{p.name}.optimized.txt").write_text(best, encoding="utf-8")
        results[p.name] = {"val": max((c.val or 0) for c in pool), "candidates": len(pool)}
    (out / "optimize_summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {out}/prompts/*.optimized.txt")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="plvr-primitives", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--traces", required=True, help="JSONL, one trace per line")
        p.add_argument("--field", default="reasoning",
                       help="assistant field to mine (default: reasoning; 'reply' for visible text)")
        p.add_argument("--repr", default="masked", choices=["masked", "verbs", "frames", "raw"])
        p.add_argument("--min-samples", type=int, default=10)
        p.add_argument("--dims", type=int, default=10)
        p.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("sweep", help="how many operations each granularity yields")
    common(s)
    s.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 40, 80])
    s.set_defaults(fn=cmd_sweep)

    m = sub.add_parser("mine", help="cluster, validate, and write primitive prompts")
    common(m)
    m.add_argument("--out", default="out")
    m.add_argument("--min-size", type=int, default=20, help="HDBSCAN min_cluster_size")
    m.add_argument("--selection", default="eom", choices=["eom", "leaf"])
    m.add_argument("--validator", default="label", choices=["label", "fanout"])
    m.add_argument("--exemplars", type=int, default=3)
    m.add_argument("--min-cluster-report", type=int, default=0,
                   help="drop clusters smaller than this from the library")
    m.add_argument("--rule", action="append", help="extra rule line for every prompt (repeatable)")
    m.add_argument("--stability", action="store_true", help="also run the split-half check")
    m.set_defaults(fn=cmd_mine)

    o = sub.add_parser("optimize", help="refine prompts against a training set (needs an API key)")
    o.add_argument("--library", required=True)
    o.add_argument("--trainset", required=True)
    o.add_argument("--out", default="out")
    o.add_argument("--name", help="optimise one primitive by name (default: all)")
    o.add_argument("--model", required=True, help="the model the prompt will run on")
    o.add_argument("--reflect-model", help="model that proposes revisions (default: --model)")
    o.add_argument("--base-url", default="https://api.openai.com/v1")
    o.add_argument("--iterations", type=int, default=10)
    o.add_argument("--minibatch", type=int, default=16)
    o.add_argument("--val-frac", type=float, default=0.2)
    o.add_argument("--workers", type=int, default=8)
    o.add_argument("--seed", type=int, default=0)
    o.set_defaults(fn=cmd_optimize)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

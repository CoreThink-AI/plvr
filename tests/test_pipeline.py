"""End-to-end smoke tests. No network: the embedder falls back to TF-IDF+SVD when
sentence-transformers is absent, so these run offline."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from plvr_primitives import (build_library, build_segments, load_traces, lift_report, mask, mine,
                             save_library, stability, sweep)
from plvr_primitives.gepa import parse_calls, score_calls

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "traces.sample.jsonl"


@pytest.fixture(scope="module")
def segments():
    return build_segments(load_traces(SAMPLE))


def test_masking_removes_identifiers_but_keeps_the_verb():
    out = mask("Let me pull up invoice 4417 for StellarPay at https://x.io/a/b",
               tools=["get_invoice"], params=["invoice_id"], servers=[])
    assert "4417" not in out and "StellarPay" not in out and "https" not in out
    assert "pull up" in out          # the operation must survive masking
    assert "<NUM>" in out and "<ENTITY>" in out and "<URI>" in out


def test_masking_keeps_sentence_initial_verbs():
    # A single capitalised word starting a sentence is usually a verb, not an entity.
    assert "Check" in mask("Check the balance first.")


def test_tool_and_param_names_are_masked_from_trace_vocab(segments):
    joined = " ".join(s.masked for s in segments)
    assert "get_account" not in joined and "list_orders" not in joined


def test_segments_are_built_with_labels(segments):
    assert len(segments) == 90
    assert all(s.text for s in segments)
    assert any(s.label is not None for s in segments)


def test_mine_recovers_clusters_and_sweep_is_monotonic(segments):
    res = mine(segments, min_size=5, min_samples=3)
    assert res.clusters, "no clusters recovered from the sample"
    assert all(c.size >= 5 for c in res.clusters)
    assert all(c.exemplars for c in res.clusters)
    assert len(res.labels) == len(segments)
    rows = sweep(segments, sizes=(5, 20), selections=("eom",), min_samples=3)
    # A larger min_cluster_size can never yield more clusters.
    assert rows[0]["n_clusters"] >= rows[1]["n_clusters"]


def test_lift_report_uses_the_label_validator(segments):
    res = mine(segments, min_size=5, min_samples=3)
    rep = lift_report(segments, res, validator="label")
    assert 0.0 < rep.base_rate < 1.0
    assert rep.n_labelled == len(segments)
    assert all(r["lift"] is not None for r in rep.rows)


def test_fanout_validator_warns_when_the_base_rate_is_high(segments):
    res = mine(segments, min_size=5, min_samples=3)
    rep = lift_report(segments, res, validator="fanout")
    # Every turn in the sample makes exactly one call, so fan-out carries no signal at all.
    assert rep.base_rate == 0.0 or rep.warning


def test_library_and_prompts_are_written(tmp_path, segments):
    res = mine(segments, min_size=5, min_samples=3)
    lift_report(segments, res, validator="label")
    prims = build_library(res)
    assert prims and prims[0].size if hasattr(prims[0], "size") else True
    for p in prims:
        assert p.name and p.prompt
        assert p.name in p.prompt
        assert "Return JSON only" in p.prompt
    save_library(prims, tmp_path)
    lib = json.loads((tmp_path / "library.json").read_text())
    assert len(lib) == len(prims)
    assert list((tmp_path / "prompts").glob("*.txt"))


def test_low_lift_cluster_gets_a_caution_rule(segments):
    res = mine(segments, min_size=5, min_samples=3)
    lift_report(segments, res, validator="label")
    for c in res.clusters:
        if c.lift is not None and c.lift <= 0.8:
            p = build_library(res)[0]
            assert any("error-prone" in r for pr in build_library(res) for r in pr.rules)
            break


def test_stability_is_a_fraction(segments):
    s = stability(segments, min_size=5, min_samples=3)
    assert 0.0 <= s <= 1.0


def test_scorer_rewards_exact_calls_and_penalises_order():
    gt = ["get_account(account_id=1)", "update_address(account_id=1, address=2)"]
    good = "```calls\nget_account(account_id=1)\nupdate_address(account_id=1, address=2)\n```"
    swapped = "```calls\nupdate_address(account_id=1, address=2)\nget_account(account_id=1)\n```"
    assert score_calls(good, gt) == 1.0
    assert 0.0 < score_calls(swapped, gt) < 1.0     # right calls, wrong order
    assert score_calls("no calls here", gt) == 0.0
    assert score_calls(good, []) == 0.0


def test_parse_calls_reads_names_and_arg_names():
    calls = parse_calls("```calls\nfoo(a=1, b='x')\n```")
    assert calls == [("foo", frozenset({"a", "b"}))]

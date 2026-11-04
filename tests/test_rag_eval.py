"""The measurement code itself: metric maths, gold-set integrity, retrieval regression gate, and the hallucination judge plumbing."""
import json

import pytest

from config import get_settings
from evals import rag_metrics as m
from evals import run_generation_eval as gen
from evals.run_rag_eval import evaluate, load_gold
from kb.store import KnowledgeBase


# ---------- retrieval metric maths ----------
def test_key_uses_file_stem():
    assert m.key("refund_policy.pdf", " Return window ") == "refund_policy#Return window"


def test_perfect_retrieval():
    s = m.score_retrieval(["a"], ["a"])
    assert (s.precision, s.p1, s.recall, s.hit, s.mrr, s.ndcg) == (1, 1, 1, 1, 1, 1)


def test_partial_retrieval_counts_precision_recall_and_rank():
    s = m.score_retrieval(["a", "x", "y"], ["a", "b"])
    assert s.precision == pytest.approx(1 / 3) and s.recall == 0.5 and s.hit == 1 and s.mrr == 1 and s.p1 == 1


def test_a_late_hit_lowers_mrr_and_ndcg_but_not_recall():
    s = m.score_retrieval(["x", "y", "a"], ["a"])
    assert s.recall == 1 and s.p1 == 0 and s.mrr == pytest.approx(1 / 3) and s.ndcg == pytest.approx(0.5)


def test_nothing_relevant_scores_zero_and_empty_results_do_not_divide_by_zero():
    for retrieved in (["x", "y"], []):
        s = m.score_retrieval(retrieved, ["a"])
        assert (s.precision, s.recall, s.hit, s.mrr, s.ndcg, s.p1) == (0, 0, 0, 0, 0, 0)


def test_only_the_top_k_count():
    s = m.score_retrieval(["x", "y", "z", "a"], ["a"], k=3)
    assert s.hit == 0 and s.recall == 0


def test_duplicate_retrieved_passages_do_not_inflate_recall():
    assert m.score_retrieval(["a", "a", "a"], ["a", "b"]).recall == 0.5


def test_aggregate_means_and_empty_input():
    agg = m.aggregate([m.score_retrieval(["a"], ["a"]), m.score_retrieval(["x"], ["a"])])
    assert agg["n"] == 2 and agg["hit"] == 0.5 and agg["recall"] == 0.5 and agg["mrr"] == 0.5
    assert m.aggregate([])["n"] == 0


def test_abstention():
    assert m.abstention([[], [], ["x"]]) == {"n": 3, "abstained": 2, "rate": pytest.approx(0.6667, abs=1e-4)}
    assert m.abstention([])["rate"] == 0.0


# ---------- hallucination and citation statistics ----------
def _c(text, supported, cited=()):
    return m.Claim(text, supported, list(cited))


def test_hallucination_rates():
    stats = m.hallucination_stats([[_c("a", True), _c("b", True)], [_c("c", True), _c("d", False)], []])
    assert stats["claims"] == 4 and stats["unsupported_claims"] == 1
    assert stats["claim_hallucination_rate"] == 0.25
    assert stats["response_hallucination_rate"] == 0.5  # 1 of the 2 replies that made claims (the empty reply is excluded)


def test_hallucination_rates_with_no_claims():
    assert m.hallucination_stats([[], []])["claim_hallucination_rate"] == 0.0


def test_citation_summary_measures_what_is_stored_not_just_inline_markers():
    def item(claims, stored, flagged=False, inline=()):
        return {"claims": claims, "stored_citations": stored, "citation_flagged": flagged, "inline_markers": list(inline)}

    s = gen._citation_summary([
        item([_c("a", True, [1]), _c("b", True, [])], "[1] a.pdf \u203a x", inline=[1]),
        item([_c("c", True, [1])], "", flagged=True),
        item([], ""),  # no factual claims (an honest "I do not know"): excluded
    ])
    assert s["replies_with_claims"] == 2 and s["with_citation"] == 0.5 and s["flagged"] == 1
    assert s["inline"] == 0.5 and s["claim_attributed"] == pytest.approx(2 / 3, abs=1e-3)


# ---------- gold set integrity ----------
@pytest.fixture
def kb():
    return KnowledgeBase(get_settings())


def test_gold_labels_all_point_at_real_sections(kb):
    """Guards against label drift: renaming a heading in a document must fail loudly here."""
    existing = {m.key(c.source, c.heading) for c in kb.chunks}
    missing = [(g["id"], r) for g in load_gold() for r in g["relevant"] if r not in existing]
    assert not missing, missing


def test_gold_set_shape():
    gold = load_gold()
    ids = [g["id"] for g in gold]
    assert len(ids) == len(set(ids)) and len({g["question"] for g in gold}) == len(gold)
    assert sum(1 for g in gold if g["relevant"]) >= 30 and sum(1 for g in gold if not g["relevant"]) >= 5
    assert all(isinstance(g["relevant"], list) and g["question"].strip() for g in gold)


def test_every_knowledge_base_document_is_covered_by_the_gold_set(kb):
    covered = {r.split("#")[0] for g in load_gold() for r in g["relevant"]}
    assert covered == {c.source.rsplit(".", 1)[0] for c in kb.chunks}


# ---------- retrieval regression gate (keyword mode, offline) ----------
def test_keyword_retrieval_has_not_regressed(kb):
    r = evaluate(kb, load_gold())
    ret = r["retrieval"]
    # measured baseline: hit 82%, recall 79%, P@1 68%, MRR 0.75; thresholds sit just below so a real regression fails
    assert ret["hit"] >= 0.80 and ret["recall"] >= 0.75 and ret["p1"] >= 0.65 and ret["mrr"] >= 0.70
    assert r["abstention"]["rate"] >= 0.5


# ---------- the judge ----------
GOOD = json.dumps({"claims": [{"text": "Returns within 30 days", "supported": True, "passages": [1]},
                              {"text": "Free returns on Sundays", "supported": False, "passages": []}]})


def test_parse_judge_json():
    claims = gen.parse_judge_json("Here you go:\n" + GOOD)
    assert [(c.text, c.supported, c.cited) for c in claims] == [("Returns within 30 days", True, [1]), ("Free returns on Sundays", False, [])]
    assert gen.parse_judge_json('{"claims": []}') == []


@pytest.mark.parametrize("bad", ["", "not json", '{"claims": "x"}', '{"claims": [{"text": "a"}]}', '{"claims": [{"text": "", "supported": true}]}',
                                 '{"claims": [{"text": "a", "supported": "yes"}]}'])
def test_malformed_judge_output_is_rejected(bad):
    with pytest.raises(ValueError):
        gen.parse_judge_json(bad)


def test_judge_retries_once_then_gives_up():
    calls = []

    def flaky(system, user):
        calls.append(1)
        return "garbage" if len(calls) == 1 else GOOD

    assert len(gen.judge([], "reply", flaky)) == 2 and len(calls) == 2

    with pytest.raises(ValueError, match="judge failed"):
        gen.judge([], "reply", lambda s, u: "garbage")


def test_judge_prompt_contains_the_passages_and_reply_and_is_strict():
    p = gen.build_judge_prompt([{"source": "refund_policy.pdf", "heading": "Return window", "passage": "30 days"}], "You have 30 days [1].")
    assert "[1] (refund_policy.pdf > Return window)" in p and "30 days" in p and "You have 30 days [1]." in p
    assert "no passages were retrieved" in gen.build_judge_prompt([], "x")
    assert "strict" in gen.JUDGE_SYSTEM.lower() and "supported=false" in gen.JUDGE_SYSTEM


def _ran(reply, hits=None, status="Resolved", citations="", reason=""):
    return {"hits": hits if hits is not None else [{"source": "refund_policy.pdf", "heading": "Return window", "passage": "Items can be returned within 30 days."}],
            "row": {"response": reply, "status": status, "escalation_reason": reason, "citations": citations}}


def test_evaluate_item_collects_claims_citations_and_figures():
    item = gen.evaluate_item({"id": "q1", "question": "How long?", "relevant": ["x#y"]}, _ran("Return within 30 days [1] or 90 days [4]."), lambda s, u: GOOD)
    assert item["inline_markers"] == [1, 4] and item["stored_citations"] == "" and not item["citation_flagged"]
    assert item["ungrounded_figures"] == ["90"]  # a figure that is in no passage
    assert len(item["claims"]) == 2 and not item["judge_error"]


def test_a_judge_failure_is_recorded_not_fatal():
    item = gen.evaluate_item({"id": "q1", "question": "q", "relevant": ["x#y"]}, _ran("A reply."), lambda s, u: "nope")
    assert item["judge_error"] and item["claims"] == []
    assert gen.summarise([item])["judge_errors"] == 1


def test_summary_and_report_for_a_mix_of_good_and_bad_replies():
    gold_a, gold_u = {"id": "q1", "question": "How long?", "relevant": ["x#y"]}, {"id": "u1", "question": "CEO?", "relevant": []}
    good = gen.evaluate_item(gold_a, _ran("30 days [1]."), lambda s, u: json.dumps({"claims": [{"text": "30 days", "supported": True, "passages": [1]}]}))
    invented = gen.evaluate_item(gold_u, _ran("The CEO is Jane.", hits=[], status="Escalated"), lambda s, u: json.dumps({"claims": [{"text": "CEO is Jane", "supported": False}]}))
    honest = gen.evaluate_item({"id": "u2", "question": "Stores?", "relevant": []}, _ran("I do not have that information.", hits=[], status="Escalated"),
                               lambda s, u: json.dumps({"claims": []}))
    summary = gen.summarise([good, invented, honest])
    assert summary["hallucination"]["claims"] == 2 and summary["hallucination"]["unsupported_claims"] == 1
    assert summary["unanswerable_correct"] == "1/2"  # the invented-fact reply is NOT counted as correct, even though it escalated
    assert summary["answerable_resolved"] == "1/1"
    report = gen.render(summary, [good, invented, honest], "m", "j", "Hybrid")
    assert "50.0%" in report and "CEO is Jane" in report and "Claim-level hallucination rate" in report

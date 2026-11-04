"""Answer-quality evaluation: citations and hallucination rate, measured on the labelled gold questions.

For every question the FULL pipeline runs (live model), then each reply is fact-checked:

  * claim-level judge: an LLM extracts every factual claim in the reply (policies, numbers, steps, contacts;
    greetings/apologies/hand-offs are ignored) and decides, strictly, whether the passages the agent actually
    retrieved support it
  * deterministic checks: citation markers that point at a real passage, factual figures absent from the passages

Reported:
  claim hallucination rate     unsupported claims / all claims
  response hallucination rate  replies with at least one unsupported claim / replies that made claims
  citation validity / coverage markers that resolve to a retrieved passage; claims that carry a citation
  unanswerable questions       must escalate AND must not state an unsupported fact

  python -m evals.run_generation_eval              # all 40 questions (about 80 model calls)
  python -m evals.run_generation_eval --limit 8    # quick smoke run

The judge is itself a language model (JUDGE_MODEL, default LLM_MODEL): treat results as a strong signal, not ground truth.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.rag_metrics import Claim, hallucination_stats  # noqa: E402
from evals.run_rag_eval import load_gold  # noqa: E402

JUDGE_SYSTEM = (
    "You are a strict fact-checker for a customer-support assistant. You receive numbered PASSAGES from the company's documents "
    "and a REPLY the assistant wrote. Extract every factual claim the REPLY makes about company policy, products, prices, fees, "
    "time limits, steps, procedures or contact details. Ignore greetings, apologies, sign-offs, offers of help, and statements that "
    "a human specialist will follow up. For each claim decide whether the PASSAGES explicitly support it. supported=false if the "
    "passages do not state it, contradict it, or the claim only follows by guessing. Be strict. "
    'Answer ONLY with JSON: {"claims": [{"text": "...", "supported": true, "passages": [1]}]}. '
    "If the reply makes no factual claims, answer {\"claims\": []}."
)


def build_judge_prompt(passages: list[dict], reply: str) -> str:
    numbered = "\n\n".join(f"[{i}] ({p['source']} > {p['heading']})\n{p['passage']}" for i, p in enumerate(passages, 1)) or "(no passages were retrieved)"
    return f"PASSAGES:\n{numbered}\n\nREPLY:\n<<<\n{reply}\n>>>"


def parse_judge_json(text: str) -> list[Claim]:
    """Parse the judge's JSON into Claim objects. Raises ValueError on anything malformed."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        raise ValueError("no JSON object in judge output")
    data = json.loads(m.group())
    claims = data.get("claims")
    if not isinstance(claims, list):
        raise ValueError("judge output has no 'claims' list")
    out = []
    for c in claims:
        if not isinstance(c, dict) or not isinstance(c.get("supported"), bool) or not str(c.get("text", "")).strip():
            raise ValueError("malformed claim")
        out.append(Claim(text=str(c["text"]).strip(), supported=c["supported"], cited=[int(n) for n in c.get("passages", []) if str(n).isdigit()]))
    return out


def judge(passages: list[dict], reply: str, call: Callable[[str, str], str], retries: int = 1) -> list[Claim]:
    """Fact-check one reply. `call(system, user) -> str` is the model call (injectable for tests)."""
    last: Exception | None = None
    for _ in range(retries + 1):
        try:
            return parse_judge_json(call(JUDGE_SYSTEM, build_judge_prompt(passages, reply)))
        except (ValueError, json.JSONDecodeError) as exc:
            last = exc
    raise ValueError(f"judge failed: {last}")


def openai_call(model: str, api_key: str, base_url: str) -> Callable[[str, str], str]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=60, max_retries=2)

    def call(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""

    return call


async def run_pipeline(question: str) -> dict:
    """Run one question through the real pipeline and collect what the judge needs."""
    from team import run_ticket

    hits, row = [], None
    async for ev in run_ticket(question):
        if ev.stage == "knowledge" and ev.kind == "done":
            hits = ev.data.get("hits", [])
        if ev.kind == "final":
            row = ev.data
    return {"hits": hits, "row": row}


def evaluate_item(gold: dict, ran: dict, call: Callable[[str, str], str]) -> dict:
    from guardrails.citations import markers
    from guardrails.output_guard import ungrounded_figures

    row, hits = ran["row"], ran["hits"]
    reply = row["response"] if row else ""
    cited = markers(reply)
    item = {
        "id": gold["id"], "question": gold["question"], "answerable": bool(gold["relevant"]),
        "status": row["status"] if row else "no ticket", "reason": row["escalation_reason"] if row else "",
        "reply": reply, "retrieved": len(hits), "inline_markers": cited,
        "stored_citations": row["citations"] if row else "",
        "citation_flagged": bool(row) and "citations" in row["escalation_reason"],
        "ungrounded_figures": ungrounded_figures(re.sub(r"\[\d{1,2}\]", "", reply), [h["passage"] for h in hits], gold["question"]) if hits else [],
        "claims": [], "judge_error": "",
    }
    if reply:
        try:
            item["claims"] = judge(hits, reply, call)
        except ValueError as exc:
            item["judge_error"] = str(exc)
    return item


def _citation_summary(ok: list[dict]) -> dict:
    """Citations as the system actually stores them (validated), not just inline [n] markers."""
    answered = [i for i in ok if i["claims"]]
    n = len(answered)
    return {
        "replies_with_claims": n,
        "with_citation": round(sum(1 for i in answered if i["stored_citations"]) / n, 4) if n else 0.0,
        "flagged": sum(1 for i in ok if i["citation_flagged"]),
        "inline": round(sum(1 for i in answered if i["inline_markers"]) / n, 4) if n else 0.0,
        "claim_attributed": round(sum(1 for i in answered for c in i["claims"] if c.cited) / max(1, sum(len(i["claims"]) for i in answered)), 4),
    }


def summarise(items: list[dict]) -> dict:
    ok = [i for i in items if not i["judge_error"]]
    answerable = [i for i in ok if i["answerable"]]
    unanswerable = [i for i in ok if not i["answerable"]]
    unans_good = [i for i in unanswerable if i["status"] == "Escalated" and all(c.supported for c in i["claims"])]
    return {
        "questions": len(items), "judge_errors": len(items) - len(ok),
        "hallucination": hallucination_stats([i["claims"] for i in ok]),
        "hallucination_answerable": hallucination_stats([i["claims"] for i in answerable]),
        "citations": _citation_summary(ok),
        "answerable_resolved": f"{sum(1 for i in answerable if i['status'] == 'Resolved')}/{len(answerable)}",
        "unanswerable_correct": f"{len(unans_good)}/{len(unanswerable)}",
        "figure_violations": sum(1 for i in ok if i["ungrounded_figures"]),
    }


def render(summary: dict, items: list[dict], model: str, judge_model: str, mode: str) -> str:
    h, ha, c = summary["hallucination"], summary["hallucination_answerable"], summary["citations"]
    lines = [
        "# Answer quality evaluation", "",
        f"- Answering model `{model}` · judge `{judge_model}` · retrieval: {mode} · {summary['questions']} questions"
        + (f" · **{summary['judge_errors']} judge failures excluded**" if summary["judge_errors"] else ""),
        "- The judge is an LLM, so this is a strong signal rather than ground truth. Gold questions: `evals/rag_gold.yaml`.", "",
        "| Metric | Value |", "|---|---|",
        f"| Claims checked | {h['claims']} across {h['replies']} replies |",
        f"| **Claim-level hallucination rate** (unsupported claims / all claims) | **{h['claim_hallucination_rate']:.1%}** ({h['unsupported_claims']} unsupported) |",
        f"| **Response-level hallucination rate** (replies with any unsupported claim) | **{h['response_hallucination_rate']:.1%}** |",
        f"| Claim-level rate, answerable questions only | {ha['claim_hallucination_rate']:.1%} |",
        f"| Replies that make claims and carry a validated citation | {c['with_citation']:.1%} of {c['replies_with_claims']} |",
        f"| Replies flagged for missing or invalid citations | {c['flagged']} |",
        f"| Replies that also use inline [n] markers | {c['inline']:.1%} |",
        f"| Claims the judge attributes to a cited passage | {c['claim_attributed']:.1%} |",
        f"| Answerable questions resolved without escalation | {summary['answerable_resolved']} |",
        f"| Unanswerable questions handled correctly (escalated, no invented fact) | {summary['unanswerable_correct']} |",
        f"| Replies with a figure that is not in the passages | {summary['figure_violations']} |", "",
    ]
    bad = [i for i in items if any(not cl.supported for cl in i["claims"])]
    if bad:
        lines += ["## Unsupported claims found", "", "| Id | Question | Unsupported claim |", "|---|---|---|"]
        for i in bad:
            for cl in i["claims"]:
                if not cl.supported:
                    lines.append(f"| {i['id']} | {i['question'][:50]} | {cl.text[:140].replace('|', '/')} |")
    esc = [i for i in items if i["answerable"] and i["status"] == "Escalated"]
    if esc:
        lines += ["", "## Answerable questions that were escalated, and why", "", "| Id | Question | Reason |", "|---|---|---|"]
        lines += [f"| {i['id']} | {i['question'][:50]} | {i['reason'][:110]} |" for i in esc]
    wrong = [i for i in items if not i["answerable"] and i["status"] != "Escalated"]
    if wrong:
        lines += ["", "## Unanswerable questions that were NOT escalated", ""] + [f"- {i['id']}: {i['question']} -> {i['status']}" for i in wrong]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--out", default=str(Path(__file__).parent / "rag_generation_report.md"))
    args = ap.parse_args()

    os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp(prefix="gendb_")) / "x.db")  # never touch the real database
    from config import get_settings

    s = get_settings()
    if not s.has_key:
        print("OPENAI_API_KEY is not set; this evaluation needs a live model.")
        return 2
    from kb.store import get_kb

    mode = get_kb(s).retrieval_label
    judge_model = os.getenv("JUDGE_MODEL", s.model)
    call = openai_call(judge_model, s.api_key, s.base_url)
    gold = load_gold()[: args.limit or None]
    items = []
    for g in gold:
        ran = asyncio.run(run_pipeline(g["question"]))
        items.append(evaluate_item(g, ran, call))
        print(f"  {g['id']}: {items[-1]['status']:9} claims={len(items[-1]['claims'])} unsupported={sum(1 for c in items[-1]['claims'] if not c.supported)}", flush=True)
    report = render(summarise(items), items, s.model, judge_model, mode)
    Path(args.out).write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

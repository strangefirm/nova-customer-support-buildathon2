"""Eval harness.

  python -m evals.run_evals            # live: real LLM end to end, writes evals/report.md
  python -m evals.run_evals --offline  # no API key: retrieval + deterministic escalation only

Checks per case: status, category, cited source, escalation reasons, security flags, language,
and groundedness (every figure in the reply must exist in the cited KB documents).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_cases() -> list[dict]:
    return yaml.safe_load((Path(__file__).parent / "cases.yaml").read_text(encoding="utf-8"))


def _judge(case: dict, row: dict, kb) -> list[str]:
    """Return a list of failure messages (empty = pass)."""
    from guardrails.output_guard import ungrounded_figures

    exp, fails = case["expect"], []
    if row["status"] != exp["status"]:
        fails.append(f"status {row['status']} != {exp['status']}")
    if "category" in exp and row["category"] not in exp["category"]:
        fails.append(f"category {row['category']} not in {exp['category']}")
    src = exp.get("source", "null_ok")
    if src is None and row["kb_sources"]:
        fails.append(f"expected no sources, got {row['kb_sources']}")
    elif src not in (None, "null_ok") and Path(src).stem not in row["kb_sources"]:
        fails.append(f"source {src} not cited (got '{row['kb_sources']}')")
    for r in exp.get("reasons", []):
        if r.lower() not in row["escalation_reason"].lower():
            fails.append(f"missing reason '{r}'")
    for f in exp.get("flags", []):
        if f not in row["security_flags"]:
            fails.append(f"missing flag '{f}'")
    if "language" in exp and exp["language"].lower() not in row["language"].lower():
        fails.append(f"language {row['language']} != {exp['language']}")
    cited = [c for c in kb.chunks if c.source in row["kb_sources"].split("; ")]
    if cited and row["response"]:
        bad = ungrounded_figures(row["response"], [c.text for c in cited], row["complaint"] + " " + row["order_id"])
        if bad:
            fails.append(f"ungrounded figures {bad}")
    return fails


async def _live(cases, settings, kb):
    from team import run_ticket

    results = []
    for c in cases:
        t0 = time.monotonic()
        row = None
        async for ev in run_ticket(c["complaint"], settings=settings):
            if ev.kind == "final":
                row = ev.data
        dt = time.monotonic() - t0
        results.append((c, row, _judge(c, row, kb) if row else ["no ticket produced"], dt))
    return results


def _offline(cases, settings, kb):
    """Deterministic subset: does retrieval find the right doc, and do the rules escalate what they should?"""
    from guardrails.escalation import decide
    from guardrails.input_guard import sanitize_complaint
    from models import QueryResult, ResolutionResult, TicketContext

    results = []
    for c in cases:
        exp = c["expect"]
        if "language" in exp:  # retrieval relies on the Knowledge Agent translating the query: LLM needed
            results.append((c, {"status": "skipped", "kb_sources": "", "escalation_reason": "needs LLM translation"}, [], 0.0))
            continue
        clean = sanitize_complaint(c["complaint"])
        hits = kb.search(c["complaint"])
        ctx = TicketContext(complaint=clean.text, security_flags=clean.flags, kb_searched=True, hits=hits)
        # Offline we have no LLM, so assume a calm, confident analysis; only code-enforced rules can fire.
        ctx.query = QueryResult(category="Other", urgency="Low", sentiment="Neutral", summary="-")
        ctx.resolution = ResolutionResult(reply="-", confidence=0.9)
        status, reasons = decide(ctx, settings.confidence_threshold)
        fails = []
        src = exp.get("source", "null_ok")
        if src is None and hits:
            fails.append(f"expected no match, got {hits[0].source}")
        elif src not in (None, "null_ok") and (not hits or Path(hits[0].source).stem != Path(src).stem):
            fails.append(f"top hit {hits[0].source if hits else None} != {src}")
        # LLM-dependent expectations (urgency/sentiment) are skipped; check code-enforced reasons only.
        code_reasons = {"Legal threat", "Payment dispute", "Possible prompt injection", "No matching knowledge-base document"}
        for r in exp.get("reasons", []):
            if r in code_reasons and r.lower() not in " + ".join(reasons).lower():
                fails.append(f"rule did not fire: {r}")
        if exp["status"] == "Resolved" and status == "Escalated":
            fails.append(f"false escalation: {reasons}")
        row = {"status": status, "category": "-", "kb_sources": hits[0].source if hits else "", "escalation_reason": " + ".join(reasons)}
        results.append((c, row, fails, 0.0))
    return results


def _report(results, mode: str, settings) -> str:
    skipped = sum(1 for _, row, _, _ in results if row and row["status"] == "skipped")
    passed = sum(1 for _, row, fails, _ in results if not fails and row and row["status"] != "skipped")
    lines = [
        "# Eval report", "",
        f"- Mode: **{mode}**  ·  Model: `{settings.model}`",
        f"- Result: **{passed}/{len(results) - skipped} passed**" + (f" ({skipped} skipped: need the LLM)" if skipped else ""), "",
        "| # | Complaint | Status | Source | Reasons | Result |", "|---|---|---|---|---|---|",
    ]
    for c, row, fails, dt in results:
        r = row or {}
        verdict = "✅" if not fails else "❌ " + "; ".join(fails)
        lines.append(
            f"| {c['id']} | {c['complaint'][:60].replace('|', '/')} | {r.get('status', '-')} | {r.get('kb_sources', '') or '-'} "
            f"| {(r.get('escalation_reason') or '-')[:70]} | {verdict} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="no LLM: retrieval and deterministic rules only")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="evals_")
    os.environ["DB_PATH"] = str(Path(tmp) / "evals.db")  # never pollute the real database
    from config import get_settings
    from kb.store import get_kb

    settings = get_settings()
    kb = get_kb(settings)
    cases = _load_cases()
    if args.offline:
        results, mode = _offline(cases, settings, kb), "offline (retrieval + rules)"
    else:
        if not settings.has_key:
            print("OPENAI_API_KEY not set. Use --offline, or add a key to .env.")
            return 2
        results, mode = asyncio.run(_live(cases, settings, kb)), "live (end to end)"
    report = _report(results, mode, settings)
    (Path(__file__).parent / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0 if all(not f for _, _, f, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

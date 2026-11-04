"""Latency check against your real model provider(s): runs sample complaints through the full pipeline and reports
seconds per agent stage and per ticket (p50 / p95), plus the triage call.

  python -m evals.latency              # 6 complaints, one after another
  python -m evals.latency --n 12       # more samples
  python -m evals.latency --parallel 3 # also time N tickets running at once (what a burst of mail or chat looks like)

Uses a throw-away database, so no tickets appear in your app. It makes real API calls (a few cents at most).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_tmp = tempfile.mkdtemp(prefix="latency_")
os.environ["DB_PATH"] = str(Path(_tmp) / "bench.db")  # must be set before config is read

import inbox  # noqa: E402
from config import get_settings  # noqa: E402
from team import run_ticket  # noqa: E402

COMPLAINTS = [
    "Where is my order? It was supposed to arrive yesterday.",
    "I want to return the shoes I bought last week. How do I get a refund?",
    "I forgot my password and can't log in.",
    "The blender I received is broken. Can I get a replacement?",
    "You charged my card three times! Fix this now or I'm contacting my bank and a lawyer.",
    "My package says delivered but I never got it, and I also want a refund for the damaged mug from my last order.",
]


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]


async def one_ticket(text: str) -> dict:
    """Seconds spent in each stage = time between consecutive finished-stage events."""
    stages: dict[str, float] = {}
    last = t0 = time.monotonic()
    status = ""
    async for ev in run_ticket(text, settings=get_settings(), owner="latency"):
        now = time.monotonic()
        if ev.kind == "done":
            stages[ev.stage] = stages.get(ev.stage, 0) + (now - last)
            last = now
        elif ev.kind == "final":
            status = ev.data.get("status", "")
    return {"total": time.monotonic() - t0, "stages": stages, "status": status}


async def triage_time(text: str) -> float:
    s = get_settings()
    t0 = time.monotonic()
    await inbox.classify_email(inbox.InboundMail("a@example.com", "", "Help", text, "<bench@x>"), s)
    return time.monotonic() - t0


async def main(n: int, parallel: int) -> None:
    s = get_settings()
    if not s.has_key:
        sys.exit("OPENAI_API_KEY is not set in .env")
    texts = [COMPLAINTS[i % len(COMPLAINTS)] for i in range(n)]
    fast = f"{getattr(s, 'fast_model', '')} (fast stages)" if getattr(s, "fast_enabled", False) else "same model for everything"
    print(f"main model: {s.model} | {fast} | tool_choice: {s.tool_choice}\n")
    tri = [await triage_time(t) for t in texts[:3]]
    print(f"triage call           p50 {statistics.median(tri):5.2f}s   (3 samples)")
    runs = [await one_ticket(t) for t in texts]
    names = ["query", "knowledge", "resolution", "log"]
    for name in names:
        v = [r["stages"].get(name, 0) for r in runs]
        print(f"{name:<21} p50 {statistics.median(v):5.2f}s   p95 {pct(v, .95):5.2f}s")
    tot = [r["total"] for r in runs]
    print(f"{'whole ticket':<21} p50 {statistics.median(tot):5.2f}s   p95 {pct(tot, .95):5.2f}s   ({n} sequential)")
    print("statuses:", {st: sum(1 for r in runs if r["status"] == st) for st in {r["status"] for r in runs}})
    if parallel > 1:
        t0 = time.monotonic()
        batch = await asyncio.gather(*(one_ticket(t) for t in texts[:parallel]))
        wall = time.monotonic() - t0
        print(f"\n{parallel} tickets at once: {wall:.1f}s wall clock (sequential would be about {sum(r['total'] for r in batch):.1f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=0)
    a = ap.parse_args()
    asyncio.run(main(a.n, a.parallel))

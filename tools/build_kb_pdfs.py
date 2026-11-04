"""Convert each knowledge_base/*.md into its own PDF in knowledge_base_pdf/.

    python tools/build_kb_pdfs.py

Headings are rendered in a larger bold font; kb/store.py relies on that to recover sections.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "knowledge_base", ROOT / "knowledge_base_pdf"

BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=15, alignment=TA_LEFT, spaceAfter=4)
H1 = ParagraphStyle("h1", parent=BODY, fontName="Helvetica-Bold", fontSize=20, leading=26, spaceAfter=10)
H2 = ParagraphStyle("h2", parent=BODY, fontName="Helvetica-Bold", fontSize=14, leading=19, spaceBefore=10, spaceAfter=4)


def _inline(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(text))


def _flush(items: list[str], ordered: bool, story: list) -> None:
    if items:
        story.append(
            ListFlowable(
                [ListItem(Paragraph(_inline(t), BODY), leftIndent=14) for t in items],
                bulletType="1" if ordered else "bullet",
                start=1 if ordered else "•",
                leftIndent=14,
            )
        )
        items.clear()


def build(md: Path, pdf: Path) -> None:
    story: list = []
    items: list[str] = []
    ordered = False
    for line in md.read_text(encoding="utf-8").splitlines():
        h = re.match(r"^(#{1,3})\s+(.*)$", line)
        num = re.match(r"^\d+\.\s+(.*)$", line)
        bul = re.match(r"^[-*]\s+(.*)$", line)
        if num or bul:
            if items and ordered != bool(num):
                _flush(items, ordered, story)
            ordered = bool(num)
            items.append((num or bul).group(1))
            continue
        _flush(items, ordered, story)
        if h:
            story.append(Paragraph(_inline(h.group(2)), H1 if len(h.group(1)) == 1 else H2))
        elif line.strip():
            story.append(Paragraph(_inline(line.strip()), BODY))
        else:
            story.append(Spacer(1, 2 * mm))
    _flush(items, ordered, story)
    title = md.stem.replace("_", " ").title()
    SimpleDocTemplate(
        str(pdf), pagesize=A4, title=title, author="NovaCart", subject="NovaCart support knowledge base",
        leftMargin=22 * mm, rightMargin=22 * mm, topMargin=20 * mm, bottomMargin=20 * mm,
    ).build(story)


def main() -> int:
    DST.mkdir(exist_ok=True)
    docs = sorted(SRC.glob("*.md"))
    if not docs:
        print(f"No .md files in {SRC}")
        return 1
    for md in docs:
        out = DST / f"{md.stem}.pdf"
        build(md, out)
        print(f"{md.name} -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

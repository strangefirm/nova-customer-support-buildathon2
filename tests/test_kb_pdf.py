"""The PDF knowledge base must behave like the Markdown one."""
from pathlib import Path

import pytest

from config import ROOT
from kb.store import KnowledgeBase, get_settings, pdf_to_markdown

PDF_DIR = ROOT / "knowledge_base_pdf"
MD_DIR = ROOT / "knowledge_base"
needs_md = pytest.mark.skipif(not MD_DIR.exists(), reason="Markdown knowledge base was removed; PDFs are the source of truth")


@pytest.fixture
def pdf_kb(monkeypatch):
    monkeypatch.setenv("KB_DIR", str(PDF_DIR))
    return KnowledgeBase(get_settings())


@needs_md
def test_one_pdf_per_markdown_document():
    md = {p.stem for p in MD_DIR.glob("*.md")}
    assert md and md == {p.stem for p in PDF_DIR.glob("*.pdf")}


def test_headings_recovered_from_font_size():
    text = pdf_to_markdown(PDF_DIR / "refund_policy.pdf")
    assert text.startswith("# NovaCart Refund and Return Policy") and "## Return window" in text
    assert "30 days" in text


@needs_md
def test_same_chunks_as_markdown(pdf_kb, monkeypatch):
    monkeypatch.setenv("KB_DIR", str(MD_DIR))
    assert len(pdf_kb.chunks) == len(KnowledgeBase(get_settings()).chunks)


@pytest.mark.parametrize("query,stem", [
    ("Where is my order? It was supposed to arrive yesterday.", "shipping_policy"),
    ("I want to return the shoes I bought last week. How do I get a refund?", "refund_policy"),
    ("I forgot my password and can't log in.", "account_and_login"),
    ("The blender I received is broken. Can I get a replacement?", "product_faq"),
])
def test_pdf_retrieval_finds_right_document(pdf_kb, query, stem):
    hits = pdf_kb.search(query)
    assert hits and Path(hits[0].source).stem == stem and hits[0].source.endswith(".pdf")


def test_pdf_out_of_scope_has_no_match(pdf_kb):
    assert pdf_kb.search("Do you sell gift cards for a company of 500 employees with custom branding?") == []

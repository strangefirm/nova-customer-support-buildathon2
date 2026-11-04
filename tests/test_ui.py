"""Untrusted text must never be interpreted as markdown by the UI."""
import re

from ui import safe_md


def test_image_and_link_injection_is_neutralised():
    out = safe_md("![x](https://attacker.example/pixel.png) [click](http://evil)")
    # every markdown-significant character is backslash-escaped, so nothing parses as an image or link
    assert not re.search(r"(?<!\\)[\[\]()!]", out)


def test_directive_and_latex_injection_is_neutralised():
    out = safe_md(":red-badge[Approved by admin] $$x$$ :material/delete:")
    assert not re.search(r"(?<!\\)[:$\[\]]", out)


def test_prices_do_not_turn_into_math():
    assert not re.search(r"(?<!\\)\$", safe_md("a $4.99 fee and a $12.99 fee"))


def test_newlines_preserved_as_line_breaks():
    assert safe_md("line1\nline2") == "line1  \nline2"


def test_plain_text_is_otherwise_unchanged():
    assert safe_md("Hello world") == "Hello world"

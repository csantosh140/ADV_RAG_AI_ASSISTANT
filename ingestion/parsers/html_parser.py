"""HTML parser for web pages and exported documentation."""

import re
from pathlib import Path
from html.parser import HTMLParser
from typing import List, Dict, Any, Tuple
from core.logger import logger
from core.exceptions import DocumentParsingError


class _HTMLTextExtractor(HTMLParser):
    """HTML Parser that separates headings and content blocks."""

    def __init__(self):
        super().__init__()
        self.sections: List[Dict[str, Any]] = []
        self.current_heading: str = "Overview"
        self.current_text: List[str] = []
        self._in_heading: bool = False
        self._skip_tag: bool = False
        self._page_num: int = 1

    def handle_starttag(self, tag, attrs):
        if tag.lower() in ["script", "style", "noscript", "svg"]:
            self._skip_tag = True
        elif tag.lower() in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            self._flush_section()
            self._in_heading = True

    def handle_endtag(self, tag):
        if tag.lower() in ["script", "style", "noscript", "svg"]:
            self._skip_tag = False
        elif tag.lower() in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            self._in_heading = False

    def handle_data(self, data):
        if self._skip_tag:
            return
        cleaned = data.strip()
        if not cleaned:
            return

        if self._in_heading:
            self.current_heading = cleaned
        else:
            self.current_text.append(cleaned)

    def _flush_section(self):
        body = " ".join(self.current_text).strip()
        if body or self.current_heading != "Overview":
            full_text = f"## {self.current_heading}\n{body}" if body else self.current_heading
            self.sections.append({
                "page_number": self._page_num,
                "text": full_text,
                "section_title": self.current_heading
            })
            self._page_num += 1
            self.current_text = []

    def close(self):
        super().close()
        self._flush_section()


def parse_html(file_path: str | Path) -> Tuple[List[Dict[str, Any]], int]:
    """
    Parse an HTML file into clean sections with headings.

    Returns:
        Tuple of (sections_data, total_sections)
    """
    path = Path(file_path)
    if not path.exists():
        raise DocumentParsingError(f"HTML file not found: {file_path}")

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw_html = f.read()
    except Exception as e:
        raise DocumentParsingError(f"Failed to read HTML file {path.name}: {str(e)}") from e

    parser = _HTMLTextExtractor()
    parser.feed(raw_html)
    parser.close()

    sections = parser.sections
    if not sections:
        # Fallback regex strip
        clean_txt = re.sub(r"<[^>]+>", " ", raw_html)
        clean_txt = " ".join(clean_txt.split())
        sections = [{
            "page_number": 1,
            "text": clean_txt,
            "section_title": path.stem
        }]

    logger.info(f"Parsed HTML file '{path.name}' into {len(sections)} sections")
    return sections, max(1, len(sections))

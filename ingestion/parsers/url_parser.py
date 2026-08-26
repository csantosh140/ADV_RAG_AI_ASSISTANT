"""Web URL parser fetching and transforming remote web pages into structured sections."""

import re
import urllib.parse
from typing import List, Dict, Any, Tuple
import requests
from html.parser import HTMLParser
from core.logger import logger
from core.exceptions import DocumentParsingError
from ingestion.parsers.html_parser import _HTMLTextExtractor


def parse_url(url: str, timeout: int = 15) -> Tuple[List[Dict[str, Any]], str, int]:
    """
    Fetch a remote webpage and parse it into structured sections with headers.

    Returns:
        Tuple of (sections_data, page_title, total_sections)
    """
    parsed_url = urllib.parse.urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise DocumentParsingError(f"Invalid URL provided: '{url}'. Must include http:// or https://")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        raise DocumentParsingError(f"Failed to fetch content from URL '{url}': {str(e)}") from e

    raw_html = response.text
    if not raw_html.strip():
        raise DocumentParsingError(f"URL '{url}' returned empty content.")

    # Extract title tag if present
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL)
    page_title = title_match.group(1).strip() if title_match else parsed_url.netloc

    parser = _HTMLTextExtractor()
    parser.feed(raw_html)
    parser.close()

    sections = parser.sections
    if not sections:
        clean_txt = re.sub(r"<[^>]+>", " ", raw_html)
        clean_txt = " ".join(clean_txt.split())
        sections = [{
            "page_number": 1,
            "text": clean_txt,
            "section_title": page_title
        }]

    # Prepend source URL context to first section
    if sections:
        sections[0]["text"] = f"[Source URL: {url}]\nTitle: {page_title}\n\n" + sections[0]["text"]

    logger.info(f"Parsed URL '{url}' ({page_title}) into {len(sections)} sections")
    return sections, page_title, max(1, len(sections))

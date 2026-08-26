"""Markdown parser extracting structured sections based on headers."""

import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from core.logger import logger
from core.exceptions import DocumentParsingError


def parse_markdown(file_path: str | Path) -> Tuple[List[Dict[str, Any]], int]:
    """
    Parse a Markdown file, splitting into logical sections based on headers.

    Returns:
        Tuple of (sections_data, total_sections)
    """
    path = Path(file_path)
    if not path.exists():
        raise DocumentParsingError(f"Markdown file not found: {file_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise DocumentParsingError(f"Failed to read Markdown file {path.name}: {str(e)}") from e

    # Regex to split on headings (# Heading)
    heading_pattern = re.compile(r"^(#{1,6}\s+.*)$", re.MULTILINE)
    splits = heading_pattern.split(content)

    sections: List[Dict[str, Any]] = []
    current_title = path.stem
    page_num = 1

    if not splits:
        sections.append({
            "page_number": page_num,
            "text": content.strip(),
            "section_title": current_title
        })
        return sections, 1

    i = 0
    while i < len(splits):
        chunk = splits[i].strip()
        if not chunk:
            i += 1
            continue

        if heading_pattern.match(chunk):
            current_title = chunk.lstrip("#").strip()
            # Next segment is the body of this section
            if i + 1 < len(splits):
                body = splits[i + 1].strip()
                full_text = f"{chunk}\n\n{body}" if body else chunk
                sections.append({
                    "page_number": page_num,
                    "text": full_text,
                    "section_title": current_title
                })
                page_num += 1
                i += 2
            else:
                sections.append({
                    "page_number": page_num,
                    "text": chunk,
                    "section_title": current_title
                })
                i += 1
        else:
            sections.append({
                "page_number": page_num,
                "text": chunk,
                "section_title": current_title
            })
            page_num += 1
            i += 1

    logger.info(f"Parsed Markdown file '{path.name}' into {len(sections)} sections")
    return sections, max(1, len(sections))

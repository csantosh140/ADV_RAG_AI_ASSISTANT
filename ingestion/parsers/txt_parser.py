"""Plaintext parser supporting multiple encodings."""

from pathlib import Path
from typing import List, Dict, Any, Tuple
from core.logger import logger
from core.exceptions import DocumentParsingError


def parse_txt(file_path: str | Path) -> Tuple[List[Dict[str, Any]], int]:
    """
    Parse a plaintext file.

    Returns:
        Tuple of (sections_data, total_units)
    """
    path = Path(file_path)
    if not path.exists():
        raise DocumentParsingError(f"File not found: {file_path}")

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    content: str = ""

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue

    if not content:
        raise DocumentParsingError(f"Unable to decode text file '{path.name}' with supported encodings.")

    logger.info(f"Parsed TXT file '{path.name}' ({len(content)} characters)")
    sections = [{
        "page_number": 1,
        "text": content.strip(),
        "section_title": path.stem
    }]
    return sections, 1

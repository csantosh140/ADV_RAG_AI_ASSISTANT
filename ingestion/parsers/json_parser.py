"""JSON parser extracting structured document sections."""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from core.logger import logger
from core.exceptions import DocumentParsingError


def _format_json_entry(key: str, value: Any, depth: int = 0) -> str:
    """Format a JSON key-value entry into readable structured text."""
    indent = "  " * depth
    if isinstance(value, dict):
        lines = [f"{indent}### {key}"]
        for k, v in value.items():
            lines.append(_format_json_entry(k, v, depth + 1))
        return "\n".join(lines)
    elif isinstance(value, list):
        lines = [f"{indent}- **{key}** ({len(value)} items):"]
        for idx, item in enumerate(value):
            if isinstance(item, dict):
                lines.append(f"{indent}  * Item {idx + 1}:")
                for ik, iv in item.items():
                    lines.append(f"{indent}    - {ik}: {iv}")
            else:
                lines.append(f"{indent}  * {item}")
        return "\n".join(lines)
    else:
        return f"{indent}- **{key}**: {value}"


def parse_json(file_path: str | Path) -> Tuple[List[Dict[str, Any]], int]:
    """
    Parse a JSON file into structured chunks/sections.

    Returns:
        Tuple of (sections_data, total_sections)
    """
    path = Path(file_path)
    if not path.exists():
        raise DocumentParsingError(f"JSON file not found: {file_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise DocumentParsingError(f"Failed to parse JSON file {path.name}: {str(e)}") from e

    sections: List[Dict[str, Any]] = []

    if isinstance(data, list):
        for idx, item in enumerate(data, start=1):
            if isinstance(item, dict):
                title = item.get("title") or item.get("name") or item.get("id") or f"Record {idx}"
                text_content = f"### {title}\n" + json.dumps(item, indent=2)
            else:
                title = f"Item {idx}"
                text_content = f"Item {idx}: {str(item)}"

            sections.append({
                "page_number": idx,
                "text": text_content,
                "section_title": str(title)
            })
    elif isinstance(data, dict):
        page_num = 1
        # Top-level keys become separate sections if they contain substantial data
        for key, value in data.items():
            section_text = _format_json_entry(key, value)
            sections.append({
                "page_number": page_num,
                "text": section_text,
                "section_title": str(key)
            })
            page_num += 1
    else:
        sections.append({
            "page_number": 1,
            "text": str(data),
            "section_title": path.stem
        })

    logger.info(f"Parsed JSON file '{path.name}' into {len(sections)} sections")
    return sections, max(1, len(sections))

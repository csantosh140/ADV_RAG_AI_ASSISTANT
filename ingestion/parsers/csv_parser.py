"""CSV and TSV tabular data parser converting rows into semantically rich structured sections."""

import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple
from core.logger import logger
from core.exceptions import DocumentParsingError


def parse_csv(file_path: str | Path, rows_per_section: int = 25) -> Tuple[List[Dict[str, Any]], int]:
    """
    Parse a CSV or TSV file, formatting rows with column headers into structured sections.

    Returns:
        Tuple of (sections_data, total_sections)
    """
    path = Path(file_path)
    if not path.exists():
        raise DocumentParsingError(f"CSV file not found: {file_path}")

    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","

    try:
        # Try reading with utf-8, fallback to latin-1
        try:
            with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                content = f.read()
        except Exception:
            with open(path, "r", encoding="latin-1") as f:
                content = f.read()

        lines = [line for line in content.splitlines() if line.strip()]
        if not lines:
            return [{
                "page_number": 1,
                "text": "Empty CSV dataset",
                "section_title": path.stem
            }], 1

        reader = csv.reader(lines, delimiter=delimiter)
        rows = list(reader)
        if not rows:
            return [{
                "page_number": 1,
                "text": "Empty CSV dataset",
                "section_title": path.stem
            }], 1

        headers = [h.strip() for h in rows[0]]
        data_rows = rows[1:]

        if not data_rows:
            # Only headers exist
            return [{
                "page_number": 1,
                "text": f"Table Headers: {', '.join(headers)} (No data rows)",
                "section_title": path.stem
            }], 1

        sections: List[Dict[str, Any]] = []
        section_index = 1

        for i in range(0, len(data_rows), rows_per_section):
            batch = data_rows[i:i + rows_per_section]
            record_lines = []
            for row_idx, row in enumerate(batch, start=i + 1):
                fields = []
                for h, val in zip(headers, row):
                    clean_val = val.strip()
                    if clean_val:
                        fields.append(f"{h}: {clean_val}")
                if fields:
                    record_lines.append(f"Record {row_idx}: {', '.join(fields)}")

            if record_lines:
                section_text = (
                    f"### Dataset: {path.name} (Rows {i + 1} to {min(i + len(batch), len(data_rows))} of {len(data_rows)})\n"
                    f"Columns: {', '.join(headers)}\n\n" +
                    "\n".join(record_lines)
                )
                sections.append({
                    "page_number": section_index,
                    "text": section_text,
                    "section_title": f"{path.stem} (Rows {i + 1}-{min(i + len(batch), len(data_rows))})"
                })
                section_index += 1

        if not sections:
            sections.append({
                "page_number": 1,
                "text": f"Dataset {path.name} with columns {', '.join(headers)}",
                "section_title": path.stem
            })

        logger.info(f"Parsed CSV '{path.name}' into {len(sections)} sections ({len(data_rows)} data rows)")
        return sections, max(1, len(sections))

    except Exception as e:
        raise DocumentParsingError(f"Failed to parse CSV file {path.name}: {str(e)}") from e

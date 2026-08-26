"""DOCX parser extracting structured sections and tables from Microsoft Word documents."""

from pathlib import Path
from typing import List, Dict, Any, Tuple
from core.logger import logger
from core.exceptions import DocumentParsingError

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


def parse_docx(file_path: str | Path) -> Tuple[List[Dict[str, Any]], int]:
    """
    Parse a .docx Word document, extracting headings, paragraphs, and tables into structured sections.

    Returns:
        Tuple of (sections_data, total_sections)
    """
    if not DOCX_AVAILABLE:
        raise DocumentParsingError("python-docx is not installed. Run 'pip install python-docx'.")

    path = Path(file_path)
    if not path.exists():
        raise DocumentParsingError(f"DOCX file not found: {file_path}")

    try:
        doc = docx.Document(path)
    except Exception as e:
        raise DocumentParsingError(f"Failed to open DOCX file {path.name}: {str(e)}") from e

    sections: List[Dict[str, Any]] = []
    current_title = path.stem
    current_paragraphs: List[str] = []
    section_index = 1

    def flush_section():
        nonlocal current_paragraphs, section_index, current_title
        if current_paragraphs:
            text = "\n\n".join(current_paragraphs).strip()
            if text:
                sections.append({
                    "page_number": section_index,
                    "text": text,
                    "section_title": current_title
                })
                section_index += 1
            current_paragraphs = []

    # Iterate over document body elements (paragraphs & tables)
    for element in doc.element.body:
        tag = element.tag
        if tag.endswith("p"):
            # Paragraph
            p = docx.text.paragraph.Paragraph(element, doc)
            text = p.text.strip()
            if not text:
                continue

            style_name = (p.style.name if p.style else "").lower()
            if "heading" in style_name or "title" in style_name:
                flush_section()
                current_title = text
                current_paragraphs.append(f"## {text}")
            else:
                current_paragraphs.append(text)

        elif tag.endswith("tbl"):
            # Table
            table = docx.table.Table(element, doc)
            table_rows: List[str] = []
            for row in table.rows:
                row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                # Avoid duplicate cell values from merged cells
                cleaned_cells = []
                for cell in row_cells:
                    if not cleaned_cells or cell != cleaned_cells[-1]:
                        cleaned_cells.append(cell)
                if any(cleaned_cells):
                    table_rows.append(" | ".join(cleaned_cells))

            if table_rows:
                table_text = "\n".join(table_rows)
                current_paragraphs.append(f"[Table Data]\n{table_text}")

    flush_section()

    if not sections:
        sections.append({
            "page_number": 1,
            "text": "Empty document",
            "section_title": current_title
        })

    logger.info(f"Parsed DOCX file '{path.name}' into {len(sections)} sections")
    return sections, max(1, len(sections))

"""PDF document parser extracting page-level text and metadata."""

from pathlib import Path
from typing import List, Dict, Any, Tuple
from core.logger import logger
from core.exceptions import DocumentParsingError

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


def parse_pdf(file_path: str | Path) -> Tuple[List[Dict[str, Any]], int]:
    """
    Parse a PDF file and extract text page-by-page.
    """
    path = Path(file_path)
    if not path.exists():
        raise DocumentParsingError(f"PDF file not found: {file_path}")

    if not PYPDF_AVAILABLE:
        logger.warning(f"pypdf library not found. Returning fallback text for '{path.name}'")
        return [{
            "page_number": 1,
            "text": f"Extracted content placeholder from {path.name} (Install pypdf for full parsing).",
            "section_title": path.stem
        }], 1

    pages_data: List[Dict[str, Any]] = []
    try:
        reader = PdfReader(str(path))
        total_pages = len(reader.pages)
        logger.info(f"Parsing PDF '{path.name}' ({total_pages} pages)")

        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            cleaned_text = "\n".join(
                line.strip() for line in text.splitlines() if line.strip()
            )

            if cleaned_text:
                pages_data.append({
                    "page_number": page_idx + 1,
                    "text": cleaned_text,
                    "section_title": f"Page {page_idx + 1}"
                })

        if not pages_data:
            pages_data.append({
                "page_number": 1,
                "text": f"Scanned or blank PDF content from {path.name}",
                "section_title": path.stem
            })

        return pages_data, total_pages
    except Exception as e:
        logger.error(f"Failed to parse PDF {path.name}: {str(e)}")
        raise DocumentParsingError(f"Error parsing PDF file {path.name}: {str(e)}") from e

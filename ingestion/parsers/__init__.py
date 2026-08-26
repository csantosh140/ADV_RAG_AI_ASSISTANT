"""Document parsers for PDF, TXT, Markdown, JSON, HTML, DOCX, CSV, and Web URLs."""

from ingestion.parsers.pdf_parser import parse_pdf
from ingestion.parsers.txt_parser import parse_txt
from ingestion.parsers.md_parser import parse_markdown
from ingestion.parsers.json_parser import parse_json
from ingestion.parsers.html_parser import parse_html
from ingestion.parsers.docx_parser import parse_docx
from ingestion.parsers.csv_parser import parse_csv
from ingestion.parsers.url_parser import parse_url

__all__ = [
    "parse_pdf",
    "parse_txt",
    "parse_markdown",
    "parse_json",
    "parse_html",
    "parse_docx",
    "parse_csv",
    "parse_url",
]


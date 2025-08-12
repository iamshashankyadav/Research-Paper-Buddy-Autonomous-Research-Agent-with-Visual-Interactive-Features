"""
utils/pdf_loader.py

Extracts text from PDFs and splits them into logical sections.
"""

import fitz  # PyMuPDF
import re
from typing import List, Dict


def extract_text(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file using PyMuPDF.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()


def split_into_sections(text: str) -> List[Dict[str, str]]:
    """
    Naively split text into sections by headings.
    Returns a list of dicts with 'heading' and 'text'.
    """
    # Regex for headings (lines in ALL CAPS or starting with numbers)
    pattern = r"(?m)^(?:[A-Z][A-Z\s]{2,}|[0-9]+\.\s+.*)$"
    matches = list(re.finditer(pattern, text))

    sections = []
    for i, match in enumerate(matches):
        heading = match.group().strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if heading and body:
            sections.append({"heading": heading, "text": body})
    return sections

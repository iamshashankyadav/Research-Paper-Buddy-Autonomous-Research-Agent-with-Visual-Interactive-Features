from crewai_tools import BaseTool
from pydantic import BaseModel
import fitz  # PyMuPDF
import re
from functools import lru_cache

# Input schema for the tool
class ParsePDFInput(BaseModel):
    file_path: str

class ParsePDFTool(BaseTool):
    name = "Parse Research PDF"
    description = "Reads a research paper PDF and extracts its metadata and sections (e.g. Title, Abstract, Introduction, Methods, etc.)"
    args_schema = ParsePDFInput
    result_as_answer = True  # Ensures tool output is used as-is

    @lru_cache(maxsize=8)
    async def _run(self, file_path: str) -> dict:
        doc = fitz.open(file_path)

        # Extract title and author from first page heuristically
        first_page_text = doc[0].get_text()
        lines = first_page_text.strip().split("\n")
        title = lines[0].strip() if lines else ""
        authors = lines[1].strip() if len(lines) > 1 else ""

        # Concatenate all page text
        full_text = "\n".join([page.get_text() for page in doc])

        # Define common section headers to extract
        section_titles = [
            "Abstract", "Introduction", "Background", "Related Work",
            "Methods", "Methodology", "Approach", "Experiments",
            "Results", "Findings", "Discussion", "Conclusion", "References"
        ]

        # Build regex to match headers (case-insensitive)
        pattern = r"(?i)(^({}))\s*\n".format("|".join(re.escape(title) for title in section_titles))
        matches = list(re.finditer(pattern, full_text, re.MULTILINE))

        # If no sections found, return the whole thing
        if not matches:
            return {
                "Title": title,
                "Authors": authors,
                "Full Text": full_text.strip()
            }

        # Otherwise split by detected headers
        sections = {}
        for i in range(len(matches)):
            start = matches[i].end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            header = matches[i].group(1).strip()
            body = full_text[start:end].strip()
            sections[header] = body

        return {
            "Title": title,
            "Authors": authors,
            "Sections": sections
        }
